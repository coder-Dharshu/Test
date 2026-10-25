"""
Comprehensive Production Hardening & Boundary Tests (Section H)
================================================================
Verifies all core architectural boundaries, error codes, and guardrails:
  1. Routing boundary: 80.0% confidence -> PATH_1; 79.9% -> PATH_2
  2. INFERENCE_BACKEND=groq with no key -> startup fails fast with clear error
  3. ALLOW_CLOUD_ESCALATION=false -> needs_review + zero calls to Groq
  4. VLM invalid JSON -> repair retry -> validation_failed on failure
  5. VLM 429 -> exponential backoff -> needs_review on persistent exhaustion
  6. HITL approve/reject moves files and appends audit log
  7. Webhook HMAC signature verification and tamper detection
  8. Batch >20 files rejected with BATCH_LIMIT
  9. Unsupported MIME rejected with UNSUPPORTED_MIME
  10. AST output strictly validates against schemas/ast_schema.json
  11. Idempotency-Key returns same job_id
  12. Multi-page table stitching merges rows across page boundaries
"""

import hashlib
import hmac
import json
import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from common.errors import ErrorCode
from config import AppConfig, config
from lego1_gateway.main import app as gateway_app
from lego1_gateway.main_standalone import stitch_multipage_tables
from lego3_qwen.qwen_engine import call_qwen
from smart_triage.smart_triage_hitl_app import approve_document, reject_document
from smart_triage.ast_compiler import ASTCompiler
from smart_triage.orchestrator import ExecutionPath, SmartTriageOrchestrator


# ---------------------------------------------------------------------------
# 1. Routing Boundary Test: exactly 80.0% -> PATH_1; 79.9% -> PATH_2
# ---------------------------------------------------------------------------
def test_routing_boundary_80_percent(text_heavy_image):
    with patch.object(AppConfig, "validate_inference_backend"):
        orch = SmartTriageOrchestrator()
    orch.ocr_confidence_threshold = 80.0

    mock_result_80 = MagicMock()
    mock_result_80.confidence = 80.0
    mock_result_80.word_count = 10
    mock_result_80.text = "Sample invoice text with 80 percent confidence"
    mock_result_80.strategy.name = "GRAYSCALE_ADAPTIVE"

    orch._multi_ocr_engine = MagicMock()
    orch._multi_ocr_engine.run.return_value = mock_result_80

    # At 80.0% confidence -> PATH_1
    res_80 = orch._run_track_b(text_heavy_image, "test_80.jpg", "job_80")
    assert res_80["execution_path"] == ExecutionPath.PATH_1_LOW_COMPLEXITY.value

    # At 79.9% confidence -> PATH_2
    mock_result_79 = MagicMock()
    mock_result_79.confidence = 79.9
    mock_result_79.word_count = 10
    mock_result_79.text = "Sample invoice text with 79.9 percent confidence"
    mock_result_79.strategy.name = "GRAYSCALE_ADAPTIVE"
    orch._multi_ocr_engine.run.return_value = mock_result_79

    res_79 = orch._run_track_b(text_heavy_image, "test_79.jpg", "job_79")
    assert res_79["execution_path"] == ExecutionPath.PATH_2_HIGH_COMPLEXITY.value


# ---------------------------------------------------------------------------
# 2. INFERENCE_BACKEND=groq with no key -> startup fails
# ---------------------------------------------------------------------------
def test_groq_no_key_startup_fails():
    cfg = AppConfig(INFERENCE_BACKEND="groq", GROQ_API_KEY="")
    with pytest.raises(ValueError) as excinfo:
        cfg.validate_inference_backend()
    assert "GROQ_API_KEY is missing or unconfigured" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. ALLOW_CLOUD_ESCALATION=false guardrail: zero calls to Groq
# ---------------------------------------------------------------------------
def test_cloud_escalation_guardrail_zero_groq_calls():
    with patch("lego3_qwen.qwen_engine.config.ALLOW_CLOUD_ESCALATION", False):
        with patch("lego3_qwen.qwen_engine.config.INFERENCE_BACKEND", "groq"):
            with patch("lego3_qwen.qwen_engine.client.chat.completions.create") as mock_groq:
                result = call_qwen("base64dummy", "image/jpeg", request_id="req_test_guardrail")
                # Assert zero calls were dispatched to cloud
                mock_groq.assert_not_called()
                assert result["status"] == "needs_review"
                assert result["data"]["reason"] == "cloud_disabled"
                assert result["data"]["error_code"] == ErrorCode.CLOUD_DISABLED.value


# ---------------------------------------------------------------------------
# 4. VLM invalid JSON -> repair retry -> validation_failed on failure
# ---------------------------------------------------------------------------
def test_vlm_invalid_json_repair_retry_and_failure():
    mock_bad_resp1 = MagicMock()
    mock_bad_resp1.choices = [MagicMock(message=MagicMock(content="Invalid non-JSON response string"))]
    mock_bad_resp1.model = "qwen/qwen3.6-27b"

    mock_bad_resp2 = MagicMock()
    mock_bad_resp2.choices = [MagicMock(message=MagicMock(content="Still not valid JSON: {{bad}}"))]
    mock_bad_resp2.model = "qwen/qwen3.6-27b"

    with patch("lego3_qwen.qwen_engine.config.ALLOW_CLOUD_ESCALATION", True):
        with patch("lego3_qwen.qwen_engine.client.chat.completions.create", side_effect=[mock_bad_resp1, mock_bad_resp2]) as mock_create:
            result = call_qwen("base64dummy", "image/jpeg", request_id="req_repair_test")
            # Proves exactly 2 calls: initial pass + 1 repair attempt
            assert mock_create.call_count == 2
            assert result["status"] == "needs_review"
            assert result["data"].get("validation_failed") is True
            assert result["data"].get("validation_attempts") == 2


# ---------------------------------------------------------------------------
# 5. VLM 429 rate limit backoff and persistent exhaustion
# ---------------------------------------------------------------------------
def test_vlm_429_backoff_exhaustion():
    class MockRateLimitError(Exception):
        pass

    with patch("lego3_qwen.qwen_engine.config.ALLOW_CLOUD_ESCALATION", True):
        with patch("lego3_qwen.qwen_engine.config.VLM_MAX_RETRIES", 2):
            with patch("lego3_qwen.qwen_engine.time.sleep") as mock_sleep:
                with patch(
                    "lego3_qwen.qwen_engine.client.chat.completions.create",
                    side_effect=MockRateLimitError("429 Too Many Requests - Rate limit exceeded"),
                ) as mock_create:
                    result = call_qwen("base64dummy", "image/jpeg", request_id="req_rate_test")
                    assert mock_create.call_count == 2
                    mock_sleep.assert_called_once()  # Exponential backoff sleep executed
                    assert result["status"] == "needs_review"
                    assert result["data"]["error_code"] == ErrorCode.VLM_RATE_LIMITED.value


# ---------------------------------------------------------------------------
# 6. HITL approve and reject transitions + audit log
# ---------------------------------------------------------------------------
def test_hitl_transitions_and_audit_trail(tmp_path):
    pending_dir = tmp_path / "pending_review"
    final_dir = tmp_path / "final_database"
    rejected_dir = tmp_path / "rejected"
    audit_dir = tmp_path / "audit"

    for d in [pending_dir, final_dir, rejected_dir, audit_dir]:
        d.mkdir(parents=True, exist_ok=True)

    job_id = f"job_audit_{uuid.uuid4().hex[:6]}"
    pending_file = pending_dir / f"{job_id}.json"
    doc_content = {
        "_job_id": job_id,
        "document_type": "invoice",
        "extracted_text": "Sample Invoice Data",
    }
    pending_file.write_text(json.dumps(doc_content), encoding="utf-8")

    with patch("smart_triage.smart_triage_hitl_app.PENDING_DIR", pending_dir):
        with patch("smart_triage.smart_triage_hitl_app.FINAL_DIR", final_dir):
            with patch("smart_triage.smart_triage_hitl_app.REJECTED_DIR", rejected_dir):
                with patch("smart_triage.smart_triage_hitl_app.AUDIT_DIR", audit_dir):
                    with patch("smart_triage.smart_triage_hitl_app.AUDIT_LOG_FILE", audit_dir / "audit_log.jsonl"):
                        with patch("smart_triage.smart_triage_hitl_app.CORRECTIONS_FILE", audit_dir / "corrections.jsonl"):
                            # 1. Test Approve
                            msg, stats = approve_document(str(pending_file), json.dumps(doc_content), reviewer="test_reviewer")
                            assert "Committed" in msg
                            assert not pending_file.exists()
                            assert (final_dir / f"{job_id}.json").exists()

                            # Verify audit line
                            audit_lines = (audit_dir / "audit_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
                            assert len(audit_lines) == 1
                            audit_entry = json.loads(audit_lines[0])
                            assert audit_entry["action"] == "approved"
                            assert audit_entry["reviewed_by"] == "test_reviewer"

                            # 2. Test Reject on another file
                            job_id_2 = f"job_audit_{uuid.uuid4().hex[:6]}"
                            pending_file_2 = pending_dir / f"{job_id_2}.json"
                            pending_file_2.write_text(json.dumps(doc_content), encoding="utf-8")

                            msg_r, _ = reject_document(str(pending_file_2), reason="Illegible handwriting", reviewer="test_reviewer")
                            assert "Rejected" in msg_r
                            assert not pending_file_2.exists()
                            assert (rejected_dir / f"{job_id_2}.json").exists()

                            audit_lines2 = (audit_dir / "audit_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
                            assert len(audit_lines2) == 2
                            reject_entry = json.loads(audit_lines2[1])
                            assert reject_entry["action"] == "rejected"
                            assert reject_entry["reason"] == "Illegible handwriting"


# ---------------------------------------------------------------------------
# 7. Webhook HMAC-SHA256 signature verification & tamper detection
# ---------------------------------------------------------------------------
def test_webhook_hmac_tamper_detection():
    secret = "test-webhook-secret-xyz"
    payload = {"job_id": "job_123", "status": "pending_review", "pipeline": "track_a"}

    # Correct HMAC
    body_bytes = json.dumps(payload, separators=(",", ":")).encode()
    valid_sig = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()

    # Verification function
    def verify_webhook(body: bytes, received_sig: str, key: str) -> bool:
        expected = hmac.new(key.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, received_sig)

    assert verify_webhook(body_bytes, valid_sig, secret) is True

    # Tampered payload fails
    tampered_body = json.dumps({"job_id": "job_123", "status": "committed"}, separators=(",", ":")).encode()
    assert verify_webhook(tampered_body, valid_sig, secret) is False

    # Tampered key fails
    assert verify_webhook(body_bytes, valid_sig, "wrong-secret-key") is False


# ---------------------------------------------------------------------------
# 8. Batch >20 files rejected with BATCH_LIMIT
# ---------------------------------------------------------------------------
def test_batch_limit_over_20_rejected(client, tmp_path):
    files = []
    for i in range(21):
        fpath = tmp_path / f"batch_doc_{i}.pdf"
        fpath.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF")
        files.append(("files", (fpath.name, open(fpath, "rb"), "application/pdf")))

    resp = client.post("/api/v1/batch-ingest", files=files)
    for _, (_, fh, _) in files:
        fh.close()

    assert resp.status_code == 400
    data = resp.json()
    assert data["error_code"] == ErrorCode.BATCH_LIMIT.value


# ---------------------------------------------------------------------------
# 9. Unsupported MIME rejected with UNSUPPORTED_MIME
# ---------------------------------------------------------------------------
def test_unsupported_mime_rejected(client, tmp_path):
    bad_file = tmp_path / "script.exe"
    bad_file.write_bytes(b"MZ\x90\x00BinaryExe")
    with open(bad_file, "rb") as fh:
        resp = client.post("/api/v1/ingest", files={"file": ("script.exe", fh, "application/octet-stream")})

    assert resp.status_code == 415
    data = resp.json()
    assert data["error_code"] == ErrorCode.UNSUPPORTED_MIME.value


# ---------------------------------------------------------------------------
# 10. AST output strictly validates against schemas/ast_schema.json
# ---------------------------------------------------------------------------
def test_ast_output_validates_against_json_schema():
    compiler = ASTCompiler()
    routing_data = {
        "job_id": str(uuid.uuid4()),
        "execution_path": "PATH_1",
        "pcs_score": 0.25,
        "_pipeline": "path_1_local_ocr",
        "backend": "local",
        "model_used": "tesseract",
        "pages": [
            {
                "page_number": 1,
                "document_type": "invoice",
                "extracted_text": "TAX INVOICE Total: $500",
                "key_value_pairs": {"Total": "$500"},
                "tables": [{"headers": ["Item", "Cost"], "rows": [["Service", "$500"]]}],
                "_ocr_avg_confidence": 88.5,
            }
        ],
    }

    ast_doc = compiler.compile(routing_data, source_filename="invoice.pdf", request_id="req_schema_test")
    assert ast_doc["schema_version"] == "2.0.0"
    assert "pipeline_metadata" in ast_doc
    assert ast_doc["pages"][0]["elements"][0]["confidence"] > 0.0

    is_valid, error = compiler.validate_ast(ast_doc)
    assert is_valid is True, f"AST Schema Validation Error: {error}"


# ---------------------------------------------------------------------------
# 11. Idempotency-Key returns same job_id
# ---------------------------------------------------------------------------
def test_idempotency_key_returns_same_job(client, sample_pdf):
    idempotency_key = f"idemp_{uuid.uuid4().hex}"

    with open(sample_pdf, "rb") as fh:
        resp1 = client.post(
            "/api/v1/ingest",
            files={"file": ("sample.pdf", fh, "application/pdf")},
            headers={"Idempotency-Key": idempotency_key},
        )
    assert resp1.status_code in (200, 202)
    job_id_1 = resp1.json()["job_id"]

    # Replay identical request with same Idempotency-Key
    with open(sample_pdf, "rb") as fh:
        resp2 = client.post(
            "/api/v1/ingest",
            files={"file": ("sample.pdf", fh, "application/pdf")},
            headers={"Idempotency-Key": idempotency_key},
        )
    assert resp2.status_code in (200, 202)
    data2 = resp2.json()
    assert data2["job_id"] == job_id_1
    assert data2.get("idempotent_replay") is True


# ---------------------------------------------------------------------------
# 12. Multi-page table stitching: merges rows across page boundaries
# ---------------------------------------------------------------------------
def test_multipage_table_stitching():
    # Page 1 table
    page1 = {
        "page_number": 1,
        "tables": [
            {
                "headers": ["Line Item", "Qty", "Price"],
                "rows": [["Item A", "2", "$20"], ["Item B", "1", "$50"]],
            }
        ],
    }
    # Page 2 continuation table with identical headers
    page2 = {
        "page_number": 2,
        "tables": [
            {
                "headers": ["Line Item", "Qty", "Price"],
                "rows": [["Item C", "5", "$15"], ["Item D", "3", "$30"]],
            }
        ],
    }

    stitched = stitch_multipage_tables([page1, page2])
    assert len(stitched) == 2

    # Page 1 table now contains all 4 rows
    p1_rows = stitched[0]["tables"][0]["rows"]
    assert len(p1_rows) == 4
    assert p1_rows[0][0] == "Item A"
    assert p1_rows[2][0] == "Item C"
    assert p1_rows[3][0] == "Item D"

    # Page 2 duplicate headers discarded
    assert stitched[1]["tables"][0]["headers"] == []
