"""
Tests for Lego 1 Gateway — batch ingest, search, stats, and webhook endpoints.
"""
import json
import os
import pytest


# ---------------------------------------------------------------------------
# Batch ingest
# ---------------------------------------------------------------------------
class TestBatchIngest:
    """Tests for /api/v1/batch-ingest endpoint."""

    def test_batch_ingest_no_files(self, client):
        """Empty batch should return 422 (no files provided)."""
        resp = client.post("/api/v1/batch-ingest")
        assert resp.status_code in (400, 422)

    def test_batch_ingest_unsupported_type(self, client, tmp_path):
        """Unsupported file extension should be flagged as rejected."""
        bad_file = tmp_path / "test.exe"
        bad_file.write_bytes(b"not a document")
        with open(bad_file, "rb") as f:
            resp = client.post(
                "/api/v1/batch-ingest",
                files=[("files", ("test.exe", f, "application/octet-stream"))],
            )
        assert resp.status_code == 202
        job = resp.json()["jobs"][0]
        assert job["status"] == "rejected"
        assert "Unsupported" in job["reason"]

    def test_batch_ingest_single_pdf(self, client, sample_pdf):
        """Single PDF should be queued successfully."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/batch-ingest",
                files=[("files", ("sample.pdf", f, "application/pdf"))],
            )
        assert resp.status_code == 202
        data = resp.json()
        assert data["batch_size"] == 1
        assert data["jobs"][0]["status"] == "queued"
        assert "job_id" in data["jobs"][0]

    def test_batch_ingest_too_many_files(self, client, tmp_path):
        """More than 20 files should be rejected with 400."""
        files = []
        for i in range(21):
            p = tmp_path / f"file{i}.pdf"
            p.write_bytes(b"%PDF-1.4 fake")
            files.append(("files", (f"file{i}.pdf", open(p, "rb"), "application/pdf")))
        resp = client.post("/api/v1/batch-ingest", files=files)
        for _, (_, fh, _) in files:
            fh.close()
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Batch status
# ---------------------------------------------------------------------------
class TestBatchStatus:
    def test_batch_status_unknown_ids(self, client):
        resp = client.get(
            "/api/v1/batch-status",
            params=[("job_ids", "abc123"), ("job_ids", "def456")],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert r["status"] == "processing"

    def test_batch_status_too_many(self, client):
        params = [("job_ids", f"id{i}") for i in range(51)]
        resp = client.get("/api/v1/batch-status", params=params)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class TestSearch:
    def test_search_requires_query(self, client):
        resp = client.get("/api/v1/search")
        assert resp.status_code == 422  # missing required param

    def test_search_empty_database(self, client):
        resp = client.get("/api/v1/search", params={"q": "invoice"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "invoice"
        assert isinstance(data["results"], list)
        assert data["total_found"] == 0

    def test_search_finds_committed_document(self, client, committed_doc):
        """If a committed doc contains the keyword, search should find it."""
        resp = client.get("/api/v1/search", params={"q": "GREENCARE_TEST_KEYWORD"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] >= 1
        job_ids = [r["job_id"] for r in data["results"]]
        assert committed_doc["job_id"] in job_ids

    def test_search_doc_type_filter(self, client, committed_doc):
        resp = client.get(
            "/api/v1/search",
            params={"q": "GREENCARE_TEST_KEYWORD", "doc_type": "invoice"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
class TestStats:
    def test_stats_returns_summary(self, client):
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "document_type_breakdown" in data
        assert "pipeline_breakdown" in data
        s = data["summary"]
        assert "pending" in s
        assert "committed" in s
        assert "rejected" in s
        assert "approval_rate_pct" in s


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------
class TestWebhooks:
    def test_register_webhook(self, client):
        resp = client.post(
            "/api/v1/webhook/register",
            json={"url": "https://example.com/hook", "description": "Test webhook"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "hook_id" in data
        assert data["status"] == "registered"
        return data["hook_id"]

    def test_list_webhooks(self, client):
        # Register one first
        client.post("/api/v1/webhook/register", json={"url": "https://test.example.com/hook"})
        resp = client.get("/api/v1/webhook/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert isinstance(data["webhooks"], list)

    def test_delete_webhook(self, client):
        # Register
        reg = client.post(
            "/api/v1/webhook/register",
            json={"url": "https://delete-me.example.com/hook"},
        )
        hook_id = reg.json()["hook_id"]
        # Delete
        resp = client.delete(f"/api/v1/webhook/{hook_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent_webhook(self, client):
        resp = client.delete("/api/v1/webhook/nonexistent-uuid")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "queue" in data
        assert "version" in data
