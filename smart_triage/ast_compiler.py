"""
Smart Triage — AST Compiler (Docling Schema)  [v2.1 Production Refactor]
========================================================================
Compiles outputs from Track A (Fast-Track), Path 1 (Local OCR), and
Path 2 (VLM Escalation) into a normalized, unified Abstract Syntax Tree (AST)
document conforming to the Docling schema specifications.

New in v2.1:
  - 'schema_version': "2.0.0" top-level identifier.
  - 'pipeline_metadata': records inference backend, model id, winning OCR strategy,
    token counts, repair attempts, and correlation request_id.
  - 'confidence' score (0.0 - 1.0) on every individual element.
  - Schema validation against schemas/ast_schema.json via jsonschema.
  - Comprehensive docstrings on all public methods.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "ast_schema.json"

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logger.warning("jsonschema not installed — schema validation will be skipped.")


class ASTCompiler:
    """
    Converts raw extraction dicts from any execution path into
    a unified, normalized Docling-schema AST document.
    """

    SCHEMA_VERSION = "2.0.0"

    def compile(
        self,
        routing_result: dict[str, Any],
        source_filename: str = "unknown",
        request_id: str = "",
    ) -> dict[str, Any]:
        """
        Takes the routing_result dict from SmartTriageOrchestrator and compiles
        it into a Docling-compliant AST document with schema_version and metadata.

        Args:
            routing_result: Dictionary containing pages, execution path, and metadata.
            source_filename: Name of the original input file.
            request_id: Unique correlation request ID for end-to-end tracing.

        Returns:
            Standardized AST document dictionary.
        """
        job_id         = routing_result.get("job_id", str(uuid.uuid4()))
        pages_raw      = routing_result.get("pages", [])
        execution_path = routing_result.get("execution_path", "PATH_2")
        pcs_score      = routing_result.get("pcs_score", 0.0)
        pipeline       = routing_result.get("_pipeline", "unknown")
        req_id         = request_id or routing_result.get("request_id") or f"req_{uuid.uuid4().hex[:8]}"

        # Pipeline metadata extraction
        backend        = routing_result.get("backend") or routing_result.get("_backend", "unknown")
        model_id       = routing_result.get("model_used") or routing_result.get("_model_used", "none")
        ocr_strategy   = routing_result.get("ocr_strategy") or routing_result.get("_ocr_strategy")
        vlm_tokens     = routing_result.get("vlm_tokens") or routing_result.get("_vlm_tokens", 0)
        val_attempts   = routing_result.get("validation_attempts", 1)

        # ── General Photo Short-Circuit ───────────────────────────────────────
        if pipeline == "general_photo":
            image_path     = routing_result.get("image_path", "")
            classification = routing_result.get("classification", {})
            detected_text  = ""
            if pages_raw and len(pages_raw) > 0:
                detected_text = pages_raw[0].get("extracted_text", "")

            logger.info("AST short-circuit: general_photo | %s | image=%s", job_id, image_path)
            ast_doc = {
                "schema_version": self.SCHEMA_VERSION,
                "document_metadata": {
                    "doc_id"              : job_id,
                    "source_filename"     : source_filename,
                    "total_pages"         : 1,
                    "processed_timestamp" : datetime.now(timezone.utc).isoformat(),
                    "pipeline"            : pipeline,
                    "execution_path"      : "GENERAL_PHOTO",
                    "pcs_score"           : 0.0,
                    "pcs_breakdown"       : None,
                },
                "pipeline_metadata": {
                    "backend"             : backend or "local_classifier",
                    "model_id"            : model_id or "image_router",
                    "ocr_strategy"        : ocr_strategy,
                    "vlm_tokens"          : vlm_tokens,
                    "validation_attempts" : val_attempts,
                    "request_id"          : req_id,
                },
                "pages": [{
                    "page_index"               : 0,
                    "page_number"              : 1,
                    "execution_path"           : "GENERAL_PHOTO",
                    "image_type"               : "general_image",
                    "image_path"               : image_path,
                    "_source_image"            : image_path,
                    "pcs_score"                : 0.0,
                    "document_type"            : "general_image",
                    "language"                 : "n/a",
                    "dimensions"               : {"width": 1000.0, "height": 1000.0},
                    "elements"                 : [
                        {
                            "element_id": "text_0",
                            "type": "text",
                            "bbox": [0.0, 0.0, 1000.0, 1000.0],
                            "confidence": 1.0,
                            "content": {"text": detected_text},
                        }
                    ] if detected_text else [],
                    "extracted_text"           : detected_text,
                    "key_value_pairs"          : {},
                    "tables"                   : [],
                    "handwriting_detected"     : False,
                    "confidence_warning"       : False,
                    "confidence_warning_reason": None,
                    "_ocr_avg_confidence"      : None,
                    "classification"           : classification,
                }],
                "_response_type" : "general_image",
                "_image_path"    : image_path,
                "_detected_text" : detected_text,
                "_classification": classification,
                "_job_id"        : job_id,
                "_filename"      : source_filename,
                "_pipeline"      : pipeline,
                "_timestamp"     : datetime.now(timezone.utc).isoformat(),
            }
            return ast_doc

        # ── Multi-Page Document Compilation ──────────────────────────────────
        compiled_pages = []
        for idx, page_raw in enumerate(pages_raw):
            compiled_page = self._compile_page(
                page_raw       = page_raw,
                page_index     = idx,
                execution_path = str(execution_path),
                pcs_score      = pcs_score if idx == 0 else 0.0,
            )
            compiled_pages.append(compiled_page)

        # If pages were empty (e.g., error fallback), construct default page
        if not compiled_pages:
            compiled_pages.append({
                "page_index": 0,
                "page_number": 1,
                "execution_path": str(execution_path),
                "pcs_score": pcs_score,
                "document_type": "unknown",
                "language": "en",
                "dimensions": {"width": 1000.0, "height": 1000.0},
                "elements": [],
                "extracted_text": "",
                "key_value_pairs": {},
                "tables": [],
                "images": [],
                "handwriting_detected": False,
                "confidence_warning": False,
                "confidence_warning_reason": None,
            })

        ast_doc = {
            "schema_version": self.SCHEMA_VERSION,
            "document_metadata": {
                "doc_id"              : job_id,
                "source_filename"     : source_filename,
                "total_pages"         : len(compiled_pages),
                "processed_timestamp" : datetime.now(timezone.utc).isoformat(),
                "pipeline"            : pipeline,
                "execution_path"      : str(execution_path),
                "pcs_score"           : pcs_score,
                "pcs_breakdown"       : routing_result.get("pcs_breakdown"),
            },
            "pipeline_metadata": {
                "backend"             : backend,
                "model_id"            : model_id,
                "ocr_strategy"        : ocr_strategy,
                "vlm_tokens"          : vlm_tokens,
                "validation_attempts" : val_attempts,
                "request_id"          : req_id,
            },
            "pages": compiled_pages,
            # Legacy compatibility fields for HITL dashboard
            "_job_id"   : job_id,
            "_filename" : source_filename,
            "_pipeline" : pipeline,
            "_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "AST compiled: %s | %d pages | path=%s | PCS=%.4f",
            job_id, len(compiled_pages), execution_path, pcs_score,
        )
        return ast_doc

    # ── Page Compilation ──────────────────────────────────────────────────────

    def _compile_page(
        self,
        page_raw      : dict[str, Any],
        page_index    : int,
        execution_path: str,
        pcs_score     : float,
    ) -> dict[str, Any]:
        """Compiles a single page extraction dictionary into normalized AST page structure."""
        elements: list[dict] = []

        # Derive element confidence based on execution path
        default_conf = 1.0
        if execution_path == "TRACK_A":
            default_conf = 1.0  # Vector text layer is 100% accurate
        elif execution_path == "PATH_1":
            ocr_conf = page_raw.get("_ocr_avg_confidence")
            default_conf = round(float(ocr_conf) / 100.0, 3) if ocr_conf is not None else 0.85
        elif execution_path == "PATH_2":
            default_conf = 0.95 if not page_raw.get("confidence_warning") else 0.65

        # ── Text Elements ──
        raw_text = page_raw.get("extracted_text", "") or ""
        cleaned_text = re.sub(r"---\s*PAGE\s*\d+\s*---", "", raw_text).strip()

        if cleaned_text:
            elements.append({
                "element_id": f"p{page_index}_text_{uuid.uuid4().hex[:6]}",
                "type"      : "text",
                "bbox"      : [0.0, 0.0, 1000.0, 1000.0],
                "confidence": default_conf,
                "content"   : {"text": cleaned_text},
            })

        # ── Key-Value Pairs ──
        kv = page_raw.get("key_value_pairs", {}) or {}
        if kv:
            kv_items = [{"field": str(k), "value": str(v)} for k, v in kv.items()]
            elements.append({
                "element_id": f"p{page_index}_kv_{uuid.uuid4().hex[:6]}",
                "type"      : "key_value",
                "bbox"      : [0.0, 0.0, 1000.0, 1000.0],
                "confidence": default_conf,
                "content"   : {"pairs": kv_items},
            })

        # ── Tables ──
        tables = page_raw.get("tables", []) or []
        for tbl_idx, tbl in enumerate(tables):
            headers = [str(h) for h in tbl.get("headers", [])]
            rows    = tbl.get("rows", [])
            if rows or headers:
                elements.append({
                    "element_id": f"p{page_index}_t{tbl_idx}_{uuid.uuid4().hex[:6]}",
                    "type"      : "table",
                    "bbox"      : [0.0, 0.0, 1000.0, 1000.0],
                    "confidence": default_conf,
                    "content"   : {
                        "headers"    : headers,
                        "rows"       : rows,
                        "table_index": tbl.get("table_index", tbl_idx),
                    },
                })

        # ── Visual Grounding / Graphics ──
        grounding = page_raw.get("visual_grounding", []) or []
        for g_idx, g in enumerate(grounding):
            box              = g.get("box_2d", [0.0, 0.0, 100.0, 100.0])
            label            = g.get("label", "graphic")
            signature_result = g.get("signature_result")
            content: dict    = {"label": label}
            if signature_result:
                content["signature_result"] = signature_result

            elements.append({
                "element_id": f"p{page_index}_g{g_idx}_{uuid.uuid4().hex[:6]}",
                "type"      : "graphic",
                "bbox"      : self._normalize_bbox(box, 1000.0, 1000.0),
                "confidence": default_conf,
                "content"   : content,
            })

        confidence_warning        = page_raw.get("confidence_warning", False)
        confidence_warning_reason = page_raw.get("confidence_warning_reason")
        handwriting_detected      = page_raw.get("handwriting_detected", False)
        markdown_tables           = page_raw.get("markdown_tables", "") or ""
        document_type             = page_raw.get("document_type", "unknown")
        language                  = page_raw.get("language", "en")
        images_out                = page_raw.get("images", [])

        return {
            "page_index"   : page_index,
            "page_number"  : page_raw.get("page_number", page_index + 1),
            "execution_path": execution_path,
            "pcs_score"    : pcs_score,
            "document_type": document_type,
            "language"     : language,
            "dimensions"   : {"width": 1000.0, "height": 1000.0},
            "elements"     : elements,
            "images"       : images_out,
            "extracted_text"             : raw_text,
            "markdown_tables"            : markdown_tables,
            "key_value_pairs"            : kv,
            "tables"                     : tables,
            "handwriting_detected"       : handwriting_detected,
            "confidence_warning"         : confidence_warning,
            "confidence_warning_reason"  : confidence_warning_reason,
            "_source_image"              : page_raw.get("_source_image"),
            "_ocr_avg_confidence"        : page_raw.get("_ocr_avg_confidence"),
        }

    # ── Helpers & Validation ─────────────────────────────────────────────────

    def validate_ast(self, ast_doc: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validates an AST document against the JSON Schema defined in schemas/ast_schema.json.

        Args:
            ast_doc: Compiled AST dictionary.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        if not JSONSCHEMA_AVAILABLE:
            return True, None

        if not SCHEMA_PATH.exists():
            return False, f"AST Schema file not found at {SCHEMA_PATH}"

        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema = json.load(f)
            jsonschema.validate(instance=ast_doc, schema=schema)
            return True, None
        except jsonschema.ValidationError as err:
            logger.error("AST validation failed: %s", err.message)
            return False, err.message
        except Exception as exc:
            logger.error("Schema error: %s", exc)
            return False, str(exc)

    def _normalize_bbox(
        self,
        box: list[float],
        img_w: float,
        img_h: float,
    ) -> list[float]:
        """Normalize pixel bbox to 0-1000 coordinate range."""
        if len(box) != 4:
            return [0.0, 0.0, 1000.0, 1000.0]
        x0, y0, x1, y1 = box
        return [
            round(float(x0) / img_w * 1000, 1),
            round(float(y0) / img_h * 1000, 1),
            round(float(x1) / img_w * 1000, 1),
            round(float(y1) / img_h * 1000, 1),
        ]

    def get_flat_tables(self, ast_doc: dict) -> list[dict]:
        """Flattens all table elements across all pages into a unified list."""
        tables = []
        for page in ast_doc.get("pages", []):
            for elem in page.get("elements", []):
                if elem.get("type") == "table":
                    tables.append({
                        "page_index": page.get("page_index", 0),
                        **elem.get("content", {}),
                    })
        return tables

    def get_full_text(self, ast_doc: dict) -> str:
        """Concatenates all extracted text blocks across all pages."""
        parts = []
        for page in ast_doc.get("pages", []):
            for elem in page.get("elements", []):
                if elem.get("type") in ("text", "paragraph", "heading"):
                    text = elem.get("content", {}).get("text", "")
                    if text:
                        parts.append(text)
        return "\n\n".join(parts)
