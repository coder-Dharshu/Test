"""
Smart Triage — AST Compiler (Docling Schema)
=============================================
Compiles outputs from Track A, Path 1, and Path 2 into a normalized,
unified Abstract Syntax Tree (AST) document conforming to the Docling schema
defined in §6 of the Developer Handover Specification.

Output Schema:
{
  "document_metadata": { "doc_id", "total_pages", "processed_timestamp" },
  "pages": [
    {
      "page_index": int,
      "execution_path": "TRACK_A" | "PATH_1" | "PATH_2",
      "pcs_score": float,
      "dimensions": { "width": float, "height": float },
      "elements": [
        {
          "element_id": str,
          "type": "paragraph" | "heading" | "table" | "key_value" | "graphic",
          "bbox": [x0, y0, x1, y1],   # normalized 0-1000
          "content": { ... }
        }
      ]
    }
  ]
}
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AST Compiler
# ---------------------------------------------------------------------------
class ASTCompiler:
    """
    Converts raw extraction dicts from any execution path into
    a unified, normalized Docling-schema AST document.
    """

    def compile(
        self,
        routing_result: dict[str, Any],
        source_filename: str = "unknown",
    ) -> dict[str, Any]:
        """
        Takes the full routing_result dict from SmartTriageOrchestrator.route_document()
        and produces a Docling-compliant AST document.
        """
        job_id         = routing_result.get("job_id", str(uuid.uuid4()))
        pages_raw      = routing_result.get("pages", [])
        execution_path = routing_result.get("execution_path", "PATH_2")
        pcs_score      = routing_result.get("pcs_score", 0.0)
        pipeline       = routing_result.get("_pipeline", "unknown")

        compiled_pages = []
        for idx, page_raw in enumerate(pages_raw):
            compiled_page = self._compile_page(
                page_raw       = page_raw,
                page_index     = idx,
                execution_path = str(execution_path),
                pcs_score      = pcs_score if idx == 0 else 0.0,
            )
            compiled_pages.append(compiled_page)

        ast_doc = {
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
            "pages": compiled_pages,
            # Legacy compatibility fields for the existing HITL dashboard
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

    # ── Internal page compiler ────────────────────────────────────────────────

    def _compile_page(
        self,
        page_raw      : dict[str, Any],
        page_index    : int,
        execution_path: str,
        pcs_score     : float,
    ) -> dict[str, Any]:
        elements: list[dict] = []

        # ── Extract text → single unified text element ──
        # Strip pdfplumber page-marker lines (e.g. "--- PAGE 1 ---") before display.
        import re as _re
        raw_text = page_raw.get("extracted_text", "") or ""
        coords   = page_raw.get("coordinates", [])

        # Remove "--- PAGE N ---" separator markers injected by pdfplumber
        cleaned_text = _re.sub(r"---\s*PAGE\s*\d+\s*---", "", raw_text).strip()

        if cleaned_text:
            # Emit one single element that preserves the full document text as-is.
            # No splitting into heading/paragraph blocks — the operator sees the
            # text exactly as it appears in the original document.
            elements.append({
                "element_id": f"p{page_index}_text_{str(uuid.uuid4())[:6]}",
                "type"      : "text",
                "bbox"      : [0.0, 0.0, 1000.0, 1000.0],
                "content"   : {"text": cleaned_text},
            })

        # ── Key-value pairs → key_value elements ──
        kv = page_raw.get("key_value_pairs", {}) or {}
        if kv:
            kv_items = [{"field": k, "value": v} for k, v in kv.items()]
            elements.append({
                "element_id": f"p{page_index}_kv_{str(uuid.uuid4())[:6]}",
                "type"      : "key_value",
                "bbox"      : [0, 0, 1000, 1000],
                "content"   : {"pairs": kv_items},
            })

        # ── Tables → table elements ──
        tables = page_raw.get("tables", []) or []
        for tbl_idx, tbl in enumerate(tables):
            headers = tbl.get("headers", [])
            rows    = tbl.get("rows", [])
            if rows:
                elements.append({
                    "element_id": f"p{page_index}_t{tbl_idx}_{str(uuid.uuid4())[:6]}",
                    "type"      : "table",
                    "bbox"      : [0, 0, 1000, 1000],
                    "content"   : {
                        "headers"    : headers,
                        "rows"       : rows,
                        "table_index": tbl.get("table_index", tbl_idx),
                    },
                })

        # ── Visual grounding → graphic elements ──
        grounding = page_raw.get("visual_grounding", []) or []
        for g_idx, g in enumerate(grounding):
            box   = g.get("box_2d", [0, 0, 100, 100])
            label = g.get("label", "graphic")
            elements.append({
                "element_id": f"p{page_index}_g{g_idx}_{str(uuid.uuid4())[:6]}",
                "type"      : "graphic",
                "bbox"      : self._normalize_bbox(box, 1000, 1000),
                "content"   : {"label": label},
            })

        # ── Confidence metadata ──
        confidence_warning        = page_raw.get("confidence_warning", False)
        confidence_warning_reason = page_raw.get("confidence_warning_reason")
        handwriting_detected      = page_raw.get("handwriting_detected", False)
        markdown_tables           = page_raw.get("markdown_tables", "") or ""
        document_type             = page_raw.get("document_type", "unknown")
        language                  = page_raw.get("language", "en")

        return {
            "page_index"   : page_index,
            "page_number"  : page_raw.get("page_number", page_index + 1),
            "execution_path": execution_path,
            "pcs_score"    : pcs_score,
            "document_type": document_type,
            "language"     : language,
            "dimensions"   : {"width": 1000.0, "height": 1000.0},
            "elements"     : elements,
            # Raw extraction fields preserved for backward compat + HITL editing
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_bbox_for_block(self, block: str, coords: list[dict]) -> list[float]:
        """
        Attempt to find a bounding box for a text block by matching first
        word in coordinate data. Falls back to full-page box [0,0,1000,1000].
        """
        if not coords or not block:
            return [0.0, 0.0, 1000.0, 1000.0]

        first_word = block.split()[0][:10].lower()
        for entry in coords:
            if entry.get("text", "").lower().startswith(first_word):
                x0 = entry.get("x0", 0)
                y0 = entry.get("y0", 0)
                x1 = entry.get("x1", x0 + 100)
                y1 = entry.get("y1", y0 + 20)
                return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
        return [0.0, 0.0, 1000.0, 1000.0]

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
            round(x0 / img_w * 1000, 1),
            round(y0 / img_h * 1000, 1),
            round(x1 / img_w * 1000, 1),
            round(y1 / img_h * 1000, 1),
        ]

    def get_flat_tables(self, ast_doc: dict) -> list[dict]:
        """
        Convenience: flatten all table elements across all pages.
        Returns list of {"page_index", "headers", "rows"} dicts.
        """
        tables = []
        for page in ast_doc.get("pages", []):
            for elem in page.get("elements", []):
                if elem["type"] == "table":
                    tables.append({
                        "page_index": page["page_index"],
                        **elem["content"],
                    })
        return tables

    def get_full_text(self, ast_doc: dict) -> str:
        """Concatenate all extracted text across all pages."""
        parts = []
        for page in ast_doc.get("pages", []):
            for elem in page.get("elements", []):
                if elem["type"] in ("paragraph", "heading"):
                    parts.append(elem["content"].get("text", ""))
        return "\n\n".join(parts)
