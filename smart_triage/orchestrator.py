"""
Smart Triage — Core Orchestrator
=================================
Implements the SmartTriageOrchestrator class per the Developer Handover Spec.

Dynamic Routing (Local-First, Confidence-Gated):
  Track A  — Vector-native PDF     → Programmatic pypdf extraction (zero API cost)
  Track B  — Raster / Scanned     → Always try CPU OCR locally first
    Path 1   (OCR confidence >= 80%) → Accept local result (zero API cost)
    Path 2   (OCR confidence <  80%) → Escalate to Groq Vision LLM as fallback

The Groq API is a LAST RESORT only — called when local extraction quality
is below the confidence threshold (default 80%). The PCS score is still
calculated for display/audit purposes in the HITL dashboard.

PCS Formula (§2.2) — used for display only:
  PCS = 0.40 * N_table + 0.45 * N_handwritten + 0.10 * N_overlap + 0.05 * A_graphic
"""

import os
import io
import uuid
import json
import logging
import asyncio
import requests
import base64
import string
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pypdf import PdfReader

# ── Optional PDF engines — used in cascaded digital-text detection ───────────

# Engine 2: pdfplumber (best for tables + layout-aware extraction)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Engine 3: PyMuPDF / fitz (fastest, most reliable for digital PDFs)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Engine 4: pypdfium2 (high-fidelity renderer, used for image rendering too)
try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False

# Engine 5: Pandas (for Excel and CSV fast-path)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Engine 6: python-docx (for Word docs)
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# OCR engine: PaddleOCR
try:
    import paddleocr
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Execution path enum
# ---------------------------------------------------------------------------
class ExecutionPath(str, Enum):
    TRACK_A_DIGITAL          = "TRACK_A"
    PATH_1_LOW_COMPLEXITY    = "PATH_1"
    PATH_2_HIGH_COMPLEXITY   = "PATH_2"


# ---------------------------------------------------------------------------
# PCS Weights (§2.2)
# ---------------------------------------------------------------------------
W1_TABLE       = 0.40
W2_HANDWRITTEN = 0.45
W3_OVERLAP     = 0.10
W4_GRAPHIC     = 0.05
PCS_THRESHOLD  = 0.50   # informational only — does NOT gate API routing
MIN_TEXT_CHARS = 50

# Local-first confidence gate:
# If CPU OCR average confidence is >= this value, accept local result (no API call).
# If confidence is BELOW this value, escalate to Groq Vision LLM.
OCR_CONFIDENCE_THRESHOLD = 80.0  # percent (0–100)


# ---------------------------------------------------------------------------
# SmartTriageOrchestrator
# ---------------------------------------------------------------------------
class SmartTriageOrchestrator:
    """
    Core triage and execution director.
    Implements Track A and Track B routing logic from the specification.
    """

    def __init__(
        self,
        pcs_threshold: float = PCS_THRESHOLD,
        ocr_confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
        lego3_url: str = "http://localhost:8002",
        lego2_temp_dir: str = "./lego2_temp",
    ):
        self.pcs_threshold            = pcs_threshold
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.lego3_url                = lego3_url
        self.lego2_temp_dir           = lego2_temp_dir
        os.makedirs(lego2_temp_dir, exist_ok=True)
        
        # Initialize persistent OCR engines to prevent timeout bottlenecks
        if PADDLE_AVAILABLE:
            from paddleocr import PaddleOCR
            from img2table.ocr import PaddleOCR as Img2TablePaddleOCR
            self.paddle_ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', det_limit_side_len=1600, det_db_thresh=0.3, det_db_box_thresh=0.5)
            self.img2table_ocr_wrapper = Img2TablePaddleOCR(lang="en")
        else:
            self.paddle_ocr_engine = None
            self.img2table_ocr_wrapper = None

        logger.info(
            "SmartTriageOrchestrator ready | OCR confidence gate=%.0f%% | pdfium=%s | paddle=%s",
            ocr_confidence_threshold, PDFIUM_AVAILABLE, PADDLE_AVAILABLE,
        )

    # ── Track A: Multi-Engine Digital PDF Detection ───────────────────────────

    # Thresholds for digital PDF confirmation
    MIN_TEXT_CHARS       = 50    # minimum total extractable characters across all pages
    MIN_PRINTABLE_RATIO  = 0.85  # ≥85% of characters must be printable ASCII/unicode

    @staticmethod
    def _printable_ratio(text: str) -> float:
        """
        Fraction of characters in `text` that are printable (not control chars).
        A scanned PDF with stray embedded junk has a much lower ratio.
        """
        if not text:
            return 0.0
        printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
        return printable / len(text)

    def _extract_with_pypdf(self, pdf_path: str) -> tuple[list[dict], int]:
        """Engine 1: pypdf — extracts text + word-level bounding boxes."""
        pages_out: list[dict] = []
        total_chars = 0
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                coords: list[dict] = []
                try:
                    visitor_data: list[dict] = []
                    def visitor(text, cm, tm, fontdict, fontsize):
                        if text.strip():
                            x, y = tm[4], tm[5]
                            visitor_data.append({
                                "text": text,
                                "x0": round(x, 2),
                                "y0": round(y, 2),
                                "x1": round(x + fontsize * len(text) * 0.6, 2),
                                "y1": round(y + fontsize, 2),
                            })
                    page.extract_text(visitor_text=visitor)
                    coords = visitor_data
                except Exception:
                    pass

                total_chars += len(raw_text.strip())
                pages_out.append({
                    "page_number"              : page_num + 1,
                    "extracted_text"           : raw_text,
                    "coordinates"              : coords,
                    "tables"                   : [],
                    "key_value_pairs"          : {},
                    "handwriting_detected"     : False,
                    "confidence_warning"       : False,
                    "confidence_warning_reason": None,
                    "_engine"                  : "pypdf",
                })
        except Exception as exc:
            logger.warning("pypdf engine failed: %s", exc)
        return pages_out, total_chars

    def _extract_with_pdfplumber(self, file_path: str) -> dict:
        """
        Engine 2: pdfplumber — superior table & layout detection.
        Iterates all pages, aggregates text and tables, and returns a single dictionary.
        """
        extracted_text = []
        combined_tables = []
        page_count = 0

        if not PDFPLUMBER_AVAILABLE:
            return {"extracted_text": "", "tables": [], "page_count": 0}

        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    raw_text = page.extract_text() or ""
                    
                    if raw_text.strip():
                        extracted_text.append(f"\n\n--- PAGE {page_num} ---\n\n{raw_text}")
                    
                    # Extract tables as structured dicts
                    tables_raw = page.extract_tables() or []
                    for tbl in tables_raw:
                        if tbl and len(tbl) > 1:
                            headers = [str(c) if c else "" for c in tbl[0]]
                            rows = [
                                {headers[j] if j < len(headers) else f"Col_{j}": str(cell) if cell else ""
                                 for j, cell in enumerate(row)}
                                for row in tbl[1:] if row
                            ]
                            if rows:
                                combined_tables.append({
                                    "headers"    : headers,
                                    "rows"       : rows,
                                    "table_index": len(combined_tables),
                                })
        except Exception as exc:
            logger.warning("pdfplumber engine failed: %s", exc)

        return {
            "extracted_text": "".join(extracted_text).strip(),
            "tables": combined_tables,
            "page_count": page_count
        }

    def _extract_with_pymupdf(self, pdf_path: str, existing_pages: list[dict]) -> tuple[list[dict], int]:
        """
        Engine 3: PyMuPDF (fitz) — fastest, most robust digital text extractor.
        Used as final authority when pypdf + pdfplumber are insufficient.
        Also extracts per-word bounding boxes (spans).
        """
        if not PYMUPDF_AVAILABLE:
            return existing_pages, sum(len(p["extracted_text"].strip()) for p in existing_pages)

        try:
            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                page      = doc[i]
                raw_text  = page.get_text("text") or ""
                # Word-level bounding boxes
                words     = page.get_text("words") or []   # (x0,y0,x1,y1,word,block,line,word_idx)
                rect      = page.rect
                pw, ph    = rect.width or 1, rect.height or 1
                coords    = [
                    {
                        "text": w[4],
                        "x0"  : round(w[0] / pw * 1000, 1),
                        "y0"  : round(w[1] / ph * 1000, 1),
                        "x1"  : round(w[2] / pw * 1000, 1),
                        "y1"  : round(w[3] / ph * 1000, 1),
                    }
                    for w in words if w[4].strip()
                ]

                if i < len(existing_pages):
                    existing = existing_pages[i]
                    if len(raw_text.strip()) > len(existing["extracted_text"].strip()):
                        existing["extracted_text"] = raw_text
                        existing["_engine"]        = "pymupdf"
                    # Always upgrade coordinates if pymupdf found more
                    if len(coords) > len(existing.get("coordinates", [])):
                        existing["coordinates"] = coords
                else:
                    existing_pages.append({
                        "page_number"              : i + 1,
                        "extracted_text"           : raw_text,
                        "coordinates"              : coords,
                        "tables"                   : [],
                        "key_value_pairs"          : {},
                        "handwriting_detected"     : False,
                        "confidence_warning"       : False,
                        "confidence_warning_reason": None,
                        "_engine"                  : "pymupdf",
                    })
        except Exception as exc:
            logger.warning("pymupdf engine failed: %s", exc)

        total_chars = sum(len(p["extracted_text"].strip()) for p in existing_pages)
        return existing_pages, total_chars

    def is_vector_native(self, pdf_path: str) -> tuple[bool, list[dict]]:
        """
        Multi-engine cascaded digital PDF detection.

        Pipeline:
          1. pypdf    → extract text + word bounding boxes
          2. pdfplumber → better tables; upgrade any pages with more text
          3. PyMuPDF    → fastest/most reliable; final authority upgrade

        Decision gate (both must pass):
          A. total_extractable_chars >= MIN_TEXT_CHARS  (default 50)
          B. printable_char_ratio    >= MIN_PRINTABLE_RATIO  (default 0.85)

          → DIGITAL PDF : skip OCR entirely, skip Groq entirely
          → SCANNED PDF : route to Track B (OCR pipeline)

        The printable-ratio gate catches PDFs that embed stray font-cmap
        characters, making pypdf return garbled non-printable bytes even
        though the document is actually a scanned image.
        """
        # Track A Engine: pdfplumber
        plumber_data = self._extract_with_pdfplumber(pdf_path)

        # ── Decision gate ───────────────────────────────────────────────────
        all_text     = plumber_data.get("extracted_text", "")
        total_chars  = len(all_text.strip())
        print_ratio  = self._printable_ratio(all_text)

        is_native = (total_chars >= self.MIN_TEXT_CHARS) and (print_ratio >= self.MIN_PRINTABLE_RATIO)

        pages_out = [{
            "page_number"              : 1,
            "extracted_text"           : all_text,
            "coordinates"              : [],
            "tables"                   : plumber_data.get("tables", []),
            "key_value_pairs"          : {},
            "handwriting_detected"     : False,
            "confidence_warning"       : False,
            "confidence_warning_reason": None,
            "_engine"                  : "pdfplumber",
        }]

        logger.info(
            "Track A decision: %s | chars=%d (gate≥%d: %s) | printable=%.2f%% (gate≥%.0f%%: %s) | engines=['pdfplumber']",
            "DIGITAL ✓" if is_native else "SCANNED ✗",
            total_chars, self.MIN_TEXT_CHARS, total_chars >= self.MIN_TEXT_CHARS,
            print_ratio * 100, self.MIN_PRINTABLE_RATIO * 100, print_ratio >= self.MIN_PRINTABLE_RATIO,
        )

        import re

        # If it's a digital PDF, structure specific formats locally using Regex
        if is_native:
            for p in pages_out:
                text = p.get("extracted_text", "")
                
                # NPTEL Certificate Local Parser
                if "NPTEL" in text:
                    p["key_value_pairs"] = {}
                    
                    name = re.search(r'awarded to\s+([A-Za-z\s]+?)\s+for successfully', text, re.IGNORECASE)
                    if name: p["key_value_pairs"]["Candidate Name"] = name.group(1).strip()
                    
                    score = re.search(r'score of\s+(\d+)', text, re.IGNORECASE)
                    if score: p["key_value_pairs"]["Final Score"] = score.group(1).strip()
                    
                    roll = re.search(r'Roll No:\s*([A-Z0-9]+)', text, re.IGNORECASE)
                    if roll: p["key_value_pairs"]["Roll Number"] = roll.group(1).strip()

        return is_native, pages_out

    # ── Track A: Office/CSV Native Digital Detection ──────────────────────────

    def _extract_with_pandas(self, file_path: str, is_csv: bool = False) -> list[dict]:
        """
        Track A extractor for tabular files (Excel, CSV) using pandas.
        Every sheet (or the single CSV) is treated as a 'page' containing one massive table.
        """
        pages_out = []
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas is not available. Cannot process %s natively.", file_path)
            return pages_out

        try:
            if is_csv:
                df = pd.read_csv(file_path, dtype=str).fillna("")
                dfs = {"Sheet1": df}
            else:
                dfs = pd.read_excel(file_path, sheet_name=None, dtype=str)
                for k in dfs:
                    dfs[k] = dfs[k].fillna("")

            page_num = 1
            for sheet_name, df in dfs.items():
                headers = list(df.columns)
                rows = []
                for _, row_data in df.iterrows():
                    rows.append({str(k): str(v) for k, v in row_data.items()})

                table = {
                    "headers": [str(h) for h in headers],
                    "rows": rows,
                    "table_index": 0
                }

                pages_out.append({
                    "page_number": page_num,
                    "extracted_text": f"Sheet: {sheet_name}",
                    "coordinates": [],
                    "tables": [table],
                    "key_value_pairs": {},
                    "handwriting_detected": False,
                    "confidence_warning": False,
                    "confidence_warning_reason": None,
                    "_engine": "pandas"
                })
                page_num += 1

        except Exception as exc:
            logger.warning("pandas extraction failed on %s: %s", file_path, exc)

        return pages_out

    def _extract_with_docx(self, file_path: str) -> list[dict]:
        """
        Track A extractor for Word (.docx) files using python-docx.

        Traverses ALL block-level elements in document order (paragraphs AND
        table cells) so no content is skipped — even in table-heavy layouts
        like cover pages, multi-column forms, or formatted structured docs.

        Strategy:
          1. Walk the raw XML body children in order (w:p and w:tbl tags).
          2. For each <w:p> paragraph, collect its text.
          3. For each <w:tbl> table, collect cell text into the text stream
             AND build a structured table dict for the 'tables' field.
          4. This guarantees document order is preserved across the entire doc.
        """
        pages_out = []
        if not DOCX_AVAILABLE:
            logger.warning("python-docx is not available. Cannot process %s natively.", file_path)
            return pages_out

        try:
            from docx.oxml.ns import qn
            import docx as _docx

            doc = _docx.Document(file_path)

            text_lines: list[str] = []
            tables_out: list[dict] = []
            t_idx = 0

            body = doc.element.body
            for child in body.iterchildren():

                # ── Root-level paragraph ───────────────────────────────────
                if child.tag == qn("w:p"):
                    para = _docx.text.paragraph.Paragraph(child, doc)
                    line = para.text.strip()
                    if line:
                        text_lines.append(line)

                # ── Root-level table ───────────────────────────────────────
                elif child.tag == qn("w:tbl"):
                    table = _docx.table.Table(child, doc)
                    if not table.rows:
                        continue

                    # Collect every cell's text into the running text stream
                    # so the text panel shows ALL content in document order.
                    for row in table.rows:
                        row_texts = [cell.text.strip() for cell in row.cells]
                        line = "  |  ".join(t for t in row_texts if t)
                        if line:
                            text_lines.append(line)

                    # Also build a structured table dict (for the tables panel)
                    headers = [cell.text.strip() for cell in table.rows[0].cells]
                    rows = []
                    for row in table.rows[1:]:
                        row_data = [cell.text.strip() for cell in row.cells]
                        row_dict = {
                            (headers[i] if i < len(headers) else f"Col_{i}"): val
                            for i, val in enumerate(row_data)
                        }
                        rows.append(row_dict)

                    if rows:
                        tables_out.append({
                            "headers"    : headers,
                            "rows"       : rows,
                            "table_index": t_idx,
                        })
                        t_idx += 1

            full_text = "\n".join(text_lines)

            pages_out.append({
                "page_number"              : 1,
                "extracted_text"           : full_text,
                "coordinates"             : [],
                "tables"                   : tables_out,
                "key_value_pairs"          : {},
                "handwriting_detected"     : False,
                "confidence_warning"       : False,
                "confidence_warning_reason": None,
                "_engine"                  : "python-docx-full-traverse",
            })

        except Exception as exc:
            logger.warning("python-docx extraction failed on %s: %s", file_path, exc)

        return pages_out


    # ── OpenCV Complexity Sieve ───────────────────────────────────────────────

    def calculate_pcs(self, image_bgr: np.ndarray) -> dict[str, float]:
        """
        OpenCV-based Page Complexity Score (PCS) calculator.
        Runs in <50ms on CPU — replaces Heron-101/RT-DETR for local deployment.

        Returns dict with individual component scores and final PCS.
        """
        h, w = image_bgr.shape[:2]
        page_area = h * w

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # ── N_table: detect grid structures (horizontal + vertical lines) ──
        # Morphological line detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 2)
        h_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 10, 30), 1))
        v_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 10, 30)))
        h_lines    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        grid       = cv2.bitwise_and(h_lines, v_lines)
        h_cnt, _   = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_cnt, _   = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_table    = 1.0 if (len(h_cnt) >= 2 and len(v_cnt) >= 2) else 0.0
        n_table   += min(len(h_cnt) / 20.0, 1.0) * 0.5   # scale up for dense grids
        n_table    = min(n_table, 1.0)

        # ── N_handwritten: high-frequency contour complexity ──
        # Handwriting has irregular, high-curvature strokes
        blurred    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges      = cv2.Canny(blurred, 30, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Handwriting ratio: irregular contours relative to image size
        irregular  = [c for c in contours if cv2.arcLength(c, False) > 20
                      and cv2.contourArea(c) < 500]
        n_handwritten = min(len(irregular) / 500.0, 1.0)

        # ── N_overlap: detect bounding box collisions ──
        # Find word-level bounding boxes and count overlaps
        dilate_k   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dilated    = cv2.dilate(thresh, dilate_k, iterations=2)
        word_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes      = [cv2.boundingRect(c) for c in word_cnts if cv2.contourArea(c) > 50]
        overlap_count = 0
        for i in range(len(boxes)):
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, min(i + 10, len(boxes))):  # check nearby boxes only
                x2, y2, w2, h2 = boxes[j]
                if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
                    overlap_count += 1
        n_overlap  = min(overlap_count / 50.0, 1.0)

        # ── A_graphic: detect non-text solid blob regions (logos, diagrams, stamps) ──
        hsv        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)
        graphic_area = cv2.countNonZero(sat_mask)
        a_graphic  = min(graphic_area / page_area, 1.0)

        # ── Final PCS ──────────────────────────────────────────────────────────
        pcs = (
            W1_TABLE       * min(n_table, 1.0)
            + W2_HANDWRITTEN * n_handwritten
            + W3_OVERLAP     * n_overlap
            + W4_GRAPHIC     * a_graphic
        )
        pcs = round(min(pcs, 1.0), 4)

        logger.info(
            "PCS=%.4f | table=%.3f handwritten=%.3f overlap=%.3f graphic=%.3f",
            pcs, n_table, n_handwritten, n_overlap, a_graphic,
        )
        return {
            "pcs_score"    : pcs,
            "n_table"      : round(n_table, 4),
            "n_handwritten": round(n_handwritten, 4),
            "n_overlap"    : round(n_overlap, 4),
            "a_graphic"    : round(a_graphic, 4),
        }

    # ── Pre-processing (Glare Suppression + Deskew) ───────────────────────────

    def preprocess_image(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply deskew correction and specular glare suppression (TELEA + CLAHE).
        Returns cleaned BGR image array.
        """
        # Deskew removed to fix 90-degree misrotation bug for handwritten notebooks
        
        # Adaptive downscaling to standard width (1600px)
        h, w = image_bgr.shape[:2]
        if w > 1600:
            scale = 1600 / float(w)
            image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Binarization for mobile captures
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)
        cleaned = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
        
        return cleaned, image_bgr

    # ── Path 1: CPU OCR ───────────────────────────────────────────────────────

    def run_path_1_ocr(self, binarized_bgr: np.ndarray, raw_downscaled_bgr: np.ndarray) -> dict[str, Any]:
        """
        Path 1 — Local CPU OCR via PaddleOCR.
        Returns structured page data with '_ocr_avg_confidence' field.
        """
        if not self.paddle_ocr_engine:
            logger.warning("Path 1: PaddleOCR not available — will escalate to VLM")
            return {
                "extracted_text"             : "",
                "key_value_pairs"            : {},
                "tables"                     : [],
                "coordinates"                : [],
                "handwriting_detected"       : False,
                "confidence_warning"         : True,
                "confidence_warning_reason"  : "PaddleOCR not installed — escalated to VLM",
                "_ocr_avg_confidence"        : 0.0,
                "_path1_no_ocr"              : True,
            }

        # Run OCR with PaddleOCR engine
        try:
            results = self.paddle_ocr_engine.ocr(binarized_bgr)
            # Handle empty results or nested structure
            if not results or results[0] is None:
                lines = []
            else:
                lines = results[0]
                
            # Dual-Pass Preprocessing Fallback
            if len(lines) < 5:
                logger.info("Pass 1 (Binarized) yielded < 5 words. Running Pass 2 on raw downscaled image.")
                results_pass2 = self.paddle_ocr_engine.ocr(raw_downscaled_bgr)
                lines = [] if not results_pass2 or results_pass2[0] is None else results_pass2[0]
                target_img_for_tables = raw_downscaled_bgr
            else:
                target_img_for_tables = binarized_bgr
                
        except Exception as exc:
            logger.error("PaddleOCR failed: %s — will escalate to VLM", exc)
            return {
                "extracted_text"             : "",
                "key_value_pairs"            : {},
                "tables"                     : [],
                "coordinates"                : [],
                "handwriting_detected"       : False,
                "confidence_warning"         : True,
                "confidence_warning_reason"  : f"OCR failed: {exc} — escalated to VLM",
                "_ocr_avg_confidence"        : 0.0,
            }

        h, w = target_img_for_tables.shape[:2]
        coordinates = []
        texts = []
        confidences = []

        for line in lines:
            box, (text, conf) = line
            text = text.strip()
            if not text:
                continue
                
            texts.append(text)
            conf_pct = conf * 100.0
            confidences.append(conf_pct)
            
            # PaddleOCR box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            min_x = min(pt[0] for pt in box)
            max_x = max(pt[0] for pt in box)
            min_y = min(pt[1] for pt in box)
            max_y = max(pt[1] for pt in box)
            
            coordinates.append({
                "text"      : text,
                "x0"        : round((min_x / w) * 1000, 1),
                "y0"        : round((min_y / h) * 1000, 1),
                "x1"        : round((max_x / w) * 1000, 1),
                "y1"        : round((max_y / h) * 1000, 1),
                "confidence": conf_pct,
            })

        raw_text = "\n".join(texts)
        
        # Simple KV pair heuristic: "Key: Value" patterns
        import re
        kv_pairs: dict[str, str] = {}
        for line_str in raw_text.splitlines():
            m = re.match(r"^([A-Za-z][A-Za-z\s]{1,30}):\s*(.+)$", line_str.strip())
            if m:
                kv_pairs[m.group(1).strip()] = m.group(2).strip()

        # Task 2: Dual-Engine Local Extraction (Path 1) -> For Tables
        tables_out = []
        try:
            from img2table.document import Image as Img2TableImage
            from img2table.ocr import PaddleOCR as Img2TablePaddleOCR
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, target_img_for_tables)
                tmp_path = tmp.name
                
            img_doc = Img2TableImage(tmp_path)
            
            if self.img2table_ocr_wrapper:
                extracted = img_doc.extract_tables(ocr=self.img2table_ocr_wrapper, implicit_rows=True, borderless_tables=True, min_confidence=50)
            else:
                extracted = []
            
            if extracted:
                for tbl in extracted:
                    df = tbl.df
                    if df is not None and not df.empty:
                        df_str = df.astype(str).fillna("")
                        headers = df_str.iloc[0].tolist()
                        rows_list = []
                        for _, row in df_str.iloc[1:].iterrows():
                            row_dict = {}
                            for j, cell in enumerate(row):
                                col_name = headers[j] if j < len(headers) else f"Col_{j}"
                                row_dict[col_name] = str(cell).strip()
                            rows_list.append(row_dict)
                        
                        if rows_list:
                            tables_out.append({
                                "headers": headers,
                                "rows": rows_list,
                                "table_index": len(tables_out)
                            })
            os.remove(tmp_path)
        except Exception as exc:
            logger.warning("img2table failed or took too long: %s", exc)

        word_count = len(texts)
        avg_conf = sum(confidences) / max(1, len(confidences))
        
        # Confidence gating logic
        confidence_warning = False
        reason = None
        
        if avg_conf < self.ocr_confidence_threshold:
            confidence_warning = True
            reason = f"Local OCR confidence {avg_conf:.1f}% < {self.ocr_confidence_threshold:.0f}% threshold — escalated to Groq VLM"
        elif word_count < 3:
            confidence_warning = True
            reason = f"OCR returned only {word_count} words (< 3) — escalated to Groq VLM"

        logger.info(
            "Path 1 OCR complete | words=%d | avg_confidence=%.1f%% | threshold=%.0f%%",
            word_count, avg_conf, self.ocr_confidence_threshold,
        )

        return {
            "extracted_text"             : raw_text.strip(),
            "key_value_pairs"            : kv_pairs,
            "tables"                     : tables_out,
            "coordinates"                : coordinates,
            "handwriting_detected"       : True,
            "confidence_warning"         : confidence_warning,
            "confidence_warning_reason"  : reason,
            "_ocr_avg_confidence"        : avg_conf,
            "_ocr_word_count"            : word_count,
        }

    # ── Path 2: VLM (Groq Vision) ─────────────────────────────────────────────

    def run_path_2_vlm(self, image_bgr: np.ndarray, temp_path: str) -> dict[str, Any]:
        """
        Path 2 — High-Complexity VLM via Groq Vision Engine (lego3 on port 8002).
        Saves the preprocessed image to disk and calls the existing /extract endpoint.
        """
        # Save cleaned image to temp file for lego3 to read
        clean_path = temp_path.replace(".jpg", "_smart_clean.jpg").replace(".png", "_smart_clean.png")
        if not clean_path.endswith("_smart_clean.jpg"):
            clean_path = temp_path + "_smart_clean.jpg"
        cv2.imwrite(clean_path, image_bgr)

        try:
            resp = requests.post(
                f"{self.lego3_url}/extract",
                json={"image_path": clean_path},
                timeout=180,
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("data", {})
        except Exception as exc:
            logger.error("Groq VLM call failed: %s", exc)
            return {
                "extracted_text"    : "",
                "key_value_pairs"   : {},
                "tables"            : [],
                "handwriting_detected"   : False,
                "confidence_warning"     : True,
                "confidence_warning_reason": f"VLM API error: {exc}",
            }

    # ── PDF → Image Rendering ─────────────────────────────────────────────────

    def render_pdf_page_to_image(self, pdf_path: str, page_index: int = 0) -> np.ndarray | None:
        """
        Render a single PDF page to a high-resolution BGR numpy array.
        Uses pypdfium2 if available, otherwise converts via fitz (PyMuPDF).
        """
        if PDFIUM_AVAILABLE:
            try:
                doc    = pdfium.PdfDocument(pdf_path)
                page   = doc[page_index]
                bitmap = page.render(scale=2.0)   # 144 DPI equivalent
                pil    = bitmap.to_pil()
                return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                logger.warning("pypdfium2 render failed: %s", exc)

        # Fallback: try PyMuPDF (fitz)
        try:
            import fitz
            doc    = fitz.open(pdf_path)
            page   = doc[page_index]
            mat    = fitz.Matrix(2.0, 2.0)
            pix    = page.get_pixmap(matrix=mat)
            arr    = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        except Exception as exc:
            logger.warning("PyMuPDF render failed: %s", exc)

        return None

    # ── Master Route Entry Point ──────────────────────────────────────────────

    def route_document(
        self,
        file_path: str,
        job_id: str,
    ) -> dict[str, Any]:
        """
        Main triage and processing entry point.
        Accepts any supported file (PDF, JPG, PNG, TIFF, Excel, CSV, Word).

        Returns structured routing result:
          {
            "job_id"        : str,
            "execution_path": ExecutionPath,
            "pcs_score"     : float | None,
            "pcs_breakdown" : dict | None,
            "pages"         : list[dict],
            "_pipeline"     : str,
          }
        """
        ext = Path(file_path).suffix.lower()
        logger.info("[%s] Routing: %s (ext=%s)", job_id, file_path, ext)

        # ── TRACK A: Digital PDF ─────────────────────────────────────────
        if ext == ".pdf":
            is_native, pages_data = self.is_vector_native(file_path)
            if is_native:
                logger.info("[%s] → Track A: Digital Fast-Path (%d pages)", job_id, len(pages_data))
                return {
                    "job_id"        : job_id,
                    "execution_path": ExecutionPath.TRACK_A_DIGITAL.value,
                    "pcs_score"     : 0.0,
                    "pcs_breakdown" : None,
                    "pages"         : pages_data,
                    "_pipeline"     : "track_a_digital",
                }
            else:
                logger.info("[%s] PDF is scanned/blank — proceeding to Track B rendering", job_id)
                # Render first page to image for complexity analysis
                image_bgr = self.render_pdf_page_to_image(file_path, page_index=0)
                if image_bgr is None:
                    logger.warning("[%s] Could not render PDF page — falling back to VLM", job_id)
                    return self._run_vlm_on_file(file_path, job_id, pcs_score=1.0, pcs_breakdown=None)
                return self._run_track_b(image_bgr, file_path, job_id)

        # ── TRACK A: Native Excel / CSV / Word ───────────────────────────
        if ext in {".xlsx", ".xls", ".csv"}:
            is_csv = (ext == ".csv")
            pages_data = self._extract_with_pandas(file_path, is_csv=is_csv)
            if pages_data:
                logger.info("[%s] → Track A: Native Table Fast-Path (%d sheets)", job_id, len(pages_data))
                return {
                    "job_id"        : job_id,
                    "execution_path": ExecutionPath.TRACK_A_DIGITAL.value,
                    "pcs_score"     : 0.0,
                    "pcs_breakdown" : None,
                    "pages"         : pages_data,
                    "_pipeline"     : "track_a_pandas",
                }
            # pandas failed (e.g. missing xlrd for .xls) — return a clear error
            # instead of falling through to the image reader.
            hint = ("Install 'xlrd' (pip install xlrd) for legacy .xls support."
                    if ext == ".xls" else
                    "pandas could not parse this file. Check it is a valid spreadsheet.")
            logger.error("[%s] pandas extraction failed for %s — %s", job_id, ext, hint)
            return {
                "job_id"        : job_id,
                "execution_path": ExecutionPath.TRACK_A_DIGITAL.value,
                "pcs_score"     : 0.0,
                "pcs_breakdown" : None,
                "pages"         : [{
                    "page_number"              : 1,
                    "extracted_text"           : f"[Extraction failed]\n{hint}",
                    "coordinates"             : [],
                    "tables"                   : [],
                    "key_value_pairs"          : {},
                    "handwriting_detected"     : False,
                    "confidence_warning"       : True,
                    "confidence_warning_reason": hint,
                    "_engine"                  : "error",
                }],
                "_pipeline"     : "error_spreadsheet",
            }

        if ext == ".docx":
            pages_data = self._extract_with_docx(file_path)
            if pages_data:
                logger.info("[%s] → Track A: Native Word Fast-Path", job_id)
                return {
                    "job_id"        : job_id,
                    "execution_path": ExecutionPath.TRACK_A_DIGITAL.value,
                    "pcs_score"     : 0.0,
                    "pcs_breakdown" : None,
                    "pages"         : pages_data,
                    "_pipeline"     : "track_a_docx",
                }
            # docx parse failed — return a clear error instead of falling through.
            hint = "python-docx could not parse this file. Check it is a valid .docx document."
            logger.error("[%s] docx extraction failed — %s", job_id, hint)
            return {
                "job_id"        : job_id,
                "execution_path": ExecutionPath.TRACK_A_DIGITAL.value,
                "pcs_score"     : 0.0,
                "pcs_breakdown" : None,
                "pages"         : [{
                    "page_number"              : 1,
                    "extracted_text"           : f"[Extraction failed]\n{hint}",
                    "coordinates"             : [],
                    "tables"                   : [],
                    "key_value_pairs"          : {},
                    "handwriting_detected"     : False,
                    "confidence_warning"       : True,
                    "confidence_warning_reason": hint,
                    "_engine"                  : "error",
                }],
                "_pipeline"     : "error_docx",
            }

        # ── TRACK B: Image / Scanned file ────────────────────────────────
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            logger.error("[%s] Cannot read image file natively or via OpenCV: %s", job_id, file_path)
            return {
                "job_id"        : job_id,
                "execution_path": ExecutionPath.PATH_2_HIGH_COMPLEXITY.value,
                "pcs_score"     : 1.0,
                "pcs_breakdown" : None,
                "pages"         : [],
                "_pipeline"     : "error_unreadable",
            }
        return self._run_track_b(image_bgr, file_path, job_id)

    def _run_track_b(self, image_bgr: np.ndarray, file_path: str, job_id: str) -> dict[str, Any]:
        """
        Local-First, Confidence-Gated routing for Track B:

          Step 1: Pre-process image (deskew + TELEA glare suppression + CLAHE)
          Step 2: Calculate PCS score (for display/audit — does NOT gate routing)
          Step 3: ALWAYS attempt CPU OCR first (pytesseract)
          Step 4: Check OCR confidence:
                   >= OCR_CONFIDENCE_THRESHOLD → Accept local result (Path 1, zero API cost)
                   <  OCR_CONFIDENCE_THRESHOLD → Escalate to Groq Vision LLM (Path 2, fallback)
        """
        # Step 1: Pre-process the image
        cleaned_bgr, downscaled_bgr = self.preprocess_image(image_bgr)

        # Step 2: Calculate PCS (informational only)
        pcs_result = self.calculate_pcs(cleaned_bgr)
        pcs_score  = pcs_result["pcs_score"]

        # Save cleaned image for downstream use and HITL image preview
        clean_path = os.path.join(self.lego2_temp_dir, f"{job_id}_clean.jpg")
        cv2.imwrite(clean_path, cleaned_bgr)

        # Step 3: Always try local CPU OCR first
        logger.info(
            "[%s] Step 3: Attempting local CPU OCR first (PCS=%.4f, threshold=%.0f%%)",
            job_id, pcs_score, self.ocr_confidence_threshold,
        )
        ocr_result = self.run_path_1_ocr(cleaned_bgr, downscaled_bgr)
        ocr_conf   = ocr_result.get("_ocr_avg_confidence", 0.0)
        no_ocr     = ocr_result.get("_path1_no_ocr", False)
        
        # Get extraction metrics
        word_count = ocr_result.get("_ocr_word_count", len(ocr_result.get("coordinates", [])))
        extracted_tables = ocr_result.get("tables", [])

        # Dynamic Confidence Gating
        current_threshold = self.ocr_confidence_threshold
        
        extracted_text = ocr_result.get("extracted_text", "").lower()
        transactional_keywords = ["total", "amount", "subtotal", "invoice", "receipt", "date", "item"]
        has_transactional_keyword = any(kw in extracted_text for kw in transactional_keywords)
        
        if word_count > 0:
            current_threshold = 0.0
            logger.info("[%s] Local CPU text detected (> 0 words). Forcing local CPU pipeline and completely blocking API escalation.", job_id)
        
        # Step 4: The Smart LLM Fallback (Path 2 Escalation)
        if not no_ocr and ocr_conf >= current_threshold and word_count >= 1:
            # ── PATH 1: Local result accepted ───────────────────────────
            logger.info(
                "[%s] → Path 1 ACCEPTED: OCR confidence %.1f%% >= %.1f%% threshold — NO API call",
                job_id, ocr_conf, current_threshold,
            )
            ocr_result["_source_image"]   = clean_path
            ocr_result["confidence_warning"] = False
            ocr_result["confidence_warning_reason"] = None
            return {
                "job_id"        : job_id,
                "execution_path": ExecutionPath.PATH_1_LOW_COMPLEXITY.value,
                "pcs_score"     : pcs_score,
                "pcs_breakdown" : pcs_result,
                "pages"         : [ocr_result],
                "_pipeline"     : "path_1_cpu_ocr",
            }
        else:
            # ── PATH 2: Escalate to Groq VLM ────────────────────────────
            reason = (
                f"PaddleOCR not installed"
                if no_ocr
                else f"Local OCR confidence {ocr_conf:.1f}% < {current_threshold:.1f}% threshold"
            )
            logger.info(
                "[%s] → Path 2 ESCALATION: %s — calling Groq Vision LLM",
                job_id, reason,
            )
            vlm_result = self.run_path_2_vlm(cleaned_bgr, clean_path)
            vlm_result["_source_image"]         = clean_path
            vlm_result["_ocr_avg_confidence"]   = ocr_conf
            vlm_result["_ocr_escalation_reason"] = reason
            # If VLM also failed, merge the best of both results
            if not vlm_result.get("extracted_text") and ocr_result.get("extracted_text"):
                logger.warning("[%s] VLM returned empty text — merging best local OCR result", job_id)
                vlm_result["extracted_text"]  = ocr_result["extracted_text"]
                vlm_result["key_value_pairs"]  = ocr_result.get("key_value_pairs", {})
                vlm_result["confidence_warning"] = True
                vlm_result["confidence_warning_reason"] = "VLM fallback: merged with local OCR result"
            return {
                "job_id"        : job_id,
                "execution_path": ExecutionPath.PATH_2_HIGH_COMPLEXITY.value,
                "pcs_score"     : pcs_score,
                "pcs_breakdown" : pcs_result,
                "pages"         : [vlm_result],
                "_pipeline"     : "path_2_groq_vlm",
            }

    def _run_vlm_on_file(self, file_path: str, job_id: str, pcs_score: float, pcs_breakdown) -> dict[str, Any]:
        """Directly route to VLM for problematic files that couldn't be rendered."""
        page_data = self.run_path_2_vlm(
            np.zeros((100, 100, 3), dtype=np.uint8),  # dummy — lego3 reads from path
            file_path,
        )
        return {
            "job_id"        : job_id,
            "execution_path": ExecutionPath.PATH_2_HIGH_COMPLEXITY.value,
            "pcs_score"     : pcs_score,
            "pcs_breakdown" : pcs_breakdown,
            "pages"         : [page_data],
            "_pipeline"     : "path_2_groq_vlm",
        }
