"""
Greencare AI — Lego 2: CPU Triage  [UPGRADED v2]
=================================================
New in v2:
  • .docx support via python-docx (fast-track, no VLM call)
  • .xlsx / .xls support via openpyxl (tables extracted natively)
  • .csv  support via Python csv module
  • Processing timestamps in all responses
  • /health/detailed endpoint with library version info
"""

import io
import os
import csv
import json
import shutil
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

# Optional: pypdfium2
try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False
    logging.warning("pypdfium2 not available — falling back to pypdf only")

# Optional: python-docx
try:
    from docx import Document as DocxDocument
    import mammoth
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx / mammoth not available — .docx fast-track disabled")

# Optional: openpyxl
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logging.warning("openpyxl not available — .xlsx fast-track disabled")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — Lego 2: CPU Triage",
    description=(
        "Fast-tracks digital PDFs, .docx, .xlsx, and .csv files. "
        "Preprocesses images with deskew and glare-suppression before Qwen 3.6 27B."
    ),
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TEMP_DIR = "./lego2_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

IMAGE_EXT      = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXT        = ".pdf"
DOCX_EXT       = ".docx"
EXCEL_EXT      = {".xlsx"}       # BUG-6 fix: openpyxl only supports .xlsx, NOT .xls
XLS_LEGACY_EXT = {".xls"}        # .xls (Excel 97) needs xlrd — not installed; reject clearly
CSV_EXT        = ".csv"
MIN_TEXT_CHARS = 50


# ---------------------------------------------------------------------------
# §5 Specular Glare Suppression Filter
# ---------------------------------------------------------------------------
def apply_glare_suppression(image_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("Cannot read image for glare suppression: %s", image_path)
        return image_path

    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val  = np.mean(gray)
    std_val   = np.std(gray)
    threshold = min(255, int(mean_val + 2.5 * std_val))
    threshold = max(threshold, 200)

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel  = np.ones((5, 5), np.uint8)
    mask    = cv2.dilate(mask, kernel, iterations=2)
    cleaned = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    lab     = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    cleaned = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    base, ext  = os.path.splitext(image_path)
    clean_path = f"{base}_glare_clean{ext}"
    cv2.imwrite(clean_path, cleaned)
    logger.info("Glare suppression → %s", clean_path)
    return clean_path


# ---------------------------------------------------------------------------
# Deskew correction
# ---------------------------------------------------------------------------
def deskew_image(image_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        return image_path

    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray   = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 10:
        return image_path

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    if abs(angle) < 0.5:
        return image_path

    h, w   = image.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)

    base, ext   = os.path.splitext(image_path)
    deskew_path = f"{base}_deskewed{ext}"
    cv2.imwrite(deskew_path, result)
    logger.info("Deskewed (%.2f°) → %s", angle, deskew_path)
    return deskew_path


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------
def extract_pdf_data(pdf_path: str) -> list:
    pages_out = []
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            pages_out.append({
                "page_number":     page_num + 1,
                "extracted_text":  raw_text,
                "coordinates":     [],
                "tables":          [],
                "key_value_pairs": {},
                "handwriting_detected":      False,
                "confidence_warning":        False,
                "confidence_warning_reason": None,
            })
    except Exception as exc:
        logger.error("pypdf extraction failed: %s", exc)

    if PDFIUM_AVAILABLE and not any(p["extracted_text"] for p in pages_out):
        try:
            doc = pdfium.PdfDocument(pdf_path)
            for i in range(len(doc)):
                page     = doc.get_page(i)
                textpage = page.get_textpage()
                text     = textpage.get_text_range()
                if i < len(pages_out):
                    pages_out[i]["extracted_text"] = text
                else:
                    pages_out.append({
                        "page_number": i + 1,
                        "extracted_text": text,
                        "coordinates": [],
                        "tables": [],
                        "key_value_pairs": {},
                        "handwriting_detected": False,
                        "confidence_warning": False,
                        "confidence_warning_reason": None,
                    })
        except Exception as exc:
            logger.error("pypdfium2 extraction failed: %s", exc)

    return pages_out


def pdf_has_enough_text(pages: list) -> bool:
    total = sum(len(p.get("extracted_text", "").strip()) for p in pages)
    return total >= MIN_TEXT_CHARS


# ---------------------------------------------------------------------------
# .docx extraction
# ---------------------------------------------------------------------------
def extract_docx_data(docx_path: str) -> dict:
    """
    Extract text and table data from a .docx file using python-docx.
    Falls back to mammoth if primary extraction yields no text.
    """
    extracted_text = ""
    tables = []

    if not DOCX_AVAILABLE:
        return {"extracted_text": "", "tables": [], "key_value_pairs": {}}

    try:
        doc = DocxDocument(docx_path)

        # Full text (paragraphs)
        para_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        extracted_text = para_text

        # Tables
        for tbl_idx, tbl in enumerate(doc.tables):
            headers = []
            rows    = []
            for row_idx, row in enumerate(tbl.rows):
                cells = [c.text.strip() for c in row.cells]
                if row_idx == 0:
                    headers = cells
                else:
                    rows.append(cells)
            tables.append({"table_index": tbl_idx, "headers": headers, "rows": rows})

    except Exception as exc:
        logger.warning("python-docx extraction failed: %s — trying mammoth", exc)

    # Mammoth fallback
    if not extracted_text.strip():
        try:
            with open(docx_path, "rb") as f:
                result = mammoth.extract_raw_text(f)
            extracted_text = result.value or ""
        except Exception as exc2:
            logger.error("mammoth extraction also failed: %s", exc2)

    return {
        "document_type":   "word_document",
        "extracted_text":  extracted_text,
        "tables":          tables,
        "key_value_pairs": {},
        "handwriting_detected":      False,
        "confidence_warning":        False,
        "confidence_warning_reason": None,
    }


# ---------------------------------------------------------------------------
# .xlsx / .xls extraction
# ---------------------------------------------------------------------------
def extract_excel_data(excel_path: str) -> dict:
    """
    Extract all sheets from an Excel file as tables.
    """
    if not OPENPYXL_AVAILABLE:
        return {"extracted_text": "", "tables": [], "key_value_pairs": {}}

    wb = None
    try:
        wb     = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
        tables = []
        all_text_parts = []

        for sheet_idx, sheet_name in enumerate(wb.sheetnames):
            ws      = wb[sheet_name]
            rows    = [[str(cell.value or "") for cell in row] for row in ws.iter_rows()]
            if not rows:
                continue
            headers = rows[0]
            data    = rows[1:]
            tables.append({
                "table_index": sheet_idx,
                "sheet_name":  sheet_name,
                "headers":     headers,
                "rows":        data,
            })
            all_text_parts.append(f"[Sheet: {sheet_name}]\n" + "\n".join(
                ", ".join(r) for r in rows
            ))

        return {
            "document_type":   "spreadsheet",
            "extracted_text":  "\n\n".join(all_text_parts),
            "tables":          tables,
            "key_value_pairs": {},
            "handwriting_detected":      False,
            "confidence_warning":        False,
            "confidence_warning_reason": None,
        }
    except Exception as exc:
        logger.error("Excel extraction failed: %s", exc)
        return {"extracted_text": "", "tables": [], "key_value_pairs": {}}
    finally:
        if wb:  # BUG-I fix: always close workbook to release file handle
            wb.close()


# ---------------------------------------------------------------------------
# .csv extraction
# ---------------------------------------------------------------------------
def extract_csv_data(csv_path: str) -> dict:
    """
    Extract a CSV file as a single table.
    """
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader  = csv.reader(f)
            all_rows = list(reader)

        if not all_rows:
            return {"extracted_text": "", "tables": [], "key_value_pairs": {}}

        headers = all_rows[0]
        rows    = all_rows[1:]
        text    = "\n".join(", ".join(r) for r in all_rows)

        return {
            "document_type":   "csv_data",
            "extracted_text":  text,
            "tables":          [{"table_index": 0, "headers": headers, "rows": rows}],
            "key_value_pairs": {},
            "handwriting_detected":      False,
            "confidence_warning":        False,
            "confidence_warning_reason": None,
        }
    except Exception as exc:
        logger.error("CSV extraction failed: %s", exc)
        return {"extracted_text": "", "tables": [], "key_value_pairs": {}}


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status":  "ok",
        "service": "lego2-triage",
        "version": "2.0.0",
        "pdfium":  PDFIUM_AVAILABLE,
        "docx":    DOCX_AVAILABLE,
        "excel":   OPENPYXL_AVAILABLE,
    }


@app.get("/health/detailed", tags=["Monitoring"])
def health_detailed():
    import sys
    return {
        "status":   "ok",
        "service":  "lego2-triage",
        "version":  "2.0.0",
        "python":   sys.version,
        "libraries": {
            "pdfium":   PDFIUM_AVAILABLE,
            "docx":     DOCX_AVAILABLE,
            "openpyxl": OPENPYXL_AVAILABLE,
        },
        "supported_extensions": list(
            {PDF_EXT, DOCX_EXT, CSV_EXT} | IMAGE_EXT | EXCEL_EXT
        ),
        "temp_files": len(os.listdir(TEMP_DIR)) if os.path.isdir(TEMP_DIR) else 0,
    }


# ---------------------------------------------------------------------------
# Triage endpoint
# ---------------------------------------------------------------------------
@app.post("/triage", tags=["Triage"])
async def triage_document(file: UploadFile = File(...)):
    """
    Routing logic:
      • Digital PDF  → fast_track_complete (CPU text extraction, no VLM)
      • .docx        → fast_track_complete (python-docx extraction)
      • .xlsx / .xls → fast_track_complete (openpyxl extraction)
      • .csv         → fast_track_complete (csv module extraction)
      • Image / scanned PDF → forward_to_vlm (clean image to Qwen 3.6 27B)
    """
    started_at    = datetime.now(timezone.utc).isoformat()
    original_name = file.filename or "unknown"
    _, ext        = os.path.splitext(original_name)
    ext           = ext.lower()

    # BUG-C fix: prefix saved file with job_id (sent as form-data filename UUID_originalname)
    # The worker sends the file with path: f"{job_id}_{original_filename}"
    # Extract job_id prefix so find_source_image() in HITL can locate by job_id.
    # If no UUID prefix is embedded in the filename, keep as-is.
    dest_filename = original_name  # already has uuid prefix if sent via worker
    dest_path = os.path.join(TEMP_DIR, dest_filename)
    with open(dest_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    logger.info("Received: %s (ext=%s)", original_name, ext)

    # ── PDF ────────────────────────────────────────────────────────────────
    if ext == PDF_EXT:
        pages = extract_pdf_data(dest_path)
        completed_at = datetime.now(timezone.utc).isoformat()
        if pdf_has_enough_text(pages):
            logger.info("PDF fast-tracked (%d pages)", len(pages))
            # Stamp timestamps on each page
            for p in pages:
                p["_started_at"]   = started_at
                p["_completed_at"] = completed_at
            return {
                "routing":        "fast_track_complete",
                "pages":          pages,
                "extracted_data": {
                    **(pages[0] if pages else {}),
                    "_started_at":   started_at,
                    "_completed_at": completed_at,
                },
            }
        else:
            logger.info("Scanned/blank PDF — forwarding to VLM")
            return {
                "routing":           "forward_to_vlm",
                "cleaned_file_path": dest_path,
                "_started_at":       started_at,
            }

    # ── Word Document (.docx) ──────────────────────────────────────────────
    elif ext == DOCX_EXT:
        if not DOCX_AVAILABLE:
            raise HTTPException(
                status_code=415,
                detail=".docx support requires python-docx. Install with: pip install python-docx mammoth",
            )
        data = extract_docx_data(dest_path)
        completed_at = datetime.now(timezone.utc).isoformat()
        data["_started_at"]   = started_at
        data["_completed_at"] = completed_at
        logger.info(".docx fast-tracked: %d chars", len(data.get("extracted_text", "")))
        return {
            "routing":        "fast_track_complete",
            "pages":          [data],
            "extracted_data": data,
        }

    # ── Excel (.xlsx) ──────────────────────────────────────────────────────
    elif ext in EXCEL_EXT:
        if not OPENPYXL_AVAILABLE:
            raise HTTPException(
                status_code=415,
                detail=".xlsx support requires openpyxl. Install with: pip install openpyxl",
            )
        data = extract_excel_data(dest_path)
        completed_at = datetime.now(timezone.utc).isoformat()
        data["_started_at"]   = started_at
        data["_completed_at"] = completed_at
        tbl_count = len(data.get("tables", []))
        logger.info("Excel fast-tracked: %d sheet(s)", tbl_count)
        return {
            "routing":        "fast_track_complete",
            "pages":          [data],
            "extracted_data": data,
        }

    # ── Legacy .xls — not supported by openpyxl ────────────────────────────
    elif ext in XLS_LEGACY_EXT:
        # BUG-6 fix: .xls requires xlrd which is not in requirements.
        # Return a clear error rather than crashing inside openpyxl.
        raise HTTPException(
            status_code=415,
            detail=(
                ".xls (Excel 97-2003) is not supported. "
                "Please convert to .xlsx and re-upload, or install xlrd: pip install xlrd"
            ),
        )

    # ── CSV ────────────────────────────────────────────────────────────────
    elif ext == CSV_EXT:
        data = extract_csv_data(dest_path)
        completed_at = datetime.now(timezone.utc).isoformat()
        data["_started_at"]   = started_at
        data["_completed_at"] = completed_at
        logger.info("CSV fast-tracked: %d rows", len(data.get("tables", [{}])[0].get("rows", [])))
        return {
            "routing":        "fast_track_complete",
            "pages":          [data],
            "extracted_data": data,
        }

    # ── Image ──────────────────────────────────────────────────────────────
    elif ext in IMAGE_EXT:
        processed = deskew_image(dest_path)
        processed = apply_glare_suppression(processed)
        return {
            "routing":           "forward_to_vlm",
            "cleaned_file_path": processed,
            "_started_at":       started_at,
        }

    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: '{ext}'")
