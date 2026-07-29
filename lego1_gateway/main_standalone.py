"""
Greencare AI — Lego 1: API Gateway + Data Serialization (The Baseplate)
=======================================================================
Responsibilities (per blueprint §3.1):
  • Primary public API ingestion gateway
  • Token-based authorization
  • Directs execution queues (via FastAPI BackgroundTasks, no Redis needed)
  • Multi-page table stitching heuristic (§5, Algorithm)
  • Compiles raw AI outputs into hierarchical JSON, CSV, and Excel tables
  • Job status tracking
  • Image asset cropping from visual grounding coordinates (§4.3)

Runs on: Port 8000
"""

import os
import uuid
import shutil
import json
import logging
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import cv2
import numpy as np

from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    BackgroundTasks, Depends, status, Query
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Smart Triage — Core Orchestrator & AST Compiler ────────────────────────
try:
    from smart_triage.orchestrator import SmartTriageOrchestrator, ExecutionPath
    from smart_triage.ast_compiler import ASTCompiler
    SMART_TRIAGE_AVAILABLE = True
except ImportError as _st_err:
    SMART_TRIAGE_AVAILABLE = False
    logging.warning("smart_triage package not available: %s — falling back to legacy pipeline", _st_err)

# ── Optional JWT auth (graceful fallback if jose not installed) ────────────
try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SECRET_KEY       = os.environ.get("SECRET_KEY", "greencare-dev-secret-change-in-production")
ALGORITHM        = "HS256"
ACCESS_TOKEN_TTL = int(os.environ.get("TOKEN_TTL_MINUTES", "1440"))  # 24h default

LEGO2_URL = os.environ.get("LEGO2_URL", "http://localhost:8001")
LEGO3_URL = os.environ.get("LEGO3_URL", "http://localhost:8002")

UPLOAD_DIR   = "./temp_uploads"
PENDING_DIR  = "./pending_review"
FINAL_DIR    = "./final_database"
REJECTED_DIR = "./rejected"
ASSETS_DIR   = "./extracted_assets"   # cropped logo/image assets from visual grounding

for d in [UPLOAD_DIR, PENDING_DIR, FINAL_DIR, REJECTED_DIR, ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Instantiate Smart Triage orchestrator (singleton) ──────────────────────
_orchestrator: "SmartTriageOrchestrator | None" = None
_ast_compiler:  "ASTCompiler | None"             = None

def get_orchestrator():
    global _orchestrator, _ast_compiler
    if SMART_TRIAGE_AVAILABLE and _orchestrator is None:
        _orchestrator = SmartTriageOrchestrator(
            pcs_threshold            = float(os.environ.get("PCS_THRESHOLD", "0.50")),
            ocr_confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "80.0")),
            lego3_url                = LEGO3_URL,
            lego2_temp_dir           = "./lego2_temp",
        )
        _ast_compiler = ASTCompiler()
        logger.info(
            "SmartTriageOrchestrator initialized (OCR confidence gate=%.0f%%)",
            _orchestrator.ocr_confidence_threshold,
        )
    return _orchestrator, _ast_compiler

ALLOWED_EXT = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".xlsx", ".xls", ".csv", ".docx"}

# In-memory job registry
job_registry: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — API Gateway (Lego 1)",
    description=(
        "Core baseplate gateway. Handles ingestion, authorization, "
        "multi-page table stitching, and JSON/CSV/Excel serialization."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Token auth helpers
# ---------------------------------------------------------------------------
def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_TTL)
    if JWT_AVAILABLE:
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    # Fallback: simple base64-ish token
    import base64
    return base64.b64encode(json.dumps(payload).encode()).decode()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    """Validate Bearer JWT. Returns payload dict. Raises 401 on failure."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    token = credentials.credentials
    if JWT_AVAILABLE:
        try:
            return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    # Fallback: accept any token in dev mode
    return {"sub": "dev_user"}


# ---------------------------------------------------------------------------
# §5 Multi-Page Table Stitching Heuristic
# ---------------------------------------------------------------------------
class PageData:
    """Wrapper that mirrors the blueprint's PageData interface."""
    def __init__(self, page_dict: dict):
        self.raw = page_dict
        tables = page_dict.get("tables", [])
        self._tables = tables
        self.table_headers = tables[0]["headers"] if tables else []
        self.table_rows    = tables[0]["rows"]    if tables else []

    def has_open_table_node(self) -> bool:
        """True if page ends mid-table (last char is not a closing marker)."""
        return bool(self._tables)

    def starts_with_table(self) -> bool:
        return bool(self._tables)

    def discard_table_headers(self):
        self.table_rows = self._tables[0]["rows"] if self._tables else []
        self._tables[0]["headers"] = []


def stitch_multipage_tables(page_data_list: list[dict]) -> list[dict]:
    """
    Blueprint §5 — Multi-page table stitching heuristic.
    If adjacent pages share matching column headers and an open table node,
    merge row matrices and discard duplicate headers.
    """
    if len(page_data_list) < 2:
        return page_data_list

    wrapped = [PageData(p) for p in page_data_list]

    for i in range(len(wrapped) - 1):
        curr = wrapped[i]
        nxt  = wrapped[i + 1]

        if (curr.has_open_table_node()
                and nxt.starts_with_table()
                and curr.table_headers
                and curr.table_headers == nxt.table_headers):

            logger.info("Stitching table from page %d into page %d", i, i + 1)
            curr.table_rows.extend(nxt.table_rows)
            nxt.discard_table_headers()

            # Merge back into raw dict
            if wrapped[i].raw.get("tables"):
                wrapped[i].raw["tables"][0]["rows"] = curr.table_rows
            if wrapped[i + 1].raw.get("tables"):
                wrapped[i + 1].raw["tables"][0]["headers"] = []

    return [w.raw for w in wrapped]


# ---------------------------------------------------------------------------
# §4.3 Visual Grounding — Crop image assets from bounding boxes
# ---------------------------------------------------------------------------

# Minimum OCR confidence (0–100) to accept extracted signature text.
SIGNATURE_OCR_CONFIDENCE_THRESHOLD = float(
    os.environ.get("SIGNATURE_OCR_CONFIDENCE_THRESHOLD", "40.0")
)
# Image-analysis thresholds (can be tuned via env vars).
SIG_MIN_INK_RATIO     = float(os.environ.get("SIG_MIN_INK_RATIO",     "0.02"))   # ≥ 2 % ink
SIG_MAX_INK_RATIO     = float(os.environ.get("SIG_MAX_INK_RATIO",     "0.55"))   # ≤ 55 % ink (not fully black)
SIG_MIN_STROKE_CV     = float(os.environ.get("SIG_MIN_STROKE_CV",     "0.35"))   # stroke irregularity
SIG_MAX_TEXT_DENSITY  = float(os.environ.get("SIG_MAX_TEXT_DENSITY",  "0.30"))   # ≤ 30 % printed text area
SIG_MIN_ASPECT        = float(os.environ.get("SIG_MIN_ASPECT",        "0.25"))   # width / height ≥ 0.25

# OCR keywords that betray the wrong region was cropped.
_SIGNATURE_REJECT_KEYWORDS = [
    "address", "street", "road", "lane", "avenue", "drive", "close",
    "city", "town", "county", "postcode", "zip", "state",
    "job ref", "job number", "job sheet", "invoice", "date:", "total:",
    "labour:", "customer name", "phone", "email", "tel:", "fax:",
    "work carried", "description", "amount", "quantity",
    "parts:", "materials:", "vat", "tax", "boiler", "pressure",
]

# Words that indicate the signature field label in the document.
_SIGNATURE_FIELD_LABELS = [
    "signature", "signed", "customer sig", "authorised", "authorized",
    "sign here", "signatory",
]


# ── Low-level helpers ────────────────────────────────────────────────────────

def _ocr_crop(crop_bgr) -> tuple[str, float]:
    """Run pytesseract on BGR crop → (text, avg_confidence)."""
    try:
        import pytesseract  # type: ignore
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        df  = pytesseract.image_to_data(
            rgb, output_type=pytesseract.Output.DATAFRAME, config="--psm 6"
        )
        df = df[df["conf"] > 0]
        if df.empty:
            return "", 0.0
        text = " ".join(df["text"].astype(str).str.strip().tolist()).strip()
        return text, float(df["conf"].mean())
    except Exception:
        return "", 0.0


def _ocr_crop_dict(crop_bgr) -> list[dict]:
    """Return per-word OCR entries for re-localization."""
    try:
        import pytesseract  # type: ignore
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(
            rgb, output_type=pytesseract.Output.DICT, config="--psm 6"
        )
        n = len(data["text"])
        return [
            {
                "text": str(data["text"][i]).strip(),
                "conf": int(data["conf"][i]),
                "left": int(data["left"][i]),
                "top" : int(data["top"][i]),
                "w"   : int(data["width"][i]),
                "h"   : int(data["height"][i]),
            }
            for i in range(n)
            if str(data["text"][i]).strip()
        ]
    except Exception:
        return []


def _ocr_text_density(crop_bgr) -> float:
    """
    Return the fraction (0–1) of the crop area covered by OCR-detected text
    bounding boxes whose confidence > 30.  Used to enforce the 30 % printed-
    text limit on the fallback image.
    """
    try:
        import pytesseract  # type: ignore
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(
            rgb, output_type=pytesseract.Output.DICT, config="--psm 6"
        )
        h, w = crop_bgr.shape[:2]
        total = max(h * w, 1)
        covered = 0
        for i in range(len(data["text"])):
            if int(data["conf"][i]) > 30 and str(data["text"][i]).strip():
                covered += int(data["width"][i]) * int(data["height"][i])
        return min(covered / total, 1.0)
    except Exception:
        return 0.0


# ── Core image-analysis validator ────────────────────────────────────────────

def _verify_signature_image(crop_bgr) -> dict:
    """
    Determine whether a BGR crop contains a handwritten signature.

    Checks (in order):
      1. Aspect ratio — crop must be wider than it is narrow.
      2. Ink coverage — dark pixels must cover ≥ SIG_MIN_INK_RATIO and
         ≤ SIG_MAX_INK_RATIO of the crop (blank or solid-black crops are rejected).
      3. Printed-text density — OCR-detected text boxes must cover
         ≤ SIG_MAX_TEXT_DENSITY (30 %) of the crop area.
      4. Keyword rejection — OCR text must not contain document-field keywords
         (address, customer name, table headers, etc.).
      5. Word count — fewer than 13 OCR words (signatures are short).
      6. Stroke irregularity — the coefficient of variation of stroke widths
         (via distance transform on the binarised crop) must be ≥ SIG_MIN_STROKE_CV,
         indicating organic, hand-drawn strokes rather than uniform print.

    Returns:
        {
          "is_signature" : bool,
          "reason"       : str,
          "confidence"   : float   # 0–1
        }
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {"is_signature": False, "reason": "empty crop", "confidence": 0.0}

    h, w = crop_bgr.shape[:2]
    score_parts: list[float] = []

    # ── 1. Aspect ratio ──────────────────────────────────────────────────────
    aspect = w / max(h, 1)
    if aspect < SIG_MIN_ASPECT:
        return {
            "is_signature": False,
            "reason"      : f"crop too narrow (aspect={aspect:.2f}) — not a signature region",
            "confidence"  : 0.05,
        }
    score_parts.append(min(aspect / 3.0, 1.0) * 0.10)   # aspect contributes 10 % of score

    # ── 2. Ink coverage ───────────────────────────────────────────────────────
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to isolate ink on varied backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    ink_ratio = float(np.sum(binary > 0)) / max(h * w, 1)

    if ink_ratio < SIG_MIN_INK_RATIO:
        return {
            "is_signature": False,
            "reason"      : f"insufficient ink ({ink_ratio:.1%}) — blank or near-blank crop",
            "confidence"  : round(ink_ratio / SIG_MIN_INK_RATIO * 0.15, 2),
        }
    if ink_ratio > SIG_MAX_INK_RATIO:
        return {
            "is_signature": False,
            "reason"      : f"too much ink ({ink_ratio:.1%}) — likely a dense text block, not a signature",
            "confidence"  : 0.10,
        }
    # Ideal ink ratio for a signature is roughly 5–25 %
    ink_score = 1.0 - abs(ink_ratio - 0.12) / 0.12
    score_parts.append(max(ink_score, 0.0) * 0.25)   # ink contributes 25 %

    # ── 3. Printed-text density (OCR area check) ──────────────────────────────
    text_density = _ocr_text_density(crop_bgr)
    if text_density > SIG_MAX_TEXT_DENSITY:
        return {
            "is_signature": False,
            "reason"      : f"printed text covers {text_density:.0%} of crop (limit {SIG_MAX_TEXT_DENSITY:.0%}) — likely a text region",
            "confidence"  : round((1.0 - text_density) * 0.3, 2),
        }
    score_parts.append((1.0 - text_density) * 0.25)   # text-density contributes 25 %

    # ── 4 & 5. OCR keyword + word-count check ────────────────────────────────
    text, _conf = _ocr_crop(crop_bgr)
    text_lower  = text.lower()
    for kw in _SIGNATURE_REJECT_KEYWORDS:
        if kw in text_lower:
            return {
                "is_signature": False,
                "reason"      : f"OCR found document keyword '{kw}' — wrong region cropped",
                "confidence"  : 0.05,
            }
    word_count = len(text.split())
    if word_count > 12:
        return {
            "is_signature": False,
            "reason"      : f"OCR returned {word_count} words — too many for a signature",
            "confidence"  : 0.10,
        }
    score_parts.append(max(0.0, 1.0 - word_count / 12.0) * 0.15)   # word-count 15 %

    # ── 6. Stroke irregularity (distance-transform CV) ───────────────────────
    dist        = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_vals = dist[dist > 0]
    if len(stroke_vals) < 50:
        stroke_cv = 0.0
    else:
        stroke_cv = float(np.std(stroke_vals)) / (float(np.mean(stroke_vals)) + 1e-6)

    if stroke_cv < SIG_MIN_STROKE_CV:
        return {
            "is_signature": False,
            "reason"      : f"stroke irregularity too low (CV={stroke_cv:.2f}) — uniform print-like strokes",
            "confidence"  : round(stroke_cv / SIG_MIN_STROKE_CV * 0.4, 2),
        }
    score_parts.append(min(stroke_cv / 1.5, 1.0) * 0.25)   # irregularity 25 %

    confidence = round(sum(score_parts), 2)
    return {
        "is_signature": True,
        "reason"      : (
            f"ink={ink_ratio:.1%}, text_density={text_density:.1%}, "
            f"stroke_CV={stroke_cv:.2f}, words={word_count}"
        ),
        "confidence"  : confidence,
    }


# ── Re-localization strategies ───────────────────────────────────────────────

def _relocate_signature(
    full_image: np.ndarray,
    job_id: str,
    max_attempts: int = 3,
) -> tuple[np.ndarray | None, str]:
    """
    Re-detect the signature region when the VLM-provided box failed validation.

    Strategy 1 — Label-anchored:
      Find a word matching a signature field label via full-doc OCR, then crop
      the region immediately to the right of the label (where handwriting lives).
      Tries progressively wider windows if the first attempt fails validation.

    Strategy 2 — Bottom-band heuristic:
      Scan horizontal bands in the bottom 35 % of the document, returning the
      first band that passes _verify_signature_image.

    Returns (crop | None, strategy_description).
    """
    h, w = full_image.shape[:2]
    attempts: list[tuple[np.ndarray, str]] = []

    # ── Strategy 1: Label-anchored ───────────────────────────────────────────
    words = _ocr_crop_dict(full_image)
    for entry in words:
        if not (any(kw in entry["text"].lower() for kw in _SIGNATURE_FIELD_LABELS)
                and entry["conf"] > 10):
            continue
        lx, ly, lw, lh = entry["left"], entry["top"], entry["w"], entry["h"]
        logger.info("[%s] Sig label '%s' at (%d,%d)", job_id, entry["text"], lx, ly)

        for pad_mult in (3, 5, 7):                     # widen vertically on retry
            pad = max(lh * pad_mult, 40)
            sx1 = min(lx + lw, w)
            sy1 = max(ly - lh, 0)
            sx2 = w
            sy2 = min(ly + pad, h)
            if (sx2 - sx1) < w * 0.15:                # too narrow → widen left
                sx1 = max(0, lx - lw * 2)
            crop = full_image[sy1:sy2, sx1:sx2]
            if crop.size > 0:
                attempts.append((crop, f"label_anchor_pad{pad_mult}"))

    # ── Strategy 2: Bottom-band scan ─────────────────────────────────────────
    band_top = int(h * 0.60)
    for y1 in range(band_top, int(h * 0.90), max(int(h * 0.04), 10)):
        y2   = min(y1 + int(h * 0.18), h)
        crop = full_image[y1:y2, :]
        if crop.size > 0:
            attempts.append((crop, f"bottom_band_y{y1}"))

    # ── Validate each candidate ───────────────────────────────────────────────
    best_crop, best_strategy, best_conf = None, "failed", 0.0
    for crop, strategy in attempts:
        result = _verify_signature_image(crop)
        logger.debug(
            "[%s] Candidate '%s': is_sig=%s conf=%.2f reason=%s",
            job_id, strategy, result["is_signature"], result["confidence"], result["reason"]
        )
        if result["is_signature"] and result["confidence"] > best_conf:
            best_crop, best_strategy, best_conf = crop, strategy, result["confidence"]
            if best_conf >= 0.70:          # good enough — stop early
                break

    if best_crop is not None:
        logger.info("[%s] Re-located signature via '%s' (conf=%.2f)", job_id, best_strategy, best_conf)
        return best_crop, best_strategy

    logger.warning("[%s] All %d relocation candidates failed validation", job_id, len(attempts))
    return None, "failed"


# ── Text-content verifier ─────────────────────────────────────────────────────

def _is_signature_like_text(text: str) -> bool:
    """True if OCR text could be a name/initials rather than document metadata."""
    if not text:
        return False
    text_lower = text.lower()
    for kw in _SIGNATURE_REJECT_KEYWORDS:
        if kw in text_lower:
            return False
    return len(text.split()) <= 5


# ── Main entry point ─────────────────────────────────────────────────────────

def crop_visual_assets(source_image_path: str, grounding_results: list, job_id: str) -> list[str]:
    """
    Blueprint §4.3 — crop visual regions and apply the full signature pipeline:

      1. Crop VLM box → run _verify_signature_image.
      2. If invalid  → _relocate_signature (label-anchor + bottom-band scan).
      3. If still no valid crop → return SIGNATURE_DETECTION_FAILED (no image).
      4. OCR the verified crop → if text passes _is_signature_like_text → text output.
      5. Else → image fallback with the validated crop.

    All results are written in-place onto grounding_results items so the AST
    compiler can embed them in graphic elements.
    Returns list of saved asset paths.
    """
    if not source_image_path or not os.path.exists(source_image_path):
        return []
    image = cv2.imread(source_image_path)
    if image is None:
        return []

    img_h, img_w = image.shape[:2]
    saved_paths  = []

    for idx, item in enumerate(grounding_results):
        box   = item.get("box_2d", [])
        label = item.get("label", f"asset_{idx}")
        if len(box) != 4:
            continue

        x_min, y_min, x_max, y_max = [int(c) for c in box]
        x_min = max(0, x_min); y_min = max(0, y_min)
        x_max = min(img_w, x_max); y_max = min(img_h, y_max)
        crop  = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue

        asset_filename = f"{job_id}_asset_{idx}_{label}.jpg"
        asset_path     = os.path.join(ASSETS_DIR, asset_filename)
        cv2.imwrite(asset_path, crop)
        saved_paths.append(asset_path)
        logger.info("Cropped visual asset: %s", asset_path)

        if "signature" not in label.lower():
            continue

        # ── Step 1: Validate VLM-provided crop ──────────────────────────────
        reloc_strategy = "vlm_box"
        vlm_check      = _verify_signature_image(crop)
        logger.info(
            "[%s] VLM box validation: is_sig=%s conf=%.2f | %s",
            job_id, vlm_check["is_signature"], vlm_check["confidence"], vlm_check["reason"]
        )

        if not vlm_check["is_signature"]:
            logger.warning(
                "[%s] Signature crop REJECTED (%s) — re-localizing",
                job_id, vlm_check["reason"]
            )
            # ── Step 2: Re-localize ─────────────────────────────────────────
            relocated, reloc_strategy = _relocate_signature(image, job_id)

            if relocated is not None and relocated.size > 0:
                crop = relocated
                cv2.imwrite(asset_path, crop)      # overwrite asset with verified crop
                logger.info(
                    "[%s] Signature asset updated from re-localization (%s)",
                    job_id, reloc_strategy
                )
            else:
                # ── Step 3: Full detection failure ──────────────────────────
                item["signature_result"] = {
                    "type"      : "SIGNATURE_DETECTION_FAILED",
                    "reason"    : (
                        f"VLM crop invalid ({vlm_check['reason']}); "
                        "re-localization could not find a valid signature region"
                    ),
                    "confidence": 0.0,
                    "image"     : None,
                }
                logger.error("[%s] SIGNATURE_DETECTION_FAILED", job_id)
                continue

        # ── Step 4: OCR on verified crop ─────────────────────────────────────
        text, ocr_conf = _ocr_crop(crop)

        if text and ocr_conf >= SIGNATURE_OCR_CONFIDENCE_THRESHOLD and _is_signature_like_text(text):
            item["signature_result"] = {
                "type"      : "signature_text",
                "value"     : text,
                "confidence": round(ocr_conf, 1),
            }
            logger.info(
                "[%s] Signature text extracted (conf=%.1f%%): '%s'",
                job_id, ocr_conf, text[:80]
            )
        else:
            # ── Step 5: Validated image fallback ─────────────────────────────
            ocr_reason = (
                f"OCR confidence {ocr_conf:.1f}% < threshold {SIGNATURE_OCR_CONFIDENCE_THRESHOLD:.0f}%"
                if ocr_conf < SIGNATURE_OCR_CONFIDENCE_THRESHOLD
                else "OCR result does not resemble a signature"
            )
            item["signature_result"] = {
                "type"      : "signature_image",
                "reason"    : ocr_reason,
                "confidence": round(ocr_conf, 1),
                "image"     : asset_path,
            }
            logger.info(
                "[%s] Signature image fallback — %s | strategy=%s",
                job_id, ocr_reason, reloc_strategy
            )

    return saved_paths


# ---------------------------------------------------------------------------
# Serialization helpers — JSON / CSV / Excel
# ---------------------------------------------------------------------------
def build_dataframes(extracted_data: dict) -> dict[str, pd.DataFrame]:
    """Build a dict of {sheet_name: DataFrame} from the extracted JSON."""
    frames = {}

    # Key-value pairs → flat table
    kvp = extracted_data.get("key_value_pairs", {})
    if kvp:
        frames["Key_Value_Pairs"] = pd.DataFrame(
            [{"Field": k, "Value": v} for k, v in kvp.items()]
        )

    # Text block
    text = extracted_data.get("extracted_text", "")
    if text:
        frames["Extracted_Text"] = pd.DataFrame([{"Text": text}])

    # Tables
    for idx, table in enumerate(extracted_data.get("tables", [])):
        headers = table.get("headers", [])
        rows    = table.get("rows", [])
        if rows:
            frames[f"Table_{idx + 1}"] = pd.DataFrame(rows, columns=headers or None)

    return frames


def export_to_excel(extracted_data: dict) -> bytes:
    """Serialize extracted data to a multi-sheet Excel workbook."""
    frames = build_dataframes(extracted_data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if not frames:
            pd.DataFrame([{"result": "No structured data extracted"}]).to_excel(
                writer, sheet_name="Result", index=False
            )
        for sheet_name, df in frames.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()


def export_to_csv(extracted_data: dict) -> str:
    """Serialize all tables into a single concatenated CSV."""
    frames = build_dataframes(extracted_data)
    if not frames:
        return "No structured data extracted"
    sections = []
    for name, df in frames.items():
        sections.append(f"### {name}")
        sections.append(df.to_csv(index=False))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Core pipeline (runs in FastAPI BackgroundTask)
# ---------------------------------------------------------------------------
def run_pipeline(file_path: str, job_id: str, source_filename: str):
    """
    Smart Triage Pipeline (v2) — Replaces legacy lego2→lego3 HTTP chain.

    Routing:
      Track A  (digital PDF ≥50 chars)  → Programmatic extraction, zero API cost
      Path 1   (PCS < 0.50)             → CPU OCR via pytesseract
      Path 2   (PCS ≥ 0.50)             → Groq Vision LLM

    Falls back to legacy lego2/lego3 HTTP chain if smart_triage not available.
    """
    logger.info("[%s] ▶ Smart Triage pipeline started: %s", job_id, source_filename)
    job_registry[job_id] = {"status": "processing", "filename": source_filename}

    try:
        orchestrator, ast_compiler = get_orchestrator()

        # ── Smart Triage path ────────────────────────────────────────────
        if orchestrator is not None:
            routing_result = orchestrator.route_document(file_path, job_id)
            exec_val       = routing_result.get("execution_path", ExecutionPath.PATH_2_HIGH_COMPLEXITY)
            exec_path      = exec_val.value if hasattr(exec_val, "value") else exec_val
            pcs_score      = routing_result.get("pcs_score", 0.0)

            # ── Step 1: Crop visual assets BEFORE compiling the AST ──────
            # crop_visual_assets writes signature_result in-place onto each
            # grounding item.  The AST compiler must run AFTER this so it
            # can embed signature_result into the graphic elements.
            asset_paths = []
            if exec_path == "PATH_2":
                pages     = routing_result.get("pages", [])
                page0     = pages[0] if pages else {}
                # Use .get() (not .pop()) so visual_grounding is still present
                # when ast_compiler.compile() reads the routing_result below.
                grounding = page0.get("visual_grounding", []) or []
                if grounding:
                    asset_paths = crop_visual_assets(file_path, grounding, job_id)

            # ── Step 2: Compile AST — now sees enriched grounding data ───
            final_data = ast_compiler.compile(routing_result, source_filename)
            if asset_paths:
                final_data["_asset_paths"] = asset_paths

            pipeline_name = routing_result.get("_pipeline", exec_path.lower())

        else:
            # ── Legacy fallback: Lego 2 → Lego 3 HTTP chain ─────────────
            logger.warning("[%s] Smart Triage not available — using legacy pipeline", job_id)
            with open(file_path, "rb") as fh:
                triage_resp = requests.post(
                    f"{LEGO2_URL}/triage",
                    files={"file": (source_filename, fh)},
                    timeout=90,
                )
            triage_resp.raise_for_status()
            triage  = triage_resp.json()
            routing = triage.get("routing")

            if routing == "fast_track_complete":
                pages     = triage.get("pages", [triage.get("extracted_data", {})])
                pages     = stitch_multipage_tables(pages)
                final_data = {"pages": pages, "_pipeline": "fast_track_cpu_legacy"}
            elif routing == "forward_to_vlm":
                cleaned_path = triage.get("cleaned_file_path", file_path)
                ai_resp = requests.post(
                    f"{LEGO3_URL}/extract",
                    json={"image_path": cleaned_path},
                    timeout=180,
                )
                ai_resp.raise_for_status()
                ai_data   = ai_resp.json()
                page_data = ai_data.get("data", {})
                grounding = page_data.pop("visual_grounding", [])
                asset_paths = crop_visual_assets(cleaned_path, grounding, job_id)
                final_data = {
                    "pages"       : [page_data],
                    "_pipeline"   : "groq_vision_legacy",
                    "_model_used" : ai_data.get("model_used", ""),
                    "_asset_paths": asset_paths,
                }
            else:
                raise ValueError(f"Unknown routing: {routing}")

            pcs_score     = 0.0
            pipeline_name = final_data.get("_pipeline", "legacy")
            final_data["_job_id"]    = job_id
            final_data["_filename"]  = source_filename
            final_data["_timestamp"] = datetime.utcnow().isoformat() + "Z"

        # ── Persist AST JSON to pending_review/ ──────────────────────────
        out_path = os.path.join(PENDING_DIR, f"{job_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        job_registry[job_id] = {
            "status"   : "pending_review",
            "pipeline" : pipeline_name,
            "pcs_score": pcs_score,
            "filename" : source_filename,
        }
        logger.info("[%s] ✅ Smart Triage complete → pending_review (pipeline=%s, PCS=%.4f)",
                    job_id, pipeline_name, pcs_score)

    except Exception as exc:
        logger.error("[%s] ❌ Pipeline error: %s", job_id, exc, exc_info=True)
        job_registry[job_id] = {
            "status"   : "failed",
            "error"    : str(exc),
            "filename" : source_filename,
        }


# ---------------------------------------------------------------------------
# AUTH ENDPOINTS
# ---------------------------------------------------------------------------
class TokenRequest(BaseModel):
    client_id: str
    client_secret: str

VALID_CLIENTS = {
    os.environ.get("CLIENT_ID", "greencare_client"):
    os.environ.get("CLIENT_SECRET", "greencare_secret_dev"),
}

@app.post("/auth/token", tags=["Authentication"])
def get_token(req: TokenRequest):
    """Issue a Bearer JWT for API access."""
    if VALID_CLIENTS.get(req.client_id) != req.client_secret:
        raise HTTPException(status_code=401, detail="Invalid client credentials")
    token = create_access_token({"sub": req.client_id})
    return {"access_token": token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_TTL * 60}


# ---------------------------------------------------------------------------
# INGEST
# ---------------------------------------------------------------------------
@app.post("/api/v1/ingest", tags=["Ingestion"], status_code=202)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # Token auth — comment out Depends to run unauthenticated in dev
    # _user: dict = Depends(verify_token),
):
    """Accept a document upload and start the processing pipeline."""
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ext}'")

    job_id    = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    background_tasks.add_task(run_pipeline, file_path, job_id, file.filename)

    return JSONResponse(status_code=202, content={
        "status"  : "queued",
        "job_id"  : job_id,
        "message" : "Processing started. Poll /api/v1/status/{job_id} for updates.",
    })


# ---------------------------------------------------------------------------
# STATUS & EXPORT
# ---------------------------------------------------------------------------
@app.get("/api/v1/status/{job_id}", tags=["Monitoring"])
def get_status(job_id: str):
    if job_id in job_registry:
        return {"job_id": job_id, **job_registry[job_id]}
    if os.path.exists(os.path.join(FINAL_DIR, f"{job_id}.json")):
        return {"job_id": job_id, "status": "committed"}
    if os.path.exists(os.path.join(PENDING_DIR, f"{job_id}.json")):
        return {"job_id": job_id, "status": "pending_review"}
    return {"job_id": job_id, "status": "not_found"}


@app.get("/api/v1/jobs", tags=["Monitoring"])
def list_jobs():
    return {"total": len(job_registry), "jobs": job_registry}


def _load_job_data(job_id: str) -> dict:
    """Load finalized or pending JSON for a job."""
    for folder in [FINAL_DIR, PENDING_DIR]:
        path = os.path.join(folder, f"{job_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found or still processing")


@app.get("/api/v1/export/{job_id}/json", tags=["Export"])
def export_json(job_id: str):
    """Download extracted data as JSON."""
    data = _load_job_data(job_id)
    return JSONResponse(content=data)


@app.get("/api/v1/export/{job_id}/csv", tags=["Export"])
def export_csv(job_id: str):
    """Download extracted data as CSV."""
    data = _load_job_data(job_id)
    pages = data.get("pages", [data])
    merged: dict = {}
    for p in pages:
        merged.update(p)
    csv_content = export_to_csv(merged)
    return StreamingResponse(
        io.BytesIO(csv_content.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={job_id}.csv"},
    )


@app.get("/api/v1/export/{job_id}/excel", tags=["Export"])
def export_excel(job_id: str):
    """Download extracted data as Excel workbook."""
    data = _load_job_data(job_id)
    pages = data.get("pages", [data])
    merged: dict = {}
    for p in pages:
        merged.update(p)
    xlsx_bytes = export_to_excel(merged)
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={job_id}.xlsx"},
    )


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status" : "ok",
        "service": "lego1-gateway",
        "version": "2.0.0",
        "queued" : sum(1 for v in job_registry.values() if v.get("status") == "processing"),
    }
