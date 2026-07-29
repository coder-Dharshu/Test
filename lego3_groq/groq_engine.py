"""
Greencare AI — Lego 3: Local VLM Engine (Ollama)
===================================================
Responsibilities (per blueprint §3.3):
  • Ingests raw image matrices (base64 encoded)
  • Executes local Ollama model (Qwen 3.6 27b)
  • Enforces strict output syntax via format=json (§4.4 CFG masking)
  • Returns formatted Markdown tables, text streams, and visual grounding coordinates

Runs on: Port 8002
"""

import os
import json
import time
import base64
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — Lego 3: Local VLM Engine",
    description="Multimodal VLM engine using local Ollama.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# §4.4 Constrained Grammar Decoding — System Prompt
# The response_format=json_object parameter is the API-level equivalent of
# the CFG masking layer described in §4.4. This prompt defines the exact
# JSON schema the model must produce; tokens outside it are masked to -∞.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an Intelligent Document Processing (IDP) engine — the equivalent of a locally 
compiled Qwen2.5-VL-7B-Instruct model running TensorRT-LLM on a Jetson edge server.

Your task is to perform a SINGLE FORWARD PASS on the provided document image and extract
ALL of the following simultaneously (no cascading — one pass, like a unified VLM):

1. Layout detection and full text extraction (print + handwriting)
2. Table structure detection with headers and row data
3. Key-value field extraction (named fields like Invoice No, Patient Name, Date, Total, etc.)
4. Visual grounding — detect and localize logos, diagrams, signatures, stamps
5. Document type classification
6. Confidence assessment

You MUST respond ONLY with a valid JSON object matching this EXACT schema 
(§4.4 Constrained Grammar Decoding — no filler text, no markdown wrappers):

{
  "document_type": "<string: invoice|medical_form|handwritten_note|shipping_label|table|receipt|form|unknown>",
  "language": "<string: ISO 639-1 code, e.g. en, hi, de, zh>",
  "extracted_text": "<string: full verbatim text, preserve line breaks with \\n>",
  "markdown_tables": "<string: all tables rendered in Markdown format, or empty string>",
  "key_value_pairs": {
    "<field_name>": "<field_value>"
  },
  "tables": [
    {
      "table_index": 0,
      "headers": ["<col1>", "<col2>"],
      "rows": [
        ["<val1>", "<val2>"]
      ]
    }
  ],
  "visual_grounding": [
    {
      "box_2d": [x_min, y_min, x_max, y_max],
      "label": "<logo|diagram|signature|stamp|image|chart>"
    }
  ],
  "handwriting_detected": false,
  "curved_text_detected": false,
  "page_count_estimate": 1,
  "confidence_warning": false,
  "confidence_warning_reason": null
}

Rules:
- DO NOT output anything outside the JSON object. No markdown code fences. No prose.
- null for inapplicable fields, [] for empty arrays, {} for empty objects.
- Preserve original spelling and casing in extracted_text.
- For visual_grounding: if no logos/diagrams found, return [].
- box_2d coordinates are pixel offsets from the image top-left corner.
- For curved or rotated text (package labels, bottles): transcribe the text as if unwarped.
"""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ImagePayload(BaseModel):
    image_path           : str            = Field(..., description="Path to the original/clean image on disk.")
    binarized_image_path : Optional[str]  = Field(None, description="Optional path to a binarized (B&W) version for better table reading.")

class ExtractionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status    : str
    model_used: str
    data      : dict



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_image(image_path: str) -> tuple[str, str]:
    """Read image file and return (base64_string, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime, _ = mimetypes.guess_type(str(path))
    if mime not in {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


# ---------------------------------------------------------------------------
# Ollama Local VLM Integration (Qwen2.5-VL)
# ---------------------------------------------------------------------------
def call_ollama(
    b64_image: str,
    b64_binarized: Optional[str] = None,
    model: str = "qwen3.6-vl:27b",
    timeout: int = 60,
) -> dict:
    """
    Call local Ollama server running Qwen3.6-VL-27B.
    Strips 'data:image/...' headers if present; Ollama expects raw base64.
    """
    import urllib.request
    import urllib.error

    images = [b64_image]
    if b64_binarized:
        images.append(b64_binarized)

    user_prompt = (
        "Perform a full single-pass extraction of this document image. "
        + (
            "I am providing TWO versions of the same image: (1) original color photo, "
            "(2) high-contrast binarized version for dense text and tables. "
            if b64_binarized else ""
        )
        + "Return the complete JSON as per the schema in the system prompt."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt, "images": images},
        ],
        "stream": False,
        "format": "json",
    }

    ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
    req = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        content = res_data.get("message", {}).get("content", "")
        return json.loads(content)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status"       : "ok",
        "service"      : "lego3-local-vlm-engine",
        "ollama_model" : os.environ.get("OLLAMA_MODEL", "qwen3.6-vl:27b"),
    }



@app.post("/extract", tags=["Extraction"], response_model=ExtractionResponse)
def extract(payload: ImagePayload):
    """
    Single-pass multimodal extraction.
    Tries local Ollama (qwen3.6-vl:27b) first.
    If Ollama is not installed/running or fails, falls back gracefully to Groq Vision.
    """
    logger.info("Extraction request: %s (binarized=%s)", payload.image_path, payload.binarized_image_path or "none")
    try:
        b64, mime = encode_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Load binarized image if provided
    b64_bin, mime_bin = None, "image/jpeg"
    if payload.binarized_image_path:
        try:
            b64_bin, mime_bin = encode_image(payload.binarized_image_path)
        except FileNotFoundError:
            logger.warning("Binarized image not found at %s — using single-image mode", payload.binarized_image_path)

    # 1. Try Ollama (qwen3.6-vl:27b)
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3.6-vl:27b")

    try:
        logger.info("Attempting local VLM via Ollama (model=%s)...", ollama_model)
        result = call_ollama(b64, b64_binarized=b64_bin, model=ollama_model)
        logger.info("Ollama extraction successful!")
        return ExtractionResponse(
            status    ="success",
            model_used=f"ollama/{ollama_model}",
            data      =result,
        )
    except Exception as exc:
        logger.error("Ollama extraction failed: %s", exc)
        raise HTTPException(
            status_code=502, 
            detail=f"Local Ollama API error or model unavailable: {exc}"
        )

