"""
IDP — Lego 3: Qwen Vision Engine  [v2.1 Production Refactor]
============================================================
Unified OpenAI-compatible multimodal inference engine.
Supports:
  - 'groq': High-speed accelerated cloud VLM (Groq API)
  - 'local': Local offline inference (Ollama / vLLM)
  - 'openai_compatible': Any OpenAI-compliant multimodal endpoint

Features:
  - Single OpenAI client code path across all backends.
  - ALLOW_CLOUD_ESCALATION guardrail: skips cloud and flags 'needs_review' when false.
  - Exponential backoff + hard timeout + max retries on 429/5xx/timeout.
  - Pydantic output schema validation with 1-shot repair retry.
  - Configurable sampling (temperature, max_tokens, response_format).
  - Dynamic logging of actual provider model ID.
  - Endpoints: GET /health, GET /models, POST /extract, POST /route.
"""

from __future__ import annotations

import os
import json
import time
import base64
import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from config import config
from common.errors import ErrorCode, IDPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="IDP — Lego 3: Vision Engine",
    description="Multimodal VLM engine with unified OpenAI client, retry backoff, and Pydantic validation.",
    version="2.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Output Pydantic Schemas for Schema Validation
# ---------------------------------------------------------------------------
class VlmTableItem(BaseModel):
    table_index: int = 0
    headers: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)


class VlmVisualGroundingItem(BaseModel):
    box_2d: List[Union[int, float]] = Field(default_factory=list)
    label: str = "item"


class VlmExtractionOutput(BaseModel):
    document_type: str = "unknown"
    language: str = "en"
    extracted_text: str = ""
    markdown_tables: Optional[str] = ""
    key_value_pairs: dict[str, Any] = Field(default_factory=dict)
    tables: List[VlmTableItem] = Field(default_factory=list)
    visual_grounding: List[VlmVisualGroundingItem] = Field(default_factory=list)
    handwriting_detected: bool = False
    curved_text_detected: Optional[bool] = False
    page_count_estimate: Optional[int] = 1
    confidence_warning: bool = False
    confidence_warning_reason: Optional[str] = None


class ImagePayload(BaseModel):
    image_path: str = Field(..., description="Path to the preprocessed image on disk.")


class ExtractionResponse(BaseModel):
    status: str
    model_used: str
    data: dict
    processing_time_ms: Optional[float] = None
    validation_attempts: Optional[int] = 1


# ---------------------------------------------------------------------------
# Single Unified Client Setup
# ---------------------------------------------------------------------------
def get_vlm_client() -> tuple[Any, str, str]:
    """
    Returns (client, client_description, base_url).
    Uses a single OpenAI-compatible client across all backends.
    """
    from openai import OpenAI

    backend = config.INFERENCE_BACKEND
    if backend == "local":
        base_url = config.LOCAL_MODEL_URL
        api_key = config.LOCAL_API_KEY or "local-qwen"
        client = OpenAI(base_url=base_url, api_key=api_key)
        desc = f"local ({base_url})"
        return client, desc, base_url

    elif backend == "groq":
        base_url = "https://api.groq.com/openai/v1"
        api_key = config.GROQ_API_KEY
        client = OpenAI(base_url=base_url, api_key=api_key or "unconfigured")
        desc = "cloud (Groq accelerated)"
        return client, desc, base_url

    else:  # openai_compatible
        base_url = config.OPENAI_COMPATIBLE_URL or "https://api.openai.com/v1"
        api_key = config.OPENAI_API_KEY or config.GROQ_API_KEY or "unconfigured"
        client = OpenAI(base_url=base_url, api_key=api_key)
        desc = f"cloud ({base_url})"
        return client, desc, base_url


client, client_desc, client_base_url = get_vlm_client()

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an Intelligent Document Processing (IDP) engine powered by a Vision Language Model.
Perform a SINGLE FORWARD PASS on the document image and extract:
1. Layout and full verbatim text (print + handwriting)
2. Table structure detection with headers and rows
3. Key-value field extraction (named entities, dates, totals, IDs)
4. Visual grounding — detect bounding boxes for logos, signatures, stamps: [ymin, xmin, ymax, xmax]
5. Document type classification & confidence assessment

You MUST respond ONLY with a valid JSON object matching this schema:
{
  "document_type": "<invoice|receipt|medical_form|shipping_label|table|form|handwritten_note|unknown>",
  "language": "en",
  "extracted_text": "<verbatim text>",
  "markdown_tables": "<markdown table or empty string>",
  "key_value_pairs": { "<field>": "<value>" },
  "tables": [
    {
      "table_index": 0,
      "headers": ["<col1>", "<col2>"],
      "rows": [["<val1>", "<val2>"]]
    }
  ],
  "visual_grounding": [
    {
      "box_2d": [ymin, xmin, ymax, xmax],
      "label": "<logo|signature|stamp|chart|image>"
    }
  ],
  "handwriting_detected": false,
  "curved_text_detected": false,
  "page_count_estimate": 1,
  "confidence_warning": false,
  "confidence_warning_reason": null
}

Rules:
- Respond strictly with JSON. No prose, no markdown code fence wrappers.
- Never invent information not visible in the document.
"""

ROUTER_SYSTEM_PROMPT = """
You are an image classifier and routing system.
1. Determine if the image is a text document or a general photo/graphic.
2. For text documents, return: {"type": "text_document", "text": "<extracted text>"}
3. For general images, return: {"type": "general_image", "image": "<original_image>", "detected_text": "<text if any>"}
Respond strictly with JSON.
"""

REPAIR_SYSTEM_PROMPT = """
You are an automated JSON schema repair engine.
You will be provided with an invalid JSON string and the schema validation error.
Repair the JSON string so that it strictly adheres to the requested schema.
Output ONLY the repaired JSON object without markdown fences or comments.
"""

# ---------------------------------------------------------------------------
# Helper Functions
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


def extract_json_from_text(text: str) -> dict:
    """Robustly parse JSON from raw LLM output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse valid JSON from text: {text[:200]}...")


# ---------------------------------------------------------------------------
# Core VLM Execution with Backoff, Guardrail & Pydantic Repair
# ---------------------------------------------------------------------------
def call_qwen(b64_image: str, mime: str, request_id: str = "") -> dict:
    """
    Executes VLM call with:
      1. Privacy guardrail (ALLOW_CLOUD_ESCALATION check)
      2. Exponential backoff on 429/5xx/timeout
      3. Pydantic validation + 1-shot repair retry
      4. Never drops job — returns 'needs_review' on fatal error
    """
    # ── Guardrail check ───────────────────────────────────────────────────────
    is_cloud = config.INFERENCE_BACKEND in ("groq", "openai_compatible")
    if is_cloud and not config.ALLOW_CLOUD_ESCALATION:
        logger.warning("[%s] Cloud escalation blocked by policy (ALLOW_CLOUD_ESCALATION=false)", request_id)
        return {
            "status": "needs_review",
            "model_used": config.QWEN_MODEL,
            "data": {
                "reason": "cloud_disabled",
                "error_code": ErrorCode.CLOUD_DISABLED.value,
                "message": "Cloud escalation disabled by policy. Document requires manual review.",
                "extracted_text": "[Cloud escalation disabled by policy — manual review required]",
                "key_value_pairs": {},
                "tables": [],
            },
            "validation_attempts": 0,
        }

    # ── Configurable parameters ───────────────────────────────────────────────
    model_to_use = config.LOCAL_MODEL_NAME if config.INFERENCE_BACKEND == "local" else config.QWEN_MODEL
    retries = config.VLM_MAX_RETRIES
    timeout_sec = config.VLM_HARD_TIMEOUT
    temperature = config.TEMPERATURE
    max_tokens = config.MAX_TOKENS

    raw_content = ""
    actual_model_id = model_to_use
    last_error: Optional[Exception] = None

    # ── Execution Loop with Exponential Backoff ──────────────────────────────
    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "[%s] VLM attempt %d/%d (backend=%s, model=%s)",
                request_id, attempt, retries, config.INFERENCE_BACKEND, model_to_use
            )

            kwargs: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all document entities adhering strictly to schema."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
                        ],
                    },
                ],
                "model": model_to_use,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout_sec,
            }

            if config.RESPONSE_FORMAT == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

            completion = client.chat.completions.create(**kwargs)
            raw_content = completion.choices[0].message.content or ""
            actual_model_id = getattr(completion, "model", None) or model_to_use
            logger.info("[%s] VLM received response (%d chars, model=%s)", request_id, len(raw_content), actual_model_id)
            break

        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()
            is_429 = "rate_limit" in err_str or "429" in err_str
            is_timeout = "timeout" in err_str or "timed out" in err_str

            if attempt < retries:
                wait_sec = 2 ** (attempt - 1)
                logger.warning("[%s] VLM retryable error on attempt %d (%s) — sleeping %ds", request_id, attempt, exc, wait_sec)
                time.sleep(wait_sec)
            else:
                logger.error("[%s] VLM exhausted all %d attempts: %s", request_id, retries, exc)
                code = ErrorCode.VLM_RATE_LIMITED.value if is_429 else (ErrorCode.VLM_TIMEOUT.value if is_timeout else ErrorCode.VLM_UNAVAILABLE.value)
                return {
                    "status": "needs_review",
                    "model_used": actual_model_id,
                    "data": {
                        "reason": "vlm_unavailable",
                        "error_code": code,
                        "message": f"VLM failure after {retries} attempts: {exc}",
                        "extracted_text": "[VLM service error — routed to human review]",
                        "key_value_pairs": {},
                        "tables": [],
                    },
                    "validation_attempts": 0,
                }

    # ── Pydantic Schema Validation & Repair Retry ────────────────────────────
    validation_attempts = 1
    parsed_json: dict[str, Any] = {}

    try:
        parsed_json = extract_json_from_text(raw_content)
        VlmExtractionOutput.model_validate(parsed_json)
        # Validation passed on first attempt
        return {
            "status": "success",
            "model_used": actual_model_id,
            "data": parsed_json,
            "validation_attempts": 1,
        }
    except (ValueError, ValidationError) as val_err:
        logger.warning("[%s] Initial JSON schema validation failed: %s. Initiating repair retry.", request_id, val_err)
        validation_attempts = 2

        # ── 1-Shot Repair Retry ──────────────────────────────────────────────
        try:
            repair_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Original output had error: {val_err}\n"
                            f"Raw invalid content:\n{raw_content}\n\n"
                            "Please output only the corrected JSON matching the schema."
                        ),
                    },
                ],
                model=model_to_use,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout_sec,
                response_format={"type": "json_object"} if config.RESPONSE_FORMAT == "json_object" else None,
            )
            repaired_raw = repair_completion.choices[0].message.content or ""
            repaired_json = extract_json_from_text(repaired_raw)
            VlmExtractionOutput.model_validate(repaired_json)
            logger.info("[%s] Repair retry successful!", request_id)
            return {
                "status": "success",
                "model_used": getattr(repair_completion, "model", None) or actual_model_id,
                "data": repaired_json,
                "validation_attempts": 2,
            }

        except Exception as repair_exc:
            logger.error("[%s] Repair retry failed: %s. Forwarding to pending_review with validation_failed=true", request_id, repair_exc)
            fallback_data = parsed_json if isinstance(parsed_json, dict) else {"raw_content": raw_content}
            fallback_data["validation_failed"] = True
            fallback_data["validation_attempts"] = 2
            fallback_data["validation_error"] = str(repair_exc)
            fallback_data["reason"] = "validation_failed"
            return {
                "status": "needs_review",
                "model_used": actual_model_id,
                "data": fallback_data,
                "validation_attempts": 2,
            }


def call_qwen_router(b64_image: str, mime: str, image_path: str, request_id: str = "") -> dict:
    """Classifies image into text document or general image."""
    model_to_use = config.LOCAL_MODEL_NAME if config.INFERENCE_BACKEND == "local" else config.QWEN_MODEL
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this image and extract text if present."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
                    ],
                },
            ],
            model=model_to_use,
            temperature=0.0,
            max_tokens=1024,
            timeout=config.VLM_HARD_TIMEOUT,
        )
        raw = completion.choices[0].message.content or "{}"
        result = extract_json_from_text(raw)
        if result.get("type") == "general_image":
            result["image"] = image_path
        return result
    except Exception as exc:
        logger.error("[%s] Router classification failed: %s", request_id, exc)
        return {"type": "text_document", "text": ""}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    """Uniform health check reporting backend, configuration, and connectivity."""
    reachable = False
    try:
        # Check endpoint reachability
        if config.INFERENCE_BACKEND == "local":
            import urllib.request
            req = urllib.request.Request(f"{config.LOCAL_MODEL_URL}/models", headers={"User-Agent": "IDP-HealthCheck"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                reachable = resp.status < 500
        else:
            reachable = bool(config.GROQ_API_KEY and len(config.GROQ_API_KEY) > 5)
    except Exception:
        reachable = False

    return {
        "service": "lego3-qwen",
        "status": "ok",
        "version": "2.1.0",
        "backend": config.INFERENCE_BACKEND,
        "model": config.LOCAL_MODEL_NAME if config.INFERENCE_BACKEND == "local" else config.QWEN_MODEL,
        "local_url": config.LOCAL_MODEL_URL,
        "cloud_allowed": config.ALLOW_CLOUD_ESCALATION,
        "reachable": reachable,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/models", tags=["Configuration"])
def list_models():
    """Returns available vision models from configuration and attempts a live provider check."""
    default_models = [
        {"model_id": config.QWEN_MODEL, "backend": config.INFERENCE_BACKEND, "default": True},
        {"model_id": "qwen/qwen3.8-27b", "backend": "groq", "vision": True},
        {"model_id": "qwen3.6:27b", "backend": "local", "vision": True},
        {"model_id": "qwen/qwen2.5-vl-72b-instruct", "backend": "groq", "vision": True},
    ]

    live_models: List[str] = []
    try:
        provider_models = client.models.list(timeout=3)
        for m in getattr(provider_models, "data", []):
            live_models.append(m.id)
    except Exception as exc:
        logger.debug("Live model check unavailable: %s", exc)

    return {
        "current_model": config.LOCAL_MODEL_NAME if config.INFERENCE_BACKEND == "local" else config.QWEN_MODEL,
        "backend": config.INFERENCE_BACKEND,
        "cloud_allowed": config.ALLOW_CLOUD_ESCALATION,
        "models": default_models,
        "provider_live_models": live_models[:20] if live_models else [],
    }


@app.post("/extract", tags=["Extraction"], response_model=ExtractionResponse)
def extract_document(payload: ImagePayload, request: Request):
    """Executes single-pass multimodal extraction with retry backoff and Pydantic validation."""
    req_id = request.headers.get("X-Request-ID", "req_" + os.urandom(4).hex())
    logger.info("[%s] Extraction requested for %s", req_id, payload.image_path)

    try:
        b64, mime = encode_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    t0 = time.monotonic()
    result = call_qwen(b64, mime, request_id=req_id)
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    return ExtractionResponse(
        status=result.get("status", "success"),
        model_used=result.get("model_used", config.QWEN_MODEL),
        data=result.get("data", {}),
        processing_time_ms=elapsed_ms,
        validation_attempts=result.get("validation_attempts", 1),
    )


@app.post("/route", tags=["Routing"])
def route_image(payload: ImagePayload, request: Request):
    """Classifies an image as text document or general image."""
    req_id = request.headers.get("X-Request-ID", "req_" + os.urandom(4).hex())
    try:
        b64, mime = encode_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    t0 = time.monotonic()
    result = call_qwen_router(b64, mime, payload.image_path, request_id=req_id)
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    return {
        "status": "success",
        "data": result,
        "processing_time_ms": elapsed_ms,
    }
