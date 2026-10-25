"""
Greencare AI — Lego 1: API Gateway (Port 8000)  [v2.1 Production Refactor]
==========================================================================
Primary ingestion gateway, batch distributor, and search index.

Features:
  - Error Taxonomy: Uniform {error_code, message, retryable} responses.
  - Ingestion Validation: Magic bytes, MIME type, max file size, max page count.
  - Idempotency: Supports 'Idempotency-Key' header on /api/v1/ingest.
  - Batch Limit: Strict 20-file cap server-side with BATCH_LIMIT code.
  - Dynamic Config: Directory paths loaded dynamically from config & environment.
  - Correlation IDs: Propagates X-Request-ID across all stages.
  - Webhooks: Signed with HMAC-SHA256.
"""

from __future__ import annotations

import glob
import hashlib
import hmac
import io
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import httpx
from fastapi import (
    FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks,
    Request, Response, Header
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from config import config
from common.errors import ErrorCode, IDPException
from lego1_gateway.worker import process_document_task

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic Paths
# ---------------------------------------------------------------------------
def get_upload_dir() -> str:
    d = os.environ.get("UPLOAD_DIR", config.UPLOAD_DIR)
    os.makedirs(d, exist_ok=True)
    return d

def get_pending_dir() -> str:
    d = os.environ.get("PENDING_DIR", config.PENDING_DIR)
    os.makedirs(d, exist_ok=True)
    return d

def get_final_dir() -> str:
    d = os.environ.get("FINAL_DIR", config.FINAL_DIR)
    os.makedirs(d, exist_ok=True)
    return d

def get_rejected_dir() -> str:
    d = os.environ.get("REJECTED_DIR", config.REJECTED_DIR)
    os.makedirs(d, exist_ok=True)
    return d

def get_webhook_file() -> str:
    return os.environ.get("WEBHOOK_FILE", "./webhook_registry.json")

def get_webhook_secret() -> str:
    return os.environ.get("WEBHOOK_SECRET", config.WEBHOOK_SECRET)

ALLOWED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
    ".docx", ".xlsx", ".xls", ".csv",
}

# In-memory Idempotency Store (key -> job_id)
IDEMPOTENCY_STORE: dict[str, str] = {}

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — API Gateway",
    description="High-throughput IDP Ingestion Gateway with idempotency and error taxonomy.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation ID Middleware
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

# Error Taxonomy Exception Handlers
@app.exception_handler(IDPException)
async def idp_exception_handler(request: Request, exc: IDPException):
    return exc.to_json_response()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Map standard HTTP exceptions to error taxonomy
    code = ErrorCode.INTERNAL_ERROR.value
    if exc.status_code == 400:
        code = ErrorCode.BATCH_LIMIT.value if "20 files" in str(exc.detail) else ErrorCode.UNSUPPORTED_MIME.value
    elif exc.status_code == 415:
        code = ErrorCode.UNSUPPORTED_MIME.value
    elif exc.status_code == 413:
        code = ErrorCode.FILE_TOO_LARGE.value
    elif exc.status_code == 404:
        code = ErrorCode.JOB_NOT_FOUND.value

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": code,
            "message": str(exc.detail),
            "retryable": exc.status_code in (429, 502, 503, 504),
        },
    )

# Static frontend mount if exists
if os.path.exists("web"):
    app.mount("/web", StaticFiles(directory="web"), name="web")

# ---------------------------------------------------------------------------
# Upload Validation
# ---------------------------------------------------------------------------
def validate_uploaded_content(content: bytes, filename: str) -> None:
    """Validates file size, extension, magic bytes, and PDF page counts."""
    # 1. Size check
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise IDPException(
            status_code=413,
            error_code=ErrorCode.FILE_TOO_LARGE,
            message=f"File exceeds maximum allowed size of {config.MAX_FILE_SIZE_MB}MB.",
            retryable=False,
        )

    # 2. Extension check
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise IDPException(
            status_code=415,
            error_code=ErrorCode.UNSUPPORTED_MIME,
            message=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            retryable=False,
        )

    # 3. Magic bytes inspection
    if ext == ".pdf":
        if not content.startswith(b"%PDF"):
            raise IDPException(
                status_code=400,
                error_code=ErrorCode.MAGIC_BYTES_MISMATCH,
                message="File content does not match PDF format (missing %PDF header).",
            )
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            if len(reader.pages) > config.MAX_PAGES_PER_DOC:
                raise IDPException(
                    status_code=400,
                    error_code=ErrorCode.PAGE_LIMIT_EXCEEDED,
                    message=f"PDF exceeds page limit ({len(reader.pages)} > {config.MAX_PAGES_PER_DOC}).",
                )
        except IDPException:
            raise
        except Exception:
            pass  # Allow corrupted PDF to be routed and handled in triage

    elif ext in (".jpg", ".jpeg"):
        if not content.startswith(b"\xff\xd8\xff"):
            raise IDPException(
                status_code=400,
                error_code=ErrorCode.MAGIC_BYTES_MISMATCH,
                message="File content does not match JPEG magic bytes.",
            )
    elif ext == ".png":
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise IDPException(
                status_code=400,
                error_code=ErrorCode.MAGIC_BYTES_MISMATCH,
                message="File content does not match PNG magic bytes.",
            )
    elif ext in (".docx", ".xlsx"):
        if not content.startswith(b"PK\x03\x04"):
            raise IDPException(
                status_code=400,
                error_code=ErrorCode.MAGIC_BYTES_MISMATCH,
                message=f"File content does not match {ext.upper()} archive structure.",
            )

# ---------------------------------------------------------------------------
# Job Status Helper
# ---------------------------------------------------------------------------
def _get_job_status(job_id: str) -> str:
    final_dir = get_final_dir()
    pending_dir = get_pending_dir()
    if os.path.exists(os.path.join(final_dir, f"{job_id}.json")):
        return "committed"
    if os.path.exists(os.path.join(pending_dir, f"{job_id}.json")):
        return "pending_review"
    return "processing"

# ---------------------------------------------------------------------------
# Webhook Subsystem
# ---------------------------------------------------------------------------
class WebhookRegisterRequest(BaseModel):
    url: HttpUrl
    description: Optional[str] = None

def _load_webhooks() -> dict:
    wfile = get_webhook_file()
    if not os.path.exists(wfile):
        return {}
    try:
        with open(wfile, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_webhooks(hooks: dict):
    with open(get_webhook_file(), "w", encoding="utf-8") as f:
        json.dump(hooks, f, indent=2)

def _sign_payload(payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":")).encode()
    return hmac.new(get_webhook_secret().encode(), body, hashlib.sha256).hexdigest()

async def _fire_webhooks(event: str, payload: dict):
    hooks = _load_webhooks()
    if not hooks:
        return
    sig = _sign_payload(payload)
    headers = {
        "Content-Type": "application/json",
        "X-Greencare-Event": event,
        "X-Greencare-Signature": f"sha256={sig}",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        for hook_id, hook in hooks.items():
            url = str(hook.get("url"))
            if not url:
                continue
            try:
                resp = await client.post(url, json=payload, headers=headers)
                logger.info("Webhook [%s] → %s: HTTP %d", hook_id, url, resp.status_code)
            except Exception as exc:
                logger.warning("Webhook [%s] failed: %s", hook_id, exc)

# ---------------------------------------------------------------------------
# Monitoring & Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
async def health_check():
    p = len(glob.glob(os.path.join(get_pending_dir(), "*.json")))
    c = len(glob.glob(os.path.join(get_final_dir(), "*.json")))
    r = len(glob.glob(os.path.join(get_rejected_dir(), "*.json")))
    return {
        "service": "lego1-gateway",
        "status": "ok",
        "version": "2.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queue": {"pending": p, "committed": c, "rejected": r},
    }

# ---------------------------------------------------------------------------
# Document Ingestion (Single & Batch)
# ---------------------------------------------------------------------------
@app.post("/api/v1/ingest", tags=["Ingestion"], status_code=202)
async def ingest_document(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Accepts a single document, verifies magic bytes, and dispatches to Celery worker.
    Supports Idempotency-Key header to avoid duplicate executions.
    """
    req_id = getattr(request.state, "request_id", f"req_{uuid.uuid4().hex[:8]}")

    # Check Idempotency Key
    if idempotency_key and idempotency_key in IDEMPOTENCY_STORE:
        existing_job_id = IDEMPOTENCY_STORE[idempotency_key]
        status = _get_job_status(existing_job_id)
        logger.info("[%s] Idempotency match: key=%s, existing_job=%s", req_id, idempotency_key, existing_job_id)
        return JSONResponse(
            content={
                "status": status,
                "job_id": existing_job_id,
                "idempotent_replay": True,
                "message": "Idempotent request recognized. Returning existing job record.",
            },
            status_code=200,
        )

    # Read and validate content
    content = await file.read()
    validate_uploaded_content(content, file.filename or "")

    job_id = str(uuid.uuid4())
    upload_dir = get_upload_dir()
    file_path = os.path.join(upload_dir, f"{job_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        buffer.write(content)

    if idempotency_key:
        IDEMPOTENCY_STORE[idempotency_key] = job_id

    task = process_document_task.delay(file_path, job_id)

    background_tasks.add_task(
        _fire_webhooks,
        "job.queued",
        {"job_id": job_id, "filename": file.filename, "status": "queued"},
    )

    logger.info("[%s] Queued file: %s (job=%s)", req_id, file.filename, job_id)
    return JSONResponse(
        content={
            "status": "queued",
            "job_id": job_id,
            "celery_task_id": task.id,
            "filename": file.filename,
            "message": "Document accepted. Poll /api/v1/status/{job_id} for updates.",
        },
        status_code=202,
    )


@app.post("/api/v1/batch-ingest", tags=["Ingestion"], status_code=202)
async def batch_ingest_documents(
    background_tasks: BackgroundTasks,
    request: Request,
    files: List[UploadFile] = File(...),
):
    """
    Accepts up to 20 documents in a single batch request.
    Enforces server-side BATCH_LIMIT code and validates each document.
    """
    req_id = getattr(request.state, "request_id", f"req_{uuid.uuid4().hex[:8]}")

    if not files:
        raise IDPException(
            status_code=400,
            error_code=ErrorCode.BATCH_LIMIT,
            message="No files provided in batch request.",
        )
    if len(files) > 20:
        raise IDPException(
            status_code=400,
            error_code=ErrorCode.BATCH_LIMIT,
            message=f"Maximum 20 files per batch request. Received {len(files)} files.",
        )

    results = []
    upload_dir = get_upload_dir()

    for file in files:
        filename = file.filename or "unknown"
        try:
            content = await file.read()
            validate_uploaded_content(content, filename)

            job_id = str(uuid.uuid4())
            file_path = os.path.join(upload_dir, f"{job_id}_{filename}")
            with open(file_path, "wb") as buf:
                buf.write(content)

            task = process_document_task.delay(file_path, job_id)
            background_tasks.add_task(
                _fire_webhooks,
                "job.queued",
                {"job_id": job_id, "filename": filename, "status": "queued"},
            )
            results.append({
                "filename": filename,
                "status": "queued",
                "job_id": job_id,
                "celery_task_id": task.id,
            })
            logger.info("[%s] Batch queued: %s (job=%s)", req_id, filename, job_id)

        except IDPException as exc:
            results.append({
                "filename": filename,
                "status": "rejected",
                "error_code": exc.error_code.value,
                "reason": exc.message,
            })
        except Exception as exc:
            results.append({
                "filename": filename,
                "status": "rejected",
                "error_code": ErrorCode.INTERNAL_ERROR.value,
                "reason": str(exc),
            })

    queued_count = sum(1 for r in results if r["status"] == "queued")
    rejected_count = sum(1 for r in results if r["status"] == "rejected")

    return JSONResponse(
        content={
            "batch_size": len(files),
            "total_submitted": len(files),
            "queued_count": queued_count,
            "rejected_count": rejected_count,
            "jobs": results,
        },
        status_code=202,
    )


@app.get("/api/v1/batch-status", tags=["Monitoring"])
async def batch_job_status(
    job_ids: List[str] = Query(..., description="List of job IDs"),
):
    if len(job_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 job IDs per batch-status call.")
    return {
        "results": [
            {"job_id": jid, "status": _get_job_status(jid)}
            for jid in job_ids
        ]
    }


@app.get("/api/v1/status/{job_id}", tags=["Monitoring"])
async def get_job_status(job_id: str):
    status = _get_job_status(job_id)
    response: dict = {"status": status, "job_id": job_id}

    if status == "committed":
        final_path = os.path.join(get_final_dir(), f"{job_id}.json")
        try:
            with open(final_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            response["approved_at"] = data.get("_approved_at")
            response["document_type"] = data.get("document_type") or (
                data.get("pages", [{}])[0].get("document_type") if "pages" in data else None
            )
            response["pipeline"] = data.get("_pipeline")
        except Exception:
            pass
    elif status == "pending_review":
        pending_path = os.path.join(get_pending_dir(), f"{job_id}.json")
        try:
            with open(pending_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            response["document_type"] = data.get("document_type") or (
                data.get("pages", [{}])[0].get("document_type") if "pages" in data else None
            )
            response["pipeline"] = data.get("_pipeline")
            response["validation_failed"] = data.get("validation_failed", False)
            response["reason"] = data.get("reason")
        except Exception:
            pass

    return response


# ---------------------------------------------------------------------------
# Search & Analytics
# ---------------------------------------------------------------------------
@app.get("/api/v1/search", tags=["Search"])
async def search_committed_documents(
    q: str = Query(..., description="Keyword search query"),
    doc_type: Optional[str] = Query(None, description="Optional document_type filter"),
    limit: int = Query(20, ge=1, le=100),
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' must not be empty.")

    q_lower = q.lower().strip()
    results = []
    final_dir = get_final_dir()

    for fpath in sorted(glob.glob(os.path.join(final_dir, "*.json"))):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        pages = data.get("pages", [data])
        all_text = " ".join((p.get("extracted_text") or "") for p in pages).lower()
        kvp_text = " ".join(str(v) for p in pages for v in (p.get("key_value_pairs") or {}).values()).lower()
        searchable = all_text + " " + kvp_text

        if q_lower not in searchable:
            continue

        first_page = pages[0] if pages else {}
        dtype = first_page.get("document_type", data.get("document_type", "unknown"))

        if doc_type and dtype != doc_type:
            continue

        idx = searchable.find(q_lower)
        start = max(0, idx - 50)
        snippet = searchable[start: idx + len(q_lower) + 50].strip()

        results.append({
            "job_id": data.get("_job_id", Path(fpath).stem),
            "document_type": dtype,
            "pipeline": data.get("_pipeline", "unknown"),
            "approved_at": data.get("_approved_at"),
            "snippet": f"...{snippet}...",
        })

        if len(results) >= limit:
            break

    return {
        "query": q,
        "total_found": len(results),
        "results": results,
    }


@app.get("/api/v1/stats", tags=["Monitoring"])
async def pipeline_stats():
    pending_files = glob.glob(os.path.join(get_pending_dir(), "*.json"))
    final_files = glob.glob(os.path.join(get_final_dir(), "*.json"))
    rejected_files = glob.glob(os.path.join(get_rejected_dir(), "*.json"))

    doc_types: dict[str, int] = {}
    pipeline_breakdown: dict[str, int] = {}
    for fp in final_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            t = d.get("document_type") or (d.get("pages", [{}])[0].get("document_type") if "pages" in d else "unknown")
            doc_types[t] = doc_types.get(t, 0) + 1
            p = d.get("_pipeline", "unknown")
            pipeline_breakdown[p] = pipeline_breakdown.get(p, 0) + 1
        except Exception:
            pass

    total_reviewed = len(final_files) + len(rejected_files)
    approval_rate = round((len(final_files) / total_reviewed * 100), 1) if total_reviewed > 0 else 100.0

    summary = {
        "pending": len(pending_files),
        "pending_review": len(pending_files),
        "committed": len(final_files),
        "rejected": len(rejected_files),
        "total": total_reviewed,
        "approval_rate_pct": approval_rate,
    }

    return {
        "summary": summary,
        "document_type_breakdown": doc_types,
        "document_types": doc_types,
        "pipeline_breakdown": pipeline_breakdown,
        "pending_review_count": len(pending_files),
        "committed_count": len(final_files),
        "rejected_count": len(rejected_files),
        "total_processed": len(final_files) + len(rejected_files),
    }


# ---------------------------------------------------------------------------
# Webhook Registration Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/webhook/register", tags=["Webhooks"], status_code=201)
async def register_webhook(body: WebhookRegisterRequest):
    hooks = _load_webhooks()
    url_str = str(body.url)
    # If already registered, update and return registered
    for hid, hook in hooks.items():
        if hook.get("url") == url_str:
            hook["description"] = body.description or hook.get("description", "")
            _save_webhooks(hooks)
            return {"hook_id": hid, "webhook_id": hid, "url": url_str, "status": "registered"}

    hid = str(uuid.uuid4())
    hooks[hid] = {
        "id": hid,
        "url": url_str,
        "description": body.description or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_webhooks(hooks)
    return {"hook_id": hid, "webhook_id": hid, "url": url_str, "status": "registered"}


@app.get("/api/v1/webhook/list", tags=["Webhooks"])
async def list_webhooks():
    hooks_list = list(_load_webhooks().values())
    return {
        "total": len(hooks_list),
        "webhooks": hooks_list,
    }


@app.delete("/api/v1/webhook/{hook_id}", tags=["Webhooks"])
async def delete_webhook(hook_id: str):
    hooks = _load_webhooks()
    if hook_id not in hooks:
        raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found.")
    del hooks[hook_id]
    _save_webhooks(hooks)
    return {"deleted": hook_id, "status": "deleted"}
