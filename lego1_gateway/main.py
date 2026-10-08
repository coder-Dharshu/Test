"""
Greencare AI — Lego 1: API Gateway (Port 8000)  [UPGRADED v2]
==============================================================
New in v2:
  • POST /api/v1/batch-ingest          — submit multiple files at once
  • GET  /api/v1/batch-status          — poll many job IDs in one call
  • GET  /api/v1/search                — full-text keyword search over final_database/
  • POST /api/v1/webhook/register      — register a callback URL for job events
  • GET  /api/v1/webhook/list          — list registered webhooks
  • DELETE /api/v1/webhook/{id}        — remove a webhook
  • GET  /api/v1/stats                 — pipeline-wide summary statistics
  • GET  /health                       — service health (unchanged)
  • POST /api/v1/ingest                — single-file ingest (unchanged)
  • GET  /api/v1/status/{job_id}       — job status (unchanged)
"""

import os
import uuid
import shutil
import json
import glob
import logging
import hmac
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from lego1_gateway.worker import process_document_task

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_DIR   = "./temp_uploads"
PENDING_DIR  = "./pending_review"
FINAL_DIR    = "./final_database"
REJECTED_DIR = "./rejected"
WEBHOOK_FILE = "./webhook_registry.json"

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "greencare-webhook-secret")

for d in [UPLOAD_DIR, PENDING_DIR, FINAL_DIR, REJECTED_DIR]:
    os.makedirs(d, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
    ".docx", ".xlsx", ".xls", ".csv",
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — API Gateway",
    description=(
        "Accepts document uploads and routes them through the IDP pipeline. "
        "Supports single & batch ingest, full-text search, webhook notifications, "
        "and pipeline statistics."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static website frontend
app.mount("/web", StaticFiles(directory="web"), name="web")


# ---------------------------------------------------------------------------
# Webhook registry helpers
# ---------------------------------------------------------------------------
def _load_webhooks() -> dict:
    if not os.path.exists(WEBHOOK_FILE):
        return {}
    try:
        with open(WEBHOOK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_webhooks(hooks: dict):
    with open(WEBHOOK_FILE, "w", encoding="utf-8") as f:
        json.dump(hooks, f, indent=2)


def _sign_payload(payload: dict) -> str:
    """HMAC-SHA256 signature for webhook payload verification."""
    body = json.dumps(payload, separators=(",", ":")).encode()
    return hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()


async def _fire_webhooks(event: str, payload: dict):
    """Dispatch webhook POST to all registered URLs (async, non-blocking)."""
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
            url = hook.get("url")
            if not url:
                continue
            try:
                resp = await client.post(url, json=payload, headers=headers)
                logger.info("Webhook [%s] → %s: HTTP %d", hook_id, url, resp.status_code)
            except Exception as exc:
                logger.warning("Webhook [%s] failed: %s", hook_id, exc)


# ---------------------------------------------------------------------------
# Job status helper
# ---------------------------------------------------------------------------
def _get_job_status(job_id: str) -> str:
    # BUG-7 fix: use os.path.join instead of f-string slashes (Windows-safe)
    if os.path.exists(os.path.join(FINAL_DIR,   f"{job_id}.json")):
        return "committed"
    if os.path.exists(os.path.join(PENDING_DIR, f"{job_id}.json")):
        return "pending_review"
    return "processing"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class WebhookRegisterRequest(BaseModel):
    url: HttpUrl                          # BUG-3 fix: validated URL, not bare str
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
async def health_check():
    p = len(glob.glob(os.path.join(PENDING_DIR, "*.json")))
    c = len(glob.glob(os.path.join(FINAL_DIR, "*.json")))
    r = len(glob.glob(os.path.join(REJECTED_DIR, "*.json")))
    return {
        "status": "ok",
        "service": "lego1-gateway",
        "version": "2.0.0",
        "queue": {"pending": p, "committed": c, "rejected": r},
    }


# ---------------------------------------------------------------------------
# Single-file ingest (unchanged behaviour, upgraded webhook dispatch)
# ---------------------------------------------------------------------------
@app.post("/api/v1/ingest", tags=["Ingestion"], status_code=202)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept a single document file, persist it, enqueue for processing.
    Returns immediately with a job_id.
    """
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = process_document_task.delay(file_path, job_id)

    # Fire webhook asynchronously
    background_tasks.add_task(
        _fire_webhooks,
        "job.queued",
        {"job_id": job_id, "filename": file.filename, "status": "queued"},
    )

    logger.info("[%s] Queued: %s", job_id, file.filename)
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


# ---------------------------------------------------------------------------
# Batch ingest — multiple files in one request
# ---------------------------------------------------------------------------
@app.post("/api/v1/batch-ingest", tags=["Ingestion"], status_code=202)
async def batch_ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Accept multiple document files in a single request.
    Returns a list of job IDs — one per file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch request.")

    results = []
    for file in files:
        _, ext = os.path.splitext(file.filename or "")
        if ext.lower() not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "rejected",
                "reason": f"Unsupported file type '{ext}'",
            })
            continue

        job_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        task = process_document_task.delay(file_path, job_id)
        background_tasks.add_task(
            _fire_webhooks,
            "job.queued",
            {"job_id": job_id, "filename": file.filename, "status": "queued"},
        )
        results.append({
            "filename": file.filename,
            "status": "queued",
            "job_id": job_id,
            "celery_task_id": task.id,
        })
        logger.info("[%s] Batch queued: %s", job_id, file.filename)

    queued_count   = sum(1 for r in results if r["status"] == "queued")
    rejected_count = sum(1 for r in results if r["status"] == "rejected")
    return JSONResponse(
        content={
            "total_submitted": len(files),
            "queued_count":    queued_count,    # BUG-E fix: clear breakdown
            "rejected_count":  rejected_count,
            "jobs":            results,
        },
        status_code=202,
    )


# ---------------------------------------------------------------------------
# Batch status — poll multiple job IDs
# ---------------------------------------------------------------------------
@app.get("/api/v1/batch-status", tags=["Monitoring"])
async def batch_job_status(
    job_ids: List[str] = Query(..., description="Comma-separated list of job IDs"),
):
    """
    Check the status of multiple jobs in a single request.
    Pass job_ids as repeated query params: ?job_ids=abc&job_ids=def
    """
    if len(job_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 job IDs per batch-status call.")
    return {
        "results": [
            {"job_id": jid, "status": _get_job_status(jid)}
            for jid in job_ids
        ]
    }


# ---------------------------------------------------------------------------
# Single job status
# ---------------------------------------------------------------------------
@app.get("/api/v1/status/{job_id}", tags=["Monitoring"])
async def get_job_status(job_id: str):
    """
    File-based status check:
    - final_database/{job_id}.json  → committed
    - pending_review/{job_id}.json  → pending_review
    - otherwise                     → processing
    """
    status = _get_job_status(job_id)
    response: dict = {"status": status, "job_id": job_id}

    if status == "committed":
        final_path = os.path.join(FINAL_DIR, f"{job_id}.json")  # BUG-7 fix
        try:
            with open(final_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            response["approved_at"] = data.get("_approved_at")
            response["document_type"] = data.get("document_type") or (
                data.get("pages", [{}])[0].get("document_type") if "pages" in data else None
            )
        except Exception:
            pass

    return response


# ---------------------------------------------------------------------------
# Document search over final_database
# ---------------------------------------------------------------------------
@app.get("/api/v1/search", tags=["Search"])
async def search_documents(
    q: str = Query(..., min_length=1, description="Keyword to search for"),
    doc_type: Optional[str] = Query(None, description="Filter by document_type"),
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
):
    """
    Full-text keyword search over committed documents in final_database/.
    Searches extracted_text, key_value_pairs values, and document metadata.
    Returns matching job IDs with snippets.
    """
    q_lower = q.lower()
    results = []

    for fpath in sorted(glob.glob(os.path.join(FINAL_DIR, "*.json"))):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # Flatten searchable text
        pages = data.get("pages", [data])
        all_text = " ".join(
            (p.get("extracted_text") or "") for p in pages
        ).lower()
        kvp_text = " ".join(
            str(v) for p in pages for v in (p.get("key_value_pairs") or {}).values()
        ).lower()
        searchable = all_text + " " + kvp_text

        if q_lower not in searchable:
            continue

        first_page = pages[0] if pages else {}
        dtype = first_page.get("document_type", "unknown")

        if doc_type and dtype != doc_type:
            continue

        # Extract a snippet (50 chars around first match)
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


# ---------------------------------------------------------------------------
# Pipeline statistics
# ---------------------------------------------------------------------------
@app.get("/api/v1/stats", tags=["Monitoring"])
async def pipeline_stats():
    """
    Return high-level pipeline statistics computed from the filesystem.
    """
    pending   = glob.glob(os.path.join(PENDING_DIR,  "*.json"))
    committed = glob.glob(os.path.join(FINAL_DIR,    "*.json"))
    rejected  = glob.glob(os.path.join(REJECTED_DIR, "*.json"))

    total_processed = len(committed) + len(rejected)
    approval_rate   = round(len(committed) / total_processed * 100, 1) if total_processed else 0.0

    # Doc-type breakdown from final_database
    doc_types: dict = {}
    pipelines: dict = {}
    for fpath in committed:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            pages   = d.get("pages", [d])
            dtype   = (pages[0] if pages else {}).get("document_type", "unknown")
            pipe    = d.get("_pipeline", "unknown")
            doc_types[dtype]  = doc_types.get(dtype, 0)  + 1
            pipelines[pipe]   = pipelines.get(pipe, 0)   + 1
        except Exception:
            pass

    return {
        "summary": {
            "pending":         len(pending),
            "committed":       len(committed),
            "rejected":        len(rejected),
            "total_processed": total_processed,
            "approval_rate_pct": approval_rate,
        },
        "document_type_breakdown": doc_types,
        "pipeline_breakdown": pipelines,
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",  # BUG-5 fix
    }


# ---------------------------------------------------------------------------
# Webhook management
# ---------------------------------------------------------------------------
@app.post("/api/v1/webhook/register", tags=["Webhooks"], status_code=201)
async def register_webhook(req: WebhookRegisterRequest):
    """
    Register a callback URL to receive job lifecycle events.
    Events: job.queued, job.pending_review, job.committed, job.rejected
    Each POST includes an X-Greencare-Signature header (HMAC-SHA256).
    """
    hooks = _load_webhooks()
    hook_id = str(uuid.uuid4())
    hooks[hook_id] = {
        "id":            hook_id,
        "url":           str(req.url),   # BUG-A fix: Pydantic v2 HttpUrl is not JSON-serialisable
        "description":   req.description or "",
        "registered_at": datetime.now(timezone.utc).isoformat() + "Z",
    }
    _save_webhooks(hooks)
    logger.info("Webhook registered: %s -> %s", hook_id, req.url)
    return {"hook_id": hook_id, "url": str(req.url), "status": "registered"}


@app.get("/api/v1/webhook/list", tags=["Webhooks"])
async def list_webhooks():
    """List all registered webhook endpoints."""
    hooks = _load_webhooks()
    return {"total": len(hooks), "webhooks": list(hooks.values())}


@app.delete("/api/v1/webhook/{hook_id}", tags=["Webhooks"])
async def delete_webhook(hook_id: str):
    """Remove a registered webhook by its ID."""
    hooks = _load_webhooks()
    if hook_id not in hooks:
        raise HTTPException(status_code=404, detail=f"Webhook '{hook_id}' not found.")
    url = hooks.pop(hook_id, {}).get("url", "")
    _save_webhooks(hooks)
    logger.info("Webhook deleted: %s (was %s)", hook_id, url)
    return {"status": "deleted", "hook_id": hook_id}


# ---------------------------------------------------------------------------
# Dashboard API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/queue", tags=["Dashboard"])
async def get_queue():
    """List pending documents in the review queue."""
    pending = []
    for fpath in glob.glob(os.path.join(PENDING_DIR, "*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            job_id = data.get("_job_id") or Path(fpath).stem
            pending.append({
                "job_id": job_id,
                "filename": data.get("_filename") or f"{job_id}.json",
                "pipeline": data.get("_pipeline"),
                "confidence_warning": data.get("pages", [data])[0].get("confidence_warning", False)
            })
        except Exception:
            continue
    # Sort by oldest first based on file modification time
    pending.sort(key=lambda x: os.path.getmtime(os.path.join(PENDING_DIR, f"{x['job_id']}.json")))
    return {"items": pending}

@app.get("/api/v1/pending/{job_id}", tags=["Dashboard"])
async def get_pending_doc(job_id: str):
    """Get the full JSON of a pending document."""
    fpath = os.path.join(PENDING_DIR, f"{job_id}.json")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Document not found in pending queue.")
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/v1/approve/{job_id}", tags=["Dashboard"])
async def approve_document(job_id: str, data: dict):
    """Approve a document and move it to final_database."""
    pending_path = os.path.join(PENDING_DIR, f"{job_id}.json")
    if not os.path.exists(pending_path):
        raise HTTPException(status_code=404, detail="Document not found in pending queue.")
    
    data["_approved_at"] = datetime.now(timezone.utc).isoformat() + "Z"
    data["_reviewed_by"] = "human_reviewer"
    
    final_path = os.path.join(FINAL_DIR, f"{job_id}.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    os.remove(pending_path)
    logger.info("[%s] Approved by human", job_id)
    return {"status": "approved", "job_id": job_id}

class RejectRequest(BaseModel):
    reason: str

@app.post("/api/v1/reject/{job_id}", tags=["Dashboard"])
async def reject_document(job_id: str, req: RejectRequest):
    """Reject a document and move it to rejected."""
    pending_path = os.path.join(PENDING_DIR, f"{job_id}.json")
    if not os.path.exists(pending_path):
        raise HTTPException(status_code=404, detail="Document not found in pending queue.")
    
    with open(pending_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    data["_rejected_at"] = datetime.now(timezone.utc).isoformat() + "Z"
    data["_rejection_reason"] = req.reason or "No reason"
    
    reject_path = os.path.join(REJECTED_DIR, f"{job_id}.json")
    with open(reject_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    os.remove(pending_path)
    logger.info("[%s] Rejected by human: %s", job_id, req.reason)
    return {"status": "rejected", "job_id": job_id}

@app.get("/api/v1/export/{job_id}", tags=["Dashboard"])
async def export_document(job_id: str, format: str = Query("json")):
    """Export a committed document as JSON or CSV."""
    fpath = os.path.join(FINAL_DIR, f"{job_id}.json")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Document not found in final database.")
    
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if format == "json":
        return Response(
            content=json.dumps(data, indent=2), 
            media_type="application/json", 
            headers={"Content-Disposition": f'attachment; filename="{job_id}.json"'}
        )
    elif format == "csv":
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        pages = data.get("pages", [data])
        first = pages[0] if pages else {}
        kvps = first.get("key_value_pairs", {})
        
        for k, v in kvps.items():
            writer.writerow([k, v])
            
        tables = first.get("tables", [])
        if tables:
            writer.writerow([])
            writer.writerow(["--- TABLE DATA ---"])
            for t in tables:
                if t.get("headers"):
                    writer.writerow(t["headers"])
                for r in t.get("rows", []):
                    writer.writerow(r)
                    
        return Response(
            content=output.getvalue(), 
            media_type="text/csv", 
            headers={"Content-Disposition": f'attachment; filename="{job_id}.csv"'}
        )
    
    raise HTTPException(status_code=400, detail="Unsupported format. Use 'json' or 'csv'.")
