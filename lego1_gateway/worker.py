"""
Greencare AI — Lego 1: Celery Worker  [UPGRADED v2]
====================================================
New in v2:
  • Webhook POST dispatch on job completion (pending_review / committed / rejected)
  • Periodic auto-cleanup task for temp_uploads/ and lego2_temp/ directories
  • Processing timestamps written into every extracted JSON record
"""

import os
import json
import time
import logging
import glob
import hmac
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from celery import Celery
from celery.schedules import crontab

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery configuration
# ---------------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "greencare_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Beat schedule for periodic cleanup
    beat_schedule={
        "auto-cleanup-every-hour": {
            # BUG-1 fix: must use the fully-qualified task name matching
            # how Celery discovers it when launched as -A lego1_gateway.worker
            "task": "lego1_gateway.worker.auto_cleanup_task",
            "schedule": crontab(minute=0),          # every hour at :00
        },
    },
)

# ---------------------------------------------------------------------------
# Service URLs
# ---------------------------------------------------------------------------
LEGO2_URL = os.environ.get("LEGO2_URL", "http://localhost:8001")
LEGO3_URL = os.environ.get("LEGO3_URL", "http://localhost:8002")

PENDING_DIR  = "./pending_review"
UPLOAD_DIR   = "./temp_uploads"
LEGO2_TEMP   = "./lego2_temp"
WEBHOOK_FILE = "./webhook_registry.json"
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "greencare-webhook-secret")

# Cleanup thresholds (hours)
UPLOAD_CLEANUP_HOURS  = int(os.environ.get("UPLOAD_CLEANUP_HOURS",  "24"))
LEGO2_TEMP_CLEANUP_HOURS = int(os.environ.get("LEGO2_TEMP_CLEANUP_HOURS", "1"))

for d in [PENDING_DIR, UPLOAD_DIR, LEGO2_TEMP]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: retry-aware HTTP request
# ---------------------------------------------------------------------------
def _post_with_retry(url: str, retries: int = 3, **kwargs) -> dict:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, timeout=120, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Helper: fire webhooks synchronously (runs inside Celery worker)
# ---------------------------------------------------------------------------
def _sign_payload(payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":")).encode()
    return hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()


def _dispatch_webhooks(event: str, payload: dict):
    """Synchronously fire all registered webhooks."""
    if not os.path.exists(WEBHOOK_FILE):
        return
    try:
        with open(WEBHOOK_FILE, "r", encoding="utf-8") as f:
            hooks: dict = json.load(f)
    except Exception:
        return

    if not hooks:
        return

    sig = _sign_payload(payload)
    headers = {
        "Content-Type": "application/json",
        "X-Greencare-Event": event,
        "X-Greencare-Signature": f"sha256={sig}",
    }
    for hook_id, hook in hooks.items():
        url = hook.get("url")
        if not url:
            continue
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            logger.info("Webhook [%s] → %s: HTTP %d", hook_id, url, r.status_code)
        except Exception as exc:
            logger.warning("Webhook [%s] failed: %s", hook_id, exc)


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------
@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def process_document_task(self, file_path: str, job_id: str):
    """
    Pipeline orchestrator.
    Returns a dict describing the final outcome.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("[%s] Pipeline started. File: %s", job_id, file_path)

    try:
        # ------------------------------------------------------------------
        # Step 1: CPU Triage (Lego 2)
        # ------------------------------------------------------------------
        logger.info("[%s] → Lego 2 triage", job_id)
        with open(file_path, "rb") as fh:
            triage_resp = _post_with_retry(
                f"{LEGO2_URL}/triage",
                files={"file": (os.path.basename(file_path), fh)},
            )

        routing = triage_resp.get("routing")
        logger.info("[%s] Lego 2 routing decision: %s", job_id, routing)

        # ------------------------------------------------------------------
        # Step 2: Route decision
        # ------------------------------------------------------------------
        if routing == "fast_track_complete":
            # BUG-4 fix: preserve ALL pages, not just page 0 (extracted_data)
            pages      = triage_resp.get("pages", [])
            final_data = triage_resp.get("extracted_data", pages[0] if pages else {})
            if pages:
                final_data["pages"] = pages   # preserve multi-page data
            final_data["_pipeline"] = "fast_track_cpu"
            logger.info("[%s] Fast-tracked (%d page(s), no Qwen API call)", job_id, len(pages))

        elif routing == "forward_to_vlm":
            cleaned_path = triage_resp.get("cleaned_file_path", file_path)
            logger.info("[%s] → Lego 3 Qwen 3.6 27B Vision. Image: %s", job_id, cleaned_path)

            ai_resp = _post_with_retry(
                f"{LEGO3_URL}/extract",
                json={"image_path": cleaned_path},
            )
            final_data = ai_resp.get("data", {})
            final_data["_pipeline"] = "qwen_vision"
            logger.info("[%s] Qwen extraction complete", job_id)

        else:
            raise ValueError(f"Unknown routing decision from Lego 2: {routing}")

        # ------------------------------------------------------------------
        # Step 3: Persist for HITL review (Lego 4)
        # ------------------------------------------------------------------
        completed_at = datetime.now(timezone.utc).isoformat()
        final_data["_job_id"]       = job_id
        final_data["_started_at"]   = started_at
        final_data["_completed_at"] = completed_at
        final_data["_filename"]     = os.path.basename(file_path)

        output_path = os.path.join(PENDING_DIR, f"{job_id}.json")
        tmp_path    = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as out_fh:
            json.dump(final_data, out_fh, indent=4, ensure_ascii=False)
        os.rename(tmp_path, output_path)

        logger.info("[%s] ✓ Written to pending_review: %s", job_id, output_path)

        # ------------------------------------------------------------------
        # Step 4: Webhook notification
        # ------------------------------------------------------------------
        _dispatch_webhooks(
            "job.pending_review",
            {
                "job_id":   job_id,
                "status":   "pending_review",
                "pipeline": final_data.get("_pipeline"),
                "filename": final_data.get("_filename"),
            },
        )

        return {"status": "success", "job_id": job_id, "pipeline": final_data.get("_pipeline")}

    except requests.RequestException as exc:
        # BUG-H fix: only retry TRANSIENT network errors, not permanent logic errors
        logger.error("[%s] Transient pipeline error: %s", job_id, exc, exc_info=True)
        raise self.retry(exc=exc)
    except (ValueError, TypeError, KeyError) as exc:
        # Permanent logic errors — do NOT retry, fail immediately
        logger.error("[%s] Permanent pipeline error (will not retry): %s", job_id, exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Periodic Auto-Cleanup Task
# ---------------------------------------------------------------------------
@celery_app.task(name="worker.auto_cleanup_task")
def auto_cleanup_task():
    """
    Periodic cleanup:
      - temp_uploads/ files older than UPLOAD_CLEANUP_HOURS → deleted
      - lego2_temp/   files older than LEGO2_TEMP_CLEANUP_HOURS → deleted
    """
    now = datetime.now(timezone.utc)
    cleaned_upload = 0
    cleaned_temp   = 0

    # Clean temp_uploads
    upload_cutoff = now - timedelta(hours=UPLOAD_CLEANUP_HOURS)
    for fpath in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        if not os.path.isfile(fpath):
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=timezone.utc)
        if mtime < upload_cutoff:
            try:
                os.remove(fpath)
                cleaned_upload += 1
                logger.info("Cleanup: removed old upload %s", fpath)
            except Exception as exc:
                logger.warning("Cleanup: could not remove %s: %s", fpath, exc)

    # Clean lego2_temp
    temp_cutoff = now - timedelta(hours=LEGO2_TEMP_CLEANUP_HOURS)
    for fpath in glob.glob(os.path.join(LEGO2_TEMP, "*")):
        if not os.path.isfile(fpath):
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=timezone.utc)
        if mtime < temp_cutoff:
            try:
                os.remove(fpath)
                cleaned_temp += 1
                logger.info("Cleanup: removed temp file %s", fpath)
            except Exception as exc:
                logger.warning("Cleanup: could not remove %s: %s", fpath, exc)

    summary = {
        "cleaned_uploads": cleaned_upload,
        "cleaned_temp_files": cleaned_temp,
        "ran_at": now.isoformat(),
    }
    logger.info("Auto-cleanup complete: %s", summary)
    return summary
