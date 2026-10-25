"""
Greencare AI — Lego 5: Analytics Service  [v2.1 Production Refactor]
=====================================================================
Observability, KPI metrics computation, cost tracking, and live dashboard.

Features:
  - Backend dimension: Tracks 'groq' vs 'local' volume and escalation rate per doc type.
  - Cost tracking: Tokens and estimated dollar cost/savings.
  - Latency percentiles: Computes p50 and p95 per stage.
  - GET /metrics/health: Evaluates reachability of all 5 IDP microservices.
  - Zero-data graceful dashboard with empty-state indicators.
  - Dynamic paths with config integration.
"""

from __future__ import annotations

import csv
import glob
import io
import json
import logging
import os
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
def get_pending_dir() -> str:
    return os.environ.get("PENDING_DIR", config.PENDING_DIR)

def get_final_dir() -> str:
    return os.environ.get("FINAL_DIR", config.FINAL_DIR)

def get_rejected_dir() -> str:
    return os.environ.get("REJECTED_DIR", config.REJECTED_DIR)

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — Analytics Service",
    description="Real-time pipeline metrics, cost accounting, and health observability.",
    version="2.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Core Metrics Computation
# ---------------------------------------------------------------------------
def _compute_metrics() -> dict:
    """Scan queues and compute comprehensive metrics including backend and cost."""
    pending_files = glob.glob(os.path.join(get_pending_dir(), "*.json"))
    final_files = glob.glob(os.path.join(get_final_dir(), "*.json"))
    rejected_files = glob.glob(os.path.join(get_rejected_dir(), "*.json"))

    n_pending = len(pending_files)
    n_committed = len(final_files)
    n_rejected = len(rejected_files)
    n_total_processed = n_committed + n_rejected

    approval_rate = round(n_committed / n_total_processed * 100, 1) if n_total_processed else 0.0

    doc_types: dict[str, int] = defaultdict(int)
    pipelines: dict[str, int] = defaultdict(int)
    languages: dict[str, int] = defaultdict(int)
    backends: dict[str, int] = defaultdict(int)

    # Escalation tracking per doc_type: total vs escalated to VLM
    doc_type_total: dict[str, int] = defaultdict(int)
    doc_type_escalated: dict[str, int] = defaultdict(int)

    processing_times: list[float] = []
    daily_counts: dict[str, int] = defaultdict(int)

    total_vlm_tokens = 0
    total_cost_usd = 0.0

    all_reviewed_files = final_files + rejected_files

    for fpath in all_reviewed_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        pages = d.get("pages", [d])
        first = pages[0] if pages else {}
        dtype = first.get("document_type", d.get("document_type", "unknown"))
        pipe = d.get("_pipeline", "unknown")
        lang = first.get("language", "unknown")

        pipe_meta = d.get("pipeline_metadata", {})
        backend = pipe_meta.get("backend") or d.get("_backend", "cpu_fasttrack" if "cpu" in pipe else "groq")

        doc_types[dtype] += 1
        pipelines[pipe] += 1
        languages[lang] += 1
        backends[backend] += 1

        doc_type_total[dtype] += 1
        if "vlm" in pipe.lower() or "qwen" in pipe.lower() or pipe_meta.get("backend") == "groq":
            doc_type_escalated[dtype] += 1
            tokens = pipe_meta.get("vlm_tokens") or 1200  # realistic estimate per page
            total_vlm_tokens += tokens
            if backend == "groq":
                total_cost_usd += (tokens / 1000.0) * 0.0005  # $0.0005 per 1k tokens

        # Latencies
        started = d.get("_started_at")
        completed = d.get("_completed_at") or d.get("_approved_at") or d.get("_rejected_at")
        if started and completed:
            try:
                t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                elapsed = max(0.01, (t1 - t0).total_seconds())
                processing_times.append(elapsed)
            except Exception:
                pass

        approved_at = d.get("_approved_at") or d.get("_timestamp")
        if approved_at:
            try:
                day = approved_at[:10]
                daily_counts[day] += 1
            except Exception:
                pass

    # Latency percentiles
    avg_proc_time = round(float(np.mean(processing_times)), 2) if processing_times else 0.0
    p50_proc_time = round(float(np.percentile(processing_times, 50)), 2) if processing_times else 0.0
    p95_proc_time = round(float(np.percentile(processing_times, 95)), 2) if processing_times else 0.0
    min_proc_time = round(float(min(processing_times)), 2) if processing_times else 0.0
    max_proc_time = round(float(max(processing_times)), 2) if processing_times else 0.0

    # Escalation rates
    escalation_rates: dict[str, float] = {}
    for dt, total in doc_type_total.items():
        esc = doc_type_escalated.get(dt, 0)
        escalation_rates[dt] = round((esc / total) * 100.0, 1)

    # Rejection reasons
    rejection_reasons: dict[str, int] = defaultdict(int)
    for fpath in rejected_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            reason = d.get("_rejection_reason") or d.get("reason", "Operator rejected")
            rejection_reasons[reason] += 1
        except Exception:
            pass

    return {
        "summary": {
            "pending": n_pending,
            "committed": n_committed,
            "rejected": n_rejected,
            "total_processed": n_total_processed,
            "approval_rate_pct": approval_rate,
        },
        "backends": dict(backends),
        "breakdowns": {
            "document_types": dict(doc_types),
            "pipelines": dict(pipelines),
            "languages": dict(languages),
            "escalation_rate_pct": escalation_rates,
        },
        "cost_tracking": {
            "total_vlm_tokens": total_vlm_tokens,
            "estimated_cost_usd": round(total_cost_usd, 4),
            "estimated_savings_usd": round((n_total_processed * 0.004) - total_cost_usd, 4) if n_total_processed else 0.0,
            "local_free_percentage": round(((n_total_processed - sum(doc_type_escalated.values())) / max(1, n_total_processed)) * 100, 1),
        },
        "performance": {
            "avg_processing_seconds": avg_proc_time,
            "p50_seconds": p50_proc_time,
            "p95_seconds": p95_proc_time,
            "min_processing_seconds": min_proc_time,
            "max_processing_seconds": max_proc_time,
            "sample_count": len(processing_times),
        },
        "daily_approved": dict(sorted(daily_counts.items())),
        "rejection_reasons": dict(rejection_reasons),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "service": "lego5-analytics",
        "status": "ok",
        "version": "2.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics", tags=["Analytics"])
def get_metrics():
    """Full JSON metrics payload with summary, backend breakdown, cost, and latency percentiles."""
    return _compute_metrics()


@app.get("/metrics/health", tags=["Monitoring"])
def metrics_service_health():
    """Probe microservices and report cluster reachability."""
    services = {
        "lego1_gateway": (config.GATEWAY_URL, "/health"),
        "lego2_triage": (config.LEGO2_URL, "/health"),
        "lego3_qwen": (config.LEGO3_URL, "/health"),
        "smart_triage_hitl": (f"http://localhost:{config.PORT_HITL_STREAMLIT}", "/healthz"),
        "lego5_analytics": (config.ANALYTICS_URL, "/health"),
    }

    results = {}
    for name, (base_url, path) in services.items():
        is_reachable = False
        try:
            req = urllib.request.Request(f"{base_url}{path}", headers={"User-Agent": "IDP-ClusterCheck"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                is_reachable = (resp.status < 500)
        except Exception:
            is_reachable = False
        results[name] = {"url": base_url, "reachable": is_reachable}

    all_healthy = all(s["reachable"] for s in results.values())
    return {
        "cluster_healthy": all_healthy,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": results,
    }


@app.get("/metrics/timeline", tags=["Analytics"])
def metrics_timeline(days: int = 30):
    """Daily document approval counts for the past `days` days."""
    m = _compute_metrics()
    data = m.get("daily_approved", {})
    today = datetime.now(timezone.utc).date()
    timeline = []
    for i in range(days - 1, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        timeline.append({"date": day, "count": data.get(day, 0)})
    return {"days": days, "timeline": timeline}


@app.get("/metrics/export", tags=["Analytics"])
def metrics_export_csv():
    """Download metrics as CSV."""
    m = _compute_metrics()
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Greencare AI IDP — Metrics Report"])
    writer.writerow(["Generated At", m["generated_at"]])
    writer.writerow([])
    writer.writerow(["Metric", "Value"])
    for k, v in m["summary"].items():
        writer.writerow([k, v])
    for k, v in m["cost_tracking"].items():
        writer.writerow([f"cost_{k}", v])
    for k, v in m["performance"].items():
        writer.writerow([f"latency_{k}", v])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=idp_metrics_report.csv"},
    )


@app.get("/metrics/dashboard", response_class=HTMLResponse, tags=["Analytics"])
def metrics_dashboard():
    """Live HTML dashboard with Chart.js and empty-state resilience."""
    m = _compute_metrics()
    s = m["summary"]
    cost = m["cost_tracking"]
    perf = m["performance"]
    is_empty = (s["total_processed"] == 0 and s["pending"] == 0)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>IDP Analytics Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 16px; margin-bottom: 24px; }}
    .title {{ font-size: 24px; font-weight: 700; color: #6366f1; }}
    .badge {{ background: #1e293b; padding: 6px 12px; border-radius: 6px; font-size: 13px; color: #94a3b8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
    .card-title {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 8px; }}
    .card-value {{ font-size: 28px; font-weight: 700; color: #f8fafc; }}
    .chart-container {{ background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; height: 320px; }}
    .empty-state {{ text-align: center; padding: 60px 20px; background: #1e293b; border-radius: 12px; border: 1px dashed #475569; margin-bottom: 24px; }}
    .empty-icon {{ font-size: 40px; margin-bottom: 12px; }}
    .empty-title {{ font-size: 18px; font-weight: 600; color: #e2e8f0; margin-bottom: 6px; }}
    .empty-desc {{ font-size: 14px; color: #94a3b8; }}
  </style>
</head>
<body>
  <div class="header">
    <div class="title">🌿 IDP Analytics & Observability</div>
    <div class="badge">Pipeline: Active • Version 2.1.0</div>
  </div>

  {"<div class='empty-state'><div class='empty-icon'>📥</div><div class='empty-title'>No Documents Processed Yet</div><div class='empty-desc'>Submit documents via <code>POST /api/v1/ingest</code> to begin tracking metrics and cost models.</div></div>" if is_empty else ""}

  <div class="grid">
    <div class="card">
      <div class="card-title">Pending Review</div>
      <div class="card-value" style="color: #f59e0b;">{s["pending"]}</div>
    </div>
    <div class="card">
      <div class="card-title">Approved & Committed</div>
      <div class="card-value" style="color: #10b981;">{s["committed"]}</div>
    </div>
    <div class="card">
      <div class="card-title">Approval Rate</div>
      <div class="card-value">{s["approval_rate_pct"]}%</div>
    </div>
    <div class="card">
      <div class="card-title">Local Free Processing</div>
      <div class="card-value" style="color: #6366f1;">{cost["local_free_percentage"]}%</div>
    </div>
    <div class="card">
      <div class="card-title">P95 Latency</div>
      <div class="card-value">{perf["p95_seconds"]}s</div>
    </div>
    <div class="card">
      <div class="card-title">Estimated Cost</div>
      <div class="card-value">${cost["estimated_cost_usd"]}</div>
    </div>
  </div>

  <div class="chart-container">
    <canvas id="pipelineChart"></canvas>
  </div>

  <script>
    const ctx = document.getElementById('pipelineChart').getContext('2d');
    const backends = {json.dumps(m.get("backends", {}))};
    const labels = Object.keys(backends).length ? Object.keys(backends) : ['Zero Data'];
    const data = Object.keys(backends).length ? Object.values(backends) : [0];

    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: [{{
          label: 'Documents Processed by Backend',
          data: data,
          backgroundColor: ['#6366f1', '#10b981', '#f59e0b', '#ec4899'],
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)
