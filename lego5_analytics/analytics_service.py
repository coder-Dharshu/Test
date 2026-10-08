"""
Greencare AI — Lego 5: Analytics Service  [NEW]
================================================
Provides real-time pipeline metrics computed from the filesystem.

Endpoints:
  GET  /health             — health check
  GET  /metrics            — full JSON metrics payload
  GET  /metrics/export     — download metrics as CSV
  GET  /metrics/dashboard  — embedded HTML analytics dashboard (port 8003)
  GET  /metrics/timeline   — daily document counts for the last 30 days

Runs on: Port 8003
"""

import os
import glob
import json
import csv
import io
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories (shared volumes — same paths as other services)
# ---------------------------------------------------------------------------
PENDING_DIR  = os.environ.get("PENDING_DIR",  "./pending_review")
FINAL_DIR    = os.environ.get("FINAL_DIR",    "./final_database")
REJECTED_DIR = os.environ.get("REJECTED_DIR", "./rejected")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Greencare AI — Analytics Service",
    description="Real-time pipeline metrics and analytics dashboard.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------
def _compute_metrics() -> dict:
    """Scan all three queues and compute comprehensive metrics."""
    pending_files  = glob.glob(os.path.join(PENDING_DIR,  "*.json"))
    final_files    = glob.glob(os.path.join(FINAL_DIR,    "*.json"))
    rejected_files = glob.glob(os.path.join(REJECTED_DIR, "*.json"))

    n_pending  = len(pending_files)
    n_committed = len(final_files)
    n_rejected  = len(rejected_files)
    n_total_processed = n_committed + n_rejected

    approval_rate = round(n_committed / n_total_processed * 100, 1) if n_total_processed else 0.0

    # Breakdowns from committed documents
    doc_types: dict  = defaultdict(int)
    pipelines: dict  = defaultdict(int)
    languages: dict  = defaultdict(int)
    processing_times: list = []
    daily_counts: dict = defaultdict(int)

    for fpath in final_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        pages   = d.get("pages", [d])
        first   = pages[0] if pages else {}
        dtype   = first.get("document_type", "unknown")
        pipe    = d.get("_pipeline", "unknown")
        lang    = first.get("language", "unknown")

        doc_types[dtype] += 1
        pipelines[pipe]  += 1
        languages[lang]  += 1

        # Processing time from timestamps
        started   = d.get("_started_at")
        completed = d.get("_completed_at") or d.get("_approved_at")
        if started and completed:
            try:
                t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                processing_times.append((t1 - t0).total_seconds())
            except Exception:
                pass

        # Daily counts from approved_at
        approved_at = d.get("_approved_at")
        if approved_at:
            try:
                day = approved_at[:10]  # YYYY-MM-DD
                daily_counts[day] += 1
            except Exception:
                pass

    avg_proc_time = round(sum(processing_times) / len(processing_times), 2) if processing_times else None
    max_proc_time = round(max(processing_times), 2) if processing_times else None
    min_proc_time = round(min(processing_times), 2) if processing_times else None

    # Rejection reasons breakdown
    rejection_reasons: dict = defaultdict(int)
    for fpath in rejected_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            reason = d.get("_rejection_reason", "No reason")
            rejection_reasons[reason] += 1
        except Exception:
            pass

    return {
        "summary": {
            "pending":            n_pending,
            "committed":          n_committed,
            "rejected":           n_rejected,
            "total_processed":    n_total_processed,
            "approval_rate_pct":  approval_rate,
        },
        "breakdowns": {
            "document_types": dict(doc_types),
            "pipelines":      dict(pipelines),
            "languages":      dict(languages),
        },
        "performance": {
            "avg_processing_seconds": avg_proc_time,
            "min_processing_seconds": min_proc_time,
            "max_processing_seconds": max_proc_time,
            "sample_count":           len(processing_times),
        },
        "daily_approved":    dict(sorted(daily_counts.items())),
        "rejection_reasons": dict(rejection_reasons),
        "generated_at":      datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok", "service": "lego5-analytics", "version": "1.0.0"}


@app.get("/metrics", tags=["Analytics"])
def metrics():
    """Full JSON metrics payload with summary, breakdowns, and performance data."""
    return _compute_metrics()


@app.get("/metrics/timeline", tags=["Analytics"])
def metrics_timeline(days: int = 30):
    """
    Return daily document approval counts for the past `days` days.
    Fills missing days with 0 for charting continuity.
    """
    m    = _compute_metrics()
    data = m.get("daily_approved", {})
    today = datetime.now(timezone.utc).date()
    timeline = []
    for i in range(days - 1, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        timeline.append({"date": day, "count": data.get(day, 0)})
    return {"days": days, "timeline": timeline}


@app.get("/metrics/export", tags=["Analytics"])
def metrics_export():
    """Download all metrics as a multi-section CSV file."""
    m = _compute_metrics()
    buf = io.StringIO()
    w   = csv.writer(buf)

    w.writerow(["=== SUMMARY ==="])
    w.writerow(["Metric", "Value"])
    for k, v in m["summary"].items():
        w.writerow([k, v])

    w.writerow([])
    w.writerow(["=== DOCUMENT TYPES ==="])
    w.writerow(["Document Type", "Count"])
    for k, v in m["breakdowns"]["document_types"].items():
        w.writerow([k, v])

    w.writerow([])
    w.writerow(["=== PIPELINES ==="])
    w.writerow(["Pipeline", "Count"])
    for k, v in m["breakdowns"]["pipelines"].items():
        w.writerow([k, v])

    w.writerow([])
    w.writerow(["=== DAILY APPROVALS ==="])
    w.writerow(["Date", "Count"])
    for k, v in m["daily_approved"].items():
        w.writerow([k, v])

    w.writerow([])
    w.writerow(["=== PERFORMANCE ==="])
    w.writerow(["Metric", "Seconds"])
    perf = m["performance"]
    w.writerow(["avg_processing", perf["avg_processing_seconds"]])
    w.writerow(["min_processing", perf["min_processing_seconds"]])
    w.writerow(["max_processing", perf["max_processing_seconds"]])

    w.writerow([])
    w.writerow([f"Generated at: {m['generated_at']}"])

    content = buf.getvalue()
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=greencare_metrics.csv"},
    )


@app.get("/metrics/dashboard", response_class=HTMLResponse, tags=["Analytics"])
def metrics_dashboard():
    """
    Serve a self-contained HTML analytics dashboard with live Chart.js charts.
    Auto-refreshes every 30 seconds.
    """
    m = _compute_metrics()

    # Prepare chart data
    pipe_labels  = list(m["breakdowns"]["pipelines"].keys())
    pipe_values  = list(m["breakdowns"]["pipelines"].values())
    dtype_labels = list(m["breakdowns"]["document_types"].keys())
    dtype_values = list(m["breakdowns"]["document_types"].values())
    timeline     = []
    today        = datetime.now(timezone.utc).date()
    daily        = m.get("daily_approved", {})
    for i in range(29, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        timeline.append({"date": day, "count": daily.get(day, 0)})

    tl_labels = [t["date"] for t in timeline]
    tl_values = [t["count"] for t in timeline]

    s   = m["summary"]
    perf = m["performance"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Greencare AI — Analytics Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:        #0f1117;
      --surface:   #1a1d2e;
      --surface2:  #242840;
      --accent:    #6366f1;
      --accent2:   #10b981;
      --accent3:   #f59e0b;
      --danger:    #ef4444;
      --text:      #e2e8f0;
      --muted:     #94a3b8;
      --border:    rgba(99,102,241,0.2);
    }}
    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 24px;
    }}
    header {{
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 32px;
    }}
    header .logo {{
      width: 44px; height: 44px; border-radius: 12px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      display: grid; place-items: center; font-size: 22px;
    }}
    header h1 {{ font-size: 1.5rem; font-weight: 700; }}
    header p  {{ color: var(--muted); font-size: 0.875rem; margin-top: 2px; }}
    .refresh-tag {{
      margin-left: auto;
      background: var(--surface2);
      border: 1px solid var(--border);
      padding: 4px 14px;
      border-radius: 20px;
      font-size: 0.75rem;
      color: var(--accent2);
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      position: relative;
      overflow: hidden;
    }}
    .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: var(--accent-color, var(--accent));
    }}
    .kpi-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi-value {{ font-size: 2rem; font-weight: 700; margin-top: 8px; }}
    .kpi-sub   {{ font-size: 0.75rem; color: var(--muted); margin-top: 4px; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
      margin-bottom: 28px;
    }}
    .chart-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
    }}
    .chart-card h3 {{
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 16px;
    }}
    .timeline-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 20px;
    }}
    .timeline-card h3 {{
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 16px;
    }}
    .perf-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}
    .perf-item {{
      background: var(--surface2);
      border-radius: 10px;
      padding: 12px 16px;
      text-align: center;
    }}
    .perf-item .val {{ font-size: 1.4rem; font-weight: 700; color: var(--accent2); }}
    .perf-item .lbl {{ font-size: 0.7rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 32px; }}
    canvas {{ max-height: 240px; }}
  </style>
  <script>
    setTimeout(() => location.reload(), 30000);
  </script>
</head>
<body>
  <header>
    <div class="logo">🌿</div>
    <div>
      <h1>Greencare AI Analytics</h1>
      <p>Pipeline Metrics Dashboard</p>
    </div>
    <div class="refresh-tag">⟳ Auto-refresh 30s</div>
  </header>

  <!-- KPI Cards -->
  <div class="kpi-grid">
    <div class="kpi-card" style="--accent-color:#6366f1">
      <div class="kpi-label">Total Processed</div>
      <div class="kpi-value">{s['total_processed']}</div>
      <div class="kpi-sub">All-time documents</div>
    </div>
    <div class="kpi-card" style="--accent-color:#10b981">
      <div class="kpi-label">Committed</div>
      <div class="kpi-value">{s['committed']}</div>
      <div class="kpi-sub">Approved &amp; in DB</div>
    </div>
    <div class="kpi-card" style="--accent-color:#f59e0b">
      <div class="kpi-label">Pending Review</div>
      <div class="kpi-value">{s['pending']}</div>
      <div class="kpi-sub">Awaiting human approval</div>
    </div>
    <div class="kpi-card" style="--accent-color:#ef4444">
      <div class="kpi-label">Rejected</div>
      <div class="kpi-value">{s['rejected']}</div>
      <div class="kpi-sub">Moved to rejected/</div>
    </div>
    <div class="kpi-card" style="--accent-color:#8b5cf6">
      <div class="kpi-label">Approval Rate</div>
      <div class="kpi-value">{s['approval_rate_pct']}%</div>
      <div class="kpi-sub">Of all processed</div>
    </div>
  </div>

  <!-- Timeline -->
  <div class="timeline-card">
    <h3>📅 Daily Approvals — Last 30 Days</h3>
    <canvas id="timelineChart"></canvas>
  </div>

  <!-- Pie / Doughnut charts -->
  <div class="charts-grid">
    <div class="chart-card">
      <h3>🔀 Pipeline Breakdown</h3>
      <canvas id="pipeChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>📄 Document Types</h3>
      <canvas id="dtypeChart"></canvas>
    </div>
  </div>

  <!-- Performance -->
  <div class="chart-card">
    <h3>⚡ Processing Performance</h3>
    <div class="perf-grid">
      <div class="perf-item">
        <div class="val">{perf['avg_processing_seconds'] or '—'}s</div>
        <div class="lbl">Avg Processing</div>
      </div>
      <div class="perf-item">
        <div class="val">{perf['min_processing_seconds'] or '—'}s</div>
        <div class="lbl">Fastest</div>
      </div>
      <div class="perf-item">
        <div class="val">{perf['max_processing_seconds'] or '—'}s</div>
        <div class="lbl">Slowest</div>
      </div>
    </div>
  </div>

  <footer>Generated {m['generated_at']} · Greencare AI v2.0</footer>

<script>
const PALETTE = ['#6366f1','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#f97316','#84cc16'];
const C = (id) => document.getElementById(id).getContext('2d');

// Timeline
new Chart(C('timelineChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(tl_labels)},
    datasets: [{{
      label: 'Approved',
      data: {json.dumps(tl_values)},
      backgroundColor: 'rgba(99,102,241,0.7)',
      borderColor: '#6366f1',
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, beginAtZero: true }}
    }}
  }}
}});

// Pipeline
new Chart(C('pipeChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(pipe_labels)},
    datasets: [{{ data: {json.dumps(pipe_values)}, backgroundColor: PALETTE, borderWidth: 0 }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#e2e8f0', padding: 12 }} }}
    }}
  }}
}});

// Doc types
new Chart(C('dtypeChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(dtype_labels)},
    datasets: [{{ data: {json.dumps(dtype_values)}, backgroundColor: PALETTE.slice(2), borderWidth: 0 }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#e2e8f0', padding: 12 }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
