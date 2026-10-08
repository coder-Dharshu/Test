# 🌿 Greencare AI — Local VLM-Powered Intelligent Document Processing  `v2.0`

A production-ready, **five-microservice** IDP pipeline that uses a **local Qwen 3.6 27B model or accelerated vision endpoint** to extract structured JSON from any document (PDFs, scanned images, handwritten forms, Word docs, Excel sheets, CSVs) with full data privacy and offline capability.

---

## What's New in v2.0

| Feature | Details |
|---|---|
| 🔄 **Batch Upload** | `/api/v1/batch-ingest` — submit up to 20 files in one request |
| 🔎 **Document Search** | `/api/v1/search` — keyword search over committed documents |
| 🔔 **Webhook Notifications** | Register callback URLs for job lifecycle events |
| 🧹 **Auto-Cleanup** | Celery Beat purges stale temp files on schedule |
| 📋 **.docx / .xlsx / .csv Support** | Fast-tracked natively — no VLM call needed |
| 📊 **Analytics Dashboard** | New Lego 5 service at port 8003 with live charts |
| 🎨 **HITL Upgrades** | Stats tab, queue search, keyboard shortcuts (A/R/N), PDF preview |
| ⏱️ **Processing Timestamps** | Every JSON record includes `_started_at`, `_completed_at` |
| 🤖 **Qwen Model Switcher** | `/models` endpoint lists all available vision models |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (HTTP POST)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │  POST /api/v1/ingest
                               │  POST /api/v1/batch-ingest
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEGO 1 — FastAPI Gateway v2  (port 8000)                        │
│  • UUID job assignment + batch processing                        │
│  • File persistence to shared volume                             │
│  • Celery task enqueue → Redis                                   │
│  • /api/v1/search  /api/v1/stats  /api/v1/webhook/register      │
└──────────────────────────────┬───────────────────────────────────┘
                               │  Celery Worker pulls task
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEGO 2 — CPU Triage v2  (port 8001)                             │
│  ┌─────────────────┐         ┌──────────────────────────────┐   │
│  │  Digital PDF    │──Text──▶│  Fast-Track (no VLM token)  │   │
│  │  .docx / .xlsx  │──Data──▶│                              │   │
│  │  .csv           │──Data──▶│                              │   │
│  └─────────────────┘         └──────────────────────────────┘   │
│  ┌─────────────────┐                                            │
│  │  Image / Scanned│──Deskew + Glare-Suppress ──────────────▶  │
│  └─────────────────┘                                            │
└──────────────────────────────┬───────────────────────────────────┘
                               │  image forwarded
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEGO 3 — Qwen 3.6 27B Vision Engine v2  (port 8002)             │
│  • Base64 image encoding                                         │
│  • Qwen 3.6 27B Vision (local / cloud)                           │
│  • response_format=json_object (grammar-locked)                  │
│  • Retry + exponential back-off on rate limits                   │
│  • /models endpoint  •  processing_time_ms in all responses     │
└──────────────────────────────┬───────────────────────────────────┘
                               │  JSON + timestamps written
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEGO 4 — HITL Dashboard v2  (port 7860)                        │
│  • Gradio web UI with tabbed layout (Review / Statistics / Help) │
│  • Queue search / filter   •  Keyboard shortcuts (A / R / N)    │
│  • PDF-to-image preview    •  Processing timestamp display       │
│  • Approve → final_database/  |  Reject → rejected/             │
│  • Export JSON / CSV / Excel                                     │
└──────────────────────────────────────────────────────────────────┘
                               │  (reads same shared volumes)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEGO 5 — Analytics Service  (port 8003)  [NEW v2]              │
│  • GET /metrics              — full JSON metrics                 │
│  • GET /metrics/dashboard    — live HTML dashboard with charts   │
│  • GET /metrics/timeline     — 30-day approval timeline          │
│  • GET /metrics/export       — CSV download                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- A **Qwen API key** (or local Ollama/vLLM inference endpoint)

### 1. Clone / navigate to the project
```bash
cd greencare-ai
```

### 2. Configure environment
```bash
# Windows PowerShell
Copy-Item .env.example .env
# Then edit .env and set QWEN_API_KEY (or LOCAL_MODEL_URL)

# Linux / macOS
cp .env.example .env && nano .env
```

### 3. Build and start all services
```bash
docker-compose up --build
```

First build takes ~3–5 minutes. Subsequent starts are instant.

### 4. Open the services

| Service | URL |
|---|---|
| 🚪 API Gateway (Swagger UI) | http://localhost:8000/docs |
| 🔍 CPU Triage (Swagger UI) | http://localhost:8001/docs |
| 🤖 VLM Vision (Swagger UI) | http://localhost:8002/docs |
| 👁 HITL Dashboard | http://localhost:7860 |
| 📊 Analytics Dashboard | http://localhost:8003/metrics/dashboard |

---

## Running Without Docker (Development Mode)

```bash
pip install -r requirements.txt
docker run -d -p 6379:6379 redis:7-alpine
```

Open five terminals:

```bash
# Terminal 1 — Triage
uvicorn lego2_triage.triage_service:app --port 8001 --reload

# Terminal 2 — Qwen 3.6 27B Engine
uvicorn lego3_qwen.qwen_engine:app --port 8002 --reload

# Terminal 3 — Celery Worker
celery -A lego1_gateway.worker.celery_app worker --loglevel=info

# Terminal 4 — Celery Beat (auto-cleanup)
celery -A lego1_gateway.worker.celery_app beat --loglevel=info

# Terminal 5 — Gateway + HITL + Analytics
uvicorn lego1_gateway.main:app --port 8000 --reload
python lego4_hitl/hitl_ui.py
uvicorn lego5_analytics.analytics_service:app --port 8003 --reload
```

---

## API Reference

### Ingestion
```bash
# Single file
curl -X POST http://localhost:8000/api/v1/ingest -F "file=@/path/to/doc.pdf"

# Batch (multiple files)
curl -X POST http://localhost:8000/api/v1/batch-ingest \
  -F "files=@doc1.pdf" -F "files=@doc2.jpg" -F "files=@report.docx"
```

### Status
```bash
# Single job
curl http://localhost:8000/api/v1/status/{job_id}

# Multiple jobs at once
curl "http://localhost:8000/api/v1/batch-status?job_ids=abc&job_ids=def"
```

### Search
```bash
# Keyword search over committed documents
curl "http://localhost:8000/api/v1/search?q=invoice&doc_type=invoice&limit=10"
```

### Statistics
```bash
curl http://localhost:8000/api/v1/stats
```

### Webhooks
```bash
# Register a callback URL
curl -X POST http://localhost:8000/api/v1/webhook/register \
  -H "Content-Type: application/json" \
  -d '{"url": "https://yourserver.com/callback", "description": "My hook"}'

# List webhooks
curl http://localhost:8000/api/v1/webhook/list

# Delete a webhook
curl -X DELETE http://localhost:8000/api/v1/webhook/{hook_id}
```

### Analytics
```bash
# Full JSON metrics
curl http://localhost:8003/metrics

# Download CSV report
curl http://localhost:8003/metrics/export -o report.csv

# Open HTML dashboard in browser
open http://localhost:8003/metrics/dashboard
```

### Qwen Vision Models
```bash
# List available models and current selection
curl http://localhost:8002/models
```

---

## Extracted JSON Schema

Every document processed through the VLM returns:

```json
{
  "document_type": "invoice",
  "language": "en",
  "extracted_text": "Full verbatim text...",
  "markdown_tables": "| Col1 | Col2 |...",
  "key_value_pairs": {
    "Invoice No": "INV-2024-001",
    "Date": "2024-01-15",
    "Total Amount": "$1,250.00"
  },
  "tables": [
    {
      "table_index": 0,
      "headers": ["Item", "Qty", "Price"],
      "rows": [["Widget A", "10", "$125.00"]]
    }
  ],
  "visual_grounding": [
    { "box_2d": [10, 20, 150, 80], "label": "logo" }
  ],
  "handwriting_detected": false,
  "confidence_warning": false,
  "confidence_warning_reason": null,
  "_pipeline": "qwen_vision",
  "_job_id": "uuid-here",
  "_started_at": "2026-08-22T10:00:00+00:00",
  "_completed_at": "2026-08-22T10:00:12+00:00",
  "_filename": "invoice.pdf"
}
```

---

## Webhook Payload

When a job transitions to `pending_review`, your registered URL receives:

```http
POST https://yourserver.com/callback
Content-Type: application/json
X-Greencare-Event: job.pending_review
X-Greencare-Signature: sha256=<hmac_hex>

{
  "job_id": "abc-123",
  "status": "pending_review",
  "pipeline": "qwen_vision",
  "filename": "document.jpg"
}
```

Verify the signature: `HMAC-SHA256(body, WEBHOOK_SECRET)`.

---

## HITL Dashboard Keyboard Shortcuts

| Key | Action |
|---|---|
| `A` | Approve & commit current document |
| `R` | Reject current document |
| `N` | Refresh queue / load next document |

---

## Trade-offs & Design Decisions

| Decision | Rationale |
|---|---|
| Qwen 3.6 27B Vision default | State-of-the-art accuracy & speed; switch via `QWEN_MODEL` env var |
| `response_format=json_object` | Grammar-level enforcement prevents hallucinated formats |
| `temperature=0.0` | Deterministic extraction; identical docs produce identical outputs |
| CPU triage first | Bypasses VLM for digital PDFs, DOCX, XLSX, CSV — ~70% of docs |
| OpenCV deskew + CLAHE | Better image quality → fewer VLM errors |
| Webhook HMAC-SHA256 | Allows receivers to verify payload authenticity |
| Celery Beat for cleanup | Keeps temp disk usage bounded without manual intervention |

> 🔒 **Data Privacy:** This architecture uses a local Qwen 3.6 27B model or cloud endpoint. For maximum privacy, configure the local model endpoint (`LOCAL_MODEL_URL`). All CPU-fast-tracked documents (PDF, DOCX, XLSX, CSV) **never** leave your server.

---

## Project Structure

```
greencare-ai/
├── Dockerfile                     # Unified image for all services
├── docker-compose.yml             # Orchestrates 8 containers
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── README.md                      # This file
│
├── lego1_gateway/
│   ├── main.py                    # FastAPI Gateway v2 (port 8000)
│   └── worker.py                  # Celery pipeline orchestrator v2
│
├── lego2_triage/
│   └── triage_service.py          # CPU triage v2 (.docx/.xlsx/.csv added)
│
├── lego3_qwen/
│   └── qwen_engine.py             # Qwen 3.6 27B Vision Engine v2
│
├── lego4_hitl/
│   └── hitl_ui.py                 # HITL Dashboard v2 (Gradio)
│
├── lego5_analytics/               # [NEW v2]
│   └── analytics_service.py       # Analytics metrics service (port 8003)
│
├── smart_triage/                  # Advanced orchestrator (SmartTriage)
│   ├── orchestrator.py
│   ├── ast_compiler.py
│   ├── ocr_engine.py
│   └── db_analyzer.py
│
└── tests/
    ├── conftest.py
    ├── test_gateway_v2.py         # [NEW v2]
    ├── test_classification.py
    ├── test_db_analyzer.py
    ├── test_excel_sheets.py
    └── test_ocr_engine.py
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```
