# Pramaan IDP — Enterprise Multimodal Document Processing & Triage Platform

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-41%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)]()

A high-throughput, production-grade **Intelligent Document Processing (IDP)** system engineered for enterprise document ingestion, cognitive multi-engine triage, zero-loss table extraction, and human-in-the-loop (HITL) auditing. 

Powered by a hybrid execution model: **Track A Native Digital Fast-Path** (<350ms programmatic extraction) and **Track B Multimodal Vision Engine** (<6.8s via Cloud LPU / Edge VLM), standardizing all outputs into an immutable, Docling-compliant Abstract Syntax Tree (AST).

---

## ⚡ Key Highlights & Benchmarks

| Capability | Benchmark / SLA | Implementation Details |
|---|---|---|
| **Digital Vector PDF (Track A)** | **~300 ms** | PyMuPDF + pdfplumber dual-engine extractor; zero API cost |
| **Tabular Spreadsheets & Office** | **~220 ms** | Native pandas / python-docx traversal across multi-sheet workbooks |
| **Scanned Receipts & Mobile Photos (Track B)** | **6.66 s** | OpenCV text-density pre-screen (15ms) + Groq Cloud LPU VLM (`qwen/qwen3.8-27b`) |
| **Human-in-the-Loop Review** | **Real-time** | Unified Streamlit Hub (Port 8501) with bounding-box visual grounding and interactive AST editing |
| **Output Standardization** | **100% Schema Valid** | Type-safe Pydantic models conforming to strict Docling-compatible AST schema |
| **Test Coverage** | **41 / 41 (100% Green)** | End-to-end integration, boundary checks, and production hardening suites |

---

## 🏗️ System Architecture

```
                          ┌─────────────────────────────────────┐
                          │         CLIENT INGESTION            │
                          │   (REST API / Webhooks / Batch)     │
                          └──────────────────┬──────────────────┘
                                             │ POST /api/v1/ingest (HTTP 202)
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  API GATEWAY — Lego 1 (Port 8000)                                                      │
│  • Idempotency key evaluation & UUID correlation tracing                               │
│  • Magic-byte verification & file size validation                                      │
│  • Background dispatch & asynchronous job polling (/api/v1/status/{job_id})            │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
│ TRACK A: Programmatic Fast-Path          │  │ TRACK B: Multimodal Vision Fast-Path     │
│ (Digital PDFs, Excel, CSV, Word)         │  │ (Scanned PDFs, Invoices, Camera Photos)  │
│                                          │  │                                          │
│ • PyMuPDF & pdfplumber extraction        │  │ • 15ms OpenCV text density classification│
│ • Multi-sheet tabular parsing            │  │ • Direct LPU acceleration (Groq Cloud)   │
│ • Zero API cost & sub-second latency     │  │ • Bounding box visual grounding          │
│ • Native paragraph & table chunking      │  │ • Complex table & key-value parsing      │
└─────────────────────┬────────────────────┘  └─────────────────────┬────────────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  AST COMPILER & SERIALIZATION ENGINE                                                   │
│  • Compiles normalized elements (headings, paragraphs, tables, key-values, graphics)   │
│  • Enforces Docling-compliant schema versioning                                        │
│  • Writes immutable state to `pending_review/{job_id}.json`                            │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
│ HITL REVIEW HUB (Port 8501)              │  │ ANALYTICS HUB — Lego 5 (Port 8003)       │
│ • Streamlit Web Dashboard                │  │ • Real-time throughput & latency telemetry│
│ • Visual bounding-box overlays           │  │ • 30-day decision breakdown              │
│ • Side-by-side AST editor                │  │ • Exportable audit metrics & SLA reports │
│ • AI-assisted summary & interactive Q&A  │  │ • REST metrics endpoints (/metrics)      │
└──────────────────────────────────────────┘  └──────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── lego1_gateway/             # API Gateway & Task Ingestion (Port 8000)
│   ├── main_standalone.py     # Standalone FastAPI gateway with embedded triage orchestrator
│   ├── main.py                # Distributed FastAPI gateway (Celery-compatible)
│   └── worker.py              # Celery background worker pipeline
├── lego2_triage/              # CPU Pre-screening Service (Port 8001)
│   └── triage_service.py      # Specular glare suppression, deskew & vector detection
├── lego3_qwen/                # Vision Language Model Engine (Port 8002)
│   └── qwen_engine.py         # Unified OpenAI/Groq/Local VLM interface (`/extract`, `/route`)
├── lego5_analytics/           # Platform Telemetry & Observability (Port 8003)
│   └── analytics_service.py   # Metrics tracking, throughput analytics, and CSV exports
├── smart_triage/              # Core Orchestrator & Human-in-the-Loop Hub
│   ├── smart_triage_hitl_app.py # Consolidated Streamlit HITL Dashboard (Port 8501)
│   ├── orchestrator.py        # SmartTriageOrchestrator: Track A / Track B decision matrix
│   ├── ast_compiler.py        # Abstract Syntax Tree compiler & schema normalizer
│   ├── ocr_engine.py          # Adaptive image preprocessor & fallback OCR cascade
│   └── db_analyzer.py         # Tabular data analyzer & ER relationship engine
├── schemas/                   # JSON schemas for AST validation
│   └── ast_schema.json        # Schema specification for IDP AST documents
├── benchmark_corpus/          # Standardized corpus for validation & latency testing
├── tests/                     # Comprehensive test suite (41 unit and integration tests)
│   ├── test_production_hardening.py
│   ├── test_gateway_v2.py
│   ├── test_classification.py
│   ├── test_excel_sheets.py
│   └── test_db_analyzer.py
├── run_all.py                 # Multi-service master launcher (development & production)
├── docker-compose.yml         # Containerized production stack configuration
└── config.py                  # Centralized, type-validated configuration
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git
- (Optional) Docker & Docker Compose for containerized deployment

### 1. Installation
```bash
git clone https://github.com/coder-Dharshu/Test.git idp-engine
cd idp-engine

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the template configuration file:
```bash
cp .env.example .env
```
Edit `.env` to configure your inference backend:
```env
# Inference Provider: 'groq' (recommended for <6s latency), 'local', or 'openai_compatible'
INFERENCE_BACKEND=groq
ALLOW_CLOUD_ESCALATION=true

# Groq Cloud Acceleration
GROQ_API_KEY=your_groq_api_key_here
QWEN_MODEL=qwen/qwen3.8-27b

# Gateway & Service Bindings
GATEWAY_URL=http://localhost:8000
LEGO3_VLM_URL=http://localhost:8002
```

### 3. Launching All Services
Start the complete 5-service cluster using the master launcher:
```bash
python run_all.py
```
This starts and monitors all microservices:
- **Port 8000**: API Gateway & Ingestion
- **Port 8001**: CPU Triage & Vector Detection
- **Port 8002**: Multimodal Vision Engine
- **Port 8003**: Analytics & Metrics Dashboard
- **Port 8501**: Streamlit Human-in-the-Loop Hub

---

## 🐳 Docker Deployment

To launch the containerized stack:
```bash
docker-compose up --build -d
```
All persistent queues, uploaded files, and committed databases are mounted via Docker volumes defined in `docker-compose.yml`.

---

## 📖 API Documentation & Usage

### 1. Ingest a Document
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@benchmark_corpus/03_clean_scanned_receipt.jpg"
```
**Response (HTTP 202 Accepted):**
```json
{
  "status": "queued",
  "job_id": "9630c61c-0d9c-4039-9810-269e3c8dd39b",
  "filename": "03_clean_scanned_receipt.jpg",
  "message": "Document accepted. Poll /api/v1/status/{job_id} for updates."
}
```

### 2. Poll Job Status
```bash
curl http://localhost:8000/api/v1/status/9630c61c-0d9c-4039-9810-269e3c8dd39b
```
**Response:**
```json
{
  "job_id": "9630c61c-0d9c-4039-9810-269e3c8dd39b",
  "status": "pending_review",
  "filename": "03_clean_scanned_receipt.jpg"
}
```

### 3. Batch Ingestion
```bash
curl -X POST http://localhost:8000/api/v1/batch-ingest \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.jpg" \
  -F "files=@sheet.xlsx"
```

### 4. Search Committed Documents
```bash
curl "http://localhost:8000/api/v1/search?q=Invoice&limit=10"
```

### 5. Webhook Subscriptions
Register a webhook endpoint to receive notifications when documents reach `pending_review` or are committed:
```bash
curl -X POST http://localhost:8000/api/v1/webhook/register \
  -H "Content-Type: application/json" \
  -d '{"url": "https://your-domain.com/api/idp-callback", "description": "Production Webhook"}'
```

---

## 🖥️ Streamlit HITL Dashboard (Port 8501)

The Human-in-the-Loop dashboard (`smart_triage/smart_triage_hitl_app.py`) provides human reviewers with tools for document validation:
- **Visual Grounding Overlays**: Highlights bounding boxes directly over the source image.
- **Side-by-Side Review**: Compares original visual document with extracted AST elements.
- **Tabular Data Editing**: Edit tables, headers, and rows with inline validation.
- **Ask AI Q&A**: Real-time semantic querying of document content using the integrated LLM.
- **Document Summaries**: Automatically generates executive summaries and key highlights.
- **Audit Logging**: Every approve/reject action is immutably recorded in `audit/audit_log.jsonl`.
- **Exporting**: Export finalized data to JSON, multi-sheet Excel workbooks, or CSV.

---

## 🧪 Testing & Quality Assurance

Run the comprehensive test suite:
```bash
python -m pytest tests/ -v
```
All 41 unit and integration tests validate:
- **Routing Boundaries**: Exact thresholding (e.g. 80.0% confidence cutoff).
- **Format Ingestion**: Validates PDF, DOCX, XLSX, CSV, JPG, and PNG handling.
- **Database Analysis**: Validates foreign key and ER inference across extracted tables.
- **Security & Hardening**: Validates API key policies, file validation, and error states.

---

## 📄 License
Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.
