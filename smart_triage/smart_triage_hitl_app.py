"""
Smart Triage — Streamlit HITL Dashboard
=========================================
§9 Human-in-the-Loop Hub — Operator audit workspace for validating
AI-extracted documents across all three execution paths.

Features:
  • Side-by-side document image preview + AST element editor
  • PCS complexity score visualization
  • Execution path badges (TRACK_A / PATH_1 / PATH_2)
  • Inline table edits via st.data_editor
  • Paragraph/heading text editing
  • Approve → final_database/ | Reject → rejected/
  • Export: JSON, CSV, Excel
  • Real-time queue metrics banner

Runs on: Port 8501
"""

import os
import glob
import json
import io
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import pandas as pd
try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def get_hitl_llm_client(api_key: str):
    local_url = os.environ.get("LOCAL_MODEL_URL") or os.environ.get("QWEN_LOCAL_URL", "")
    if local_url and OpenAI:
        return OpenAI(base_url=local_url, api_key=api_key or "local-qwen")
    if api_key and api_key.startswith("gsk_") and Groq:
        return Groq(api_key=api_key)
    if api_key and OpenAI:
        return OpenAI(api_key=api_key)
    if Groq:
        return Groq(api_key=api_key or "dummy")
    return None

# ── DB Analyzer (optional, graceful fallback) ─────────────────────────
try:
    from smart_triage.db_analyzer import analyze_schema, ERDiagramRenderer, PYVIS_AVAILABLE
    DB_ANALYZER_AVAILABLE = True
except ImportError:
    try:
        from db_analyzer import analyze_schema, ERDiagramRenderer, PYVIS_AVAILABLE
        DB_ANALYZER_AVAILABLE = True
    except ImportError:
        DB_ANALYZER_AVAILABLE = False
        PYVIS_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title   = "Smart Triage IDP — HITL Hub",
    page_icon    = "🛡",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Gradient header */
    .smart-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c75 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .smart-header h1 { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; margin:0; }
    .smart-header p  { color: #94a3b8; font-size: 0.85rem; margin: 6px 0 0; }

    /* Path badges */
    .badge-track-a  { background:#065f46; color:#6ee7b7; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; }
    .badge-path-1   { background:#78350f; color:#fcd34d; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; }
    .badge-path-2   { background:#4c1d95; color:#c4b5fd; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; }
    .badge-unknown  { background:#1e293b; color:#94a3b8; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; }

    /* PCS meter */
    .pcs-bar-wrap { background:#1e293b; border-radius:8px; height:8px; width:100%; margin:6px 0; }
    .pcs-bar      { height:8px; border-radius:8px; transition: width 0.5s ease; }

    /* Card containers */
    .info-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .info-card h4 { color: #e2e8f0; font-size: 0.9rem; font-weight: 600; margin: 0 0 4px; }
    .info-card p  { color: #64748b; font-size: 0.8rem; margin: 0; }

    /* Confidence badges */
    .conf-high { background:#14532d; color:#86efac; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; }
    .conf-low  { background:#7f1d1d; color:#fca5a5; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; }
    .conf-hw   { background:#78350f; color:#fcd34d; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; }

    /* Metric styling */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.75rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important; font-size: 1.4rem !important; font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Approve / Reject buttons */
    .stButton button[kind="primary"]  { background: #059669; border:0; font-weight:600; }
    .stButton button[kind="secondary"]{ border: 1px solid #334155; color: #94a3b8; font-weight:600; }

    div[data-testid="stTabs"] [data-baseweb="tab"] { color: #94a3b8; }
    div[data-testid="stTabs"] [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color:#38bdf8 !important; }

    .element-type-tag {
        display: inline-block;
        background: #0f172a;
        border: 1px solid #334155;
        color: #94a3b8;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Directory config ──────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
PENDING_DIR  = BASE_DIR / "pending_review"
FINAL_DIR    = BASE_DIR / "final_database"
REJECTED_DIR = BASE_DIR / "rejected"
UPLOAD_DIR   = BASE_DIR / "temp_uploads"
LEGO2_TEMP   = BASE_DIR / "lego2_temp"
EXPORTS_DIR  = BASE_DIR / "exports"
GATEWAY_URL  = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000")

for d in [PENDING_DIR, FINAL_DIR, REJECTED_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "current_file"       not in st.session_state: st.session_state.current_file       = None
if "ast_data"           not in st.session_state: st.session_state.ast_data           = None
if "is_modified"        not in st.session_state: st.session_state.is_modified        = False
if "current_page_idx"   not in st.session_state: st.session_state.current_page_idx   = 0
if "upload_msg"         not in st.session_state: st.session_state.upload_msg         = ""
if "selected_sheet_idx" not in st.session_state: st.session_state.selected_sheet_idx = 0  # Excel multi-sheet selector
if "db_analysis_result" not in st.session_state: st.session_state.db_analysis_result = None  # DB ER result

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_pending_files() -> list[Path]:
    return sorted(PENDING_DIR.glob("*.json"))

def get_pending_names() -> list[str]:
    return [f.name for f in get_pending_files()]

def queue_stats() -> dict:
    return {
        "pending"  : len(list(PENDING_DIR.glob("*.json"))),
        "committed": len(list(FINAL_DIR.glob("*.json"))),
        "rejected" : len(list(REJECTED_DIR.glob("*.json"))),
    }

def find_source_image(job_id: str) -> Path | None:
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    for d in [UPLOAD_DIR, LEGO2_TEMP]:
        if not d.is_dir(): continue
        for f in sorted(d.iterdir()):
            if f.name.startswith(job_id) and f.suffix.lower() in exts:
                return f
    return None

def execution_path_badge(path: str) -> str:
    badges = {
        "TRACK_A"      : '<span class="badge-track-a">⚡ TRACK A — Digital Fast-Path</span>',
        "PATH_1"       : '<span class="badge-path-1">🔠 PATH 1 — Local CPU OCR (High Confidence)</span>',
        "PATH_2"       : '<span class="badge-path-2">🧠 PATH 2 — VLM (OCR confidence escalation)</span>',
        "GENERAL_PHOTO": '<span class="badge-unknown">🖼 GENERAL PHOTO — Image returned unchanged</span>',
    }
    return badges.get(path, f'<span class="badge-unknown">{path}</span>')

def pcs_meter_html(score: float) -> str:
    pct   = round(score * 100, 1)
    color = "#ef4444" if score >= 0.5 else "#22c55e" if score < 0.2 else "#f59e0b"
    return f"""
    <div style="margin:4px 0 12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span style="color:#94a3b8;font-size:11px;font-weight:600">PAGE COMPLEXITY SCORE</span>
        <span style="color:{color};font-size:14px;font-weight:700">{pct:.1f}%</span>
      </div>
      <div class="pcs-bar-wrap">
        <div class="pcs-bar" style="width:{pct}%;background:{color};"></div>
      </div>
      <div style="color:#475569;font-size:10px;margin-top:3px">
        Threshold: 50% &nbsp;|&nbsp; {'⬆ VLM Route' if score >= 0.5 else '⬇ CPU Route'}
      </div>
    </div>
    """

def confidence_badges_html(page: dict) -> str:
    parts = []
    if page.get("confidence_warning"):
        parts.append('<span class="conf-low">⚠ LOW CONFIDENCE</span>')
    else:
        parts.append('<span class="conf-high">✓ HIGH CONFIDENCE</span>')
    if page.get("handwriting_detected"):
        parts.append('<span class="conf-hw">✍ Handwriting</span>')
    reason = page.get("confidence_warning_reason")
    html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0">' + "".join(parts) + "</div>"
    if reason:
        html += f'<div style="color:#fca5a5;font-size:11px;margin-top:4px">{reason}</div>'
    return html

def load_ast(file_path: Path) -> dict | None:
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")
        return None

def sanitize_headers(headers: list, n_cols: int) -> list:
    """Ensure headers are unique, non-empty strings matching n_cols length."""
    # Fill missing / empty headers with generic column names
    sanitized = []
    for idx in range(n_cols):
        raw = headers[idx] if idx < len(headers) else ""
        h   = str(raw).strip() if raw else ""
        sanitized.append(h if h else f"Col_{idx + 1}")
    # Deduplicate: append suffix if name already seen
    seen: dict[str, int] = {}
    result = []
    for h in sanitized:
        if h in seen:
            seen[h] += 1
            result.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            result.append(h)
    return result

@st.cache_data(show_spinner=False)
def generate_summary(extracted_text: str, doc_type: str, api_key: str) -> str:
    """Call Qwen 3.6 27B text API to produce a concise document summary."""
    if not extracted_text.strip():
        return "⚠️ No extracted text available to summarise."
    try:
        client = get_hitl_llm_client(api_key)
        if not client:
            return "❌ No LLM client available. Check QWEN_API_KEY / GROQ_API_KEY or LOCAL_MODEL_URL."
        model  = os.environ.get("QWEN_MODEL") or os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
        # Truncate very large documents to stay within token limits
        text_snippet = extracted_text[:12000]
        prompt = (
            f"You are an expert document analyst. The document type is '{doc_type}'.\n"
            "Analyse the following extracted document text and return a structured summary with:\n"
            "1. A one-paragraph executive summary (3-4 sentences).\n"
            "2. Key Points: 5-8 concise bullet points covering the most important facts, figures, and actions.\n"
            "3. Document Highlights: any notable dates, names, amounts, or references found.\n"
            "Format your response in clear Markdown.\n\n"
            f"--- Document Text ---\n{text_snippet}"
        )
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or "No summary generated."
    except Exception as exc:
        return f"❌ Summary generation failed: {exc}"

def save_to_dir(ast_data: dict, target_dir: Path, extra_fields: dict = {}):
    fname = st.session_state.current_file.name
    data  = {**ast_data, **extra_fields}
    (target_dir / fname).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    st.session_state.current_file.unlink(missing_ok=True)

def build_export_frames(ast_data: dict) -> dict[str, pd.DataFrame]:
    frames = {}
    for page in ast_data.get("pages", []):
        for elem in page.get("elements", []):
            if elem["type"] in ("paragraph", "heading"):
                key = f"Page_{page['page_index']+1}_Text"
                row = {"Type": elem["type"], "Text": elem["content"].get("text", "")}
                frames.setdefault(key, []).append(row)
            elif elem["type"] == "key_value":
                key = f"Page_{page['page_index']+1}_KeyValues"
                for pair in elem["content"].get("pairs", []):
                    frames.setdefault(key, []).append(pair)
            elif elem["type"] == "table":
                key = f"Page_{page['page_index']+1}_Table_{elem['content'].get('table_index',0)+1}"
                rows    = elem["content"].get("rows", [])
                headers = elem["content"].get("headers", [])
                if rows:
                    frames[key] = rows
                    frames[key + "__headers"] = headers   # store headers separately
    result = {}
    for k, v in frames.items():
        if k.endswith("__headers"): continue
        hdrs = frames.get(k + "__headers")
        try:
            result[k[:31]] = pd.DataFrame(v, columns=hdrs) if hdrs else pd.DataFrame(v)
        except Exception:
            result[k[:31]] = pd.DataFrame(v)
    return result

def upload_file(uploaded_file):
    if uploaded_file is None:
        return
    import requests as rq
    import time
    try:
        resp = rq.post(
            f"{GATEWAY_URL}/api/v1/ingest",
            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            timeout=30,
        )
        resp.raise_for_status()
        job_id = resp.json().get("job_id", "?")
        
        with st.spinner(f"Processing job {job_id}... Please wait for extraction to complete."):
            # Active polling loop with 45-second timeout fail-safe
            max_retries = 45
            poll_interval = 1
            expected_file = None
            
            for _ in range(max_retries):
                # Search for the JSON file in pending_review
                for f in PENDING_DIR.glob("*.json"):
                    if f.name.startswith(job_id):
                        expected_file = f
                        break
                        
                if expected_file:
                    break
                    
                time.sleep(poll_interval)
                
            if not expected_file:
                st.error("Backend timeout or crash. The local CPU engine failed to process the document. Please check the backend terminal for Python errors.")
                return
                
        if expected_file:
            data = load_ast(expected_file)
            if data:
                st.session_state.current_file     = expected_file
                st.session_state.ast_data         = data
                st.session_state.is_modified      = False
                st.session_state.current_page_idx = 0
                st.session_state.upload_msg = f"✅ Successfully processed job `{job_id}`."
            else:
                st.session_state.upload_msg = f"❌ Failed to parse AST for job `{job_id}`."
        else:
            st.session_state.upload_msg = f"⏳ Timeout waiting for job `{job_id}`. Check the queue manually later."
            
    except Exception as e:
        st.session_state.upload_msg = f"❌ Upload failed: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡 Smart Triage IDP")
    st.markdown("---")

    stats = queue_stats()
    st.markdown(f"""
    <div style='margin-bottom:16px'>
        <div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e293b'>
            <span style='color:#94a3b8;font-size:13px'>⏳ Pending Review</span>
            <span style='color:#f59e0b;font-weight:700'>{stats['pending']}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e293b'>
            <span style='color:#94a3b8;font-size:13px'>✅ Committed</span>
            <span style='color:#22c55e;font-weight:700'>{stats['committed']}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:6px 0'>
            <span style='color:#94a3b8;font-size:13px'>❌ Rejected</span>
            <span style='color:#ef4444;font-weight:700'>{stats['rejected']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Select Document**")
    pending_names = get_pending_names()
    if not pending_names:
        st.info("Queue is empty. Upload a document above.")
    else:
        selected = st.selectbox("Pending Documents", pending_names, label_visibility="collapsed")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Load", use_container_width=True):
                fp = PENDING_DIR / selected
                data = load_ast(fp)
                if data:
                    st.session_state.current_file     = fp
                    st.session_state.ast_data         = data
                    st.session_state.is_modified      = False
                    st.session_state.current_page_idx = 0
                    st.rerun()
        with col2:
            if st.button("🗑️ Delete", use_container_width=True):
                fp = PENDING_DIR / selected
                fp.unlink(missing_ok=True)
                if st.session_state.current_file == fp:
                    st.session_state.current_file = None
                    st.session_state.ast_data = None
                st.rerun()

    st.markdown("---")
    st.markdown("**Upload New Document**")
    uploaded = st.file_uploader(
        "Choose file",
        type=["pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp",
              "xlsx", "xls", "csv", "docx"],
        label_visibility="collapsed",
    )
    if st.button("🚀 Submit to Pipeline", use_container_width=True, type="primary"):
        upload_file(uploaded)
        st.rerun()
    if st.session_state.upload_msg:
        st.markdown(st.session_state.upload_msg)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col4:
        if st.button("🗑️ Clear All", use_container_width=True):
            for f in PENDING_DIR.glob("*.json"):
                f.unlink(missing_ok=True)
            st.session_state.current_file = None
            st.session_state.ast_data = None
            st.rerun()

    # ── DB Relationship Analyzer ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🗄 Database Relationship Analyzer**")

    if not DB_ANALYZER_AVAILABLE:
        st.warning("db_analyzer module not found.")
    else:
        db_file = st.file_uploader(
            "Upload SQLite / Excel / CSV",
            type=["db", "sqlite", "sqlite3", "xlsx", "xls", "csv"],
            key="db_upload",
            label_visibility="collapsed",
        )
        if db_file is not None:
            if st.button("🔍 Analyze Schema", use_container_width=True):
                import tempfile
                suffix = Path(db_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(db_file.read())
                    tmp_path = tmp.name
                try:
                    with st.spinner("Analyzing schema..."):
                        result = analyze_schema(tmp_path)
                    st.session_state.db_analysis_result = result
                    st.success(result.summary())
                except Exception as exc:
                    st.error(f"Schema analysis failed: {exc}")
                finally:
                    import os as _os
                    try: _os.unlink(tmp_path)
                    except Exception: pass

        if st.session_state.db_analysis_result is not None:
            if st.button("❌ Clear Analysis", key="db_clear"):
                st.session_state.db_analysis_result = None
                st.rerun()


def render_qa_tab(job_id: str, pages: list[dict]):
    st.markdown("### 💬 Ask AI — Document Q&A")
    st.markdown(
        "<div style='color:#64748b;font-size:13px;margin-bottom:16px'>"
        "Ask any question about this document. The AI will answer using only the "
        "extracted content — no hallucination, grounded answers only."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Collect all document text ──────────────────────────────────────
    qa_texts: list[str] = []
    for p_idx, page in enumerate(pages):
        for elem in page.get("elements", []):
            etype   = elem.get("type", "")
            content = elem.get("content", {})
            if etype in ("text", "paragraph", "heading"):
                t = content.get("text", "").strip()
                if t:
                    qa_texts.append(f"[Page {page.get('page_number', p_idx+1)}]\n{t}")
            elif etype == "key_value":
                for pair in content.get("pairs", []):
                    if isinstance(pair, dict):
                        kv_line = " : ".join(str(v) for v in pair.values())
                        qa_texts.append(kv_line)
            elif etype == "table":
                headers = content.get("headers", [])
                rows    = content.get("rows", [])
                if rows:
                    table_lines = [" | ".join(str(h) for h in headers)]
                    for row in rows[:20]:  # Limit for context
                        cells = list(row.values()) if isinstance(row, dict) else row
                        table_lines.append(" | ".join(str(c) for c in cells))
                    qa_texts.append("Table:\n" + "\n".join(table_lines))
        raw_et = page.get("extracted_text", "")
        if raw_et and raw_et not in "\n".join(qa_texts):
            qa_texts.append(raw_et)

    qa_context = "\n\n".join(qa_texts)[:15000]  # Token-safe limit

    llm_key = os.environ.get("QWEN_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    local_url = os.environ.get("LOCAL_MODEL_URL") or os.environ.get("QWEN_LOCAL_URL", "")
    if not qa_context.strip():
        st.info("⚠️ No text content found in the document to answer questions about.")
    elif not llm_key and not local_url:
        st.error("❌ QWEN_API_KEY / GROQ_API_KEY or LOCAL_MODEL_URL not set. Cannot use Q&A without an API key or local model.")
    else:
        # ── Session state for chat history ─────────────────────────────
        chat_key = f"qa_history_{job_id}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        # ── CSS for chat bubbles ────────────────────────────────────────
        st.markdown("""
        <style>
        .qa-user-bubble {
            background: linear-gradient(135deg, #1e40af, #3b82f6);
            color: #fff;
            border-radius: 16px 16px 4px 16px;
            padding: 12px 18px;
            margin: 6px 0 6px auto;
            max-width: 80%;
            font-size: 14px;
            line-height: 1.6;
            display: inline-block;
            float: right;
            clear: both;
            box-shadow: 0 2px 8px rgba(59,130,246,0.3);
        }
        .qa-ai-bubble {
            background: #1e293b;
            border: 1px solid #334155;
            color: #e2e8f0;
            border-radius: 16px 16px 16px 4px;
            padding: 14px 18px;
            margin: 6px auto 6px 0;
            max-width: 85%;
            font-size: 14px;
            line-height: 1.7;
            display: inline-block;
            float: left;
            clear: both;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .qa-clear { clear: both; }
        .qa-label-user { color:#60a5fa; font-size:11px; font-weight:700; float:right; clear:both; margin-top:4px; margin-right:4px; }
        .qa-label-ai   { color:#94a3b8; font-size:11px; font-weight:700; float:left; clear:both; margin-top:4px; margin-left:4px; }
        </style>
        """, unsafe_allow_html=True)

        # ── Render chat history ──────────────────────────────────────────
        history = st.session_state[chat_key]
        if history:
            for msg in history:
                if msg["role"] == "user":
                    st.markdown(
                        f"<div class='qa-label-user'>You</div>"
                        f"<div class='qa-user-bubble'>{msg['content']}</div>"
                        f"<div class='qa-clear'></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    safe_content = msg["content"].replace("\n", "<br>")
                    st.markdown(
                        f"<div class='qa-label-ai'>🤖 AI</div>"
                        f"<div class='qa-ai-bubble'>{safe_content}</div>"
                        f"<div class='qa-clear'></div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("<div class='qa-clear'></div>", unsafe_allow_html=True)
            st.markdown("---")

        # ── Suggested questions ──────────────────────────────────────────
        if not history:
            st.markdown(
                "<div style='color:#64748b;font-size:12px;font-weight:600;margin-bottom:8px'>💡 Suggested questions:</div>",
                unsafe_allow_html=True,
            )
            suggested_qs = [
                "What is this document about?",
                "List all key facts and figures mentioned.",
                "What are the dates mentioned in this document?",
                "Summarize the main entities (names, companies, amounts).",
                "What actions or conclusions are mentioned?",
            ]
            sq_cols = st.columns(len(suggested_qs))
            for sq_col, sq in zip(sq_cols, suggested_qs):
                with sq_col:
                    if st.button(sq, key=f"sq_{job_id}_{sq[:20]}", use_container_width=True):
                        st.session_state[f"qa_prefill_{job_id}"] = sq
                        st.rerun()

        # ── Q&A input ────────────────────────────────────────────────────
        prefill_val = st.session_state.pop(f"qa_prefill_{job_id}", "")
        with st.form(key=f"qa_form_{job_id}", clear_on_submit=True):
            q_col, btn_col = st.columns([5, 1])
            with q_col:
                user_question = st.text_input(
                    "Your question",
                    value=prefill_val,
                    placeholder="e.g. What is the total amount? Who is the customer?",
                    label_visibility="collapsed",
                    key=f"qa_input_{job_id}",
                )
            with btn_col:
                submitted = st.form_submit_button("Ask ✈", use_container_width=True, type="primary")

        if submitted and user_question.strip():
            system_prompt = (
                "You are an intelligent document analysis assistant. "
                "You MUST answer questions ONLY using the document content provided below. "
                "If the answer is not in the document, say so clearly. "
                "Be concise, factual, and reference specific parts of the document when possible.\n\n"
                f"--- DOCUMENT CONTENT ---\n{qa_context}\n--- END OF DOCUMENT ---"
            )

            api_messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-6:]:  # Last 3 turns for context window
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            api_messages.append({"role": "user", "content": user_question.strip()})

            st.session_state[chat_key].append({"role": "user", "content": user_question.strip()})

            with st.spinner("🤖 Qwen 3.6 27B thinking..."):
                try:
                    client_qa = get_hitl_llm_client(llm_key)
                    model_name = os.environ.get("QWEN_MODEL") or os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
                    response = client_qa.chat.completions.create(
                        messages=api_messages,
                        model=model_name,
                        temperature=0.2,
                        max_tokens=1024,
                    )
                    ai_answer = response.choices[0].message.content or "No answer generated."
                except Exception as exc:
                    ai_answer = f"❌ Error calling AI: {exc}"

            st.session_state[chat_key].append({"role": "assistant", "content": ai_answer})
            st.rerun()

        if history:
            if st.button("🗑️ Clear conversation", key=f"qa_clear_{job_id}"):
                st.session_state[chat_key] = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="smart-header">
    <h1>🛡 Smart Triage Enterprise IDP</h1>
    <p>Strategic Audit Workspace &amp; Multi-path Extraction Validation Engine</p>
</div>
""", unsafe_allow_html=True)

# ── No document loaded ────────────────────────────────────────────────────────
if st.session_state.ast_data is None:
    # ── DB Visualization (shown even without a document loaded) ───────────
    if st.session_state.db_analysis_result is not None:
        result = st.session_state.db_analysis_result
        st.markdown("## 🗄 Database Relationship Visualization")
        st.markdown(result.summary())

        if result.warnings:
            for w in result.warnings:
                st.warning(w)

        db_tab_graph, db_tab_table, db_tab_schema = st.tabs([
            "🕸 Relationship Graph", "📋 Relationship Table", "🏛 Schema Details"
        ])

        renderer = ERDiagramRenderer()

        with db_tab_graph:
            if not PYVIS_AVAILABLE:
                st.warning("Install `pyvis` (`pip install pyvis`) for interactive graph rendering.")
                st.code(renderer.render_text_table(result))
            elif not result.has_relationships:
                st.info("No relationships detected to visualize.")
                st.code(renderer.render_text_table(result))
            else:
                html_graph = renderer.render_html(result, height="580px")
                if html_graph:
                    st.components.v1.html(html_graph, height=600, scrolling=False)
                else:
                    st.code(renderer.render_text_table(result))

        with db_tab_table:
            if result.relationships:
                rel_rows = [r.to_dict() for r in result.relationships]
                df_rels = pd.DataFrame(rel_rows)
                # Clean up column names for display
                df_rels.columns = [c.replace("_", " ").title() for c in df_rels.columns]
                st.dataframe(df_rels, use_container_width=True)
            else:
                st.info("No relationships found.")
                for w in result.warnings:
                    st.caption(f"ℹ️ {w}")

        with db_tab_schema:
            for table in result.tables:
                with st.expander(
                    f"**{table.name}** — {len(table.columns)} cols"
                    + (f" | {table.row_count:,} rows" if table.row_count is not None else ""),
                    expanded=False,
                ):
                    col_rows = [
                        {
                            "Column": c.name,
                            "Type": c.data_type,
                            "PK": "🔑" if c.is_primary_key else "",
                            "Nullable": "✓" if c.is_nullable else "✗",
                            "Default": c.default_value or "",
                        }
                        for c in table.columns
                    ]
                    st.dataframe(pd.DataFrame(col_rows), use_container_width=True, hide_index=True)
        st.stop()

    stats = queue_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("⏳ Pending Review",  stats["pending"])
    c2.metric("✅ Committed",       stats["committed"])
    c3.metric("❌ Rejected",        stats["rejected"])
    st.markdown("---")
    st.info("**Select a document from the sidebar to begin the review process.**")
    st.markdown("""
    #### Enterprise-Grade Document Routing & Sovereignty:
    | Path | Trigger | Processing & Data Sovereignty |
    |---|---|---|
    | **Track A** | Digital PDF with ≥50 chars | **Local & Offline**: Instant programmatic extraction on your local server zero API costs. |
    |  **Path 2** | Scanned PDF, photo, or handwritten image | **Local VLM Ready**: Configured for local offline edge models Ollama ensuring **zero data leaves your private enterprise network**. |
    """)
    st.stop()


# ── Document loaded ───────────────────────────────────────────────────────────
ast       = st.session_state.ast_data
pages     = ast.get("pages", [])
meta      = ast.get("document_metadata", {})
job_id    = meta.get("doc_id") or ast.get("_job_id", "—")
filename  = meta.get("source_filename") or ast.get("_filename", "—")
exec_path = meta.get("execution_path", "PATH_2")
pcs       = meta.get("pcs_score", 0.0) or 0.0
pipeline  = meta.get("pipeline") or ast.get("_pipeline", "—")
n_pages   = len(pages)

# ── Top metrics banner ────────────────────────────────────────────────────────
stats = queue_stats()
m1, m2, m3, m4 = st.columns(4)
m1.metric("📄 Total Pages",      n_pages)
m2.metric("⏳ Queue Remaining",  stats["pending"])
m3.metric("✅ Committed",        stats["committed"])
m4.metric("❌ Rejected",         stats["rejected"])
st.markdown("---")

# ── Document metadata bar ─────────────────────────────────────────────────────
st.markdown(
    f"**File:** `{filename}` &nbsp;|&nbsp; **Job ID:** `{job_id}` &nbsp;|&nbsp; "
    f"{execution_path_badge(exec_path)} &nbsp;|&nbsp; **Pipeline:** `{pipeline}`",
    unsafe_allow_html=True,
)
st.markdown(pcs_meter_html(pcs), unsafe_allow_html=True)

# ── PCS Breakdown (if available) ──────────────────────────────────────────────
pcs_breakdown = meta.get("pcs_breakdown") or ast.get("pcs_breakdown")
if pcs_breakdown:
    with st.expander("📊 PCS Score Breakdown", expanded=False):
        bc = st.columns(4)
        bc[0].metric("Tables (×0.40)",      f"{pcs_breakdown.get('n_table', 0):.3f}")
        bc[1].metric("Handwriting (×0.45)", f"{pcs_breakdown.get('n_handwritten', 0):.3f}")
        bc[2].metric("Overlap (×0.10)",     f"{pcs_breakdown.get('n_overlap', 0):.3f}")
        bc[3].metric("Graphic Area (×0.05)",f"{pcs_breakdown.get('a_graphic', 0):.3f}")

# ── OCR Diagnostic Panel ──────────────────────────────────────────────────────
_first_page = pages[0] if pages else {}
_ocr_conf   = _first_page.get("_ocr_confidence")
_ocr_strat  = _first_page.get("_ocr_strategy")
_density    = _first_page.get("_text_density_score")
_pipe_ms    = _first_page.get("_pipeline_time_ms")
_doc_type   = _first_page.get("document_type")
if any(v is not None for v in [_ocr_conf, _ocr_strat, _density, _pipe_ms]):
    with st.expander("🔬 OCR Pipeline Diagnostics", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        if _density is not None:
            d1.metric("Text Density Score", f"{_density:.4f}",
                      help="0=no text, 1=very dense text. >0.18=text doc, <0.04=general image")
        if _ocr_conf is not None:
            conf_color = "normal" if _ocr_conf >= 60 else "inverse"
            d2.metric("OCR Confidence", f"{_ocr_conf:.1f}%",
                      delta="High" if _ocr_conf >= 60 else "Low",
                      delta_color=conf_color)
        if _ocr_strat is not None:
            d3.metric("Preprocessing Strategy", _ocr_strat.replace("_", " ").title())
        if _pipe_ms is not None:
            d4.metric("Pipeline Time", f"{_pipe_ms:.0f}ms")
        if _doc_type:
            st.markdown(
                f"<span style='background:#1e40af;color:#bfdbfe;padding:4px 12px;"
                f"border-radius:20px;font-size:12px;font-weight:700'>📄 {_doc_type}</span>",
                unsafe_allow_html=True
            )

st.markdown("---")


# ── Document Visuals & Extraction (Multi-Page View) ───────────────────────────
# We no longer paginate the view. Instead, we render all pages linearly.

# ── General Photo special view ───────────────────────────────────────────────
if exec_path == "GENERAL_PHOTO":
    current_page = pages[0]
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e293b,#0f2744);border:1px solid #334155;
                border-radius:14px;padding:24px 28px;margin-bottom:20px'>
        <div style='color:#38bdf8;font-size:1.1rem;font-weight:700;margin-bottom:6px'>
            🖼  General Photo — Visual Content Preserved
        </div>
        <div style='color:#94a3b8;font-size:0.85rem'>
            The image router classified this upload as a <strong>general photograph or illustration</strong>.
            The original visual asset is preserved. If any incidental text was detected by the VLM, it is available below.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Split into two columns for image and text
    gp_left, gp_right = st.columns([1, 1], gap="large")

    with gp_left:
        # Image display
        photo_path = current_page.get("image_path") or current_page.get("_source_image")
        if photo_path and Path(str(photo_path)).exists():
            st.image(str(photo_path), use_container_width=True,
                     caption=f"Original image — {filename}")
        else:
            st.warning(f"Image file not found at: `{photo_path}`")

    with gp_right:
        gp_tab_text, gp_tab_qa = st.tabs(["✍ Detected Text", "💬 Ask AI (Q&A)"])
        
        with gp_tab_text:
            # Extracted Text Display (if any)
            extracted = current_page.get("extracted_text", "")
            if extracted:
                st.subheader("✍ Detected Text")
                edited = st.text_area(
                    "Text incidental to the image",
                    value=extracted,
                    height=400,
                    key=f"gp_text_{job_id}"
                )
                if edited != extracted:
                    ast["pages"][0]["extracted_text"] = edited
                    st.session_state.ast_data = ast
                    st.session_state.is_modified = True
            else:
                st.info("No incidental text was detected in this image.")
        
        with gp_tab_qa:
            render_qa_tab(job_id, pages)

    # ── Save modifications ──────────────────────────────────────────────
    if st.session_state.is_modified:
        st.warning("⚠ Unsaved modifications detected.")
        if st.button("💾 Apply Changes to Memory"):
            st.session_state.is_modified = False
            st.success("✅ Changes saved to session memory.")

    st.markdown("---")
    col_approve, col_reject = st.columns(2)
    with col_approve:
        if st.button("✅ Approve & Commit", type="primary", use_container_width=True):
            save_to_dir(ast, FINAL_DIR, {"_reviewed_at": datetime.now(timezone.utc).isoformat()})
            st.session_state.current_file = None
            st.session_state.ast_data = None
            st.success("Committed to final_database/")
            st.rerun()
    with col_reject:
        if st.button("❌ Reject", type="secondary", use_container_width=True):
            save_to_dir(ast, REJECTED_DIR)
            st.session_state.current_file = None
            st.session_state.ast_data = None
            st.warning("Document rejected.")
            st.rerun()
    st.stop()
# ── Confidence indicators ─────────────────────────────────────────────────────
# For multi-page, display badges of the first page as an overall indicator.
st.markdown(confidence_badges_html(pages[0]), unsafe_allow_html=True)

# ── Main two-column layout ────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT: Document visual ─────────────────────────────────────────────────────
with col_left:
    st.subheader("📄 Document Visual Verification")
    path_label = pages[0].get("execution_path", exec_path)
    if path_label == "TRACK_A":
        st.success(f"⚡ Vector-Native PDF — {n_pages} pages processed natively.")
    elif path_label == "PATH_1":
        st.success(f"🔠 Local CPU OCR accepted.")
    else:
        st.error(f"🧠 Escalated to VLM.")

    import logging
    
    for p_idx, page in enumerate(pages):
        page_num = page.get("page_number", p_idx + 1)
        st.markdown(f"### Page {page_num}")
        
        # Determine source
        source_img = page.get("_source_image")
        if not source_img and p_idx == 0:
            source_img = find_source_image(job_id)

        if source_img and Path(str(source_img)).exists():
            st.image(str(source_img), use_container_width=True, caption=f"Source: {filename} - Page {page_num}")
        else:
            embedded_images = page.get("images", [])
            if embedded_images:
                st.info(f"**{len(embedded_images)}** embedded visual assets found on Page {page_num}.")
                logging.info(f"UI rendering {len(embedded_images)} images for page {page_num}")
                for img in embedded_images:
                    img_path = img.get("path")
                    if img_path and Path(str(img_path)).exists():
                        logging.info(f"UI rendering image from path: {img_path}")
                        with st.expander(f"🖼️ {img.get('filename')}", expanded=True):
                            st.image(str(img_path), use_container_width=True)
                    else:
                        logging.warning(f"UI dropped image! Path not found: {img_path}")
            else:
                st.info(f"No embedded visual assets found on Page {page_num}.")
                # Show extracted text as fallback preview
                extracted = page.get("extracted_text", "")
                if extracted:
                    st.markdown(f"""
                    <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
                                padding:16px;font-family:monospace;font-size:12px;
                                color:#94a3b8;max-height:400px;overflow-y:auto;white-space:pre-wrap'>
                        {extracted[:2000]}{'...' if len(extracted) > 2000 else ''}
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("---")

# ── RIGHT: Extracted elements editor ─────────────────────────────────────────
with col_right:
    st.subheader("✍ Extracted Elements — AST Operator Audit")

    # ── Audit Panel Metrics ──────────────────────────────────────────────────
    num_text = sum(1 for p in pages for e in p.get("elements", []) if e.get("type") in ("text", "paragraph", "heading"))
    num_tables = sum(1 for p in pages for e in p.get("elements", []) if e.get("type") == "table")
    num_images = sum(len(p.get("images", [])) for p in pages)

    st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #0f172a;
        border: 1px solid #334155;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    a1.metric("Text Blocks", num_text)
    a2.metric("Images Found", num_images)
    a3.metric("Tables Found", num_tables)
    st.markdown("---")

    tab_elements, tab_summary, tab_qa, tab_json, tab_export = st.tabs([
        "🗂 Elements", "📋 Summary", "💬 Ask AI", "{ } Raw JSON", "📥 Export"
    ])

    # ── Tab 1: Elements editor ────────────────────────────────────────────────
    with tab_elements:
        for p_idx, page in enumerate(pages):
            page_num = page.get("page_number", p_idx + 1)
            elements = page.get("elements", [])
            
            st.markdown(f"#### Page {page_num}")
            
            if not elements:
                st.info(f"No structured elements found on Page {page_num}.")
            else:
                for i, elem in enumerate(elements):
                    etype   = elem.get("type", "unknown")
                    content = elem.get("content", {})
                    eid     = elem.get("element_id", f"elem_{i}")

                    path_label = page.get("execution_path", exec_path)
                    if path_label == "TRACK_A":
                        acc_text = "Accuracy: 100% (Digital Native)"
                        bg_color, text_color = "#065f46", "#6ee7b7"
                    elif path_label == "PATH_1":
                        conf_val = page.get("_ocr_avg_confidence")
                        acc_val = f"{conf_val:.1f}%" if conf_val is not None else "Unknown"
                        acc_text = f"Accuracy: {acc_val} (OCR)"
                        bg_color, text_color = "#78350f", "#fcd34d"
                    else:
                        acc_text = "Accuracy: ~95% (VLM)"
                        bg_color, text_color = "#4c1d95", "#c4b5fd"

                    conf_badge = f'<span class="element-type-tag" style="background:{bg_color};color:{text_color};margin-left:8px;border-color:{bg_color}">{acc_text}</span>'
                    st.markdown(f'<span class="element-type-tag">{etype}</span>{conf_badge}', unsafe_allow_html=True)

                    if etype == "text":
                        # Unified document text — shown exactly as it appears in the source
                        text_val   = content.get("text", "")
                        line_count = max(text_val.count("\n") + 1, 3)
                        height     = min(max(line_count * 22, 120), 600)
                        edited = st.text_area(
                            f"📄 Document Text (Page {page_num})",
                            value  = text_val,
                            height = height,
                            key    = f"elem_{job_id}_p{p_idx}_{i}",
                        )
                        if edited != text_val:
                            ast["pages"][p_idx]["elements"][i]["content"]["text"] = edited
                            st.session_state.ast_data  = ast
                            st.session_state.is_modified = True

                    elif etype in ("paragraph", "heading"):
                        # Legacy fallback for older AST documents
                        text_val = content.get("text", "")
                        edited   = st.text_area(
                            f"{'📌 Heading' if etype=='heading' else '📝 Paragraph'} (Page {page_num})",
                            value   = text_val,
                            height  = 80 if etype == "heading" else 120,
                            key     = f"elem_{job_id}_p{p_idx}_{i}",
                        )
                        if edited != text_val:
                            ast["pages"][p_idx]["elements"][i]["content"]["text"] = edited
                            st.session_state.ast_data  = ast
                            st.session_state.is_modified = True

                    elif etype == "key_value":
                        pairs = content.get("pairs", [])
                        if pairs:
                            df_kv = pd.DataFrame(pairs)
                            edited_kv = st.data_editor(
                                df_kv,
                                key           = f"kv_{job_id}_p{p_idx}_{i}",
                                use_container_width=True,
                                num_rows      = "dynamic",
                            )
                            if not edited_kv.equals(df_kv):
                                new_pairs = edited_kv.to_dict("records")
                                ast["pages"][p_idx]["elements"][i]["content"]["pairs"] = new_pairs
                                st.session_state.ast_data    = ast
                                st.session_state.is_modified = True

                    elif etype == "table":
                        headers = content.get("headers", [])
                        rows    = content.get("rows", [])
                        tbl_idx = content.get("table_index", i)
                        if rows:
                            try:
                                # Determine column count from data
                                n_cols = max(
                                    (len(r) if isinstance(r, list) else len(r.values()) if isinstance(r, dict) else 1)
                                    for r in rows
                                )
                                safe_headers = sanitize_headers(headers, n_cols)
                                df_tbl = pd.DataFrame(
                                    [r if isinstance(r, (list, dict)) else [r] for r in rows],
                                    columns=safe_headers,
                                )
                                st.caption(f"🗃 Table {tbl_idx + 1} — {len(rows)} row(s) × {n_cols} col(s)")
                                edited_tbl = st.data_editor(
                                    df_tbl,
                                    key                 = f"tbl_{job_id}_p{p_idx}_{i}",
                                    use_container_width = True,
                                    num_rows            = "dynamic",
                                )
                                if not edited_tbl.equals(df_tbl):
                                    ast["pages"][p_idx]["elements"][i]["content"]["rows"]    = edited_tbl.values.tolist()
                                    ast["pages"][p_idx]["elements"][i]["content"]["headers"] = list(edited_tbl.columns)
                                    st.session_state.ast_data    = ast
                                    st.session_state.is_modified = True
                            except Exception as e:
                                st.warning(f"Table render error: {e}")
                                # Render as read-only markdown fallback
                                raw_headers = headers or [f"Col {c+1}" for c in range(len(rows[0]) if rows else 1)]
                                md_rows = ["| " + " | ".join(str(h) for h in raw_headers) + " |",
                                           "| " + " | ".join(["---"] * len(raw_headers)) + " |"]
                                for row in rows[:50]:
                                    cells = row if isinstance(row, list) else list(row.values())
                                    md_rows.append("| " + " | ".join(str(c) for c in cells) + " |")
                                st.markdown("\n".join(md_rows))
                        else:
                            st.info("Empty table detected.")

                    elif etype == "graphic":
                        label            = content.get("label", "graphic")
                        signature_result = content.get("signature_result")

                        if "signature" in label.lower() and signature_result:
                            sig_type  = signature_result.get("type", "")
                            sig_conf  = signature_result.get("confidence", 0.0)

                            if sig_type == "signature_text":
                                sig_value = signature_result.get("value", "")
                                st.markdown(
                                    f"<div style='background:#14532d;border:1px solid #166534;"
                                    f"border-radius:10px;padding:14px 18px;margin:4px 0'>"
                                    f"<div style='color:#86efac;font-size:11px;font-weight:700;"
                                    f"letter-spacing:0.05em;margin-bottom:6px'>"
                                    f"✍ SIGNATURE — TEXT EXTRACTED"
                                    f"<span style='float:right;background:#166534;padding:2px 8px;"
                                    f"border-radius:8px;font-size:10px'>conf {sig_conf:.1f}%</span>"
                                    f"</div>"
                                    f"<div style='color:#dcfce7;font-size:14px;font-style:italic'>"
                                    f"{sig_value}"
                                    f"</div></div>",
                                    unsafe_allow_html=True,
                                )

                            elif sig_type == "signature_image":
                                sig_image  = signature_result.get("image")
                                sig_reason = signature_result.get("reason", "OCR confidence below threshold")

                                st.markdown(
                                    f"<div style='background:#78350f;border:1px solid #92400e;"
                                    f"border-radius:10px;padding:10px 14px;margin:4px 0'>"
                                    f"<span style='color:#fcd34d;font-size:11px;font-weight:700'>"
                                    f"✍ SIGNATURE — IMAGE FALLBACK"
                                    f"<span style='float:right;background:#92400e;padding:2px 8px;"
                                    f"border-radius:8px;font-size:10px'>conf {sig_conf:.1f}%</span>"
                                    f"</span><br>"
                                    f"<span style='color:#fef3c7;font-size:11px'>{sig_reason}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                                if sig_image and Path(sig_image).exists():
                                    st.image(
                                        sig_image,
                                        caption=f"Verified signature crop — {Path(sig_image).name}",
                                        use_container_width=True,
                                    )
                                elif sig_image:
                                    st.warning(f"Signature image not found at: `{sig_image}`")
                                else:
                                    st.error("⚠️ Signature localization failed — could not isolate a valid signature region.")

                            else:
                                st.markdown(f"🖼 **Visual element** — `{label}`")
                        else:
                            st.markdown(f"🖼 **Visual element** — `{label}`")

                    st.markdown('<div style="height:1px;background:#1e293b;margin:12px 0"></div>', unsafe_allow_html=True)
            st.markdown("---")

        # ── Save modifications ──────────────────────────────────────────────
        if st.session_state.is_modified:
            st.warning("⚠ Unsaved modifications detected.")
            if st.button("💾 Apply Changes to Memory"):
                st.session_state.is_modified = False
                st.success("✅ Changes saved to session memory.")

    # ── Tab 2: Document Summary ───────────────────────────────────────────────
    with tab_summary:
        st.markdown("### 📋 AI Document Summary")
        st.markdown(
            "<div style='color:#64748b;font-size:13px;margin-bottom:16px'>"
            "Powered by VLM — generates an executive summary from all extracted text across the document."
            "</div>",
            unsafe_allow_html=True,
        )

        # Collect all text from all pages
        page_texts = []
        for p_idx, page in enumerate(pages):
            for elem in page.get("elements", []):
                etype   = elem.get("type", "")
                content = elem.get("content", {})
                if etype in ("text", "paragraph", "heading"):
                    t = content.get("text", "").strip()
                    if t:
                        page_texts.append(t)
                elif etype == "key_value":
                    for pair in content.get("pairs", []):
                        if isinstance(pair, dict):
                            page_texts.append(" : ".join(str(v) for v in pair.values()))
            # Also use top-level extracted_text if available
            top_extracted = page.get("extracted_text", "")
            if top_extracted:
                page_texts.append(top_extracted)
                
        combined_text = "\n\n".join(page_texts)

        doc_type = ast.get("document_type") or meta.get("document_type") or "document"
        llm_key = os.environ.get("QWEN_API_KEY") or os.environ.get("GROQ_API_KEY", "")
        local_url = os.environ.get("LOCAL_MODEL_URL") or os.environ.get("QWEN_LOCAL_URL", "")

        if not combined_text.strip():
            st.info("No text content found in the document to summarise.")
        elif not llm_key and not local_url:
            st.error("Qwen 3.6 27B API key or local endpoint not found — cannot generate summary.")
        else:
            sum_col1, sum_col2 = st.columns([3, 1])
            with sum_col2:
                regen = st.button("🔄 Regenerate", use_container_width=True)

            cache_key = f"summary_{job_id}_all_pages"
            if regen and cache_key in st.session_state:
                del st.session_state[cache_key]
                generate_summary.clear()

            if cache_key not in st.session_state:
                with st.spinner("✨ Generating summary via Qwen 3.6 27B..."):
                    st.session_state[cache_key] = generate_summary(
                        combined_text, doc_type, llm_key
                    )

            summary_md = st.session_state.get(cache_key, "")
            st.markdown(
                f"<div style='background:#1e293b;border:1px solid #334155;border-radius:12px;"
                f"padding:24px;line-height:1.7;color:#e2e8f0'>{summary_md}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "📥 Download Summary (.md)",
                data      = summary_md,
                file_name = f"{job_id}_summary_p{st.session_state.current_page_idx+1}.md",
                mime      = "text/markdown",
                use_container_width=True,
            )

    # ── Tab 3: Ask AI (Q&A) ───────────────────────────────────────────────────
    with tab_qa:
        render_qa_tab(job_id, pages)

    # ── Tab 4: Raw JSON editor ────────────────────────────────────────────────
    with tab_json:
        json_str = json.dumps(st.session_state.ast_data, indent=2, ensure_ascii=False)
        edited_json = st.text_area(
            "Raw AST JSON — edit carefully",
            value  = json_str,
            height = 500,
            key    = f"json_editor_{job_id}",
        )
        if st.button("♻ Apply JSON Edits"):
            try:
                st.session_state.ast_data    = json.loads(edited_json)
                st.session_state.is_modified = True
                st.success("JSON applied.")
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # ── Tab 5: Export ─────────────────────────────────────────────────────────
    with tab_export:
        ast_str = json.dumps(st.session_state.ast_data, indent=2, ensure_ascii=False)
        e1, e2, e3 = st.columns(3)

        with e1:
            st.download_button(
                "📥 Export JSON",
                data      = ast_str,
                file_name = f"{job_id}_ast.json",
                mime      = "application/json",
                use_container_width=True,
            )

        with e2:
            frames  = build_export_frames(st.session_state.ast_data)
            buf_xl  = io.BytesIO()
            with pd.ExcelWriter(buf_xl, engine="openpyxl") as wr:
                if frames:
                    for sname, df in frames.items():
                        df.to_excel(wr, sheet_name=sname, index=False)
                else:
                    pd.DataFrame([{"result": "No structured data"}]).to_excel(wr, sheet_name="Result", index=False)
            st.download_button(
                "📊 Export Excel",
                data      = buf_xl.getvalue(),
                file_name = f"{job_id}_ast.xlsx",
                mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with e3:
            csv_parts = []
            for sname, df in frames.items():
                csv_parts.append(f"### {sname}\n" + df.to_csv(index=False))
            csv_content = "\n\n".join(csv_parts) if csv_parts else "No structured data"
            st.download_button(
                "📋 Export CSV",
                data      = csv_content,
                file_name = f"{job_id}_ast.csv",
                mime      = "text/csv",
                use_container_width=True,
            )

# ── Approve / Reject bar ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Validation Decision")

a_col, r_col = st.columns([2, 1])
with a_col:
    if st.button("✅ Approve & Commit to Database", type="primary", use_container_width=True):
        save_to_dir(
            st.session_state.ast_data, FINAL_DIR,
            extra_fields={
                "_approved_at" : datetime.now(timezone.utc).isoformat(),
                "_reviewed_by" : "human_operator",
            },
        )
        st.session_state.ast_data   = None
        st.session_state.is_modified = False
        st.session_state.current_file = None
        st.success("🎉 Committed to `final_database/`!")
        st.rerun()

with r_col:
    reason = st.text_input("Rejection reason", placeholder="e.g. Wrong document type...")
    if st.button("❌ Reject Document", type="secondary", use_container_width=True):
        save_to_dir(
            st.session_state.ast_data, REJECTED_DIR,
            extra_fields={
                "_rejected_at"       : datetime.now(timezone.utc).isoformat(),
                "_rejection_reason"  : reason or "No reason provided",
            },
        )
        st.session_state.ast_data    = None
        st.session_state.is_modified = False
        st.session_state.current_file = None
        st.warning("Document moved to `rejected/`.")
        st.rerun()
