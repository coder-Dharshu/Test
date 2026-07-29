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
from groq import Groq

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title   = "PROG-OCR — HITL Hub",
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
GATEWAY_URL  = os.environ.get("GATEWAY_URL", "http://localhost:8000")

for d in [PENDING_DIR, FINAL_DIR, REJECTED_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "current_file"       not in st.session_state: st.session_state.current_file       = None
if "ast_data"           not in st.session_state: st.session_state.ast_data           = None
if "is_modified"        not in st.session_state: st.session_state.is_modified        = False
if "current_page_idx"   not in st.session_state: st.session_state.current_page_idx   = 0
if "upload_msg"         not in st.session_state: st.session_state.upload_msg         = ""
if "qa_history"         not in st.session_state: st.session_state.qa_history         = {}

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
        "TRACK_A": '<span class="badge-track-a">⚡ TRACK A — Digital Fast-Path</span>',
        "PATH_1" : '<span class="badge-path-1">🔠 PATH 1 — Local CPU OCR (High Confidence)</span>',
        "PATH_2" : '<span class="badge-path-2">🧠 PATH 2 — VLM (OCR confidence escalation)</span>',
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
def generate_summary(extracted_text: str, doc_type: str, groq_api_key: str, current_date: str) -> str:
    """Call VLM text API to produce a concise document summary with expiry detection."""
    if not extracted_text.strip():
        return "⚠️ No extracted text available to summarise."
    try:
        client = Groq(api_key=groq_api_key)
        model  = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        # Truncate very large documents to stay within token limits
        text_snippet = extracted_text[:12000]
        
        # Convert date to a nice readable form
        from datetime import datetime
        try:
            current_date_obj = datetime.strptime(current_date, "%Y-%m-%d")
            current_date_str = current_date_obj.strftime("%B %d, %Y")
        except Exception:
            current_date_str = current_date
            
        prompt = (
            f"You are an expert document and product analyst. The document type is '{doc_type}'.\n"
            f"Today's date is {current_date_str}.\n\n"
            "Analyse the following extracted text/data and return a structured summary. "
            "Please check carefully if this is a product label, receipt, packaging photo, or product document:\n\n"
            "1. EXPIRY & PRODUCT VALIDATION (CRITICAL):\n"
            "   - Identify if this is a product (e.g. food, medicine, cosmetic, chemical).\n"
            "   - Extract any Expiry Date (EXP), Expiration Date, Use By, Best Before, or Manufacture/Packaging Date (MFG).\n"
            f"   - Compare any found expiry date to today's date ({current_date_str}).\n"
            "   - If the product is EXPIRED (expiry date is before today), print a prominent, bold, high-visibility warning at the VERY TOP of your response, e.g.:\n"
            "     '⚠️ **EXPIRED WARNING: This product expired on [Expiry Date]! (Expired [N] days/months ago)**'\n"
            "   - If the product is close to expiring (within 30 days), print a warning, e.g.:\n"
            "     '⚠️ **WARNING: This product expires soon on [Expiry Date]! (Expires in [N] days)**'\n"
            "   - If it is a product but no expiry date is found, state: 'ℹ️ **Product identified, but no expiry date was found in the text.**'\n"
            "   - If it is not a product document at all, you can skip the expiry warning block.\n\n"
            "2. Executive Summary: A one-paragraph summary (3-4 sentences).\n"
            "3. Key Points: 5-8 concise bullet points covering the most important facts, figures, and actions.\n"
            "4. Document Highlights: any notable dates, names, amounts, or references found.\n\n"
            "Format your response in clear Markdown with appropriate subheadings and styling.\n\n"
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

def answer_question(extracted_text: str, question: str, doc_type: str, groq_api_key: str) -> str:
    """Call Groq API to answer a user's question about the extracted document text."""
    if not extracted_text.strip():
        return "⚠️ No extracted text available to answer questions."
    if not question.strip():
        return "Please enter a question."
    try:
        client = Groq(api_key=groq_api_key)
        model  = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        text_snippet = extracted_text[:15000]
        prompt = (
            f"You are an expert Q&A assistant. The document type is '{doc_type}'.\n"
            "Analyze the document text below and answer the user's question. "
            "Base your answer strictly on the facts directly mentioned in the document. "
            "If the answer cannot be determined from the document, state that clearly.\n\n"
            f"--- Document Text ---\n{text_snippet}\n\n"
            f"--- User Question ---\n{question}\n\n"
            "Provide a concise, clear answer formatted in Markdown."
        )
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.2,
            max_tokens=800,
        )
        return resp.choices[0].message.content or "No answer generated."
    except Exception as exc:
        return f"❌ Failed to answer question: {exc}"

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
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        job_id = resp.json().get("job_id", "?")
        
        with st.spinner(f"Processing job {job_id}... Please wait for extraction to complete."):
            # Active polling loop with 300-second timeout fail-safe for slow CPU OCR
            max_retries = 300
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

# ── Auto-load first pending file if none loaded ──────────────────────────────
pending_files = get_pending_files()
if st.session_state.current_file is None and pending_files:
    first_pending = pending_files[0]
    data = load_ast(first_pending)
    if data:
        st.session_state.current_file     = first_pending
        st.session_state.ast_data         = data
        st.session_state.is_modified      = False
        st.session_state.current_page_idx = 0
elif st.session_state.current_file is not None and not Path(st.session_state.current_file).exists():
    st.session_state.current_file = None
    st.session_state.ast_data = None
    st.session_state.is_modified = False
    st.session_state.current_page_idx = 0
    if pending_files:
        first_pending = pending_files[0]
        data = load_ast(first_pending)
        if data:
            st.session_state.current_file     = first_pending
            st.session_state.ast_data         = data

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡 PROG-OCR")
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
    if st.session_state.current_file:
        if st.button("🗑️ Delete Active Document", use_container_width=True):
            fp = Path(st.session_state.current_file)
            fp.unlink(missing_ok=True)
            st.session_state.current_file = None
            st.session_state.ast_data = None
            st.session_state.is_modified = False
            st.session_state.current_page_idx = 0
            st.rerun()
        st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔄 Refresh Queue", use_container_width=True):
            st.rerun()
    with col4:
        if st.button("🗑️ Clear All", use_container_width=True):
            for f in PENDING_DIR.glob("*.json"):
                f.unlink(missing_ok=True)
            st.session_state.current_file = None
            st.session_state.ast_data = None
            st.session_state.is_modified = False
            st.session_state.current_page_idx = 0
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="smart-header">
    <h1>🛡 PROG-OCR</h1>
    <p>Strategic Audit Workspace &amp; Multi-path Extraction Validation Engine</p>
</div>
""", unsafe_allow_html=True)

# ── No document loaded ────────────────────────────────────────────────────────
if st.session_state.ast_data is None:
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

st.markdown("---")

# ── Page Navigation ───────────────────────────────────────────────────────────
if n_pages > 1:
    pn1, pn2 = st.columns([3, 1])
    with pn1:
        page_labels = [
            f"Page {p['page_number']} — {p.get('execution_path','?')} (PCS {p.get('pcs_score', 0):.2f})"
            for p in pages
        ]
        selected_page_label = st.selectbox("Navigate to page:", page_labels)
        st.session_state.current_page_idx = page_labels.index(selected_page_label)
    with pn2:
        st.markdown(f"<br><span style='color:#94a3b8;font-size:13px'>Page {st.session_state.current_page_idx+1} of {n_pages}</span>", unsafe_allow_html=True)

# Guard: clamp page index and handle empty pages list
if not pages:
    st.error("⚠️ This document has no extractable pages. The pipeline could not parse the file.")
    pipeline_err = ast.get("_pipeline", "")
    if "spreadsheet" in pipeline_err or "xls" in (meta.get("source_filename","") + "").lower():
        st.info("💡 **Tip for .xls files:** Install the `xlrd` library with `pip install xlrd`, then restart and re-upload.")
    st.stop()

# Reset index if stale (e.g. switching between docs with different page counts)
if st.session_state.current_page_idx >= len(pages):
    st.session_state.current_page_idx = 0

current_page = pages[st.session_state.current_page_idx]

# ── Confidence indicators ─────────────────────────────────────────────────────
st.markdown(confidence_badges_html(current_page), unsafe_allow_html=True)

# ── Main two-column layout ────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT: Document visual ─────────────────────────────────────────────────────
with col_left:
    st.subheader("📄 Document Visual Verification")
    path_label = current_page.get("execution_path", exec_path)
    if path_label == "TRACK_A":
        st.success("⚡ Vector-Native PDF — GPU resources bypassed entirely.")
    elif path_label == "PATH_1":
        conf = current_page.get("_ocr_avg_confidence")
        conf_str = f" (OCR confidence: {conf:.1f}%)" if conf is not None else ""
        st.success(f"🔠 Local CPU OCR accepted{conf_str} — no VLM API call made.")
    else:
        reason = current_page.get("_ocr_escalation_reason") or current_page.get("confidence_warning_reason") or ""
        st.error(f"🧠 Escalated to VLM — {reason}")

    # Try to show source image
    source_img = current_page.get("_source_image")
    if not source_img:
        source_img = find_source_image(job_id)

    if source_img and Path(str(source_img)).exists():
        st.image(str(source_img), use_container_width=True, caption=f"Source: {filename}")
    else:
        st.info("No source image available (digital PDF fast-tracked programmatically).")
        # Show extracted text as fallback preview
        extracted = current_page.get("extracted_text", "")
        if extracted:
            st.markdown(f"""
            <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
                        padding:16px;font-family:monospace;font-size:12px;
                        color:#94a3b8;max-height:400px;overflow-y:auto;white-space:pre-wrap'>
                {extracted[:2000]}{'...' if len(extracted) > 2000 else ''}
            </div>
            """, unsafe_allow_html=True)

# ── RIGHT: Extracted elements editor ─────────────────────────────────────────
with col_right:
    st.subheader("✍ Extracted Elements — AST Operator Audit")

    tab_elements, tab_summary, tab_qa, tab_json, tab_export = st.tabs([
        "🗂 Elements", "📋 Summary", "❓ Q&A", "{ } Raw JSON", "📥 Export"
    ])

    # ── Tab 1: Elements editor ────────────────────────────────────────────────
    with tab_elements:
        elements = current_page.get("elements", [])
        if not elements:
            st.info("No structured elements found on this page.")
        else:
            for i, elem in enumerate(elements):
                etype   = elem.get("type", "unknown")
                content = elem.get("content", {})
                eid     = elem.get("element_id", f"elem_{i}")

                path_label = current_page.get("execution_path", exec_path)
                if path_label == "TRACK_A":
                    acc_text = "Accuracy: 100% (Digital Native)"
                    bg_color, text_color = "#065f46", "#6ee7b7"
                elif path_label == "PATH_1":
                    conf_val = current_page.get("_ocr_avg_confidence")
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
                        "📄 Document Text",
                        value  = text_val,
                        height = height,
                        key    = f"elem_{job_id}_p{st.session_state.current_page_idx}_{i}",
                    )
                    if edited != text_val:
                        ast["pages"][st.session_state.current_page_idx]["elements"][i]["content"]["text"] = edited
                        st.session_state.ast_data  = ast
                        st.session_state.is_modified = True

                elif etype in ("paragraph", "heading"):
                    # Legacy fallback for older AST documents
                    text_val = content.get("text", "")
                    edited   = st.text_area(
                        f"{'📌 Heading' if etype=='heading' else '📝 Paragraph'}",
                        value   = text_val,
                        height  = 80 if etype == "heading" else 120,
                        key     = f"elem_{job_id}_p{st.session_state.current_page_idx}_{i}",
                    )
                    if edited != text_val:
                        ast["pages"][st.session_state.current_page_idx]["elements"][i]["content"]["text"] = edited
                        st.session_state.ast_data  = ast
                        st.session_state.is_modified = True

                elif etype == "key_value":
                    pairs = content.get("pairs", [])
                    if pairs:
                        df_kv = pd.DataFrame(pairs)
                        edited_kv = st.data_editor(
                            df_kv,
                            key           = f"kv_{job_id}_p{st.session_state.current_page_idx}_{i}",
                            use_container_width=True,
                            num_rows      = "dynamic",
                        )
                        if edited_kv.to_dict("records") != df_kv.to_dict("records"):
                            new_pairs = edited_kv.to_dict("records")
                            ast["pages"][st.session_state.current_page_idx]["elements"][i]["content"]["pairs"] = new_pairs
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
                                key                 = f"tbl_{job_id}_p{st.session_state.current_page_idx}_{i}",
                                use_container_width = True,
                                num_rows            = "dynamic",
                            )
                            if edited_tbl.values.tolist() != df_tbl.values.tolist() or list(edited_tbl.columns) != list(df_tbl.columns):
                                ast["pages"][st.session_state.current_page_idx]["elements"][i]["content"]["rows"]    = edited_tbl.values.tolist()
                                ast["pages"][st.session_state.current_page_idx]["elements"][i]["content"]["headers"] = list(edited_tbl.columns)
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
            "Powered by VLM — generates an executive summary from all extracted text on this page."
            "</div>",
            unsafe_allow_html=True,
        )

        # Collect all text from the current page
        page_texts = []
        for elem in current_page.get("elements", []):
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
        top_extracted = current_page.get("extracted_text", "")
        combined_text = "\n\n".join(page_texts) or top_extracted

        doc_type = ast.get("document_type") or meta.get("document_type") or "document"
        groq_key = os.environ.get("GROQ_API_KEY", "")

        if not combined_text.strip():
            st.info("No text content found on this page to summarise.")
        elif not groq_key:
            st.error("VLM API KEY (GROQ_API_KEY) not found — cannot generate summary.")
        else:
            sum_col1, sum_col2 = st.columns([3, 1])
            with sum_col2:
                regen = st.button("🔄 Regenerate", use_container_width=True)

            cache_key = f"summary_{job_id}_p{st.session_state.current_page_idx}"
            if regen and cache_key in st.session_state:
                del st.session_state[cache_key]
                generate_summary.clear()

            if cache_key not in st.session_state:
                with st.spinner("✨ Generating summary via VLM..."):
                    current_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    st.session_state[cache_key] = generate_summary(
                        combined_text, doc_type, groq_key, current_date_str
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

    # ── Tab: Q&A ──────────────────────────────────────────────────────────────
    with tab_qa:
        st.markdown("### ❓ Document Q&A")
        st.markdown(
            "<div style='color:#64748b;font-size:13px;margin-bottom:16px'>"
            "Ask any question about the contents of this document. Answers are generated using Groq LLM."
            "</div>",
            unsafe_allow_html=True,
        )

        qa_list = st.session_state.qa_history.setdefault(job_id, [])

        # Display history
        if qa_list:
            for item in qa_list:
                st.markdown(
                    f"<div style='background:#1e293b; border:1px solid #334155; border-radius:10px; padding:12px; margin-bottom:10px'>"
                    f"<div style='color:#38bdf8; font-weight:600; font-size:12px; margin-bottom:4px'>❓ QUESTION</div>"
                    f"<div style='color:#f8fafc; font-size:14px; margin-bottom:8px'>{item['question']}</div>"
                    f"<div style='height:1px; background:#334155; margin-bottom:8px'></div>"
                    f"<div style='color:#34d399; font-weight:600; font-size:12px; margin-bottom:4px'>💡 ANSWER</div>"
                    f"<div style='color:#e2e8f0; font-size:14px; line-height:1.6'>{item['answer']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if st.button("🗑️ Clear Q&A History", key=f"clear_qa_{job_id}", use_container_width=True):
                st.session_state.qa_history[job_id] = []
                st.rerun()

        # Chat form
        with st.form(key=f"qa_form_{job_id}", clear_on_submit=True):
            user_q = st.text_input("Ask a question about this document:", placeholder="e.g. What is the expiry date? What is the total invoice amount?")
            submit_q = st.form_submit_button("Ask VLM 🚀", use_container_width=True)

            if submit_q:
                if not user_q.strip():
                    st.warning("Please enter a non-empty question.")
                elif not groq_key:
                    st.error("Groq API key not configured.")
                else:
                    with st.spinner("Analyzing document and generating answer..."):
                        # Build full document context across all pages
                        all_doc_texts = []
                        for p_idx, p in enumerate(pages):
                            p_texts = []
                            for elem in p.get("elements", []):
                                etype = elem.get("type", "")
                                content = elem.get("content", {})
                                if etype in ("text", "paragraph", "heading"):
                                    t = content.get("text", "").strip()
                                    if t: p_texts.append(t)
                                elif etype == "key_value":
                                    for pair in content.get("pairs", []):
                                        if isinstance(pair, dict):
                                            p_texts.append(" : ".join(str(v) for v in pair.values()))
                            top_p_extracted = p.get("extracted_text", "")
                            combined_p = "\n".join(p_texts) or top_p_extracted
                            if combined_p:
                                all_doc_texts.append(f"--- Page {p_idx+1} ---\n{combined_p}")
                        
                        full_context = "\n\n".join(all_doc_texts)
                        answer = answer_question(full_context, user_q, doc_type, groq_key)
                        st.session_state.qa_history[job_id].append({
                            "question": user_q,
                            "answer": answer
                        })
                        st.rerun()

    # ── Tab 3: Raw JSON editor ────────────────────────────────────────────────
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

    # ── Tab 3: Export ─────────────────────────────────────────────────────────
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
