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
GATEWAY_URL  = os.environ.get("GATEWAY_URL", "http://localhost:8000")

for d in [PENDING_DIR, FINAL_DIR, REJECTED_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "current_file"       not in st.session_state: st.session_state.current_file       = None
if "ast_data"           not in st.session_state: st.session_state.ast_data           = None
if "is_modified"        not in st.session_state: st.session_state.is_modified        = False
if "current_page_idx"   not in st.session_state: st.session_state.current_page_idx   = 0
if "upload_msg"         not in st.session_state: st.session_state.upload_msg         = ""

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
        "PATH_2" : '<span class="badge-path-2">🧠 PATH 2 — Groq VLM (OCR confidence escalation)</span>',
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="smart-header">
    <h1>🛡 Smart Triage Enterprise IDP — Human-In-The-Loop Hub</h1>
    <p>Strategic Audit Workspace &amp; Multi-path Extraction Validation Engine &nbsp;|&nbsp; §9 Developer Specification</p>
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
    st.info("👈 **Select a document from the sidebar to begin the review process.**")
    st.markdown("""
    #### How Smart Triage routes your documents:
    | Path | Trigger | Processing |
    |---|---|---|
    | ⚡ **Track A** | Digital PDF with ≥50 chars | Instant programmatic extraction — zero API cost |
    | 🔠 **Path 1** | Scanned/image, PCS < 0.50 | CPU OCR via pytesseract |
    | 🧠 **Path 2** | Complex/handwritten, PCS ≥ 0.50 | Groq Llama Vision LLM |
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
        st.success(f"🔠 Local CPU OCR accepted{conf_str} — no Groq API call made.")
    else:
        reason = current_page.get("_ocr_escalation_reason") or current_page.get("confidence_warning_reason") or ""
        st.error(f"🧠 Escalated to Groq Vision LLM — {reason}")

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

    tab_elements, tab_json, tab_export = st.tabs(["🗂 Elements", "{ } Raw JSON", "📥 Export"])

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

                st.markdown(f'<span class="element-type-tag">{etype}</span>', unsafe_allow_html=True)

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
                        if not edited_kv.equals(df_kv):
                            new_pairs = edited_kv.to_dict("records")
                            ast["pages"][st.session_state.current_page_idx]["elements"][i]["content"]["pairs"] = new_pairs
                            st.session_state.ast_data    = ast
                            st.session_state.is_modified = True

                elif etype == "table":
                    headers = content.get("headers", [])
                    rows    = content.get("rows", [])
                    if rows:
                        try:
                            df_tbl    = pd.DataFrame(rows, columns=headers or None)
                            edited_tbl = st.data_editor(
                                df_tbl,
                                key           = f"tbl_{job_id}_p{st.session_state.current_page_idx}_{i}",
                                use_container_width=True,
                                num_rows      = "dynamic",
                            )
                            if not edited_tbl.equals(df_tbl):
                                ast["pages"][st.session_state.current_page_idx]["elements"][i]["content"]["rows"] = edited_tbl.values.tolist()
                                st.session_state.ast_data    = ast
                                st.session_state.is_modified = True
                        except Exception as e:
                            st.warning(f"Table render error: {e}")
                            st.json(content)
                    else:
                        st.info("Empty table detected.")

                elif etype == "graphic":
                    st.markdown(f"🖼 **Visual element** — `{content.get('label','graphic')}`")

                st.markdown('<div style="height:1px;background:#1e293b;margin:12px 0"></div>', unsafe_allow_html=True)

        # ── Save modifications ──────────────────────────────────────────────
        if st.session_state.is_modified:
            st.warning("⚠ Unsaved modifications detected.")
            if st.button("💾 Apply Changes to Memory"):
                st.session_state.is_modified = False
                st.success("✅ Changes saved to session memory.")

    # ── Tab 2: Raw JSON editor ────────────────────────────────────────────────
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
