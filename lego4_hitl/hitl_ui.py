"""
Greencare AI — Lego 4: Human-in-the-Loop Validation Dashboard  [UPGRADED v2]
=============================================================================
New in v2:
  • 📊 Statistics tab with real-time pipeline KPIs
  • 🔎 Queue search / filter by filename keyword
  • ⌨️  Keyboard shortcuts: A = Approve, R = Reject, N = Next doc
  • 🗂️  PDF-to-image preview (renders first page as image for PDF jobs)
  • 🌙 Dark-mode styling via Gradio Soft theme (was already present, improved)
  • Display processing timestamps (_started_at, _completed_at)

Runs on: Port 7860
"""

import json
import os
import glob
import io
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

# Optional: PDF → image preview
try:
    from PIL import Image as PILImage
    import pypdfium2 as pdfium
    PDF_PREVIEW_AVAILABLE = True
except ImportError:
    PDF_PREVIEW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
PENDING_DIR  = "./pending_review"
FINAL_DIR    = "./final_database"
REJECTED_DIR = "./rejected"
UPLOAD_DIR   = "./temp_uploads"
LEGO2_TEMP   = "./lego2_temp"
ASSETS_DIR   = "./extracted_assets"
EXPORT_DIR   = "./exports"
GATEWAY_URL  = os.environ.get("GATEWAY_URL", "http://localhost:8000")

for d in [PENDING_DIR, FINAL_DIR, REJECTED_DIR, ASSETS_DIR, EXPORT_DIR, LEGO2_TEMP]:  # BUG-9 fix: added LEGO2_TEMP
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def get_pending_files() -> list[str]:
    return sorted(glob.glob(os.path.join(PENDING_DIR, "*.json")))


def get_pending_names(filter_text: str = "") -> list[str]:
    filter_text = filter_text or ""       # BUG-J fix: Gradio may pass None for empty Textbox
    names = [os.path.basename(f) for f in get_pending_files()]
    if filter_text.strip():
        ft = filter_text.strip().lower()
        names = [n for n in names if ft in n.lower()]
    return names


def format_stats() -> str:
    p = len(get_pending_files())
    c = len(glob.glob(os.path.join(FINAL_DIR,    "*.json")))
    r = len(glob.glob(os.path.join(REJECTED_DIR, "*.json")))
    total = c + r
    rate  = f"{round(c / total * 100, 1)}%" if total else "—"
    return (
        f"**Queue:** {p} pending  |  **Committed:** {c}  |  "
        f"**Rejected:** {r}  |  **Approval Rate:** {rate}"
    )


def find_source_image(job_id: str) -> str | None:
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    for d in [UPLOAD_DIR, LEGO2_TEMP, ASSETS_DIR]:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.startswith(job_id) and Path(fname).suffix.lower() in exts:
                return os.path.join(d, fname)
    return None


def find_source_pdf(job_id: str) -> str | None:
    for d in [UPLOAD_DIR, LEGO2_TEMP]:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.startswith(job_id) and fname.lower().endswith(".pdf"):
                return os.path.join(d, fname)
    return None


def pdf_to_image(pdf_path: str):
    """Render first page of a PDF as a PIL image for Gradio display."""
    if not PDF_PREVIEW_AVAILABLE:
        return None
    try:
        doc  = pdfium.PdfDocument(pdf_path)
        page = doc[0]
        bmp  = page.render(scale=1.5)
        img  = bmp.to_pil()
        return img
    except Exception as exc:
        logger.warning("PDF preview failed: %s", exc)
        return None


def build_confidence_html(page: dict) -> str:
    warn   = page.get("confidence_warning", False)
    reason = page.get("confidence_warning_reason") or ""
    hw     = page.get("handwriting_detected", False)
    curved = page.get("curved_text_detected", False)

    parts = []
    color = "#ef4444" if warn else "#22c55e"
    label = "LOW CONFIDENCE" if warn else "HIGH CONFIDENCE"
    parts.append(
        f'<span style="background:{color};color:#fff;padding:3px 10px;'
        f'border-radius:20px;font-size:12px;font-weight:700">{label}</span>'
    )
    if hw:
        parts.append(
            '<span style="background:#f59e0b;color:#fff;padding:3px 10px;'
            'border-radius:20px;font-size:12px">Handwriting Detected</span>'
        )
    if curved:
        parts.append(
            '<span style="background:#8b5cf6;color:#fff;padding:3px 10px;'
            'border-radius:20px;font-size:12px">Curved Text</span>'
        )

    html = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0">' + "".join(parts) + "</div>"
    if warn and reason:
        html += (
            f'<div style="background:#fef2f2;border:1px solid #ef4444;border-radius:8px;'
            f'padding:8px;font-size:12px;color:#991b1b;margin-top:4px">{reason}</div>'
        )
    return html


def build_first_table_df(data: dict) -> pd.DataFrame:
    pages = data.get("pages", [data])
    for page in pages:
        for tbl in page.get("tables", []):
            rows    = tbl.get("rows", [])
            headers = tbl.get("headers", [])
            if rows:
                try:
                    return pd.DataFrame(rows, columns=headers or None)
                except Exception:
                    pass
    return pd.DataFrame([{"info": "No tables detected in this document"}])


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def _export_path(job_id: str, ext: str) -> str:
    return os.path.join(EXPORT_DIR, f"{job_id}.{ext}")


def build_export_frames(data: dict) -> dict[str, pd.DataFrame]:
    pages = data.get("pages", [data])
    merged: dict = {}
    for p in pages:
        merged.update(p)

    frames: dict[str, pd.DataFrame] = {}
    kvp = merged.get("key_value_pairs", {})
    if kvp:
        frames["Key_Value_Pairs"] = pd.DataFrame([{"Field": k, "Value": v} for k, v in kvp.items()])
    text = merged.get("extracted_text", "")
    if text:
        frames["Extracted_Text"] = pd.DataFrame([{"Text": text}])
    for tbl in merged.get("tables", []):
        headers = tbl.get("headers", [])
        rows    = tbl.get("rows",    [])
        if rows:
            idx = tbl.get("table_index", len(frames))
            frames[f"Table_{idx + 1}"] = pd.DataFrame(rows, columns=headers or None)
    return frames


def do_export_json(file_path: str, edited_json: str) -> str | None:
    if not file_path:
        return None
    try:
        data = json.loads(edited_json)
    except Exception:
        return None
    job_id = Path(file_path).stem
    out    = _export_path(job_id, "json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return out


def do_export_excel(file_path: str, edited_json: str) -> str | None:
    if not file_path:
        return None
    try:
        data = json.loads(edited_json)
    except Exception:
        return None
    frames = build_export_frames(data)
    job_id = Path(file_path).stem
    out    = _export_path(job_id, "xlsx")
    buf    = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if not frames:
            pd.DataFrame([{"result": "No structured data"}]).to_excel(writer, sheet_name="Result", index=False)
        for name, df in frames.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    with open(out, "wb") as f:
        f.write(buf.getvalue())
    return out


def do_export_csv(file_path: str, edited_json: str) -> str | None:
    if not file_path:
        return None
    try:
        data = json.loads(edited_json)
    except Exception:
        return None
    frames   = build_export_frames(data)
    job_id   = Path(file_path).stem
    out      = _export_path(job_id, "csv")
    sections = []
    for name, df in frames.items():
        sections.append(f"### {name}\n" + df.to_csv(index=False))
    content = "\n\n".join(sections) if sections else "No structured data"
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    return out


# ---------------------------------------------------------------------------
# Core pipeline actions
# ---------------------------------------------------------------------------
def load_document(file_name: str):
    """
    Load a document from pending_review by filename.
    Returns: status_md, json_str, image, conf_html,
             file_path, table_df, text_md, stats, timing_md
    """
    EMPTY = (
        "Select a document from the queue above.",
        "{}",
        None,
        "",
        None,
        pd.DataFrame([{"info": "No document loaded"}]),
        "",
        format_stats(),
        "",
    )

    if not file_name:
        return EMPTY

    file_path = os.path.join(PENDING_DIR, file_name)
    if not os.path.exists(file_path):
        return EMPTY

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to load %s: %s", file_path, exc)
        return EMPTY

    job_id   = data.get("_job_id",   Path(file_path).stem)
    pipeline = data.get("_pipeline", "unknown")
    fname    = data.get("_filename", file_name)
    pages    = data.get("pages",     [data])
    first    = pages[0] if pages else {}
    doc_type = first.get("document_type", "unknown")

    status_md = (
        f"### {fname}\n"
        f"**Job ID:** `{job_id}`   **Pipeline:** `{pipeline}`   "
        f"**Type:** `{doc_type}`   **Pages:** {len(pages)}"
    )

    # Timing info
    started_at   = data.get("_started_at")   or first.get("_started_at")
    completed_at = data.get("_completed_at") or first.get("_completed_at")
    timing_md = ""
    if started_at or completed_at:
        timing_parts = []
        if started_at:
            timing_parts.append(f"**Started:** `{started_at[:19].replace('T', ' ')} UTC`")
        if completed_at:
            timing_parts.append(f"**Completed:** `{completed_at[:19].replace('T', ' ')} UTC`")
        if started_at and completed_at:
            try:
                t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                elapsed = round((t1 - t0).total_seconds(), 2)
                timing_parts.append(f"**Processing time:** `{elapsed}s`")
            except Exception:
                pass
        timing_md = "  |  ".join(timing_parts)

    conf_html  = build_confidence_html(first)
    json_str   = json.dumps(data, indent=4, ensure_ascii=False)

    # Image preview: try image first, then PDF render
    image_path = find_source_image(job_id)
    if image_path is None and PDF_PREVIEW_AVAILABLE:
        pdf_path = find_source_pdf(job_id)
        if pdf_path:
            image_path = pdf_to_image(pdf_path)

    table_df  = build_first_table_df(data)

    raw_text  = (first.get("extracted_text", "") or "")[:1200]
    md_tables = first.get("markdown_tables", "") or ""
    text_md   = f"**Extracted Text:**\n```\n{raw_text}\n```\n\n{md_tables}"

    return (status_md, json_str, image_path, conf_html,
            file_path, table_df, text_md, format_stats(), timing_md)


def refresh_queue(filter_text: str = ""):
    """Refresh + auto-load first queued document. Returns all display outputs."""
    names      = get_pending_names(filter_text)
    dd_update  = gr.update(choices=names, value=names[0] if names else None)
    if not names:
        return (
            dd_update, format_stats(),
            "No documents in queue. Upload one above.",
            "{}", None, "",
            None,
            pd.DataFrame([{"info": "Queue is empty"}]),
            "", "",
        )
    status, js, img, conf, fpath, tbl, txt, stats, timing = load_document(names[0])
    return (dd_update, stats, status, js, img, conf, fpath, tbl, txt, timing)


def approve_document(file_path: str, edited_json: str):
    if not file_path or not os.path.exists(file_path):
        return "No document loaded — nothing to approve.", format_stats()
    try:
        final = json.loads(edited_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON — fix the error before approving:\n`{e}`", format_stats()

    final["_approved_at"]  = datetime.now(timezone.utc).isoformat() + "Z"  # BUG-5 fix
    final["_reviewed_by"]  = "human_reviewer"

    dest = os.path.join(FINAL_DIR, os.path.basename(file_path))
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    os.remove(file_path)
    logger.info("Approved: %s", dest)
    return "✅ Committed! Record saved to final_database/.", format_stats()


def reject_document(file_path: str, reason: str):
    if not file_path or not os.path.exists(file_path):
        return "No document loaded.", format_stats()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    data["_rejected_at"]      = datetime.now(timezone.utc).isoformat() + "Z"  # BUG-5 fix
    data["_rejection_reason"] = reason or "No reason"

    dest = os.path.join(REJECTED_DIR, os.path.basename(file_path))
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.remove(file_path)
    logger.info("Rejected: %s", dest)
    return "🗑 Rejected. Moved to rejected/.", format_stats()


def upload_and_process(file_obj, filter_text: str = ""):
    """Upload file to gateway, wait for processing, then refresh queue."""
    if file_obj is None:
        return ("No file selected.",) + _empty_refresh(filter_text)

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    filename  = os.path.basename(file_path)

    try:
        with open(file_path, "rb") as fh:
            resp = requests.post(
                f"{GATEWAY_URL}/api/v1/ingest",
                files={"file": (filename, fh)},
                timeout=30,
            )
        resp.raise_for_status()
        job_id = resp.json().get("job_id", "?")
        upload_msg = f"✅ Submitted! Job ID: `{job_id}` — waiting for processing..."
    except requests.ConnectionError:
        return (
            "❌ Cannot reach the API Gateway. Make sure run_all.py is running.",
        ) + _empty_refresh(filter_text)
    except Exception as exc:
        return (f"❌ Upload failed: {exc}",) + _empty_refresh(filter_text)

    # Poll up to 30 seconds
    expected = os.path.join(PENDING_DIR, f"{job_id}.json")
    for _ in range(30):
        if os.path.exists(expected):
            break
        time.sleep(1)

    result = refresh_queue(filter_text)
    return (upload_msg,) + result


def _empty_refresh(filter_text: str = ""):
    names = get_pending_names(filter_text)
    dd    = gr.update(choices=names, value=None)
    return (
        dd, format_stats(),
        "Select a document from the queue.", "{}", None, "",
        None,
        pd.DataFrame([{"info": "No document loaded"}]),
        "", "",
    )


# ---------------------------------------------------------------------------
# Statistics tab helpers
# ---------------------------------------------------------------------------
def get_stats_html() -> str:
    pending_files  = glob.glob(os.path.join(PENDING_DIR,  "*.json"))
    final_files    = glob.glob(os.path.join(FINAL_DIR,    "*.json"))
    rejected_files = glob.glob(os.path.join(REJECTED_DIR, "*.json"))
    n_p  = len(pending_files)
    n_c  = len(final_files)
    n_r  = len(rejected_files)
    n_t  = n_c + n_r
    rate = f"{round(n_c / n_t * 100, 1)}%" if n_t else "—"

    doc_types: dict = {}
    pipelines: dict = {}
    for fpath in final_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            pages = d.get("pages", [d])
            dtype = (pages[0] if pages else {}).get("document_type", "unknown")
            pipe  = d.get("_pipeline", "unknown")
            doc_types[dtype] = doc_types.get(dtype, 0) + 1
            pipelines[pipe]  = pipelines.get(pipe,  0) + 1
        except Exception:
            pass

    def kpi(label, value, color="#6366f1"):
        return (
            f'<div style="background:#1a1d2e;border:1px solid rgba(99,102,241,0.2);'
            f'border-radius:12px;padding:16px 20px;flex:1;min-width:140px;">'
            f'<div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em">{label}</div>'
            f'<div style="font-size:2rem;font-weight:700;color:{color};margin-top:6px">{value}</div>'
            f'</div>'
        )

    def table_rows(d: dict):
        return "".join(
            f'<tr><td style="padding:6px 12px;color:#94a3b8">{k}</td>'
            f'<td style="padding:6px 12px;font-weight:600">{v}</td></tr>'
            for k, v in d.items()
        )

    html = f"""
    <div style="font-family:Inter,sans-serif;color:#e2e8f0">
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px">
        {kpi("Pending",   n_p,  "#f59e0b")}
        {kpi("Committed", n_c,  "#10b981")}
        {kpi("Rejected",  n_r,  "#ef4444")}
        {kpi("Approval",  rate, "#8b5cf6")}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div>
          <h4 style="margin-bottom:8px;color:#94a3b8;font-size:12px;text-transform:uppercase">Pipeline Breakdown</h4>
          <table style="width:100%;border-collapse:collapse">
            {table_rows(pipelines) or '<tr><td style="color:#64748b">No data yet</td></tr>'}
          </table>
        </div>
        <div>
          <h4 style="margin-bottom:8px;color:#94a3b8;font-size:12px;text-transform:uppercase">Document Types</h4>
          <table style="width:100%;border-collapse:collapse">
            {table_rows(doc_types) or '<tr><td style="color:#64748b">No data yet</td></tr>'}
          </table>
        </div>
      </div>
      <p style="color:#475569;font-size:11px;margin-top:16px">Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>"""
    return html


# ---------------------------------------------------------------------------
# Keyboard shortcuts JS (injected via gr.HTML)
# ---------------------------------------------------------------------------
KEYBOARD_SHORTCUTS_JS = """
<script>
document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === 'a' || e.key === 'A') {
    const btn = document.querySelector('#approve-btn');
    if (btn) btn.click();
  } else if (e.key === 'r' || e.key === 'R') {
    const btn = document.querySelector('#reject-btn');
    if (btn) btn.click();
  } else if (e.key === 'n' || e.key === 'N') {
    const btn = document.querySelector('#refresh-btn');
    if (btn) btn.click();
  }
});
</script>
<div style="font-size:11px;color:#64748b;padding:4px 0">
  ⌨️ Keyboard shortcuts: <kbd style="background:#1e293b;padding:1px 6px;border-radius:4px">A</kbd> Approve &nbsp;
  <kbd style="background:#1e293b;padding:1px 6px;border-radius:4px">R</kbd> Reject &nbsp;
  <kbd style="background:#1e293b;padding:1px 6px;border-radius:4px">N</kbd> Next/Refresh
</div>
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Greencare AI HITL Dashboard",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
) as dashboard:

    current_file = gr.State(value=None)

    # Header
    gr.Markdown(
        "# 🌿 Greencare AI — Human Review Dashboard  `v2.0`\n"
        "*Blueprint §3.4 — Review, correct, and approve AI extractions.*"
    )
    stats_bar = gr.Markdown(value=format_stats())

    # Keyboard shortcuts injected
    gr.HTML(value=KEYBOARD_SHORTCUTS_JS)

    # ── Tab layout ────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Review ────────────────────────────────────────────────────
        with gr.Tab("📋 Review Queue"):

            # Step 1: Upload
            with gr.Accordion("Step 1 — Upload Document", open=True):
                gr.Markdown(
                    "Select a file (PDF, JPG, PNG, TIFF, BMP, DOCX, XLSX, CSV). "
                    "It will be sent through the AI pipeline and appear below once processed (~5–15s)."
                )
                with gr.Row():
                    upload_widget = gr.File(
                        label="Choose File",
                        file_types=[
                            ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif",
                            ".bmp", ".webp", ".docx", ".xlsx", ".xls", ".csv",
                        ],
                        scale=3,
                    )
                    submit_btn = gr.Button("Submit to AI Pipeline", variant="primary", scale=1, size="lg")
                upload_msg_box = gr.Markdown(value="")

            # Step 2: Review Queue
            gr.Markdown("---")
            gr.Markdown("### Step 2 — Select Document from Queue to Review")

            with gr.Row():
                search_box = gr.Textbox(
                    label="Filter queue by filename",
                    placeholder="Type to filter...",
                    scale=3,
                    container=True,
                )
                refresh_btn = gr.Button("🔄 Refresh Queue", variant="secondary", scale=1, elem_id="refresh-btn")

            queue_dropdown = gr.Dropdown(
                label="Pending Documents",
                choices=get_pending_names(),
                interactive=True,
            )

            doc_status = gr.Markdown(value="*Upload a document or click Refresh to load the queue.*")
            timing_bar = gr.Markdown(value="")

            with gr.Row():
                conf_box = gr.HTML(value="")

            # Side-by-side: image + JSON
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("**Original Document / Preview**")
                    img_view = gr.Image(label="Source Image", interactive=False, height=520)

                with gr.Column(scale=1):
                    gr.Markdown("**AI Extracted JSON — edit to fix errors**")
                    json_view = gr.Code(label="JSON", language="json", lines=26, interactive=True)

            # Table grid
            with gr.Accordion("Extracted Table (Editable Grid)", open=False):
                table_grid = gr.Dataframe(label="Table Data", interactive=True, wrap=True)

            # Text preview
            with gr.Accordion("Extracted Text & Markdown Tables", open=False):
                text_view = gr.Markdown(value="")

            # Step 3: Approve / Reject
            gr.Markdown("---")
            gr.Markdown("### Step 3 — Approve or Reject")

            with gr.Row():
                approve_btn = gr.Button(
                    "✅ Approve & Commit to Database",
                    variant="primary",
                    scale=2,
                    elem_id="approve-btn",
                )
                reject_btn = gr.Button(
                    "🗑 Reject Document",
                    variant="stop",
                    scale=1,
                    elem_id="reject-btn",
                )

            rejection_box = gr.Textbox(
                label="Rejection Reason (fill in before rejecting)",
                placeholder="e.g. Wrong document type, illegible scan, duplicate entry...",
            )
            action_result = gr.Markdown(value="")

            # Step 4: Export
            with gr.Accordion("Step 4 — Export Data", open=False):
                gr.Markdown("Generate a download file from the current JSON (before or after approval).")
                with gr.Row():
                    btn_json  = gr.Button("Generate JSON",  scale=1)
                    btn_excel = gr.Button("Generate Excel", scale=1)
                    btn_csv   = gr.Button("Generate CSV",   scale=1)
                export_out = gr.File(label="Download", interactive=False)

        # ── Tab 2: Statistics ─────────────────────────────────────────────────
        with gr.Tab("📊 Statistics"):
            gr.Markdown("### Pipeline Statistics\nReal-time counts from the filesystem.")
            stats_refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
            stats_html_box    = gr.HTML(value=get_stats_html())

            stats_refresh_btn.click(
                fn=get_stats_html,
                outputs=[stats_html_box],
            )

        # ── Tab 3: Help ───────────────────────────────────────────────────────
        with gr.Tab("❓ Help"):
            gr.Markdown("""
## How to use the HITL Dashboard

### Uploading Documents
- **Supported formats:** PDF, JPG, PNG, TIFF, BMP, WEBP, DOCX, XLSX, XLS, CSV
- Drop a file in the upload box and click **Submit to AI Pipeline**
- Digital PDFs, Word docs, Excel sheets, and CSVs are fast-tracked (no VLM call)
- Images and scanned PDFs go through Qwen 3.6 27B Vision extraction

### Reviewing Documents
1. Click **Refresh Queue** to load pending documents
2. Use the **filter box** to narrow down by filename
3. Select a document from the dropdown
4. Review the extracted JSON in the right panel — you can edit it directly
5. Check the **confidence badge** (green = high, red = low)
6. View the original document image on the left

### Approving / Rejecting
- **Approve** → moves to `final_database/` with your edits preserved
- **Reject** → moves to `rejected/` with a reason logged
- Fill in the rejection reason textbox before clicking Reject

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `A` | Approve current document |
| `R` | Reject current document |
| `N` | Refresh queue (load next) |

### Exporting
- Use **Generate JSON / Excel / CSV** to download extracted data
- Files are saved to `exports/` directory

### Statistics Tab
- Shows real-time counts, pipeline breakdown, and document type distribution
- Click **Refresh Stats** to update
            """)

    # ── Shared output list for refresh/load operations ───────────────────────
    REFRESH_OUTPUTS = [
        queue_dropdown, stats_bar,
        doc_status, json_view, img_view, conf_box,
        current_file,
        table_grid, text_view, timing_bar,
    ]

    # ── Event wiring ──────────────────────────────────────────────────────────

    def on_dropdown_change(selected_name):
        if not selected_name:
            return (
                format_stats(),
                "Select a document from the queue.",
                "{}", None, "", None,
                pd.DataFrame([{"info": "No document loaded"}]),
                "", "",
            )
        status, js, img, conf, fpath, tbl, txt, stats, timing = load_document(selected_name)
        return stats, status, js, img, conf, fpath, tbl, txt, timing

    def on_search_change(filter_text):
        names     = get_pending_names(filter_text)
        dd_update = gr.update(choices=names, value=names[0] if names else None)
        return dd_update

    submit_btn.click(
        fn=upload_and_process,
        inputs=[upload_widget, search_box],
        outputs=[upload_msg_box] + REFRESH_OUTPUTS,
    )

    refresh_btn.click(
        fn=refresh_queue,
        inputs=[search_box],
        outputs=REFRESH_OUTPUTS,
    )

    search_box.change(
        fn=on_search_change,
        inputs=[search_box],
        outputs=[queue_dropdown],
    )

    queue_dropdown.change(
        fn=on_dropdown_change,
        inputs=[queue_dropdown],
        outputs=[
            stats_bar, doc_status, json_view, img_view, conf_box,
            current_file, table_grid, text_view, timing_bar,
        ],
    )

    approve_btn.click(
        fn=approve_document,
        inputs=[current_file, json_view],
        outputs=[action_result, stats_bar],
    )

    reject_btn.click(
        fn=reject_document,
        inputs=[current_file, rejection_box],
        outputs=[action_result, stats_bar],
    )

    btn_json.click(fn=do_export_json,  inputs=[current_file, json_view], outputs=[export_out])
    btn_excel.click(fn=do_export_excel, inputs=[current_file, json_view], outputs=[export_out])
    btn_csv.click(fn=do_export_csv,   inputs=[current_file, json_view], outputs=[export_out])


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
