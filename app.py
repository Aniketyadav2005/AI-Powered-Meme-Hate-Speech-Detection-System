from typing import List
import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import torch
import easyocr
import csv
import io
import json
import os
import datetime
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meme Hate Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; }

    /* Metric cards */
    .metric-card {
        background: #1e2530;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #4f8ef7;
    }
    .metric-card.hate { border-left-color: #e05252; }
    .metric-card.safe { border-left-color: #4caf7d; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .badge-hate { background: #3b1a1a; color: #e05252; }
    .badge-safe { background: #1a3b2b; color: #4caf7d; }

    /* History table */
    .history-row {
        background: #1e2530;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Progress bar label */
    .conf-label { font-size: 12px; color: #aaa; margin-bottom: 2px; }

    h1, h2, h3 { color: #e8eaf6 !important; }
    .stSubheader { color: #b0bec5 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State: Upload History ──────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts

# ─── Model Loading (cached) ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner="Loading image captioning model…")
def load_blip():
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    mdl = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    return proc, mdl

@st.cache_resource(show_spinner="Loading hate-speech classifier…")
def load_classifier():
    return pipeline(
        "text-classification",
        model="facebook/roberta-hate-speech-dynabench-r4-target",
        return_all_scores=True,   # ← get probabilities for ALL labels
    )

@st.cache_resource(show_spinner="Loading OCR engine…")
def load_ocr(languages: List[str]):
    return easyocr.Reader(languages, gpu=False)

processor, blip_model = load_blip()
classifier_pipeline = load_classifier()

# ─── Sidebar: Settings ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("**🌐 OCR Languages**")
    lang_options = {
        "English": "en", "French": "fr", "German": "de",
        "Spanish": "es", "Hindi": "hi", "Chinese (Simplified)": "ch_sim",
        "Arabic": "ar", "Japanese": "ja", "Korean": "ko",
    }
    selected_langs = st.multiselect(
        "Select languages to detect",
        options=list(lang_options.keys()),
        default=["English"],
    )
    ocr_lang_codes = [lang_options[l] for l in selected_langs] or ["en"]
    reader = load_ocr(tuple(ocr_lang_codes))

    st.markdown("---")
    st.markdown("**🔢 Batch Settings**")
    max_batch = st.slider("Max images per batch", 1, 20, 10)

    st.markdown("---")
    st.markdown("**📊 History**")
    total = len(st.session_state.history)
    hate_count = sum(1 for r in st.session_state.history if r["is_hate"])
    safe_count = total - hate_count
    st.metric("Total Analysed", total)
    col1, col2 = st.columns(2)
    col1.metric("🚨 Hateful", hate_count)
    col2.metric("✅ Safe", safe_count)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ─── Helper Functions ────────────────────────────────────────────────────────

def blur_image(img_array: np.ndarray, radius: int = 15) -> np.ndarray:
    pil_img = Image.fromarray(img_array)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


def classify_text(text: str):
    """Returns (label, hate_score, safe_score).

    Handles both pipeline output shapes:
      - return_all_scores=True  → [[{label, score}, ...]]  (list of lists)
      - return_all_scores=False → [{label, score}]          (list of dicts)
    """
    raw = classifier_pipeline(text)

    # Unwrap outer list if it's a list-of-lists
    if isinstance(raw[0], list):
        results = raw[0]           # [[{...}, {...}]]  → [{...}, {...}]
    elif isinstance(raw[0], dict):
        results = raw              # [{...}, {...}]    → use as-is
    else:
        results = raw[0]

    # Build a score map, lower-casing all label names for safety
    scores = {r["label"].lower(): r["score"] for r in results}

    # The facebook/roberta model uses "hate" and "nothate"
    hate_score = scores.get("hate", scores.get("label_1", 0.0))
    safe_score = scores.get("nothate", scores.get("label_0", 1.0 - hate_score))

    is_hate = hate_score >= 0.5
    label = "hate" if is_hate else "nothate"
    return label, hate_score, safe_score


def analyse_image(uploaded: UploadedFile):
    """Full pipeline for one image. Returns a result dict."""
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)

    # OCR
    extracted_text = " ".join([t[1] for t in reader.readtext(image_np)])

    # Caption
    inputs = processor(image_pil, return_tensors="pt").to(DEVICE)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Classify
    combined = caption + " " + extracted_text
    label, hate_score, safe_score = classify_text(combined)
    is_hate = label == "hate"

    # Blurred copy if hateful
    blurred_np = blur_image(image_np) if is_hate else None

    return {
        "filename": uploaded.name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_np": image_np,
        "blurred_np": blurred_np,
        "extracted_text": extracted_text or "No text detected.",
        "caption": caption,
        "label": label,
        "is_hate": is_hate,
        "hate_score": round(hate_score * 100, 1),
        "safe_score": round(safe_score * 100, 1),
    }


def render_result_card(result: dict, show_image: bool = True):
    """Renders a single result card."""
    is_hate = result["is_hate"]
    card_class = "hate" if is_hate else "safe"
    badge_class = "badge-hate" if is_hate else "badge-safe"
    verdict = "🚨 HATEFUL" if is_hate else "✅ SAFE"

    col_img, col_info = st.columns([1, 1.6])

    with col_img:
        if show_image:
            display_img = result["blurred_np"] if is_hate else result["image_np"]
            st.image(display_img, caption=result["filename"], use_column_width=True)
            if is_hate:
                with st.expander("⚠️ Show original (blurred for safety)"):
                    st.image(result["image_np"], use_column_width=True)

    with col_info:
        st.markdown(
            f'<span class="badge {badge_class}">{verdict}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**📄 Caption:** {result['caption']}")
        st.markdown(f"**🔤 OCR Text:** {result['extracted_text']}")

        st.markdown("**📊 Confidence Scores**")
        st.markdown('<p class="conf-label">Hate probability</p>', unsafe_allow_html=True)
        st.progress(result["hate_score"] / 100)
        st.caption(f"{result['hate_score']}%")

        st.markdown('<p class="conf-label">Safe probability</p>', unsafe_allow_html=True)
        st.progress(result["safe_score"] / 100)
        st.caption(f"{result['safe_score']}%")

        if is_hate:
            highlighted = result["extracted_text"]
            if highlighted and highlighted != "No text detected.":
                st.markdown(
                    f'<div style="background:#3b1a1a;border-radius:8px;padding:10px;'
                    f'color:#e05252;font-weight:600;">⚠️ Flagged Text:<br>{highlighted}</div>',
                    unsafe_allow_html=True,
                )


# ─── Export Helpers ──────────────────────────────────────────────────────────

def export_csv(results: List[dict]) -> bytes:
    output = io.StringIO()
    fields = ["filename", "timestamp", "label", "hate_score", "safe_score", "caption", "extracted_text"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue().encode("utf-8")


def export_pdf(results: List[dict]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], textColor=colors.HexColor("#1a237e"), fontSize=20)
    heading_style = ParagraphStyle("heading", parent=styles["Heading2"], textColor=colors.HexColor("#283593"))
    body_style = styles["Normal"]
    hate_style = ParagraphStyle("hate", parent=body_style, textColor=colors.HexColor("#c62828"))
    safe_style = ParagraphStyle("safe", parent=body_style, textColor=colors.HexColor("#2e7d32"))

    story = []
    story.append(Paragraph("🛡️ Meme Hate Speech Detection — Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 4))

    hate_count = sum(1 for r in results if r["is_hate"])
    summary_data = [
        ["Total Images", "Hateful", "Safe"],
        [str(len(results)), str(hate_count), str(len(results) - hate_count)],
    ]
    tbl = Table(summary_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3949ab")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#e8eaf6")),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#e8eaf6"), colors.white]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#9fa8da")))
    story.append(Spacer(1, 8))

    for i, r in enumerate(results, 1):
        story.append(Paragraph(f"{i}. {r['filename']}", heading_style))
        story.append(Paragraph(f"Timestamp: {r['timestamp']}", body_style))
        verdict_style = hate_style if r["is_hate"] else safe_style
        verdict_text = "🚨 HATEFUL" if r["is_hate"] else "✅ SAFE"
        story.append(Paragraph(f"Verdict: {verdict_text}", verdict_style))
        story.append(Paragraph(f"Hate Score: {r['hate_score']}%  |  Safe Score: {r['safe_score']}%", body_style))
        story.append(Paragraph(f"Caption: {r['caption']}", body_style))
        story.append(Paragraph(f"OCR Text: {r['extracted_text']}", body_style))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buffer.getvalue()


# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ Meme Hate Speech Detector")
st.markdown("Upload one or multiple meme images — get captions, OCR text, hate-speech scores, and a downloadable report.")
st.markdown("---")

tab_analyse, tab_history = st.tabs(["📤 Analyse Images", "📋 Upload History"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE
# ══════════════════════════════════════════════════════════════════
with tab_analyse:
    uploaded_files: List[UploadedFile] = st.file_uploader(
        "Upload image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help=f"Upload up to {max_batch} images at once.",
    )

    if uploaded_files:
        uploaded_files = uploaded_files[:max_batch]

        if st.button("🚀 Analyse All Images", type="primary"):
            batch_results = []
            progress = st.progress(0, text="Starting analysis…")

            for idx, uf in enumerate(uploaded_files):
                progress.progress(
                    (idx + 1) / len(uploaded_files),
                    text=f"Analysing {uf.name} ({idx + 1}/{len(uploaded_files)})…",
                )
                with st.spinner(f"Processing {uf.name}…"):
                    result = analyse_image(uf)
                    batch_results.append(result)
                    st.session_state.history.append(result)   # add to history

            progress.empty()
            st.success(f"✅ Done! Analysed {len(batch_results)} image(s).")

            # Summary stats bar
            hate_n = sum(1 for r in batch_results if r["is_hate"])
            safe_n = len(batch_results) - hate_n
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Analysed", len(batch_results))
            c2.metric("🚨 Hateful", hate_n)
            c3.metric("✅ Safe", safe_n)
            st.markdown("---")

            # Individual result cards
            for result in batch_results:
                render_result_card(result)
                st.markdown("---")

            # ── Export buttons ──────────────────────────────────────
            st.markdown("### 📥 Export Results")
            export_col1, export_col2 = st.columns(2)

            with export_col1:
                csv_bytes = export_csv(batch_results)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_bytes,
                    file_name=f"meme_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with export_col2:
                pdf_bytes = export_pdf(batch_results)
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"meme_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📋 Upload History")

    if not st.session_state.history:
        st.info("No images analysed yet. Go to the **Analyse Images** tab to get started.")
    else:
        history = st.session_state.history

        # Summary metrics
        total = len(history)
        hate_total = sum(1 for r in history if r["is_hate"])
        safe_total = total - hate_total
        avg_hate = round(sum(r["hate_score"] for r in history) / total, 1)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📦 Total", total)
        m2.metric("🚨 Hateful", hate_total)
        m3.metric("✅ Safe", safe_total)
        m4.metric("📊 Avg Hate Score", f"{avg_hate}%")
        st.markdown("---")

        # Filter
        filter_opt = st.radio("Filter by", ["All", "Hateful Only", "Safe Only"], horizontal=True)

        filtered = history
        if filter_opt == "Hateful Only":
            filtered = [r for r in history if r["is_hate"]]
        elif filter_opt == "Safe Only":
            filtered = [r for r in history if not r["is_hate"]]

        st.markdown(f"**Showing {len(filtered)} result(s)**")

        for r in reversed(filtered):   # newest first
            badge = "🚨 HATE" if r["is_hate"] else "✅ SAFE"
            score_bar = "█" * int(r["hate_score"] / 10) + "░" * (10 - int(r["hate_score"] / 10))
            with st.expander(f"{badge}  |  {r['filename']}  —  {r['timestamp']}"):
                render_result_card(r, show_image=True)

        st.markdown("---")
        st.markdown("### 📥 Export Full History")
        h1, h2 = st.columns(2)
        with h1:
            st.download_button(
                "⬇️ Export History CSV",
                data=export_csv(filtered),
                file_name="full_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with h2:
            st.download_button(
                "⬇️ Export History PDF",
                data=export_pdf(filtered),
                file_name="full_history_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
