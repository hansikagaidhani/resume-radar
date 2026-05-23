import streamlit as st
from extractor import extract_text
from matcher import analyse_gaps

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeRadar",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.score-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px; padding: 2rem; text-align: center;
    color: white; margin-bottom: 1.25rem;
}
.score-number { font-size: 4rem; font-weight: 800; line-height: 1; }
.score-label  { font-size: 1.1rem; opacity: 0.9; margin-top: 0.25rem; }

.chip {
    display: inline-block; border-radius: 9999px;
    padding: 0.2rem 0.75rem; margin: 0.2rem;
    font-size: 0.82rem; font-weight: 600;
}
.chip-high   { background: #fee2e2; color: #991b1b; }
.chip-medium { background: #fef9c3; color: #854d0e; }
.chip-low    { background: #e0f2fe; color: #0c4a6e; }
.chip-present{ background: #dcfce7; color: #166534; }

.tip {
    border-radius: 0 8px 8px 0; padding: 0.7rem 1rem;
    margin-bottom: 0.55rem; font-size: 0.93rem;
}
.tip-Summary    { background:#eff6ff; border-left: 4px solid #3b82f6; }
.tip-Skills     { background:#f0fdf4; border-left: 4px solid #16a34a; }
.tip-Experience { background:#fff7ed; border-left: 4px solid #f97316; }
.tip-General    { background:#faf5ff; border-left: 4px solid #9333ea; }

.section-hdr { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.6rem; color: #1e293b; }
.legend      { font-size: 0.78rem; color: #64748b; margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 ResumeRadar")
st.markdown(
    "**Scan your resume. Spot the gaps. Land the job.** &nbsp;|&nbsp; "
    "Upload or paste a **job description** and your **resume** to get a "
    "semantic match score, prioritised missing keywords, and section-specific edit suggestions."
)
st.divider()

# ── Helper: build a text input widget (tabs: upload file | paste text) ────────
def _text_input_widget(label: str, icon: str, file_types: list[str], key_prefix: str) -> str:
    """Render upload+paste tabs; return the resolved text (or empty string)."""
    st.subheader(f"{icon} {label}")
    tab_upload, tab_paste = st.tabs(["📁 Upload File", "✏️ Paste Text"])

    text = ""
    with tab_upload:
        uploaded = st.file_uploader(
            f"Upload {label} (.pdf or .docx)",
            type=file_types,
            key=f"{key_prefix}_file",
            label_visibility="collapsed",
        )
        if uploaded:
            try:
                text = extract_text(uploaded)
                st.success(f"Extracted {len(text.split())} words from **{uploaded.name}**.")
                with st.expander("Preview extracted text"):
                    st.text(text[:1500] + ("…" if len(text) > 1500 else ""))
            except ValueError as exc:
                st.error(str(exc))

    with tab_paste:
        pasted = st.text_area(
            label=label,
            label_visibility="collapsed",
            placeholder=f"Paste the full {label.lower()} here…",
            height=300,
            key=f"{key_prefix}_text",
        )
        if pasted.strip():
            text = pasted

    return text


# ── Inputs ────────────────────────────────────────────────────────────────────
col_jd, col_res = st.columns(2, gap="large")

with col_jd:
    job_desc = _text_input_widget("Job Description", "📋", ["pdf", "docx"], "jd")

with col_res:
    resume = _text_input_widget("Resume", "📝", ["pdf", "docx"], "res")

# ── Analyse button ─────────────────────────────────────────────────────────────
st.markdown("")
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    analyse = st.button("🔍 Analyse Match", use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
if analyse:
    if not job_desc.strip() or not resume.strip():
        st.warning("Please provide both a job description and a resume (upload or paste).")
        st.stop()

    with st.spinner("Running semantic analysis…"):
        result = analyse_gaps(job_desc, resume)

    score   = result["score"]
    pct     = result["pct"]
    verdict = result["verdict"]
    missing = result["missing"]
    present = result["present"]
    suggestions = result["suggestions"]

    st.divider()

    # ── Score panel ───────────────────────────────────────────────────────────
    score_col, results_col = st.columns([1, 2], gap="large")

    with score_col:
        st.markdown(
            f'<div class="score-box">'
            f'<div class="score-number">{pct}%</div>'
            f'<div class="score-label">{verdict}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(score, text=f"Cosine similarity: {score:.4f}")

        st.markdown("**How is this calculated?**")
        st.caption(
            "Both texts are encoded into 384-dimensional vectors using "
            "**all-MiniLM-L6-v2** (sentence-transformers). "
            "The score is the **cosine similarity** between those vectors — "
            "measuring semantic closeness, not just word overlap."
        )

    # ── Keywords & suggestions ────────────────────────────────────────────────
    with results_col:

        # Missing keywords with priority chips
        st.markdown('<div class="section-hdr">🔑 Missing Keywords</div>', unsafe_allow_html=True)
        if missing:
            st.markdown(
                '<div class="legend">'
                '<span class="chip chip-high">● High priority</span> &nbsp;'
                '<span class="chip chip-medium">● Medium</span> &nbsp;'
                '<span class="chip chip-low">● Lower</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            chips = "".join(
                f'<span class="chip chip-{m["priority"]}">{m["keyword"]}</span>'
                for m in missing[:24]
            )
            st.markdown(chips, unsafe_allow_html=True)
            if len(missing) > 24:
                st.caption(f"…and {len(missing) - 24} more. See full list below.")
        else:
            st.success("All major JD keywords are already present in your resume!")

        # Keywords already present
        if present:
            with st.expander(f"✅ {len(present)} JD keywords already in your resume"):
                present_chips = "".join(
                    f'<span class="chip chip-present">{kw}</span>' for kw in present
                )
                st.markdown(present_chips, unsafe_allow_html=True)

        st.markdown("")

        # Section-grouped suggestions
        st.markdown('<div class="section-hdr">💡 Edit Suggestions</div>', unsafe_allow_html=True)

        section_order = ["Summary", "Skills", "Experience", "General"]
        grouped: dict[str, list[str]] = {s: [] for s in section_order}
        for tip in suggestions:
            grouped.setdefault(tip["section"], []).append(tip["text"])

        section_icons = {"Summary": "📝", "Skills": "🛠️", "Experience": "💼", "General": "🔎"}
        for section in section_order:
            tips = grouped.get(section, [])
            if not tips:
                continue
            st.markdown(f"**{section_icons.get(section, '')} {section}**")
            for tip_text in tips:
                st.markdown(
                    f'<div class="tip tip-{section}">{tip_text}</div>',
                    unsafe_allow_html=True,
                )

    # ── Full keyword table ────────────────────────────────────────────────────
    if missing:
        with st.expander("View all missing keywords"):
            cols = st.columns(4)
            for i, m in enumerate(missing):
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(m["priority"], "")
                cols[i % 4].markdown(f"{priority_icon} `{m['keyword']}`")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🎯 **ResumeRadar** · Built with "
    "[Streamlit](https://streamlit.io) · "
    "[sentence-transformers](https://www.sbert.net) · "
    "scikit-learn · pdfplumber · python-docx"
)
