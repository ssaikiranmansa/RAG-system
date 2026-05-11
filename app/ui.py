import streamlit as st
import requests
import json

# ── config ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }

/* ── title ── */
.rag-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.rag-subtitle {
    font-size: 0.95rem;
    color: #6b6b8a;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* ── section headers ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.8rem;
}

/* ── upload zone ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #2a2a3e !important;
    border-radius: 12px !important;
    background: #0f0f1a !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #a78bfa !important;
}
            
[data-testid="stFileUploader"] small {
    display: none !important;
}

/* ── buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    width: 100%;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.85;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── text input ── */
.stTextArea textarea, .stTextInput input {
    background: #0f0f1a !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
}

/* ── answer card ── */
.answer-card {
    background: linear-gradient(135deg, #0f0f1a 0%, #12102a 100%);
    border: 1px solid #2a2a3e;
    border-left: 3px solid #a78bfa;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
    line-height: 1.7;
    font-size: 0.97rem;
}

/* ── source card ── */
.source-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.source-card:hover { border-color: #3a3a5e; }
.source-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6b6b8a;
    margin-bottom: 0.5rem;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.source-meta span {
    background: #1a1a2e;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
}
.source-content {
    font-size: 0.88rem;
    color: #9090b0;
    line-height: 1.6;
}
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed22, #2563eb22);
    border: 1px solid #7c3aed44;
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.1rem 0.5rem;
    border-radius: 20px;
    float: right;
}

/* ── status pills ── */
.pill-success {
    display: inline-block;
    background: #05140d;
    border: 1px solid #166534;
    color: #34d399;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
}
.pill-cached {
    display: inline-block;
    background: #0f0a1a;
    border: 1px solid #4c1d95;
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
}
.pill-info {
    display: inline-block;
    background: #0a1628;
    border: 1px solid #1e3a5f;
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
}

/* ── divider ── */
.thin-divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1.5rem 0;
}

/* ── ingested docs list ── */
.doc-chip {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    color: #9090c0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    margin: 0.2rem;
}

/* ── slider ── */
.stSlider [data-baseweb="slider"] { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── helpers ───────────────────────────────────────────────────────────────────

def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def ingest_files(files):
    file_tuples = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    try:
        r = requests.post(f"{API_BASE}/ingest", files=file_tuples, timeout=500)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is the FastAPI server running on port 8000?"
    except Exception as e:
        return None, str(e)


def query_api(query, top_k):
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"query": query, "top_k": top_k},
            timeout=60
        )
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is the FastAPI server running on port 8000?"
    except Exception as e:
        return None, str(e)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="rag-title">RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subtitle">Document Intelligence</div>', unsafe_allow_html=True)

    # API status
    api_ok = check_api_health()
    if api_ok:
        st.markdown('<span class="pill-success">● API connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#f87171;font-size:0.8rem;">● API offline — start uvicorn first</span>', unsafe_allow_html=True)

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="section-label">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("Ingest Documents"):
            with st.spinner("Extracting, chunking, embedding..."):
                results, error = ingest_files(uploaded_files)
            if error:
                st.error(error)
            else:
                for r in results:
                    if r.get("skipped"):
                        st.info(f"'{r['source']}' already ingested — skipped.")
                    else:
                        st.success(f"✓ '{r['source']}' — {r['chunks_ingested']} chunks stored.")
                    if r["source"] not in st.session_state.ingested_docs:
                        st.session_state.ingested_docs.append(r["source"])

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    # Ingested docs
    if st.session_state.ingested_docs:
        st.markdown('<div class="section-label">Ingested Documents</div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="doc-chip">📄 {d}</span>' for d in st.session_state.ingested_docs)
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Settings
    # st.markdown('<div class="section-label">Retrieval Settings</div>', unsafe_allow_html=True)
    # top_k = st.slider("Top-K chunks", min_value=1, max_value=15, value=5,
    #                   help="How many chunks to retrieve before reranking")

    # st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    # Query history
    if st.session_state.query_history:
        st.markdown('<div class="section-label">Recent Queries</div>', unsafe_allow_html=True)
        for q in reversed(st.session_state.query_history[-5:]):
            if st.button(f"↩ {q[:40]}{'...' if len(q) > 40 else ''}", key=f"hist_{q}"):
                st.session_state.prefill_query = q
                st.rerun()


# ── main area ─────────────────────────────────────────────────────────────────

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)

    prefill = st.session_state.get("prefill_query", "")
    query = st.text_area(
        "Query",
        value=prefill,
        placeholder="What does this document say about...?",
        height=100,
        label_visibility="collapsed"
    )
    if prefill:
        st.session_state.prefill_query = ""

    ask_col, clear_col = st.columns([3, 1])
    with ask_col:
        ask_clicked = st.button("Ask", disabled=not api_ok or not query.strip())
    with clear_col:
        if st.button("Clear"):
            st.session_state.last_result = None
            st.rerun()

    # ── answer ──
    if ask_clicked and query.strip():
        with st.spinner("Searching and generating answer..."):
            result, error = query_api(query.strip(), top_k)

        if error:
            st.error(error)
        else:
            st.session_state.last_result = result
            if query not in st.session_state.query_history:
                st.session_state.query_history.append(query)

    if st.session_state.last_result:
        result = st.session_state.last_result

        # status pills
        pills = []
        if result.get("cached"):
            pills.append('<span class="pill-cached">⚡ cached</span>')
        else:
            pills.append('<span class="pill-success">✓ fresh</span>')
        src_count = len(result.get("sources", []))
        pills.append(f'<span class="pill-info">{src_count} sources</span>')
        st.markdown(" &nbsp; ".join(pills), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # answer
        st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-card">{result["answer"]}</div>',
            unsafe_allow_html=True
        )

with col2:
    if st.session_state.last_result:
        sources = st.session_state.last_result.get("sources", [])
        if sources:
            st.markdown('<div class="section-label">Retrieved Sources</div>', unsafe_allow_html=True)
            for i, src in enumerate(sources):
                page = src.get("page_number") or "—"
                section = src.get("section_header") or "—"
                score = src.get("rerank_score", "—")
                sim = round(src.get("similarity", 0) * 100)
                content_preview = src["content"][:220] + "..." if len(src["content"]) > 220 else src["content"]

                st.markdown(f"""
                <div class="source-card">
                    <span class="score-badge">score {score}/10</span>
                    <div class="source-meta">
                        <span>📄 {src['source']}</span>
                        <span>pg {page}</span>
                        <span>{section}</span>
                        <span>~{sim}% sim</span>
                    </div>
                    <div class="source-content">{content_preview}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        # empty state
        st.markdown("""
        <div style="margin-top:3rem;text-align:center;color:#3a3a5a;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">🔍</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#4a4a6a;">
                Sources appear here
            </div>
            <div style="font-size:0.82rem;margin-top:0.4rem;color:#2a2a4a;">
                Ask a question to see retrieved chunks
            </div>
        </div>
        """, unsafe_allow_html=True)
