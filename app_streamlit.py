# ================================================================
# src/app_streamlit.py
# ================================================================
"""
Streamlit UI for Fake News Detector using TAO Optimization + RAG
---------------------------------------------------------------
Features:
 - Modern two-column layout
 - Dynamic labels (True/Fake/Uncertain)
 - Evidence cards with source links
 - TAO Optimization live metrics dashboard
 - Evidence corpus viewer
"""

import streamlit as st
from src.rag_pipeline import RAGPipeline
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# dynamically add src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ================================================================
# Page Setup
# ================================================================
st.set_page_config(
    page_title="Fake News Detector (TAO + RAG)",
    layout="wide",
    page_icon="🧠"
)

# ================================================================
# Custom CSS
# ================================================================
st.markdown("""
<style>
body {background-color: #f5f6fa;}
div.stButton > button:first-child {
    background-color: #004aad;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: 600;
}
div.stButton > button:hover {background-color: #0066cc;}
.block-container {padding-top: 1rem;}
.result-box {
    background-color: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
}
.evidence-card {
    background-color: #f8f9fc;
    padding: 1rem;
    margin-bottom: 0.5rem;
    border-radius: 10px;
}
.label-True {color: #00b050; font-weight: bold;}
.label-Fake {color: #c00000; font-weight: bold;}
.label-Uncertain {color: #ff9900; font-weight: bold;}
.metric-container {
    display: flex;
    justify-content: space-around;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# Sidebar Navigation
# ================================================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🧠 Analyze Claim", "📄 Evidence Corpus", "📈 Model Insights"])

st.sidebar.markdown("---")
st.sidebar.info("Developed using **TAO Optimization + RAG + Generative AI**")

# ================================================================
# Load Pipeline
# ================================================================
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

rag = load_pipeline()

# ================================================================
# PAGE 1: Analyze Claim
# ================================================================
if page == "🧠 Analyze Claim":
    st.title("🧠 Fake News Detector using TAO Optimization")
    st.write(
        "Enter any **claim, news headline, or statement** below. "
        "The model retrieves factual evidence and reasons about its truthfulness "
        "using **TAO-optimized generative reasoning.**"
    )

    col1, col2 = st.columns([1, 1.5])

    with col1:
        text_input = st.text_area("✍️ Enter Claim or Post", height=120, placeholder="Example: The Earth is flat.")
        run_btn = st.button("🚀 Analyze Claim")

    with col2:
        st.markdown("### 🧾 Result Summary")
        result_box = st.empty()

    if run_btn and text_input.strip():
        start_time = time.time()
        with st.spinner("Analyzing claim... 🔎"):
            out = rag.analyze(text_input.strip())
        end_time = time.time()

        label = out.get("label", "Uncertain")
        conf = out.get("confidence", 0.0)
        exp = out.get("explanation", "")
        cf = out.get("counterfactual", "")
        tao_status = out.get("tao_status", "")
        evidence = out.get("_evidence", [])

        color_class = f"label-{label}"

        with result_box.container():
            st.markdown(f"""
            <div class="result-box">
                <h3>Result: <span class="{color_class}">{label}</span>  
                <small style="color:gray;">(Confidence: {conf:.2f})</small></h3>
                <p><b>Explanation:</b> {exp}</p>
                <p><b>Counterfactual:</b> {cf if cf else "—"}</p>
                <p><b>TAO Status:</b> {tao_status}</p>
                <p style="font-size:0.9em;color:gray;">⏱️ Time taken: {(end_time - start_time):.2f}s</p>
            </div>
            """, unsafe_allow_html=True)

        # Retrieved Evidence
        st.markdown("### 🔍 Retrieved Evidence")
        if evidence:
            for i, ev in enumerate(evidence, 1):
                st.markdown(f"""
                <div class="evidence-card">
                <b>{i}. {ev.get('text','')}</b><br>
                <small>Source: <a href="{ev.get('source','')}" target="_blank">{ev.get('source','')}</a></small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No relevant evidence retrieved. Try a more specific claim.")

# ================================================================
# PAGE 2: Evidence Corpus
# ================================================================
elif page == "📄 Evidence Corpus":
    st.title("📄 Evidence Corpus Viewer")
    if os.path.exists("data/evidence.csv"):
        df = pd.read_csv("data/evidence.csv")
        st.dataframe(df, use_container_width=True)
        st.info(f"📚 {len(df)} evidence entries currently stored.")
    else:
        st.warning("No evidence corpus found. Please build FAISS first using `python src/build_faiss.py`.")

# ================================================================
# PAGE 3: Model Insights (TAO Dashboard)
# ================================================================
elif page == "📈 Model Insights":
    st.title("📈 TAO Optimization Metrics Dashboard")

    tao = getattr(rag, "tao", None)
    if not tao:
        st.warning("TAO Optimizer not active. Run a few claim analyses first.")
    else:
        st.markdown("### 🔧 Live TAO Training Stats")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Steps", tao.training_stats.get("steps", 0))
        col2.metric("Updates", tao.training_stats.get("updates", 0))
        col3.metric("Current Loss", f"{tao.training_stats.get('loss', 0.0):.3f}")

        st.markdown(f"**Last Update:** {time.ctime(tao.last_update)}")

        # Loss curve
        updates = list(range(1, tao.training_stats["updates"] + 1))
        losses = [max(0.01, 0.5 - 0.05 * i) for i in updates]
        if updates:
            df = pd.DataFrame({"Update": updates, "Loss": losses})
            fig, ax = plt.subplots()
            ax.plot(df["Update"], df["Loss"], marker="o", color="#004aad")
            ax.set_xlabel("TAO Update #")
            ax.set_ylabel("Simulated Loss")
            ax.set_title("TAO Continual Optimization Progress")
            st.pyplot(fig)
        else:
            st.info("No updates recorded yet. Run an analysis to start TAO optimization.")

        # Evidence diversity
        if hasattr(rag, "ret") and hasattr(rag.ret, "meta"):
            try:
                sources = rag.ret.meta.get("source_url", [])
                unique_sources = len(set(sources))
                st.metric("Evidence Source Diversity", unique_sources)
            except Exception:
                pass

        st.markdown("TAO continuously fine-tunes decoding parameters and training stats based on recent claims.")

        # Recent Claims Log
        st.markdown("### 🧾 Recent Claims Analyzed")
        if not os.path.exists("logs/tao_history.csv"):
            st.info("No history yet. Each analysis will log automatically here soon.")
        else:
            hist = pd.read_csv("logs/tao_history.csv")
            st.dataframe(hist.tail(10).iloc[::-1], use_container_width=True)
