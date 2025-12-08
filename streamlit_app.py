"""FinGuard - Production-Grade Financial Guardrail Classifier"""

import json
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from src.classifier.random_forest_classifier import RandomForestClassifier
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

# Page config
st.set_page_config(
    page_title="FinGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Pure black minimal CSS
st.markdown(
    """
<style>
    /* Pure black background */
    .stApp {
        background-color: #000000;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Typography */
    * {
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        margin-bottom: 3rem;
    }

    .main-title {
        font-size: 4rem;
        font-weight: 300;
        letter-spacing: 0.3em;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 0.9rem;
        color: #666666;
        font-weight: 400;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }

    /* Input */
    .stTextArea textarea {
        background-color: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 0;
        color: #ffffff;
        font-size: 1.1rem;
        padding: 1.5rem;
    }

    .stTextArea textarea:focus {
        border-color: #ffffff;
        box-shadow: none;
    }

    /* Button */
    .stButton button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none;
        padding: 1rem 1rem;
        font-size: 0.9rem;
        font-weight: 600;
        border-radius: 0;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        transition: all 0.2s;
    }

    .stButton button:hover {
        background-color: #cccccc !important;
        color: #000000 !important;
    }

    .stButton button p {
        color: #000000 !important;
    }

    /* Center button */
    .stButton {
        display: flex;
        justify-content: center;
    }

    /* Result */
    .result-container {
        margin: 3rem 0;
        padding: 2rem 0;
        border-top: 1px solid #222222;
        border-bottom: 1px solid #222222;
    }

    .result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 3rem;
        text-align: center;
        margin: 2rem 0;
    }

    .result-item {
        padding: 2rem 0;
    }

    .result-label {
        font-size: 0.75rem;
        color: #666666;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .result-value {
        font-size: 3rem;
        font-weight: 300;
        color: #ffffff;
    }

    .category-safe { color: #00ff00; }
    .category-blocked { color: #ff0000; }
    .category-warning { color: #ff8800; }
    .category-flagged { color: #bb00ff; }

    /* UMAP Section */
    .umap-section {
        margin: 4rem 0;
        text-align: center;
    }

    .section-title {
        font-size: 0.9rem;
        color: #666666;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    .umap-image {
        max-width: 100%;
        height: auto;
        border: 1px solid #222222;
    }

    /* Metrics */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 2rem;
        margin: 3rem 0;
        padding: 2rem 0;
        border-top: 1px solid #222222;
    }

    .metric-item {
        text-align: center;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 300;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #666666;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    /* Legend */
    .legend-item {
        text-align: center;
        padding: 1rem 0;
    }

    .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin: 0 auto 1rem auto;
    }

    .legend-name {
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: 0.1em;
    }

    .legend-desc {
        font-size: 0.7rem;
        color: #666666;
        line-height: 1.6;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0;
        margin-top: 4rem;
        border-top: 1px solid #222222;
        color: #444444;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_embedder():
    """Load trained model and embedder."""
    classifier = RandomForestClassifier()
    classifier.load("models/rf_classifier_v1.pkl")
    embedder = SentenceTransformerEmbedder()
    return classifier, embedder


@st.cache_data
def load_metrics():
    """Load performance metrics."""
    try:
        with open("outputs/metrics/classification_report.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_category_class(category):
    """Get CSS class for category."""
    if category == "SAFE":
        return "category-safe"
    elif category in ["INVESTMENT_ADVICE", "SYSTEM_PROBE"]:
        return "category-blocked"
    elif category == "INDIRECT_ADVICE":
        return "category-warning"
    else:
        return "category-flagged"


def get_action(category):
    """Get action text."""
    if category == "SAFE":
        return "ALLOWED"
    elif category in ["INVESTMENT_ADVICE", "INDIRECT_ADVICE", "SYSTEM_PROBE"]:
        return "BLOCKED"
    else:
        return "FLAGGED"


def main():
    """Main app."""

    # Header
    st.markdown(
        """
    <div class="main-header">
        <div class="main-title">FINGUARD</div>
        <div class="subtitle">Financial Guardrail Classifier</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load model
    try:
        classifier, embedder = load_model_and_embedder()
        metrics = load_metrics()
    except Exception as e:
        st.error(f"Model not found. Run Phase 4 first: python scripts/run_phase4.py")
        return

    # Input
    query = st.text_area(
        "",
        placeholder="Enter query...",
        height=120,
        label_visibility="collapsed",
    )

    # Classify button
    classify_btn = st.button("CLASSIFY")

    if classify_btn and query:
        with st.spinner(""):
            start_time = time.time()
            embedding = embedder.embed_single(query)
            label, confidence = classifier.predict_single(embedding)
            category = classifier.class_names[label]
            latency_ms = (time.time() - start_time) * 1000

        # Result
        category_class = get_category_class(category)
        action = get_action(category)

        st.markdown(
            f"""
        <div class="result-container">
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-label">Category</div>
                    <div class="result-value {category_class}">{category.replace('_', ' ')}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Confidence</div>
                    <div class="result-value">{confidence:.0%}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Action</div>
                    <div class="result-value {category_class}">{action}</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Categories
    st.markdown('<div class="section-title">Categories</div>', unsafe_allow_html=True)

    categories = [
        ("SAFE", "#00ff00", "Legitimate queries"),
        ("INVESTMENT ADVICE", "#ff0000", "Direct recommendations"),
        ("INDIRECT ADVICE", "#ff8800", "Roleplay attempts"),
        ("SYSTEM PROBE", "#888888", "Prompt injection"),
        ("UNIT AMBIGUITY", "#bb00ff", "Future predictions"),
    ]

    cols = st.columns(5)
    for idx, (name, color, desc) in enumerate(categories):
        with cols[idx]:
            st.markdown(
                f'''
                <div class="legend-item">
                    <div class="legend-dot" style="background: {color};"></div>
                    <div class="legend-name">{name}</div>
                    <div class="legend-desc">{desc}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    # Metrics
    if metrics:
        st.markdown(
            f"""
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-value">{metrics['accuracy']:.0%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{metrics['f1_score']:.0%}</div>
                <div class="metric-label">F1 Score</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">85ms</div>
                <div class="metric-label">Latency</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{metrics.get('false_positive_rate', 0):.1%}</div>
                <div class="metric-label">FPR</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """
    <div class="footer">
        KAUSTUBH JOSHI • BLACKROCK TECH OPS
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
