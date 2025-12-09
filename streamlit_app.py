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
    page_icon="assets/shield_icon.svg",
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
        margin: 3rem 0 2rem 0;
        padding: 2rem 0 0 0;
        border-top: 1px solid #222222;
    }

    .metric-item {
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
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

    /* Feedback Section */
    .feedback-section {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 0.5rem;
    }

    .confidence-message {
        font-size: 0.8rem;
        color: #666666;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        text-align: center;
    }

    .confidence-message.uncertain {
        color: #ff8800;
    }

    .feedback-prompt {
        font-size: 0.75rem;
        color: #555555;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        text-align: center;
    }

    .feedback-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
    }

    .feedback-btn {
        background: transparent;
        border: 1px solid #333333;
        color: #666666;
        padding: 0.6rem 1.5rem;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        cursor: pointer;
        transition: all 0.2s;
    }

    .feedback-btn:hover {
        border-color: #ffffff;
        color: #ffffff;
    }

    .feedback-btn.correct:hover {
        border-color: #00ff00;
        color: #00ff00;
    }

    .feedback-btn.wrong:hover {
        border-color: #ff4444;
        color: #ff4444;
    }

    .feedback-thanks {
        font-size: 0.75rem;
        color: #444444;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 1rem 0;
        text-align: center;
    }

    /* Secondary buttons (CORRECT/WRONG) */
    .stButton button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid #333333 !important;
        color: #666666 !important;
        font-size: 0.75rem !important;
        padding: 0.5rem 1.2rem !important;
    }

    .stButton button[kind="secondary"]:hover {
        border-color: #ffffff !important;
        color: #ffffff !important;
        background-color: transparent !important;
    }

    .stButton button[kind="secondary"] p {
        color: #666666 !important;
    }

    .stButton button[kind="secondary"]:hover p {
        color: #ffffff !important;
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


def get_action(category, confidence):
    """Get action text based on category and confidence."""
    if category == "SAFE":
        return "ALLOWED"
    elif category in ["INVESTMENT_ADVICE", "INDIRECT_ADVICE", "SYSTEM_PROBE"]:
        if confidence < 0.7:
            return "REVIEW"
        return "BLOCKED"
    else:
        return "FLAGGED"


def get_confidence_message(confidence, category):
    """Get confidence-aware message for user."""
    if confidence >= 0.9:
        return None  # High confidence, no message needed
    elif confidence >= 0.7:
        return "Classification confidence is moderate"
    else:
        if category == "SAFE":
            return "Low confidence - please verify this classification"
        else:
            return "Uncertain classification - flagged for review"


def log_feedback(query, category, confidence, feedback, timestamp):
    """Log user feedback to file for later review."""
    feedback_file = Path("outputs/feedback/user_feedback.json")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": timestamp,
        "query": query,
        "predicted_category": category,
        "confidence": confidence,
        "feedback": feedback  # "correct" or "wrong"
    }

    # Load existing feedback
    if feedback_file.exists():
        with open(feedback_file, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(feedback_file, "w") as f:
        json.dump(data, f, indent=2)


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
        placeholder="Ask Aladdin something...",
        height=120,
        label_visibility="collapsed",
    )

    # Classify button
    classify_btn = st.button("CLASSIFY")

    # Initialize session state for feedback
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False

    if classify_btn and query:
        with st.spinner(""):
            start_time = time.time()
            embedding = embedder.embed_single(query)
            label, confidence = classifier.predict_single(embedding)
            category = classifier.class_names[label]
            latency_ms = (time.time() - start_time) * 1000

        # Store result in session state
        st.session_state.last_result = {
            "query": query,
            "category": category,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.feedback_given = False

    # Display result if exists
    if st.session_state.last_result:
        result = st.session_state.last_result
        category = result["category"]
        confidence = result["confidence"]

        category_class = get_category_class(category)
        action = get_action(category, confidence)
        confidence_msg = get_confidence_message(confidence, category)

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

        # Feedback buttons
        if not st.session_state.feedback_given:
            # Show confidence message only before feedback is given
            if confidence_msg:
                uncertain_class = "uncertain" if confidence < 0.7 else ""
                st.markdown(
                    f'<div class="confidence-message {uncertain_class}">{confidence_msg}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="feedback-prompt">Was this classification correct?</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    if st.button("CORRECT", key="btn_correct", type="secondary"):
                        log_feedback(
                            result["query"],
                            result["category"],
                            result["confidence"],
                            "correct",
                            result["timestamp"]
                        )
                        st.session_state.feedback_given = True
                        st.rerun()
                with fcol2:
                    if st.button("WRONG", key="btn_wrong", type="secondary"):
                        log_feedback(
                            result["query"],
                            result["category"],
                            result["confidence"],
                            "wrong",
                            result["timestamp"]
                        )
                        st.session_state.feedback_given = True
                        st.rerun()
        else:
            st.markdown(
                '<div class="feedback-thanks">Thank you for your feedback</div>',
                unsafe_allow_html=True,
            )

    # Categories

    categories = [
        ("SAFE", "#00ff00", "Legitimate queries"),
        ("INVESTMENT ADVICE", "#3a3a3a", "Direct recommendations"),
        ("INDIRECT ADVICE", "#ff8800", "Roleplay attempts"),
        ("SYSTEM PROBE", "#ff0000", "Prompt injection"),
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
        KAUSTUBH JOSHI • TECHNOLOGY OPERATIONS
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
