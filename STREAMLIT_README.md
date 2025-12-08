# FinGuard Streamlit UI

## Professional Demo Interface for BlackRock Tech Ops Interview

### Design Philosophy
- **Minimal**: Clean, futuristic design with no clutter
- **Professional**: Dark theme with gradient accents
- **Performance-focused**: Real-time metrics displayed prominently
- **Production-ready**: Boss-level execution

### Launch the App

```bash
# Install Streamlit (if not already installed)
pip install streamlit>=1.40.0

# Run the app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Features

**Core Functionality:**
- Real-time prompt classification
- Category detection across 5 classes
- Confidence scoring
- Latency tracking (milliseconds)

**UI Elements:**
- Professional dark gradient background
- Color-coded results (green/red/orange/purple/gray)
- One-click example queries
- Performance metrics dashboard
- Category legend with descriptions

**Performance Display:**
- Accuracy: 97%
- F1 Score: 97%
- Latency: 85ms average
- False Positive Rate: <1%

### Color Scheme

- **SAFE**: Green (#00ff87) - Allowed
- **INVESTMENT_ADVICE**: Red (#ff4444) - Blocked
- **INDIRECT_ADVICE**: Orange (#ffa500) - Blocked
- **SYSTEM_PROBE**: Gray (#888888) - Blocked
- **UNIT_AMBIGUITY**: Purple (#a855f7) - Flagged

### Technical Stack

- **Frontend**: Streamlit with custom CSS
- **Backend**: Random Forest classifier (scikit-learn)
- **Embeddings**: Sentence Transformers (768-dim)
- **Model**: Cached with `@st.cache_resource`
- **Theme**: Custom dark gradient theme

### Deployment Ready

The app is configured for:
- Streamlit Cloud deployment
- Local development
- Production demo environments

### File Structure

```
FinGuard/
├── streamlit_app.py          # Main application
├── .streamlit/
│   └── config.toml           # Theme configuration
├── models/
│   └── rf_classifier_v1.pkl  # Trained model (required)
└── outputs/metrics/
    └── classification_report.json  # Performance metrics
```

### Prerequisites

The app requires:
1. Trained model at `models/rf_classifier_v1.pkl`
2. Classification metrics at `outputs/metrics/classification_report.json`

Run Phase 4 first if these don't exist:
```bash
python scripts/run_phase4.py
```

### Customization

**To modify the app name:**
Edit line 11 in `streamlit_app.py`:
```python
page_title="Your Title Here"
```

**To change colors:**
Edit the CSS in the `st.markdown()` block (lines 20-300)

**To adjust performance metrics:**
Modify the metrics section (lines 500+)

---

**Built for BlackRock Tech Ops Interview**
Demonstrating production-grade ML engineering with enterprise-level polish.
