# FinGuard Streamlit UI

## Professional Demo Interface for BlackRock Tech Ops Interview

### Design Philosophy
- **Minimal**: Pure black background with clean typography
- **Professional**: Minimalist design with centered layout
- **Performance-focused**: Real-time metrics displayed prominently
- **Production-ready**: Boss-level execution

### Launch the App

```bash
# Install Streamlit (if not already installed)
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

**Note:** The app is compatible with older versions of Streamlit. If you encounter compatibility issues, ensure you have at least Streamlit 0.89.0 installed.

### Features

**Core Functionality:**
- Real-time prompt classification
- Category detection across 5 classes
- Confidence scoring
- Latency tracking (milliseconds)

**UI Elements:**
- Pure black minimal background (#000000)
- Centered classify button with tight padding
- Color-coded results (green/red/orange/purple/gray)
- Performance metrics dashboard
- Category legend with descriptions
- No UMAP visualization (removed for cleaner interface)

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

- **Frontend**: Streamlit with custom CSS (pure black minimal theme)
- **Backend**: Random Forest classifier (scikit-learn)
- **Embeddings**: Sentence Transformers (768-dim)
- **Model**: Cached with `@st.cache_resource`
- **Theme**: Pure black background with centered elements
- **Styling**: Custom CSS with flexbox for centered button layout

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

### Recent Updates

**v1.1 - UI Fixes (2025-12-08)**
- Fixed `use_container_width` compatibility issue with older Streamlit versions
- Removed UMAP visualization section for cleaner minimal interface
- Fixed button text visibility (black text on white button background)
- Centered classify button with reduced horizontal padding
- Updated footer with correct name

**Customization:**
- **App name**: Edit `page_title` in `st.set_page_config()` (line 15)
- **Colors**: Modify CSS in `st.markdown()` block (lines 22-220)
- **Button styling**: Adjust `.stButton button` CSS (lines 80-100)
- **Metrics**: Update metrics section (lines 384-407)

---

**Built for BlackRock Tech Ops Interview**
Demonstrating production-grade ML engineering with enterprise-level polish.
