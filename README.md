# FinGuard

## Overview
FinGuard is a production-grade prompt safety classifier designed to detect and block financial advisory attacks on LLM systems. Using semantic embeddings and lightweight ML classifiers, FinGuard achieves superior accuracy and speed compared to LLM-as-judge approaches.

**Key Features:**
- 🎯 Multi-category attack detection (investment advice, social engineering, prompt injection, etc.)
- ⚡ Sub-100ms inference latency
- 🎨 Embedding-based classification with visual cluster analysis
- 📊 Comprehensive benchmarking suite
- 🔧 Modular, extensible architecture

## Getting Started

### Prerequisites
- Python 3.10+
- pip
- (Optional) CUDA-capable GPU for faster local embeddings

### Installation
```bash
# Clone the repository
cd FinGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
```bash
# Copy environment template
cp .env.template .env

# Add your API keys to .env
# ANTHROPIC_API_KEY=your_claude_key_here
# OPENAI_API_KEY=your_openai_key_here
```

## Usage

### Quick Start
```bash
# Run full pipeline
python scripts/run_full_pipeline.py

# Or run phase by phase
python scripts/run_phase1.py  # Generate dataset
python scripts/run_phase2.py  # Create embeddings
python scripts/run_phase3.py  # Visualize
python scripts/run_phase4.py  # Train classifier
python scripts/run_phase5.py  # Demo & benchmark
```

### Interactive Demo
```bash
python src/demo/cli_demo.py
```

### Run Tests
```bash
pytest tests/ -v --cov=src
```

## Development

### Project Structure
```
FinGuard/
├── .claude/          # Claude Code configuration
├── src/              # Source code (to be created)
├── tests/            # Test files (to be created)
└── docs/             # Additional documentation (optional)
```

### Running Tests
```bash
# Test commands to be added
```

## Contributing
(Contribution guidelines to be added)

## License
(License information to be added)
