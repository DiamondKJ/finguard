#!/usr/bin/env python
"""Run Phase 5: Demo and Benchmarks"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.demo.cli_demo import main as run_demo
from src.utils.logger import setup_logging

if __name__ == "__main__":
    setup_logging(log_level="INFO")
    run_demo()
