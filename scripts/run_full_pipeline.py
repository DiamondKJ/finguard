#!/usr/bin/env python
"""Run complete FinGuard pipeline (all phases)."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Execute all phases sequentially."""
    setup_logging(log_level="INFO")

    phases = [
        ("Phase 1: Dataset Generation", "scripts.run_phase1"),
        ("Phase 2: Embedding Generation", "scripts.run_phase2"),
        ("Phase 3: Visualization", "scripts.run_phase3"),
        ("Phase 4: Classifier Training", "scripts.run_phase4"),
    ]

    for phase_name, module_name in phases:
        logger.info("=" * 80)
        logger.info(f"STARTING: {phase_name}")
        logger.info("=" * 80)

        try:
            # Dynamically import and run phase
            import importlib

            module = importlib.import_module(module_name)
            module.main()

            logger.info(f"✓ {phase_name} completed successfully")

        except Exception as e:
            logger.error(f"✗ {phase_name} failed: {str(e)}", exc_info=True)
            logger.error("Pipeline aborted")
            sys.exit(1)

    logger.info("=" * 80)
    logger.info("ALL PHASES COMPLETE - FINGUARD READY")
    logger.info("=" * 80)
    logger.info("\nRun the demo with: python scripts/run_phase5.py")


if __name__ == "__main__":
    main()
