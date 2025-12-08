#!/usr/bin/env python
"""Run Phase 1: Dataset Generation"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.generator import PromptGenerator
from src.dataset.validator import DatasetValidator
from src.utils.config import ensure_directories, get_settings, load_config
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Generate and validate dataset."""
    setup_logging(log_level="INFO")
    ensure_directories()

    logger.info("=" * 60)
    logger.info("PHASE 1: DATASET GENERATION")
    logger.info("=" * 60)

    # Load configuration
    config = load_config("config/dataset_config.yaml")
    output_path = Path(config["dataset"]["output_path"])

    # Initialize generator
    logger.info("Initializing prompt generator...")
    generator = PromptGenerator()

    # Generate dataset
    logger.info("Generating full dataset...")
    dataset = generator.generate_full_dataset()

    # Save raw dataset
    generator.save_dataset(dataset, output_path)

    # Validate dataset
    logger.info("Validating dataset...")
    validator = DatasetValidator(config["validation"])
    validated_dataset, report = validator.validate_full_dataset(dataset)

    # Save validation report
    report_path = Path("outputs/reports/dataset_validation_report.json")
    validator.generate_validation_report(report, report_path)

    # Save processed dataset
    processed_path = Path(config["dataset"]["processed_path"])
    generator.save_dataset(validated_dataset, processed_path)

    # Summary
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info(f"Total examples: {report['total_examples']}")
    logger.info(f"Passed validation: {report['passed']}")
    logger.info(f"Failed validation: {report['failed']}")
    logger.info(f"Duplicates found: {report['duplicate_count']}")
    logger.info(f"Dataset saved to: {processed_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
