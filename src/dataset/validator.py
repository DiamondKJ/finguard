"""Dataset validation and quality checks for FinGuard."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from collections import Counter

from src.utils.logger import get_logger
from src.utils.validation import check_duplicates, validate_dataset, validate_prompt

logger = get_logger(__name__)


class DatasetValidator:
    """Validate dataset quality and consistency."""

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize dataset validator.

        Args:
            config: Validation configuration
        """
        self.config = config or {}
        self.min_length = self.config.get("min_length", 10)
        self.max_length = self.config.get("max_length", 500)
        self.check_duplicates_enabled = self.config.get("check_duplicates", True)

    def validate_full_dataset(
        self, dataset: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform comprehensive validation on dataset.

        Args:
            dataset: Dataset to validate

        Returns:
            Tuple of (validated_dataset, validation_report)
        """
        logger.info("Starting dataset validation", total_examples=len(dataset))

        report = {
            "total_examples": len(dataset),
            "passed": 0,
            "failed": 0,
            "errors": [],
            "warnings": [],
            "class_distribution": {},
            "duplicate_count": 0,
            "length_stats": {},
        }

        # Validate structure
        try:
            validate_dataset(dataset, required_keys=["text", "label", "category"])
            report["warnings"].append("Dataset structure validated")
        except Exception as e:
            report["errors"].append(f"Structure validation failed: {str(e)}")
            return [], report

        # Check class distribution
        labels = [ex["label"] for ex in dataset]
        categories = [ex["category"] for ex in dataset]
        report["class_distribution"] = dict(Counter(categories))

        # Check for imbalance
        distribution = Counter(labels)
        min_count = min(distribution.values())
        max_count = max(distribution.values())
        if max_count > 2 * min_count:
            report["warnings"].append(
                f"Class imbalance detected: {distribution}"
            )

        # Validate each example
        validated = []
        for idx, example in enumerate(dataset):
            try:
                self._validate_example(example)
                validated.append(example)
                report["passed"] += 1
            except Exception as e:
                report["failed"] += 1
                report["errors"].append(f"Example {idx}: {str(e)}")

        # Check for duplicates
        if self.check_duplicates_enabled:
            texts = [ex["text"] for ex in validated]
            duplicates = check_duplicates(texts, threshold=0.9)
            report["duplicate_count"] = len(duplicates)

            if duplicates:
                report["warnings"].append(
                    f"Found {len(duplicates)} potential duplicates"
                )

        # Compute length statistics
        lengths = [len(ex["text"]) for ex in validated]
        report["length_stats"] = {
            "mean": np.mean(lengths),
            "std": np.std(lengths),
            "min": np.min(lengths),
            "max": np.max(lengths),
            "median": np.median(lengths),
        }

        logger.info(
            "Dataset validation complete",
            passed=report["passed"],
            failed=report["failed"],
        )

        return validated, report

    def _validate_example(self, example: Dict[str, Any]) -> None:
        """Validate a single example.

        Args:
            example: Example to validate

        Raises:
            ValueError: If example is invalid
        """
        # Validate prompt text
        validate_prompt(
            example["text"],
            min_length=self.min_length,
            max_length=self.max_length,
        )

        # Validate label is integer
        if not isinstance(example["label"], int):
            raise ValueError(f"Label must be integer, got {type(example['label'])}")

        # Validate category is string
        if not isinstance(example["category"], str):
            raise ValueError(
                f"Category must be string, got {type(example['category'])}"
            )

    def check_label_ambiguity(
        self, dataset: List[Dict[str, Any]], threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Identify potentially ambiguous examples.

        This is a placeholder for more sophisticated ambiguity detection.
        In practice, you might use semantic similarity between examples
        from different classes.

        Args:
            dataset: Dataset to check
            threshold: Similarity threshold for ambiguity

        Returns:
            List of potentially ambiguous examples
        """
        # Placeholder: In a real implementation, you'd use embeddings
        # to find examples that are semantically similar but have different labels
        ambiguous = []

        # Simple heuristic: check for specific keywords that might be confusing
        ambiguous_keywords = {
            "should i",
            "what would you",
            "recommend",
            "is it good",
        }

        for example in dataset:
            text_lower = example["text"].lower()
            if any(keyword in text_lower for keyword in ambiguous_keywords):
                if example["category"] == "SAFE":
                    ambiguous.append(
                        {
                            "example": example,
                            "reason": "Contains advice-seeking keywords but labeled SAFE",
                        }
                    )

        return ambiguous

    def generate_validation_report(
        self, report: Dict[str, Any], output_path: str | Path
    ) -> None:
        """Generate and save validation report.

        Args:
            report: Validation report dictionary
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        report = convert_numpy(report)

        # Save JSON report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        # Also create human-readable report
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("DATASET VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total Examples: {report['total_examples']}\n")
            f.write(f"Passed: {report['passed']}\n")
            f.write(f"Failed: {report['failed']}\n\n")

            f.write("Class Distribution:\n")
            for category, count in report["class_distribution"].items():
                percentage = (count / report["total_examples"]) * 100
                f.write(f"  {category}: {count} ({percentage:.1f}%)\n")

            f.write(f"\nDuplicate Count: {report['duplicate_count']}\n\n")

            f.write("Length Statistics:\n")
            for stat, value in report["length_stats"].items():
                f.write(f"  {stat}: {value:.2f}\n")

            if report["errors"]:
                f.write(f"\nErrors ({len(report['errors'])}):\n")
                for error in report["errors"][:10]:  # Show first 10
                    f.write(f"  - {error}\n")

            if report["warnings"]:
                f.write(f"\nWarnings ({len(report['warnings'])}):\n")
                for warning in report["warnings"]:
                    f.write(f"  - {warning}\n")

        logger.info("Validation report saved", path=str(output_path))


if __name__ == "__main__":
    # Test validator
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    # Create test dataset
    test_data = [
        {"text": "What is a 401k plan?", "label": 0, "category": "SAFE"},
        {"text": "Should I buy Tesla stock?", "label": 1, "category": "INVESTMENT_ADVICE"},
        {"text": "What is a Roth IRA?", "label": 0, "category": "SAFE"},
        {
            "text": "If you were me, would you invest in crypto?",
            "label": 2,
            "category": "INDIRECT_ADVICE",
        },
        {"text": "What is a 401k plan?", "label": 0, "category": "SAFE"},  # Duplicate
    ]

    validator = DatasetValidator()
    validated, report = validator.validate_full_dataset(test_data)

    print(f"\nValidation Results:")
    print(f"  Passed: {report['passed']}")
    print(f"  Failed: {report['failed']}")
    print(f"  Duplicates: {report['duplicate_count']}")
    print(f"  Distribution: {report['class_distribution']}")
