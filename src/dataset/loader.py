"""Data loading utilities for FinGuard."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.validation import validate_dataset, validate_labels

logger = get_logger(__name__)


class DatasetLoader:
    """Load and prepare datasets for training and evaluation."""

    def __init__(self):
        """Initialize dataset loader."""
        pass

    def load_json(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """Load dataset from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of examples

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading dataset from JSON", path=str(file_path))

        with open(file_path, "r") as f:
            data = json.load(f)

        validate_dataset(data, required_keys=["text", "label", "category"])

        logger.info(f"Loaded dataset", count=len(data))
        return data

    def save_json(self, data: List[Dict[str, Any]], file_path: str | Path) -> None:
        """Save dataset to JSON file.

        Args:
            data: Dataset to save
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved dataset", path=str(file_path), count=len(data))

    def to_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame.

        Args:
            data: Dataset as list of dictionaries

        Returns:
            DataFrame with columns: text, label, category
        """
        df = pd.DataFrame(data)
        logger.info(f"Converted to DataFrame", shape=df.shape)
        return df

    def from_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to dataset format.

        Args:
            df: DataFrame with text, label, category columns

        Returns:
            List of dictionaries
        """
        data = df.to_dict("records")
        logger.info(f"Converted from DataFrame", count=len(data))
        return data

    def split_dataset(
        self,
        data: List[Dict[str, Any]],
        test_size: float = 0.2,
        val_size: float = 0.0,
        random_seed: int = 42,
        stratify: bool = True,
    ) -> Tuple[List[Dict[str, Any]], ...]:
        """Split dataset into train/test or train/val/test.

        Args:
            data: Dataset to split
            test_size: Proportion for test set
            val_size: Proportion for validation set (from training data)
            random_seed: Random seed for reproducibility
            stratify: Whether to stratify split by labels

        Returns:
            Tuple of (train, test) or (train, val, test) datasets
        """
        labels = [ex["label"] for ex in data]
        stratify_labels = labels if stratify else None

        # First split: train+val vs test
        train_val, test = train_test_split(
            data,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_labels,
        )

        logger.info(
            f"Split dataset",
            total=len(data),
            train_val=len(train_val),
            test=len(test),
        )

        # Second split: train vs val (if needed)
        if val_size > 0:
            train_val_labels = [ex["label"] for ex in train_val]
            stratify_labels = train_val_labels if stratify else None

            # Adjust val_size to be proportion of train_val
            adjusted_val_size = val_size / (1 - test_size)

            train, val = train_test_split(
                train_val,
                test_size=adjusted_val_size,
                random_state=random_seed,
                stratify=stratify_labels,
            )

            logger.info(f"Created validation split", train=len(train), val=len(val))
            return train, val, test

        return train_val, test

    def extract_features_labels(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[List[str], np.ndarray]:
        """Extract texts and labels from dataset.

        Args:
            data: Dataset

        Returns:
            Tuple of (texts, labels)
        """
        texts = [ex["text"] for ex in data]
        labels = np.array([ex["label"] for ex in data])

        validate_labels(labels, num_classes=5)

        return texts, labels

    def get_class_names(self, data: List[Dict[str, Any]]) -> List[str]:
        """Get ordered list of class names.

        Args:
            data: Dataset

        Returns:
            List of class names ordered by label
        """
        # Create label -> category mapping
        label_to_category = {}
        for ex in data:
            label_to_category[ex["label"]] = ex["category"]

        # Sort by label
        class_names = [label_to_category[i] for i in sorted(label_to_category.keys())]

        logger.info(f"Class names", classes=class_names)
        return class_names

    def get_dataset_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistical summary of dataset.

        Args:
            data: Dataset

        Returns:
            Dictionary of statistics
        """
        texts = [ex["text"] for ex in data]
        labels = [ex["label"] for ex in data]
        categories = [ex["category"] for ex in data]

        from collections import Counter

        stats = {
            "total_examples": len(data),
            "num_classes": len(set(labels)),
            "class_distribution": dict(Counter(categories)),
            "text_lengths": {
                "mean": np.mean([len(t) for t in texts]),
                "std": np.std([len(t) for t in texts]),
                "min": min([len(t) for t in texts]),
                "max": max([len(t) for t in texts]),
            },
            "label_distribution": dict(Counter(labels)),
        }

        return stats


if __name__ == "__main__":
    # Test loader
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    loader = DatasetLoader()

    # Create test dataset
    test_data = [
        {"text": f"Example {i}", "label": i % 5, "category": f"CAT_{i%5}"}
        for i in range(100)
    ]

    # Save and load
    test_path = Path("data/test_dataset.json")
    loader.save_json(test_data, test_path)
    loaded = loader.load_json(test_path)

    print(f"Loaded {len(loaded)} examples")

    # Split dataset
    train, test = loader.split_dataset(loaded, test_size=0.2)
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Get stats
    stats = loader.get_dataset_stats(loaded)
    print(f"Dataset stats: {stats}")

    # Cleanup
    test_path.unlink()
