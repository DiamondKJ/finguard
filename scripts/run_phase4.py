#!/usr/bin/env python
"""Run Phase 4: Classifier Training"""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.classifier.random_forest_classifier import RandomForestClassifier
from src.dataset.loader import DatasetLoader
from src.embeddings.cache import EmbeddingCache
from src.utils.config import ensure_directories, get_settings, load_config
from src.utils.logger import get_logger, setup_logging
from src.utils.metrics import MetricsCalculator

logger = get_logger(__name__)


def main() -> None:
    """Train and evaluate classifier."""
    setup_logging(log_level="INFO")
    ensure_directories()

    logger.info("=" * 60)
    logger.info("PHASE 4: CLASSIFIER TRAINING")
    logger.info("=" * 60)

    # Load dataset
    dataset_config = load_config("config/dataset_config.yaml")
    training_config = load_config("config/training_config.yaml")

    dataset_path = Path(dataset_config["dataset"]["processed_path"])

    loader = DatasetLoader()
    dataset = loader.load_json(dataset_path)
    texts, labels = loader.extract_features_labels(dataset)
    class_names = loader.get_class_names(dataset)

    logger.info(f"Loaded dataset with {len(texts)} examples")

    # Load embeddings (prefer Sentence Transformer for faster local training)
    cache = EmbeddingCache()

    try:
        embeddings, metadata = cache.load_dataset_embeddings(
            "labeled_dataset", "sentence-transformers_all-mpnet-base-v2"
        )
        logger.info(f"Loaded Sentence Transformer embeddings: {embeddings.shape}")
    except FileNotFoundError:
        try:
            embeddings, metadata = cache.load_dataset_embeddings(
                "labeled_dataset", "openai_text-embedding-3-small"
            )
            logger.info(f"Loaded OpenAI embeddings: {embeddings.shape}")
        except FileNotFoundError:
            logger.error("No embeddings found. Run Phase 2 first.")
            return

    # Split dataset
    test_size = training_config["training"]["test_size"]
    random_seed = training_config["training"]["random_seed"]

    # Split by indices to maintain alignment
    from sklearn.model_selection import train_test_split

    indices = list(range(len(embeddings)))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_seed, stratify=labels
    )

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train classifier
    logger.info("Training Random Forest classifier...")
    rf_config = training_config["classifiers"]["random_forest"]

    classifier = RandomForestClassifier(
        n_estimators=rf_config["n_estimators"],
        max_depth=rf_config["max_depth"],
        min_samples_split=rf_config["min_samples_split"],
        min_samples_leaf=rf_config["min_samples_leaf"],
        max_features=rf_config["max_features"],
        class_weight=rf_config["class_weight"],
        random_state=rf_config["random_state"],
        n_jobs=rf_config["n_jobs"],
    )

    classifier.fit(X_train, y_train, class_names=class_names)

    # Evaluate
    logger.info("Evaluating classifier...")
    metrics = classifier.evaluate(X_test, y_test)

    # Save metrics
    metrics_path = Path(training_config["evaluation"]["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    model_path = Path(training_config["evaluation"]["output"]["model_path"])
    classifier.save(model_path)

    # Print results
    calculator = MetricsCalculator(class_names)
    report = calculator.format_metrics_report(metrics)
    print("\n" + report)

    # Check thresholds
    thresholds = training_config["thresholds"]
    passed = True

    if metrics["accuracy"] < thresholds["min_accuracy"]:
        logger.warning(
            f"Accuracy {metrics['accuracy']:.3f} below threshold {thresholds['min_accuracy']}"
        )
        passed = False

    if metrics["f1_score"] < thresholds["min_f1_macro"]:
        logger.warning(
            f"F1 {metrics['f1_score']:.3f} below threshold {thresholds['min_f1_macro']}"
        )
        passed = False

    # Summary
    logger.info("=" * 60)
    logger.info("PHASE 4 COMPLETE")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    logger.info(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"Threshold check: {'PASSED' if passed else 'FAILED'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
