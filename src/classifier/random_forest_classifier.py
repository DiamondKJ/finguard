"""Random Forest classifier for FinGuard."""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier

from src.utils.logger import get_logger
from src.utils.metrics import MetricsCalculator
from src.utils.validation import validate_embedding, validate_labels

logger = get_logger(__name__)


class RandomForestClassifier:
    """Random Forest classifier for prompt classification."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        class_weight: str | Dict = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """Initialize Random Forest classifier.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features for best split
            class_weight: Class weights ('balanced' or dict)
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.model = SKRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.class_names = None
        self.is_fitted = False

        logger.info(
            "Initialized RandomForestClassifier",
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        class_names: list[str] | None = None,
    ) -> None:
        """Train the classifier.

        Args:
            X_train: Training embeddings (n_samples, n_features)
            y_train: Training labels (n_samples,)
            class_names: Optional class names for interpretability
        """
        logger.info(f"Training classifier", X_shape=X_train.shape, y_shape=y_train.shape)

        # Validate inputs
        validate_labels(y_train, num_classes=5)

        # Train model
        self.model.fit(X_train, y_train)
        self.class_names = class_names
        self.is_fitted = True

        logger.info("Training complete", n_classes=len(np.unique(y_train)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Embeddings to classify (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Embeddings to classify (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        probabilities = self.model.predict_proba(X)
        return probabilities

    def predict_single(self, embedding: np.ndarray) -> tuple[int, float]:
        """Predict class for single embedding.

        Args:
            embedding: Single embedding vector

        Returns:
            Tuple of (predicted_label, confidence)
        """
        # Ensure 2D array
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        label = self.predict(embedding)[0]
        probabilities = self.predict_proba(embedding)[0]
        confidence = float(probabilities[label])

        return int(label), confidence

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate classifier on test set.

        Args:
            X_test: Test embeddings
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating classifier", test_size=len(y_test))

        y_pred = self.predict(X_test)

        calculator = MetricsCalculator(self.class_names or [f"Class_{i}" for i in range(5)])
        metrics = calculator.compute_all_metrics(y_test, y_pred)

        logger.info(
            "Evaluation complete",
            accuracy=metrics["accuracy"],
            f1_score=metrics["f1_score"],
        )

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.

        Returns:
            Feature importance array

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.feature_importances_

    def save(self, model_path: str | Path) -> None:
        """Save trained model to disk.

        Args:
            model_path: Path to save model
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "class_names": self.class_names,
            "is_fitted": self.is_fitted,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved", path=str(model_path))

    def load(self, model_path: str | Path) -> None:
        """Load trained model from disk.

        Args:
            model_path: Path to model file

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.class_names = model_data["class_names"]
        self.is_fitted = model_data["is_fitted"]

        logger.info(f"Model loaded", path=str(model_path))


if __name__ == "__main__":
    # Test classifier
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    # Create dummy training data
    X_train = np.random.rand(100, 768).astype(np.float32)
    y_train = np.random.randint(0, 5, 100)

    class_names = ["SAFE", "INVESTMENT_ADVICE", "INDIRECT_ADVICE", "SYSTEM_PROBE", "UNIT_AMBIGUITY"]

    # Train classifier
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, y_train, class_names=class_names)

    # Test prediction
    X_test = np.random.rand(20, 768).astype(np.float32)
    y_test = np.random.randint(0, 5, 20)

    predictions = clf.predict(X_test)
    print(f"Predictions: {predictions[:5]}")

    # Evaluate
    metrics = clf.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.3f}")

    # Test single prediction
    single_embedding = np.random.rand(768).astype(np.float32)
    label, confidence = clf.predict_single(single_embedding)
    print(f"\nSingle prediction: {class_names[label]} ({confidence:.2%} confidence)")

    # Save and load
    clf.save("models/test_model.pkl")
    clf_loaded = RandomForestClassifier()
    clf_loaded.load("models/test_model.pkl")
    print(f"Model loaded successfully: {clf_loaded.is_fitted}")
