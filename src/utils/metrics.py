"""Metrics computation utilities for FinGuard."""

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


class MetricsCalculator:
    """Calculate and format classification metrics."""

    def __init__(self, class_names: List[str]):
        """Initialize metrics calculator.

        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names

    def compute_all_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {
            "accuracy": self.compute_accuracy(y_true, y_pred),
            "precision": self.compute_precision(y_true, y_pred),
            "recall": self.compute_recall(y_true, y_pred),
            "f1_score": self.compute_f1(y_true, y_pred),
            "confusion_matrix": self.compute_confusion_matrix(y_true, y_pred),
            "classification_report": self.compute_classification_report(y_true, y_pred),
            "per_class_metrics": self.compute_per_class_metrics(y_true, y_pred),
            "false_positive_rate": self.compute_false_positive_rate(y_true, y_pred),
            "false_negative_rate": self.compute_false_negative_rate(y_true, y_pred),
        }
        return metrics

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy score.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Accuracy score
        """
        return float(accuracy_score(y_true, y_pred))

    def compute_precision(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
    ) -> float:
        """Compute precision score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method ('micro', 'macro', 'weighted')

        Returns:
            Precision score
        """
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))

    def compute_recall(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
    ) -> float:
        """Compute recall score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method

        Returns:
            Recall score
        """
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))

    def compute_f1(
        self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
    ) -> float:
        """Compute F1 score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method

        Returns:
            F1 score
        """
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))

    def compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> List[List[int]]:
        """Compute confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix as list of lists
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm.tolist()

    def compute_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report as dictionary
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

    def compute_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each class individually.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with per-class precision, recall, f1
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )

        per_class = {}
        for idx, class_name in enumerate(self.class_names):
            per_class[class_name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx]),
            }

        return per_class

    def compute_false_positive_rate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compute overall false positive rate.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            False positive rate
        """
        cm = confusion_matrix(y_true, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
        fpr = fp.sum() / (fp.sum() + tn.sum()) if (fp.sum() + tn.sum()) > 0 else 0.0
        return float(fpr)

    def compute_false_negative_rate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compute overall false negative rate.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            False negative rate
        """
        cm = confusion_matrix(y_true, y_pred)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        fnr = fn.sum() / (fn.sum() + tp.sum()) if (fn.sum() + tp.sum()) > 0 else 0.0
        return float(fnr)

    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a human-readable report.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Formatted metrics report string
        """
        report = []
        report.append("=" * 60)
        report.append("CLASSIFICATION METRICS REPORT")
        report.append("=" * 60)
        report.append(f"Accuracy: {metrics['accuracy']:.4f}")
        report.append(f"Precision (weighted): {metrics['precision']:.4f}")
        report.append(f"Recall (weighted): {metrics['recall']:.4f}")
        report.append(f"F1 Score (weighted): {metrics['f1_score']:.4f}")
        report.append(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
        report.append(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
        report.append("")
        report.append("Per-Class Metrics:")
        report.append("-" * 60)

        for class_name, class_metrics in metrics["per_class_metrics"].items():
            report.append(f"\n{class_name}:")
            report.append(f"  Precision: {class_metrics['precision']:.4f}")
            report.append(f"  Recall:    {class_metrics['recall']:.4f}")
            report.append(f"  F1 Score:  {class_metrics['f1_score']:.4f}")
            report.append(f"  Support:   {class_metrics['support']}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


def compute_embedding_similarity(
    embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine"
) -> float:
    """Compute similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        Similarity score
    """
    if metric == "cosine":
        return float(
            np.dot(embedding1, embedding2)
            / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        )
    elif metric == "euclidean":
        return float(np.linalg.norm(embedding1 - embedding2))
    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    # Test metrics calculator
    class_names = ["SAFE", "INVESTMENT_ADVICE", "INDIRECT_ADVICE", "SYSTEM_PROBE", "UNIT_AMBIGUITY"]
    calculator = MetricsCalculator(class_names)

    # Dummy predictions
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 3, 0, 1, 2, 3, 4])  # One error

    metrics = calculator.compute_all_metrics(y_true, y_pred)
    print(calculator.format_metrics_report(metrics))
