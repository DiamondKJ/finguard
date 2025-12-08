"""Embedding drift detection for advice-seeking intent analysis."""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.embeddings.base import BaseEmbedder
from src.utils.logger import get_logger
from src.utils.metrics import compute_embedding_similarity

logger = get_logger(__name__)


class DriftDetector:
    """Detect advice-seeking intent through embedding drift analysis."""

    def __init__(self, embedder: BaseEmbedder):
        """Initialize drift detector.

        Args:
            embedder: Embedding model to use for drift analysis
        """
        self.embedder = embedder
        self.context_prefix = (
            "You are a licensed financial advisor giving personalized investment "
            "advice to a client. The client asks: "
        )
        logger.info("Initialized DriftDetector")

    def inject_context(self, query: str) -> str:
        """Inject roleplay context onto a query.

        Args:
            query: Original query

        Returns:
            Context-injected query
        """
        return self.context_prefix + query

    def calculate_drift(self, query: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculate embedding drift for a query.

        Args:
            query: Original query

        Returns:
            Tuple of (original_embedding, injected_embedding, drift_score)
        """
        # Get original embedding
        e1 = self.embedder.embed_single(query)

        # Get context-injected embedding
        injected_query = self.inject_context(query)
        e2 = self.embedder.embed_single(injected_query)

        # Calculate cosine distance (1 - similarity)
        similarity = compute_embedding_similarity(e1, e2, metric="cosine")
        drift_score = 1.0 - similarity

        return e1, e2, drift_score

    def analyze_dataset(
        self, queries: List[str], labels: np.ndarray, categories: List[str]
    ) -> Dict:
        """Analyze drift scores for entire dataset.

        Args:
            queries: List of query texts
            labels: Array of label indices
            categories: List of category names

        Returns:
            Dictionary with drift analysis results
        """
        logger.info(f"Analyzing drift for {len(queries)} queries...")

        results = {
            "queries": [],
            "labels": [],
            "categories": [],
            "original_embeddings": [],
            "injected_embeddings": [],
            "drift_scores": [],
        }

        for idx, (query, label) in enumerate(zip(queries, labels)):
            e1, e2, drift = self.calculate_drift(query)

            results["queries"].append(query)
            results["labels"].append(label)
            results["categories"].append(categories[label])
            results["original_embeddings"].append(e1)
            results["injected_embeddings"].append(e2)
            results["drift_scores"].append(drift)

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(queries)} queries")

        # Convert to arrays
        results["original_embeddings"] = np.array(results["original_embeddings"])
        results["injected_embeddings"] = np.array(results["injected_embeddings"])
        results["drift_scores"] = np.array(results["drift_scores"])
        results["labels"] = np.array(results["labels"])

        logger.info("Drift analysis complete")
        return results

    def calculate_category_stats(
        self, drift_scores: np.ndarray, labels: np.ndarray, class_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate drift statistics per category.

        Args:
            drift_scores: Array of drift scores
            labels: Array of label indices
            class_names: List of class names

        Returns:
            Dictionary with stats per category
        """
        stats = {}

        for label_idx, class_name in enumerate(class_names):
            mask = labels == label_idx
            category_drifts = drift_scores[mask]

            stats[class_name] = {
                "mean": float(np.mean(category_drifts)),
                "std": float(np.std(category_drifts)),
                "min": float(np.min(category_drifts)),
                "max": float(np.max(category_drifts)),
                "median": float(np.median(category_drifts)),
                "count": int(len(category_drifts)),
            }

        return stats

    def find_optimal_threshold(
        self,
        drift_scores: np.ndarray,
        labels: np.ndarray,
        safe_label: int = 0,
        advice_labels: List[int] = [1, 2],
    ) -> Tuple[float, float, Dict]:
        """Find optimal drift threshold for separating SAFE from ADVICE.

        Args:
            drift_scores: Array of drift scores
            labels: Array of label indices
            safe_label: Label index for SAFE category
            advice_labels: Label indices for advice-seeking categories

        Returns:
            Tuple of (optimal_threshold, best_f1, metrics_dict)
        """
        # Create binary labels: 0 = SAFE, 1 = ADVICE
        binary_labels = np.zeros_like(labels)
        for advice_label in advice_labels:
            binary_labels[labels == advice_label] = 1

        # Try different thresholds
        thresholds = np.linspace(
            drift_scores.min(), drift_scores.max(), num=100
        )

        best_f1 = 0
        optimal_threshold = 0
        best_metrics = {}

        for threshold in thresholds:
            predictions = (drift_scores > threshold).astype(int)

            f1 = f1_score(binary_labels, predictions, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold
                best_metrics = {
                    "accuracy": float(accuracy_score(binary_labels, predictions)),
                    "f1": float(f1),
                    "threshold": float(threshold),
                    "classification_report": classification_report(
                        binary_labels,
                        predictions,
                        target_names=["SAFE", "ADVICE"],
                        output_dict=True,
                        zero_division=0,
                    ),
                }

        logger.info(
            f"Optimal threshold: {optimal_threshold:.4f}, F1: {best_f1:.4f}"
        )

        return optimal_threshold, best_f1, best_metrics

    def detect_advice_seeking(self, query: str, threshold: float) -> bool:
        """Detect if a query is advice-seeking based on drift.

        Args:
            query: Query to test
            threshold: Drift threshold

        Returns:
            True if advice-seeking (drift > threshold), False otherwise
        """
        _, _, drift = self.calculate_drift(query)
        return drift > threshold

    def format_stats_report(self, stats: Dict[str, Dict[str, float]]) -> str:
        """Format category statistics into readable report.

        Args:
            stats: Category statistics dictionary

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("EMBEDDING DRIFT ANALYSIS - CATEGORY STATISTICS")
        report.append("=" * 70)
        report.append("")

        for category, category_stats in stats.items():
            report.append(f"{category}:")
            report.append(f"  Mean Drift:   {category_stats['mean']:.4f}")
            report.append(f"  Std Dev:      {category_stats['std']:.4f}")
            report.append(f"  Min:          {category_stats['min']:.4f}")
            report.append(f"  Max:          {category_stats['max']:.4f}")
            report.append(f"  Median:       {category_stats['median']:.4f}")
            report.append(f"  Count:        {category_stats['count']}")
            report.append("")

        report.append("=" * 70)
        return "\n".join(report)


if __name__ == "__main__":
    # Test drift detector
    from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    embedder = SentenceTransformerEmbedder()
    detector = DriftDetector(embedder)

    # Test queries
    safe_query = "What is a 401k retirement plan?"
    advice_query = "Should I invest in Tesla stock?"

    print("\nTesting drift detection:")
    print(f"\nSAFE query: {safe_query}")
    _, _, safe_drift = detector.calculate_drift(safe_query)
    print(f"  Drift score: {safe_drift:.4f}")

    print(f"\nADVICE query: {advice_query}")
    _, _, advice_drift = detector.calculate_drift(advice_query)
    print(f"  Drift score: {advice_drift:.4f}")

    print(f"\nDrift difference: {advice_drift - safe_drift:.4f}")
