"""Visualization utilities for drift analysis."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftVisualizer:
    """Create visualizations for embedding drift analysis."""

    def __init__(self):
        """Initialize visualizer."""
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (14, 10)

    def plot_drift_distributions(
        self,
        drift_scores: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        output_path: str | Path,
    ) -> None:
        """Create box plot and violin plot of drift distributions.

        Args:
            drift_scores: Array of drift scores
            labels: Array of label indices
            class_names: List of class names
            output_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Prepare data for plotting
        plot_data = []
        plot_labels = []
        for label_idx, class_name in enumerate(class_names):
            mask = labels == label_idx
            category_drifts = drift_scores[mask]
            plot_data.append(category_drifts)
            plot_labels.append(class_name)

        # Color scheme
        colors = ["blue", "red", "orange", "black", "purple"]

        # Box plot
        bp = ax1.boxplot(
            plot_data,
            labels=plot_labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="yellow", markersize=8),
        )

        for patch, color in zip(bp["boxes"], colors[: len(plot_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax1.set_ylabel("Drift Score", fontsize=12)
        ax1.set_title("Drift Score Distribution by Category (Box Plot)", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Violin plot
        parts = ax2.violinplot(
            plot_data,
            positions=range(len(plot_data)),
            showmeans=True,
            showmedians=True,
        )

        for pc, color in zip(parts["bodies"], colors[: len(plot_data)]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        ax2.set_xticks(range(len(plot_labels)))
        ax2.set_xticklabels(plot_labels, rotation=45, ha="right")
        ax2.set_ylabel("Drift Score", fontsize=12)
        ax2.set_title("Drift Score Distribution by Category (Violin Plot)", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved drift distribution plot: {output_path}")
        plt.close()

    def plot_drift_histogram(
        self,
        drift_scores: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        output_path: str | Path,
        optimal_threshold: float | None = None,
    ) -> None:
        """Create histogram of drift scores with category overlay.

        Args:
            drift_scores: Array of drift scores
            labels: Array of label indices
            class_names: List of class names
            output_path: Path to save plot
            optimal_threshold: Optional threshold line to draw
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = ["blue", "red", "orange", "black", "purple"]

        for label_idx, class_name in enumerate(class_names):
            mask = labels == label_idx
            category_drifts = drift_scores[mask]

            ax.hist(
                category_drifts,
                bins=30,
                alpha=0.5,
                label=class_name,
                color=colors[label_idx],
                edgecolor="white",
                linewidth=0.5,
            )

        if optimal_threshold is not None:
            ax.axvline(
                optimal_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Optimal Threshold: {optimal_threshold:.4f}",
            )

        ax.set_xlabel("Drift Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Drift Score Histogram - All Categories", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved drift histogram: {output_path}")
        plt.close()

    def plot_umap_with_drift(
        self,
        umap_projections: np.ndarray,
        drift_scores: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        output_path: str | Path,
    ) -> None:
        """Create UMAP scatter plot colored by drift score.

        Args:
            umap_projections: 2D UMAP projections
            drift_scores: Array of drift scores
            labels: Array of label indices
            class_names: List of class names
            output_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Left: Original UMAP colored by category
        colors = ["blue", "red", "orange", "black", "purple"]
        for label_idx, class_name in enumerate(class_names):
            mask = labels == label_idx
            ax1.scatter(
                umap_projections[mask, 0],
                umap_projections[mask, 1],
                c=colors[label_idx],
                label=class_name,
                alpha=0.6,
                s=50,
                edgecolors="white",
                linewidth=0.5,
            )

        ax1.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax1.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax1.set_title("Original UMAP - Colored by Category", fontsize=14, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Right: UMAP colored by drift score (heatmap style)
        scatter = ax2.scatter(
            umap_projections[:, 0],
            umap_projections[:, 1],
            c=drift_scores,
            cmap="coolwarm",
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Drift Score", fontsize=12)

        ax2.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax2.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax2.set_title("UMAP - Colored by Drift Score", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved UMAP drift visualization: {output_path}")
        plt.close()

    def plot_drift_comparison(
        self,
        stats: dict,
        output_path: str | Path,
    ) -> None:
        """Create bar chart comparing mean drift across categories.

        Args:
            stats: Category statistics dictionary
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        categories = list(stats.keys())
        means = [stats[cat]["mean"] for cat in categories]
        stds = [stats[cat]["std"] for cat in categories]

        colors = ["blue", "red", "orange", "black", "purple"]

        bars = ax.bar(
            categories,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.7,
            color=colors[: len(categories)],
            edgecolor="white",
            linewidth=1.5,
        )

        ax.set_ylabel("Mean Drift Score", fontsize=12)
        ax.set_title("Mean Embedding Drift by Category", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{mean:.4f}\n±{std:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved drift comparison plot: {output_path}")
        plt.close()


if __name__ == "__main__":
    # Test visualizer
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    visualizer = DriftVisualizer()

    # Create dummy data
    np.random.seed(42)
    n_samples = 500
    drift_scores = np.random.rand(n_samples) * 0.5
    labels = np.random.randint(0, 5, n_samples)
    class_names = ["SAFE", "INVESTMENT_ADVICE", "INDIRECT_ADVICE", "SYSTEM_PROBE", "UNIT_AMBIGUITY"]

    # Test plots
    visualizer.plot_drift_distributions(
        drift_scores, labels, class_names, "outputs/test_drift_dist.png"
    )
    print("Test visualization created")
