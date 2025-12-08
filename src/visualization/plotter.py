"""Plotting utilities for embedding visualizations."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingPlotter:
    """Create visualizations of embedding spaces."""

    def __init__(self):
        """Initialize plotter with style settings."""
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def plot_2d_scatter(
        self,
        projections: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        texts: List[str] | None = None,
        output_path: str | Path | None = None,
        title: str = "Embedding Space Visualization",
    ) -> None:
        """Create 2D scatter plot of embeddings.

        Args:
            projections: 2D projections (n_samples, 2)
            labels: Class labels (n_samples,)
            class_names: List of class names
            texts: Optional texts for hover tooltips
            output_path: Path to save plot
            title: Plot title
        """
        color_map = {
            "SAFE": "blue",
            "INVESTMENT_ADVICE": "red",
            "INDIRECT_ADVICE": "orange",
            "SYSTEM_PROBE": "black",
            "UNIT_AMBIGUITY": "purple",
        }

        plt.figure(figsize=(14, 10))

        for label_idx, class_name in enumerate(class_names):
            mask = labels == label_idx
            color = color_map.get(class_name, f"C{label_idx}")

            plt.scatter(
                projections[mask, 0],
                projections[mask, 1],
                c=color,
                label=class_name,
                alpha=0.6,
                s=50,
                edgecolors="white",
                linewidth=0.5,
            )

        plt.xlabel("UMAP Dimension 1", fontsize=12)
        plt.ylabel("UMAP Dimension 2", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="best", frameon=True, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved static plot", path=str(output_path))

        plt.show()
        plt.close()

    def plot_interactive_2d(
        self,
        projections: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        texts: List[str],
        output_path: str | Path | None = None,
        title: str = "Interactive Embedding Space",
    ) -> go.Figure:
        """Create interactive 2D scatter plot with Plotly.

        Args:
            projections: 2D projections (n_samples, 2)
            labels: Class labels (n_samples,)
            class_names: List of class names
            texts: Texts for hover tooltips
            output_path: Path to save HTML
            title: Plot title

        Returns:
            Plotly figure object
        """
        color_map = {
            "SAFE": "blue",
            "INVESTMENT_ADVICE": "red",
            "INDIRECT_ADVICE": "orange",
            "SYSTEM_PROBE": "black",
            "UNIT_AMBIGUITY": "purple",
        }

        fig = go.Figure()

        for label_idx, class_name in enumerate(class_names):
            mask = labels == label_idx
            color = color_map.get(class_name, f"rgb({label_idx * 50}, 100, 200)")

            # Truncate texts for hover display
            hover_texts = [
                text[:100] + "..." if len(text) > 100 else text
                for text in np.array(texts)[mask]
            ]

            fig.add_trace(
                go.Scatter(
                    x=projections[mask, 0],
                    y=projections[mask, 1],
                    mode="markers",
                    name=class_name,
                    text=hover_texts,
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                    marker=dict(
                        color=color,
                        size=8,
                        line=dict(width=0.5, color="white"),
                        opacity=0.7,
                    ),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            hovermode="closest",
            template="plotly_white",
            width=1200,
            height=800,
            font=dict(size=12),
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.info(f"Saved interactive plot", path=str(output_path))

        return fig


if __name__ == "__main__":
    # Test plotter
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    # Create dummy data
    np.random.seed(42)
    projections = np.random.rand(100, 2) * 10
    labels = np.random.randint(0, 5, 100)
    class_names = ["SAFE", "INVESTMENT_ADVICE", "INDIRECT_ADVICE", "SYSTEM_PROBE", "UNIT_AMBIGUITY"]
    texts = [f"Example text {i}" for i in range(100)]

    plotter = EmbeddingPlotter()

    # Test static plot
    plotter.plot_2d_scatter(
        projections,
        labels,
        class_names,
        texts,
        output_path="outputs/visualizations/test_scatter.png",
    )

    # Test interactive plot
    fig = plotter.plot_interactive_2d(
        projections,
        labels,
        class_names,
        texts,
        output_path="outputs/visualizations/test_interactive.html",
    )
