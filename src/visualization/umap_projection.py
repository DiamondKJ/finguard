"""UMAP dimensionality reduction for embedding visualization."""

from pathlib import Path
from typing import List

import numpy as np
import umap

from src.utils.logger import get_logger

logger = get_logger(__name__)


class UMAPProjector:
    """Project high-dimensional embeddings to 2D using UMAP."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ):
        """Initialize UMAP projector.

        Args:
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            metric: Distance metric ('cosine' or 'euclidean')
            random_state: Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

        self.reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

        logger.info(
            "Initialized UMAP projector",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
        )

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit UMAP and transform embeddings to 2D.

        Args:
            embeddings: High-dimensional embeddings (n_samples, n_features)

        Returns:
            2D projections (n_samples, 2)
        """
        logger.info(f"Projecting embeddings to 2D", input_shape=embeddings.shape)

        projections = self.reducer.fit_transform(embeddings)

        logger.info(f"Projection complete", output_shape=projections.shape)
        return projections

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using fitted UMAP.

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            2D projections
        """
        return self.reducer.transform(embeddings)

    def save_projections(
        self, projections: np.ndarray, output_path: str | Path
    ) -> None:
        """Save 2D projections to file.

        Args:
            projections: 2D projection array
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, projections)
        logger.info(f"Saved projections", path=str(output_path))

    def load_projections(self, input_path: str | Path) -> np.ndarray:
        """Load 2D projections from file.

        Args:
            input_path: Input file path

        Returns:
            2D projection array
        """
        projections = np.load(input_path)
        logger.info(f"Loaded projections", shape=projections.shape)
        return projections


if __name__ == "__main__":
    # Test UMAP projector
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    # Create dummy high-dimensional data
    embeddings = np.random.rand(100, 768).astype(np.float32)

    projector = UMAPProjector()
    projections = projector.fit_transform(embeddings)

    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {projections.shape}")
    print(f"Projection range X: [{projections[:, 0].min():.2f}, {projections[:, 0].max():.2f}]")
    print(f"Projection range Y: [{projections[:, 1].min():.2f}, {projections[:, 1].max():.2f}]")
