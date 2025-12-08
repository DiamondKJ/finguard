#!/usr/bin/env python
"""Run Phase 3: Visualization"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.loader import DatasetLoader
from src.embeddings.cache import EmbeddingCache
from src.utils.config import ensure_directories, get_settings, load_config
from src.utils.logger import get_logger, setup_logging
from src.visualization.plotter import EmbeddingPlotter
from src.visualization.umap_projection import UMAPProjector

logger = get_logger(__name__)


def main() -> None:
    """Visualize embedding space with UMAP."""
    setup_logging(log_level="INFO")
    ensure_directories()

    logger.info("=" * 60)
    logger.info("PHASE 3: VISUALIZATION")
    logger.info("=" * 60)

    # Load dataset
    config = load_config("config/dataset_config.yaml")
    dataset_path = Path(config["dataset"]["processed_path"])

    loader = DatasetLoader()
    dataset = loader.load_json(dataset_path)
    texts, labels = loader.extract_features_labels(dataset)
    class_names = loader.get_class_names(dataset)

    logger.info(f"Loaded dataset with {len(texts)} examples")

    # Load embeddings (try OpenAI first, fallback to Sentence Transformer)
    cache = EmbeddingCache()

    try:
        embeddings, metadata = cache.load_dataset_embeddings(
            "labeled_dataset", "openai_text-embedding-3-small"
        )
        model_name = "OpenAI"
        logger.info(f"Loaded OpenAI embeddings: {embeddings.shape}")
    except FileNotFoundError:
        try:
            embeddings, metadata = cache.load_dataset_embeddings(
                "labeled_dataset", "sentence-transformers_all-mpnet-base-v2"
            )
            model_name = "Sentence Transformer"
            logger.info(f"Loaded Sentence Transformer embeddings: {embeddings.shape}")
        except FileNotFoundError:
            logger.error("No embeddings found. Run Phase 2 first.")
            return

    # UMAP projection
    logger.info("Projecting to 2D with UMAP...")
    projector = UMAPProjector(n_neighbors=15, min_dist=0.1, metric="cosine")
    projections = projector.fit_transform(embeddings)

    # Save projections
    proj_path = Path(f"outputs/visualizations/{model_name.lower()}_projections.npy")
    projector.save_projections(projections, proj_path)

    # Create visualizations
    logger.info("Creating visualizations...")
    plotter = EmbeddingPlotter()

    # Static plot
    static_path = Path(f"outputs/visualizations/embedding_space_{model_name.lower()}.png")
    plotter.plot_2d_scatter(
        projections,
        labels,
        class_names,
        texts,
        output_path=static_path,
        title=f"FinGuard Embedding Space ({model_name})",
    )

    # Interactive plot
    interactive_path = Path(
        f"outputs/visualizations/embedding_space_{model_name.lower()}_interactive.html"
    )
    plotter.plot_interactive_2d(
        projections,
        labels,
        class_names,
        texts,
        output_path=interactive_path,
        title=f"FinGuard Embedding Space ({model_name}) - Interactive",
    )

    # Summary
    logger.info("=" * 60)
    logger.info("PHASE 3 COMPLETE")
    logger.info(f"Projections saved to: {proj_path}")
    logger.info(f"Static plot saved to: {static_path}")
    logger.info(f"Interactive plot saved to: {interactive_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
