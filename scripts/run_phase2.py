#!/usr/bin/env python
"""Run Phase 2: Embedding Generation"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.dataset.loader import DatasetLoader
from src.embeddings.cache import EmbeddingCache
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.utils.config import ensure_directories, get_settings, load_config
from src.utils.logger import get_logger, setup_logging
from src.utils.metrics import compute_embedding_similarity

logger = get_logger(__name__)


def main() -> None:
    """Generate embeddings using multiple models."""
    setup_logging(log_level="INFO")
    ensure_directories()

    logger.info("=" * 60)
    logger.info("PHASE 2: EMBEDDING GENERATION")
    logger.info("=" * 60)

    # Load dataset
    config = load_config("config/dataset_config.yaml")
    dataset_path = Path(config["dataset"]["processed_path"])

    loader = DatasetLoader()
    dataset = loader.load_json(dataset_path)
    texts, labels = loader.extract_features_labels(dataset)

    logger.info(f"Loaded dataset with {len(texts)} examples")

    # Initialize cache
    cache = EmbeddingCache()

    # Generate embeddings with multiple models
    embedders = []

    # 1. OpenAI embeddings (if API key available)
    try:
        logger.info("Generating OpenAI embeddings...")
        openai_embedder = OpenAIEmbedder()
        openai_embeddings = openai_embedder.embed_batch(texts)
        cache.save_dataset_embeddings(
            "labeled_dataset",
            openai_embeddings,
            "openai_text-embedding-3-small",
        )
        embedders.append(("OpenAI", openai_embeddings))
        logger.info(f"OpenAI embeddings complete: {openai_embeddings.shape}")
    except ValueError as e:
        logger.warning(f"Skipping OpenAI embeddings: {e}")
    except Exception as e:
        logger.error(f"Error with OpenAI embeddings: {e}")

    # 2. Sentence Transformer embeddings (always available)
    try:
        logger.info("Generating Sentence Transformer embeddings...")
        st_embedder = SentenceTransformerEmbedder()
        st_embeddings = st_embedder.embed_batch(texts)
        cache.save_dataset_embeddings(
            "labeled_dataset",
            st_embeddings,
            "sentence-transformers_all-mpnet-base-v2",
        )
        embedders.append(("Sentence Transformer", st_embeddings))
        logger.info(f"Sentence Transformer embeddings complete: {st_embeddings.shape}")
    except Exception as e:
        logger.error(f"Error with Sentence Transformer embeddings: {e}")

    # Compare embeddings (within same model only, different dimensions can't be compared)
    if len(embedders) >= 2:
        logger.info("Comparing embedding models...")
        model1_name, emb1 = embedders[0]
        model2_name, emb2 = embedders[1]

        # Compare within-model similarities for same examples
        sample_idx1, sample_idx2 = 0, 1

        sim1 = compute_embedding_similarity(emb1[sample_idx1], emb1[sample_idx2])
        sim2 = compute_embedding_similarity(emb2[sample_idx1], emb2[sample_idx2])

        logger.info(
            f"Within-model similarity (examples 0 vs 1):"
        )
        logger.info(f"  {model1_name}: {sim1:.4f}")
        logger.info(f"  {model2_name}: {sim2:.4f}")

    # Summary
    logger.info("=" * 60)
    logger.info("PHASE 2 COMPLETE")
    logger.info(f"Embeddings generated for {len(texts)} examples")
    logger.info(f"Models used: {len(embedders)}")
    for name, emb in embedders:
        logger.info(f"  {name}: {emb.shape}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
