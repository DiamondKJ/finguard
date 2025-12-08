"""Sentence Transformer embedding model for FinGuard."""

from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.embeddings.base import BaseEmbedder
from src.utils.logger import get_logger
from src.utils.validation import validate_embedding

logger = get_logger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformer embedding model (local, free)."""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """Initialize Sentence Transformer embedder.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for embedding
            normalize: Whether to normalize embeddings
        """
        # Model dimensions
        dimensions_map = {
            "all-mpnet-base-v2": 768,
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
        }
        dimensions = dimensions_map.get(model_name, 768)

        super().__init__(model_name, dimensions)

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Sentence Transformer model", device=device)

        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info(f"Model loaded successfully", device=self.device)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        return validate_embedding(
            np.array(embedding, dtype=np.float32), expected_dim=self.dimensions
        )

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts with progress tracking.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings with shape (n_texts, dimensions)
        """
        logger.info(f"Embedding batch", total=len(texts), batch_size=self.batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )

        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Embeddings created", shape=embeddings.shape)

        return embeddings

    def get_model_info(self) -> dict:
        """Get information about the embedding model.

        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update(
            {
                "device": self.device,
                "normalize": self.normalize,
                "max_seq_length": self.model.max_seq_length,
            }
        )
        return info


if __name__ == "__main__":
    # Test Sentence Transformer embedder
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    try:
        embedder = SentenceTransformerEmbedder()

        # Test single embedding
        text = "What is a 401k plan?"
        embedding = embedder.embed_single(text)
        print(f"Single embedding shape: {embedding.shape}")
        print(f"First 10 values: {embedding[:10]}")

        # Test batch embedding
        texts = [
            "What is a 401k?",
            "Should I buy stocks?",
            "How do bonds work?",
        ]
        embeddings = embedder.embed_batch(texts)
        print(f"\nBatch embeddings shape: {embeddings.shape}")

        # Check similarity
        from src.utils.metrics import compute_embedding_similarity

        sim = compute_embedding_similarity(embeddings[0], embeddings[1])
        print(f"\nCosine similarity between first two: {sim:.4f}")

    except Exception as e:
        logger.error(
            "Error testing Sentence Transformer embedder", error=str(e), exc_info=True
        )
