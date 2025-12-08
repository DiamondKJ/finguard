"""Abstract base class for embedding models in FinGuard."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""

    def __init__(self, model_name: str, dimensions: int):
        """Initialize embedder.

        Args:
            model_name: Name of the embedding model
            dimensions: Dimensionality of embeddings
        """
        self.model_name = model_name
        self.dimensions = dimensions
        logger.info(
            f"Initialized {self.__class__.__name__}",
            model=model_name,
            dimensions=dimensions,
        )

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings with shape (n_texts, dimensions)
        """
        pass

    def __call__(self, text_or_texts: str | List[str]) -> np.ndarray:
        """Embed text(s) using the model.

        Args:
            text_or_texts: Single text string or list of texts

        Returns:
            Embedding vector or array of embeddings
        """
        if isinstance(text_or_texts, str):
            return self.embed_single(text_or_texts)
        elif isinstance(text_or_texts, list):
            return self.embed_batch(text_or_texts)
        else:
            raise TypeError(
                f"Expected str or List[str], got {type(text_or_texts)}"
            )

    def get_model_info(self) -> dict:
        """Get information about the embedding model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "class": self.__class__.__name__,
        }
