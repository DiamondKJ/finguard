"""OpenAI embedding model for FinGuard."""

import time
from typing import List

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.embeddings.base import BaseEmbedder
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.validation import validate_embedding

logger = get_logger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text embedding model."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        """Initialize OpenAI embedder.

        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (if None, loads from settings)
            batch_size: Batch size for embedding
            max_retries: Maximum number of retries on failure
        """
        # Model dimensions
        dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        dimensions = dimensions_map.get(model_name, 1536)

        super().__init__(model_name, dimensions)

        settings = get_settings()
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=api_key)
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            Exception: If embedding fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name, input=[text]
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                return validate_embedding(embedding, expected_dim=self.dimensions)

            except Exception as e:
                logger.warning(
                    f"Embedding attempt {attempt + 1} failed", error=str(e)
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    raise

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts with progress tracking.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings with shape (n_texts, dimensions)
        """
        logger.info(f"Embedding batch", total=len(texts), batch_size=self.batch_size)

        all_embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch = texts[i : i + self.batch_size]

            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name, input=batch
                    )

                    batch_embeddings = [
                        np.array(item.embedding, dtype=np.float32)
                        for item in response.data
                    ]
                    all_embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    logger.warning(
                        f"Batch embedding attempt {attempt + 1} failed", error=str(e)
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise

            # Rate limiting
            time.sleep(0.1)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Embeddings created", shape=embeddings.shape)

        return embeddings


if __name__ == "__main__":
    # Test OpenAI embedder
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    try:
        embedder = OpenAIEmbedder()

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

    except ValueError as e:
        logger.error("API key not set. Please set OPENAI_API_KEY in .env file")
    except Exception as e:
        logger.error("Error testing OpenAI embedder", error=str(e), exc_info=True)
