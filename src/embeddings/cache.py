"""Embedding cache for FinGuard to avoid recomputation."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Cache embeddings to disk to avoid recomputation."""

    def __init__(self, cache_dir: str | Path = "data/embeddings"):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized embedding cache", cache_dir=str(self.cache_dir))

    def _text_hash(self, text: str) -> str:
        """Compute hash of text for cache key.

        Args:
            text: Text to hash

        Returns:
            Hash string
        """
        return hashlib.md5(text.encode()).hexdigest()

    def _batch_cache_path(self, model_name: str, batch_id: str) -> Path:
        """Get cache file path for a batch.

        Args:
            model_name: Name of embedding model
            batch_id: Unique identifier for the batch

        Returns:
            Path to cache file
        """
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_model_name}_{batch_id}.npz"

    def save_batch(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        model_name: str,
        metadata: Dict | None = None,
    ) -> str:
        """Save batch of embeddings to cache.

        Args:
            texts: List of texts
            embeddings: Array of embeddings
            model_name: Name of embedding model
            metadata: Optional metadata to save with embeddings

        Returns:
            Batch ID for retrieval
        """
        # Create batch ID from hash of all texts
        combined_text = "".join(texts)
        batch_id = self._text_hash(combined_text)

        cache_path = self._batch_cache_path(model_name, batch_id)

        # Save embeddings and metadata
        metadata = metadata or {}
        metadata.update(
            {
                "model_name": model_name,
                "num_texts": len(texts),
                "dimensions": embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings),
            }
        )

        np.savez(
            cache_path,
            embeddings=embeddings,
            texts=np.array(texts),
            metadata=json.dumps(metadata),
        )

        logger.info(
            f"Cached embeddings",
            batch_id=batch_id,
            num_texts=len(texts),
            path=str(cache_path),
        )

        return batch_id

    def load_batch(self, model_name: str, batch_id: str) -> tuple[np.ndarray, List[str], Dict]:
        """Load batch of embeddings from cache.

        Args:
            model_name: Name of embedding model
            batch_id: Batch identifier

        Returns:
            Tuple of (embeddings, texts, metadata)

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        cache_path = self._batch_cache_path(model_name, batch_id)

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        texts = data["texts"].tolist()
        metadata = json.loads(str(data["metadata"]))

        logger.info(f"Loaded cached embeddings", batch_id=batch_id, num_texts=len(texts))

        return embeddings, texts, metadata

    def save_dataset_embeddings(
        self,
        dataset_name: str,
        embeddings: np.ndarray,
        model_name: str,
        metadata: Dict | None = None,
    ) -> None:
        """Save embeddings for entire dataset.

        Args:
            dataset_name: Name of the dataset
            embeddings: Array of embeddings
            model_name: Name of embedding model
            metadata: Optional metadata
        """
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        cache_path = self.cache_dir / f"{dataset_name}_{safe_model_name}.npz"

        metadata = metadata or {}
        metadata.update(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "num_examples": embeddings.shape[0],
                "dimensions": embeddings.shape[1],
            }
        )

        np.savez(cache_path, embeddings=embeddings, metadata=json.dumps(metadata))

        logger.info(
            f"Cached dataset embeddings",
            dataset=dataset_name,
            model=model_name,
            shape=embeddings.shape,
        )

    def load_dataset_embeddings(
        self, dataset_name: str, model_name: str
    ) -> tuple[np.ndarray, Dict]:
        """Load embeddings for entire dataset.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of embedding model

        Returns:
            Tuple of (embeddings, metadata)

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        cache_path = self.cache_dir / f"{dataset_name}_{safe_model_name}.npz"

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Dataset embeddings not cached: {dataset_name} with {model_name}"
            )

        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        metadata = json.loads(str(data["metadata"]))

        logger.info(
            f"Loaded dataset embeddings",
            dataset=dataset_name,
            model=model_name,
            shape=embeddings.shape,
        )

        return embeddings, metadata

    def clear_cache(self, model_name: str | None = None) -> None:
        """Clear cache files.

        Args:
            model_name: If specified, only clear cache for this model
        """
        if model_name:
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            pattern = f"{safe_model_name}_*.npz"
        else:
            pattern = "*.npz"

        removed_count = 0
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            removed_count += 1

        logger.info(f"Cleared cache", files_removed=removed_count)


if __name__ == "__main__":
    # Test embedding cache
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    cache = EmbeddingCache()

    # Test batch caching
    texts = ["What is a 401k?", "Should I buy stocks?", "How do bonds work?"]
    embeddings = np.random.rand(3, 768).astype(np.float32)

    batch_id = cache.save_batch(texts, embeddings, "test-model")
    print(f"Saved batch with ID: {batch_id}")

    loaded_embeddings, loaded_texts, metadata = cache.load_batch("test-model", batch_id)
    print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
    print(f"Loaded texts: {loaded_texts}")
    print(f"Metadata: {metadata}")

    # Test dataset caching
    dataset_embeddings = np.random.rand(100, 768).astype(np.float32)
    cache.save_dataset_embeddings("test_dataset", dataset_embeddings, "test-model")

    loaded_dataset, metadata = cache.load_dataset_embeddings("test_dataset", "test-model")
    print(f"\nLoaded dataset embeddings shape: {loaded_dataset.shape}")

    # Cleanup
    cache.clear_cache()
