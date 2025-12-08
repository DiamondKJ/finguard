"""Input validation utilities for FinGuard."""

import re
from typing import Any, Dict, List

import numpy as np


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_prompt(
    prompt: str,
    min_length: int = 10,
    max_length: int = 500,
    allow_empty: bool = False,
) -> str:
    """Validate a user prompt.

    Args:
        prompt: Input prompt to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        allow_empty: Whether to allow empty prompts

    Returns:
        Validated and cleaned prompt

    Raises:
        ValidationError: If prompt fails validation
    """
    if not allow_empty and not prompt:
        raise ValidationError("Prompt cannot be empty")

    if not isinstance(prompt, str):
        raise ValidationError(f"Prompt must be a string, got {type(prompt)}")

    # Strip whitespace
    prompt = prompt.strip()

    # Check length
    if len(prompt) < min_length:
        raise ValidationError(
            f"Prompt too short ({len(prompt)} chars). Minimum: {min_length}"
        )

    if len(prompt) > max_length:
        raise ValidationError(
            f"Prompt too long ({len(prompt)} chars). Maximum: {max_length}"
        )

    return prompt


def validate_embedding(
    embedding: np.ndarray, expected_dim: int | None = None
) -> np.ndarray:
    """Validate an embedding vector.

    Args:
        embedding: Embedding vector to validate
        expected_dim: Expected dimensionality (optional)

    Returns:
        Validated embedding

    Raises:
        ValidationError: If embedding fails validation
    """
    if not isinstance(embedding, np.ndarray):
        raise ValidationError(f"Embedding must be numpy array, got {type(embedding)}")

    if embedding.ndim != 1:
        raise ValidationError(
            f"Embedding must be 1-dimensional, got {embedding.ndim} dimensions"
        )

    if expected_dim and embedding.shape[0] != expected_dim:
        raise ValidationError(
            f"Embedding dimension mismatch. Expected {expected_dim}, got {embedding.shape[0]}"
        )

    if np.isnan(embedding).any():
        raise ValidationError("Embedding contains NaN values")

    if np.isinf(embedding).any():
        raise ValidationError("Embedding contains infinite values")

    return embedding


def validate_dataset(
    data: List[Dict[str, Any]], required_keys: List[str] | None = None
) -> List[Dict[str, Any]]:
    """Validate dataset structure.

    Args:
        data: List of data examples
        required_keys: Required keys in each example

    Returns:
        Validated dataset

    Raises:
        ValidationError: If dataset fails validation
    """
    if not isinstance(data, list):
        raise ValidationError(f"Dataset must be a list, got {type(data)}")

    if len(data) == 0:
        raise ValidationError("Dataset cannot be empty")

    if required_keys:
        for idx, example in enumerate(data):
            if not isinstance(example, dict):
                raise ValidationError(
                    f"Example {idx} must be a dictionary, got {type(example)}"
                )

            missing_keys = set(required_keys) - set(example.keys())
            if missing_keys:
                raise ValidationError(
                    f"Example {idx} missing required keys: {missing_keys}"
                )

    return data


def validate_labels(
    labels: np.ndarray | List[int], num_classes: int
) -> np.ndarray:
    """Validate classification labels.

    Args:
        labels: Array of labels
        num_classes: Expected number of classes

    Returns:
        Validated labels as numpy array

    Raises:
        ValidationError: If labels fail validation
    """
    labels = np.array(labels)

    if labels.ndim != 1:
        raise ValidationError(f"Labels must be 1-dimensional, got {labels.ndim} dimensions")

    if len(labels) == 0:
        raise ValidationError("Labels array cannot be empty")

    if labels.min() < 0:
        raise ValidationError(f"Labels must be non-negative, got min: {labels.min()}")

    if labels.max() >= num_classes:
        raise ValidationError(
            f"Labels must be < {num_classes}, got max: {labels.max()}"
        )

    return labels


def check_duplicates(
    texts: List[str], threshold: float = 0.9
) -> List[tuple[int, int]]:
    """Check for duplicate or near-duplicate texts.

    Args:
        texts: List of text strings
        threshold: Similarity threshold for considering duplicates

    Returns:
        List of (index1, index2) tuples for duplicate pairs
    """
    duplicates = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # Simple string similarity (can be enhanced with embedding similarity)
            if texts[i] == texts[j]:
                duplicates.append((i, j))
            elif _jaccard_similarity(texts[i], texts[j]) > threshold:
                duplicates.append((i, j))

    return duplicates


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity score
    """
    # Tokenize by splitting on whitespace
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union)


def validate_config(config: Dict[str, Any], schema: Dict[str, type]) -> Dict[str, Any]:
    """Validate configuration against a schema.

    Args:
        config: Configuration dictionary
        schema: Schema defining expected types

    Returns:
        Validated configuration

    Raises:
        ValidationError: If config fails validation
    """
    for key, expected_type in schema.items():
        if key not in config:
            raise ValidationError(f"Missing required config key: {key}")

        if not isinstance(config[key], expected_type):
            raise ValidationError(
                f"Config key '{key}' must be {expected_type.__name__}, "
                f"got {type(config[key]).__name__}"
            )

    return config


if __name__ == "__main__":
    # Test validation functions
    try:
        validate_prompt("What is a 401k?")
        print("✓ Valid prompt")
    except ValidationError as e:
        print(f"✗ {e}")

    try:
        validate_prompt("Short")  # Should fail
    except ValidationError as e:
        print(f"✓ Caught short prompt: {e}")

    try:
        embedding = np.random.rand(768)
        validate_embedding(embedding, expected_dim=768)
        print("✓ Valid embedding")
    except ValidationError as e:
        print(f"✗ {e}")

    try:
        labels = np.array([0, 1, 2, 3, 4])
        validate_labels(labels, num_classes=5)
        print("✓ Valid labels")
    except ValidationError as e:
        print(f"✗ {e}")
