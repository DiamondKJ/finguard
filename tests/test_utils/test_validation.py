"""Tests for validation utilities."""

import numpy as np
import pytest

from src.utils.validation import (
    ValidationError,
    check_duplicates,
    validate_embedding,
    validate_labels,
    validate_prompt,
)


class TestPromptValidation:
    """Test prompt validation functions."""

    def test_valid_prompt(self):
        """Test validation of valid prompt."""
        prompt = "What is a 401k retirement plan?"
        result = validate_prompt(prompt)
        assert result == prompt

    def test_empty_prompt(self):
        """Test validation rejects empty prompts."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_prompt("")

    def test_short_prompt(self):
        """Test validation rejects too-short prompts."""
        with pytest.raises(ValidationError, match="too short"):
            validate_prompt("Short", min_length=10)

    def test_long_prompt(self):
        """Test validation rejects too-long prompts."""
        long_prompt = "x" * 1000
        with pytest.raises(ValidationError, match="too long"):
            validate_prompt(long_prompt, max_length=500)

    def test_prompt_whitespace_stripping(self):
        """Test that whitespace is stripped."""
        prompt = "  What is a 401k?  "
        result = validate_prompt(prompt)
        assert result == "What is a 401k?"


class TestEmbeddingValidation:
    """Test embedding validation functions."""

    def test_valid_embedding(self):
        """Test validation of valid embedding."""
        embedding = np.random.rand(768).astype(np.float32)
        result = validate_embedding(embedding, expected_dim=768)
        assert result.shape == (768,)

    def test_wrong_dimension(self):
        """Test validation rejects wrong dimensions."""
        embedding = np.random.rand(512).astype(np.float32)
        with pytest.raises(ValidationError, match="dimension mismatch"):
            validate_embedding(embedding, expected_dim=768)

    def test_multidimensional_embedding(self):
        """Test validation rejects multidimensional arrays."""
        embedding = np.random.rand(10, 768).astype(np.float32)
        with pytest.raises(ValidationError, match="1-dimensional"):
            validate_embedding(embedding)

    def test_nan_values(self):
        """Test validation rejects NaN values."""
        embedding = np.random.rand(768).astype(np.float32)
        embedding[0] = np.nan
        with pytest.raises(ValidationError, match="NaN"):
            validate_embedding(embedding)

    def test_inf_values(self):
        """Test validation rejects infinite values."""
        embedding = np.random.rand(768).astype(np.float32)
        embedding[0] = np.inf
        with pytest.raises(ValidationError, match="infinite"):
            validate_embedding(embedding)


class TestLabelValidation:
    """Test label validation functions."""

    def test_valid_labels(self):
        """Test validation of valid labels."""
        labels = np.array([0, 1, 2, 3, 4])
        result = validate_labels(labels, num_classes=5)
        assert len(result) == 5

    def test_negative_labels(self):
        """Test validation rejects negative labels."""
        labels = np.array([0, 1, -1, 2])
        with pytest.raises(ValidationError, match="non-negative"):
            validate_labels(labels, num_classes=5)

    def test_out_of_range_labels(self):
        """Test validation rejects labels >= num_classes."""
        labels = np.array([0, 1, 5, 2])
        with pytest.raises(ValidationError, match="must be <"):
            validate_labels(labels, num_classes=5)


class TestDuplicateDetection:
    """Test duplicate detection."""

    def test_exact_duplicates(self):
        """Test detection of exact duplicates."""
        texts = ["Hello", "World", "Hello", "Test"]
        duplicates = check_duplicates(texts, threshold=0.9)
        assert len(duplicates) == 1
        assert (0, 2) in duplicates

    def test_no_duplicates(self):
        """Test no false positives for unique texts."""
        texts = ["Alpha", "Beta", "Gamma", "Delta"]
        duplicates = check_duplicates(texts, threshold=0.9)
        assert len(duplicates) == 0
