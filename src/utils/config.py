"""Configuration management for FinGuard."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # Paths
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    models_dir: str = Field(default="./models", alias="MODELS_DIR")
    output_dir: str = Field(default="./outputs", alias="OUTPUT_DIR")

    # Dataset
    dataset_size: int = Field(default=500, alias="DATASET_SIZE")
    examples_per_category: int = Field(default=100, alias="EXAMPLES_PER_CATEGORY")

    # Embedding
    default_embedding_model: str = Field(default="openai", alias="DEFAULT_EMBEDDING_MODEL")
    embedding_cache_enabled: bool = Field(default=True, alias="EMBEDDING_CACHE_ENABLED")
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")

    # Classifier
    classifier_type: str = Field(default="random_forest", alias="CLASSIFIER_TYPE")
    n_estimators: int = Field(default=100, alias="N_ESTIMATORS")
    test_split: float = Field(default=0.2, alias="TEST_SPLIT")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")

    # Benchmark
    benchmark_iterations: int = Field(default=100, alias="BENCHMARK_ITERATIONS")
    warmup_iterations: int = Field(default=10, alias="WARMUP_ITERATIONS")

    # Demo
    demo_model_path: str = Field(
        default="./models/rf_classifier_v1.pkl", alias="DEMO_MODEL_PATH"
    )


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_settings() -> Settings:
    """Get global settings instance.

    Returns:
        Settings object with environment variables loaded
    """
    # Load .env file if it exists
    load_dotenv(override=True)
    return Settings()


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    settings = get_settings()
    root = get_project_root()

    directories = [
        settings.data_dir,
        f"{settings.data_dir}/raw",
        f"{settings.data_dir}/processed",
        f"{settings.data_dir}/embeddings",
        settings.models_dir,
        settings.output_dir,
        f"{settings.output_dir}/visualizations",
        f"{settings.output_dir}/metrics",
        f"{settings.output_dir}/benchmarks",
        f"{settings.output_dir}/reports",
    ]

    for directory in directories:
        dir_path = root / directory
        dir_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration loading
    settings = get_settings()
    print(f"Settings loaded successfully")
    print(f"Log level: {settings.log_level}")
    print(f"Data dir: {settings.data_dir}")
    ensure_directories()
    print("All directories ensured")
