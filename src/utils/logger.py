"""Logging utilities for FinGuard."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from rich.console import Console
from rich.logging import RichHandler

from src.utils.config import get_settings


def setup_logging(
    log_level: str | None = None,
    log_format: str | None = None,
    log_file: Path | None = None,
) -> None:
    """Configure structured logging with rich formatting.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'console')
        log_file: Optional file path for logging output
    """
    settings = get_settings()
    log_level = log_level or settings.log_level
    log_format = log_format or settings.log_format

    # Configure standard library logging
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                console=Console(stderr=True),
                show_time=True,
                show_level=True,
                show_path=True,
            )
        ],
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level.upper())
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding context to logs."""

    def __init__(self, logger: structlog.BoundLogger, **context: Any):
        """Initialize log context.

        Args:
            logger: Logger instance
            **context: Context key-value pairs to add to logs
        """
        self.logger = logger
        self.context = context
        self.bound_logger = None

    def __enter__(self) -> structlog.BoundLogger:
        """Enter context and bind context to logger."""
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        pass


def log_function_call(func):
    """Decorator to log function calls with arguments and results.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    logger = get_logger(func.__module__)

    def wrapper(*args, **kwargs):
        logger.info(
            f"Calling {func.__name__}",
            args=args[:3] if len(args) > 3 else args,  # Limit arg logging
            kwargs=list(kwargs.keys()),
        )
        try:
            result = func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}", error=str(e), exc_info=True)
            raise

    return wrapper


if __name__ == "__main__":
    # Test logging setup
    setup_logging(log_level="DEBUG", log_format="console")
    logger = get_logger(__name__)

    logger.debug("Debug message")
    logger.info("Info message", key="value")
    logger.warning("Warning message")
    logger.error("Error message", error_code=500)

    # Test context manager
    with LogContext(logger, phase="test", iteration=1) as ctx_logger:
        ctx_logger.info("Message with context")
