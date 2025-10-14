"""Logging configuration and utilities for TranPy.

This module provides centralized logging configuration to ensure consistent
logging behavior across all TranPy modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Default logging format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_FORMAT = '%(levelname)s: %(message)s'


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger for a TranPy module.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (defaults to INFO)

    Returns:
        Configured logger instance

    Examples:
        >>> from tranpy.utils.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.warning("Configuration not found, using defaults")
    """
    logger = logging.getLogger(name)

    # Set level if not already configured
    if level is None:
        level = logging.INFO

    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(SIMPLE_FORMAT)
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def configure_logging(
    level: int = logging.INFO,
    format_string: str = DEFAULT_FORMAT,
    log_file: Optional[Path] = None
):
    """
    Configure global logging settings for TranPy.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Format string for log messages
        log_file: Optional file path for logging output

    Examples:
        >>> from tranpy.utils.logging import configure_logging
        >>> import logging
        >>>
        >>> # Enable debug logging
        >>> configure_logging(level=logging.DEBUG)
        >>>
        >>> # Log to file
        >>> configure_logging(log_file='tranpy.log')
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[]
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logging.root.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.root.addHandler(file_handler)


def set_level(level: int):
    """
    Set the logging level for all TranPy loggers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Examples:
        >>> import logging
        >>> from tranpy.utils.logging import set_level
        >>>
        >>> # Enable verbose logging
        >>> set_level(logging.DEBUG)
        >>>
        >>> # Quiet mode
        >>> set_level(logging.WARNING)
    """
    logging.getLogger('tranpy').setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)


def disable_logging():
    """
    Disable all logging output.

    Examples:
        >>> from tranpy.utils.logging import disable_logging
        >>> disable_logging()
    """
    logging.disable(logging.CRITICAL)


def enable_logging():
    """
    Re-enable logging after it has been disabled.

    Examples:
        >>> from tranpy.utils.logging import enable_logging
        >>> enable_logging()
    """
    logging.disable(logging.NOTSET)
