"""Logging utilities."""

import sys
import os
from pathlib import Path
from loguru import logger


def setup_logger(
    name: str = "3d_scene",
    level: str = "INFO",
    log_file: str = None,
) -> logger:
    """
    Set up logger with console and file output.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging

    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}",
            level=level,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )

    return logger


def get_logger(name: str = None) -> logger:
    """
    Get logger instance.

    Args:
        name: Optional logger name

    Returns:
        Logger instance
    """
    level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", None)

    if log_file is None:
        log_dir = os.getenv("LOGS_DIR", "/data/logs")
        if name:
            log_file = f"{log_dir}/{name}.log"

    return setup_logger(name or "3d_scene", level, log_file)
