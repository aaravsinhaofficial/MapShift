"""Shared logging helpers for benchmark scripts and runners."""

from __future__ import annotations

import logging as std_logging


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide logging once."""

    std_logging.basicConfig(
        level=getattr(std_logging, level.upper(), std_logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> std_logging.Logger:
    """Get a logger after ensuring the default logging format exists."""

    if not std_logging.getLogger().handlers:
        configure_logging()
    return std_logging.getLogger(name)
