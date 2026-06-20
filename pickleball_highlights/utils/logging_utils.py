"""Logging utilities for the pickleball highlight extractor."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    extra_handlers: Optional[list[logging.Handler]] = None,
) -> None:
    """Configure root logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write log output to a file in addition to
                  stdout.
        extra_handlers: Optional logging handlers to attach alongside the
                        default stdout/file handlers.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(levelname)s] %(name)s: %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    if extra_handlers:
        handlers.extend(extra_handlers)

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name (usually ``__name__`` of the calling module).

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
