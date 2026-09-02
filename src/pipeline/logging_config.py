"""Logging setup for command-line pipeline execution."""
import logging
import os
import sys


def configure_logging() -> None:
    """Configure concise console logging once for pipeline commands."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
