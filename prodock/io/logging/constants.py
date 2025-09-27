"""Constants for logging package (ANSI codes and mappings)."""

from __future__ import annotations

ANSI = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
}

_LEVEL_TO_ANSI = {
    "DEBUG": ANSI["CYAN"],
    "INFO": ANSI["GREEN"],
    "WARNING": ANSI["YELLOW"],
    "ERROR": ANSI["RED"],
    "CRITICAL": ANSI["MAGENTA"],
}

__all__ = ["ANSI", "_LEVEL_TO_ANSI"]
