"""Public API for prodock.io.logging package.

Export a compact and convenient surface for consumers.
"""

from __future__ import annotations

from .manager import LoggerManager, setup_logging, get_logger
from .adapters import StructuredAdapter
from .timing import Timer
from .decorators import log_step
from .formatters import SimpleColorFormatter, JSONFormatter

__all__ = [
    "LoggerManager",
    "setup_logging",
    "get_logger",
    "StructuredAdapter",
    "Timer",
    "log_step",
    "SimpleColorFormatter",
    "JSONFormatter",
]
