# prodock/io/logging/compat_logging.py
"""Compatibility shim for older imports that used a single-file logging module.

This module re-exports the public API from the refactored package while emitting
a DeprecationWarning so callers know to update their imports to
`prodock.io.logging`.
"""

from __future__ import annotations

import warnings

# Explicit (non-star) imports so linters can statically detect names.
from .manager import LoggerManager, setup_logging, get_logger
from .adapters import StructuredAdapter
from .timing import Timer
from .decorators import log_step
from .formatters import SimpleColorFormatter, JSONFormatter

# Warn users that this is a compatibility shim.
warnings.warn(
    "prodock.io.logging.compat_logging is a compatibility shim; "
    "please update imports to `from prodock.io.logging import ...`",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the same public API as the refactored package.
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
