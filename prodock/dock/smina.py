from __future__ import annotations

from .common_binary import BaseBinaryEngine


class SminaEngine(BaseBinaryEngine):
    """smina command-line backend."""

    exe_name = "smina"
    supports_autobox = True
