from __future__ import annotations

from .common_binary import BaseBinaryEngine


class GninaEngine(BaseBinaryEngine):
    """gnina command-line backend."""

    exe_name = "gnina"
    supports_autobox = True
