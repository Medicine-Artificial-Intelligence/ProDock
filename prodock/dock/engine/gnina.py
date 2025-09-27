from __future__ import annotations
from .common_binary import BaseBinaryEngine


class GninaEngine(BaseBinaryEngine):
    """
    gnina CLI wrapper (no score parsing).

    Supports autoboxing via the same flags used by smina.
    """

    exe_name = "gnina"
    supports_autobox = True
    # keep default flag_map (includes autobox flags)
