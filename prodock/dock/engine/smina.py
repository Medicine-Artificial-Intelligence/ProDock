from __future__ import annotations
from .common_binary import BaseBinaryEngine


class SminaEngine(BaseBinaryEngine):
    """
    smina CLI wrapper (no score parsing).

    Supports autoboxing via ``--autobox_ligand`` and ``--autobox_add``.
    """

    exe_name = "smina"
    supports_autobox = True
    # keep default flag_map (includes autobox flags)
