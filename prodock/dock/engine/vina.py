from __future__ import annotations
from .common_binary import BaseBinaryEngine


class VinaEngine(BaseBinaryEngine):
    """
    AutoDock Vina **CLI** wrapper (no score parsing).

    Notes
    -----
    - Vina CLI does **not** offer ``--autobox_*`` flags, so autobox is disabled.
    - If you want autobox, use :class:`SminaEngine` or :class:`GninaEngine`.
    """

    exe_name = "vina"
    supports_autobox = False
    # Remove autobox flags so they never appear
    flag_map = {
        k: v
        for k, v in BaseBinaryEngine.flag_map.items()
        if k not in ("autobox_ligand", "autobox_add")
    }
