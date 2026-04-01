from __future__ import annotations

from .common_binary import BaseBinaryEngine


class QVinaWEngine(BaseBinaryEngine):
    """QuickVina-W command-line backend."""

    exe_name = "qvina-w"
    supports_autobox = False
    flag_map = {
        key: value
        for key, value in BaseBinaryEngine.flag_map.items()
        if key not in {"autobox_ligand", "autobox_add"}
    }
