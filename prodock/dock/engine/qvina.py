from __future__ import annotations
from .common_binary import BaseBinaryEngine


class QVinaEngine(BaseBinaryEngine):
    """
    qvina CLI wrapper (no score parsing).

    Notes
    -----
    - qvina has **no** autoboxing flags; they are removed from ``flag_map``.
    """

    exe_name = "qvina"
    supports_autobox = False
    flag_map = {
        k: v
        for k, v in BaseBinaryEngine.flag_map.items()
        if k not in ("autobox_ligand", "autobox_add")
    }
