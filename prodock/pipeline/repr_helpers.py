# prodock/pipeline/repr_helpers.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class ProDockReprMixin:
    """
    Provide readable ``__repr__`` and ``__str__`` for pipeline-like objects.

    Expected attributes on the host:
    ``target_path``, ``target``, ``project_dir``, ``ligand_dir``, ``receptor_dir``,
    ``output_dir``, ``cfg_box``.
    """

    def _safe_path(self, attr: str) -> Optional[Path]:
        v = getattr(self, attr, None)
        if v is None:
            return None
        try:
            return Path(v)
        except Exception:
            return None

    def _basic_info(self) -> Dict[str, Any]:
        target_path = self._safe_path("target_path")
        prepared = self._safe_path("target")
        project_dir = self._safe_path("project_dir")
        ligand_dir = self._safe_path("ligand_dir")
        receptor_dir = self._safe_path("receptor_dir")
        output_dir = self._safe_path("output_dir")
        cfg_box = getattr(self, "cfg_box", None)

        ligand_count = 0
        if ligand_dir and ligand_dir.exists():
            try:
                ligand_count = sum(1 for _ in ligand_dir.iterdir())
            except Exception:  # pragma: no cover - filesystem edge
                ligand_count = -1

        return {
            "target_path": str(target_path) if target_path else "<none>",
            "prepared_target": str(prepared) if prepared else "<not prepared>",
            "project_dir": str(project_dir) if project_dir else "<none>",
            "ligand_dir": str(ligand_dir) if ligand_dir else "<none>",
            "ligand_count": ligand_count,
            "receptor_dir": str(receptor_dir) if receptor_dir else "<none>",
            "output_dir": str(output_dir) if output_dir else "<none>",
            "cfg_box_present": bool(cfg_box),
        }

    def __repr__(self) -> str:
        info = self._basic_info()
        box_w = 72

        # Break long lines to satisfy E501
        lig_line = "│ Ligand dir: " + info["ligand_dir"]
        lig_line += f" ({info['ligand_count']} files)"

        lines = [
            "┌" + "─" * box_w + "┐",
            ("│ ProDock status").ljust(box_w) + " │",
            "├" + "─" * box_w + "┤",
            ("│ Target input: " + info["target_path"]).ljust(box_w) + " │",
            ("│ Prepared target: " + info["prepared_target"]).ljust(box_w) + " │",
            ("│ Project dir: " + info["project_dir"]).ljust(box_w) + " │",
            lig_line.ljust(box_w) + " │",
            ("│ Receptor dir: " + info["receptor_dir"]).ljust(box_w) + " │",
            ("│ Output dir: " + info["output_dir"]).ljust(box_w) + " │",
            (
                "│ cfg_box present: " + ("yes" if info["cfg_box_present"] else "no")
            ).ljust(box_w)
            + " │",
            "└" + "─" * box_w + "┘",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        info = self._basic_info()
        target_name = (
            Path(info["target_path"]).name
            if info["target_path"] != "<none>"
            else "<none>"
        )
        prepared = "yes" if info["prepared_target"] != "<not prepared>" else "no"
        has_box = "yes" if info["cfg_box_present"] else "no"
        return (
            f"ProDock(target={target_name}, prepared={prepared}, "
            f"ligands={info['ligand_count']}, cfg_box={has_box})"
        )
