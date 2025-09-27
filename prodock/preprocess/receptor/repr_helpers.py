"""
Small ReprMixin that provides pretty __repr__ / __str__ for ReceptorPrep.

Keep it tiny and dependency-free so it can be unit tested on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class ReprMixin:
    """
    Mixin that provides a compact box-style __repr__ and friendly __str__.

    Expects the host class to provide the following attributes (as used in the original):
      - _final_artifact: Optional[pathlib.Path]
      - _last_simulation_report: Optional[dict]
      - _used_obabel: bool
      - _minimized_stage: Optional[str]
      - _use_meeko: bool

    The mixin does not set any state; it only reads it.
    """

    def _repr_basic_info(self) -> Dict[str, Any]:
        """Return a minimal dict of values derived from instance state."""
        info: Dict[str, Any] = {}
        info["final_artifact"] = (
            str(self._final_artifact)
            if getattr(self, "_final_artifact", None)
            else "<none>"
        )
        info["artifact_name"] = (
            Path(info["final_artifact"]).name
            if getattr(self, "_final_artifact", None)
            else "<none>"
        )
        info["use_meeko"] = bool(getattr(self, "_use_meeko", False))
        info["used_obabel"] = bool(getattr(self, "_used_obabel", False))
        info["minimized_stage"] = getattr(self, "_minimized_stage", None) or "<none>"
        info["out_fmt"] = (getattr(self, "_last_simulation_report", {}) or {}).get(
            "out_fmt", "<none>"
        )
        info["mekoo_info"] = (getattr(self, "_last_simulation_report", {}) or {}).get(
            "mekoo_info", {}
        ) or {}
        return info

    def _repr_converter_status(self, info: Dict[str, Any]) -> str:
        """
        Return short converter label: 'OpenBabel', 'mekoo' or 'none'.

        Priority:
          1. if used_obabel True -> 'OpenBabel'
          2. elif mekoo_info produced entries -> 'mekoo'
          3. elif out_fmt requested 'pdbqt' but no produced -> 'none'
          4. else -> 'none'
        """
        if info.get("used_obabel"):
            return "OpenBabel"
        mek = info.get("mekoo_info", {})
        if isinstance(mek, dict) and mek.get("produced"):
            return "mekoo"
        if str(info.get("out_fmt", "")).lower() == "pdbqt":
            return "none"
        return "none"

    def __repr__(self) -> str:
        """
        Compact, low-complexity pretty repr for ReceptorPrep.

        This method is intentionally short and uses two helpers above to keep
        cyclomatic complexity low and make the code easy to unit-test.
        """
        info = self._repr_basic_info()
        converter = self._repr_converter_status(info)

        # statuses
        protein_fix_status = (
            "success"
            if getattr(self, "_final_artifact", None)
            else (
                "not-run"
                if getattr(self, "_last_simulation_report", None) is None
                else "fail"
            )
        )
        minimizer_used = (
            "OpenBabel"
            if info["used_obabel"]
            else (
                "OpenMM" if info["minimized_stage"] not in ("<none>", None) else "none"
            )
        )
        stage = info["minimized_stage"]

        # build small multi-line box
        box_width = 64
        lines = [
            "┌" + "─" * box_width + "┐",
            ("│ ReceptorPrep Summary").ljust(box_width) + " │",
            "├" + "─" * box_width + "┤",
            ("│ Protein Fix: " + protein_fix_status).ljust(box_width) + " │",
            ("│ Minimiz: " + minimizer_used).ljust(box_width) + " │",
            ("│ Stage: " + stage).ljust(box_width) + " │",
            ("│ Converter: " + converter).ljust(box_width) + " │",
            ("│ Final artifact: " + info["final_artifact"]).ljust(box_width) + " │",
            ("│ Out format: " + str(info["out_fmt"])).ljust(box_width) + " │",
            ("│ mekoo enabled: " + ("yes" if info["use_meeko"] else "no")).ljust(
                box_width
            )
            + " │",
            "└" + "─" * box_width + "┘",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Short friendly single-line string representation."""
        fa = (
            str(getattr(self, "_final_artifact", "<none>"))
            if getattr(self, "_final_artifact", None)
            else "<none>"
        )
        stage = getattr(self, "_minimized_stage", "<none>") or "<none>"
        meeko = "on" if bool(getattr(self, "_use_meeko", False)) else "off"
        ob = "obabel" if bool(getattr(self, "_used_obabel", False)) else "openmm"
        return f"ReceptorPrep(final={fa}, stage={stage}, meeko={meeko}, last_minimizer={ob})"
