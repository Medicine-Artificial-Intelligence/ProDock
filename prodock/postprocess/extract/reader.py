from __future__ import annotations
import re
from typing import Optional

from .engines import (
    detect_engine as _auto_detect_engine,
    VINA_TABLE_HEADER,
    VINA_ROW_RE,
    GNINA_TABLE_HEADER,
    GNINA_ROW_RE,
)


def _iter_lines(text: str):
    for line in text.splitlines():
        yield line.rstrip("\n")


def _parse_vina_family(text: str) -> list[dict]:
    """
    Parse Vina-like tables (vina/smina/qvina/vina-gpu/qvina-gpu).
    Returns list of dicts with keys: mode, affinity_kcal_mol, rmsd_lb, rmsd_ub
    """
    lines = list(_iter_lines(text))
    rows: list[dict] = []
    # Find header index
    header_idx = None
    for i, ln in enumerate(lines):
        if VINA_TABLE_HEADER.search(ln.lower()):
            header_idx = i
            break
    if header_idx is None:
        return rows
    # Scan subsequent lines for numeric rows
    # fmt: off
    for ln in lines[header_idx + 1:]:  # fmt: on
        m = VINA_ROW_RE.match(ln)
        if not m:
            continue
        mode = int(m.group(1))
        aff = float(m.group(2))
        rmsd_lb = float(m.group(3))
        rmsd_ub = float(m.group(4))
        rows.append(
            {
                "mode": mode,
                "affinity_kcal_mol": aff,
                "rmsd_lb": rmsd_lb,
                "rmsd_ub": rmsd_ub,
            }
        )
    return rows


def _parse_gnina(text: str) -> list[dict]:
    """
    Parse GNINA tables (affinity + CNN columns).
    Keys: mode, affinity_kcal_mol, cnn_pose, cnn_affinity
    """
    lines = list(_iter_lines(text))
    rows: list[dict] = []
    header_idx = None
    for i, ln in enumerate(lines):
        if GNINA_TABLE_HEADER.search(ln.lower()):
            header_idx = i
            break
    if header_idx is None:
        return rows
    # fmt: off
    for ln in lines[header_idx + 1:]:  # fmt: on
        m = GNINA_ROW_RE.match(ln)
        if not m:
            continue
        mode = int(m.group(1))
        aff = float(m.group(2))
        cnn_pose = float(m.group(3))
        cnn_aff = float(m.group(4))
        rows.append(
            {
                "mode": mode,
                "affinity_kcal_mol": aff,
                "cnn_pose": cnn_pose,
                "cnn_affinity": cnn_aff,
            }
        )
    return rows


def parse_log_text(
    text: str,
    engine: Optional[str] = None,
    regex: Optional[dict[str, str]] = None,
) -> list[dict]:
    """
    Parse docking log text with built-in or user-provided regex.

    Parameters
    ----------
    text : str
        Full text content of a docking log.
    engine : str, optional
        Force a particular engine ('vina', 'smina', 'qvina', 'vina-gpu', 'qvina-gpu', 'gnina').
        When not provided, the function attempts auto-detection.
    regex : dict[str, str], optional
        Mapping from field-name to row-regex pattern. When provided, the custom
        regex is tried **first**. The pattern must capture groups for the fields:
          - for vina-like: (mode, affinity, rmsd_lb, rmsd_ub)
          - for gnina: (mode, affinity, cnn_pose, cnn_affinity)

        Example:
            {'vina_row': r'^\\s*(\\d+)\\s+(-?\\d+(?:\\.\\d+)?)\\s+(\\d+(?:\\.\\d+)?)\\s+(\\d+(?:\\.\\d+)?)\\s*$'}

    Returns
    -------
    list[dict]
        Parsed rows with numeric fields.
    """
    # Custom regex path (lets users bring their own)
    if regex:
        patt_key = "vina_row" if (engine is None or engine != "gnina") else "gnina_row"
        patt = regex.get(patt_key)
        if patt:
            row_re = re.compile(patt)
            rows = []
            for ln in text.splitlines():
                m = row_re.match(ln)
                if not m:
                    continue
                if engine == "gnina":
                    rows.append(
                        {
                            "mode": int(m.group(1)),
                            "affinity_kcal_mol": float(m.group(2)),
                            "cnn_pose": float(m.group(3)),
                            "cnn_affinity": float(m.group(4)),
                        }
                    )
                else:
                    rows.append(
                        {
                            "mode": int(m.group(1)),
                            "affinity_kcal_mol": float(m.group(2)),
                            "rmsd_lb": float(m.group(3)),
                            "rmsd_ub": float(m.group(4)),
                        }
                    )
            if rows:
                return rows
        # If custom regex given but didn't match, fall through to built-ins

    eng = engine or _auto_detect_engine(text)

    if eng == "gnina":
        rows = _parse_gnina(text)
        if rows:
            return rows
        # fallback: try vina parser (some gnina builds can print vina-like table)
        return _parse_vina_family(text)

    # Vina family (vina/smina/qvina/vina-gpu/qvina-gpu or generic 'vina')
    return _parse_vina_family(text)
