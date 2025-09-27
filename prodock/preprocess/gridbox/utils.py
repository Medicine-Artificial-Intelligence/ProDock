# prodock/process/gridbox/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
from rdkit import Chem


def is_pathlike(x: str | Path) -> bool:
    """Return True if x refers to an existing path."""
    try:
        return Path(str(x)).exists()
    except Exception:
        return False


def coords_from_mol(mol: Chem.Mol) -> np.ndarray:
    """
    Return (N,3) numpy array of atom coordinates for an RDKit mol.
    Raises ValueError if no conformer present.
    """
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    return np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(n)
        ],
        dtype=float,
    )


def gb_coords_from_mol(mol: Chem.Mol, heavy_only: bool = False) -> np.ndarray:
    """
    Coordinates for bounding calculations. When heavy_only True, exclude hydrogens.
    If no atoms remain after filtering, fall back to all atoms.
    """
    conf = mol.GetConformer()
    out = []
    for i in range(mol.GetNumAtoms()):
        if heavy_only and mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(i)
        out.append((p.x, p.y, p.z))
    if not out:
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            out.append((p.x, p.y, p.z))
    return np.asarray(out, dtype=float)


def round_tuple(t: Iterable[float], nd: int) -> Tuple[float, float, float]:
    a, b, c = t
    return float(round(a, nd)), float(round(b, nd)), float(round(c, nd))


def snap_val(v: float, step: float) -> float:
    return float(round(v / step) * step)


def snap_tuple(t: Iterable[float], step: float) -> Tuple[float, float, float]:
    a, b, c = t
    return snap_val(a, step), snap_val(b, step), snap_val(c, step)


def ensure_pos_size(size: Iterable[float]) -> Tuple[float, float, float]:
    sx, sy, sz = tuple(float(x) for x in size)
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError("All size components must be positive.")
    return sx, sy, sz


def center_and_span(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return center (3,) and span (3,) arrays from coordinates (N,3)."""
    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    span = xyz_max - xyz_min
    return center, span
