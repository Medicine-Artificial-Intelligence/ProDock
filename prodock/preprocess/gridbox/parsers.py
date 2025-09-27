# prodock/process/gridbox/parsers.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
from rdkit import Chem

try:
    from prodock.io.parser import (
        _parse_sdf_text,
        _parse_pdb_text,
        _parse_mol2_text,
        _parse_xyz_text,
    )

    def _parse_with_project(text: str, fmt: str) -> Chem.Mol | None:
        dispatch = {
            "sdf": _parse_sdf_text,
            "pdb": _parse_pdb_text,
            "mol2": _parse_mol2_text,
            "xyz": _parse_xyz_text,
        }
        fn = dispatch.get(fmt)
        return fn(text) if fn else None

except Exception:
    _parse_with_project = None  # type: ignore


def _parse_with_rdkit(text: str, fmt: str) -> Chem.Mol | None:
    fmt = fmt.lower()
    try:
        if fmt == "sdf":
            sup = Chem.SDMolSupplier()
            sup.SetData(text, sanitize=True)
            mols = [m for m in sup if m is not None]
            return mols[0] if mols else None
        if fmt == "pdb":
            return Chem.MolFromPDBBlock(text, removeHs=False)
        if fmt == "mol2":
            return Chem.MolFromMol2Block(text, sanitize=True, removeHs=False)
        if fmt == "xyz":
            return Chem.MolFromXYZBlock(text)
    except Exception:
        return None
    return None


def parse_text_to_mol(
    text_or_path: str | Path, fmt: Optional[str] = None
) -> Chem.Mol | None:
    """
    Parse a string containing molecule data or a file path into an RDKit Mol.

    Behaviour:
      - If `text_or_path` is an existing file path, read from file.
      - Otherwise treat it as raw text containing molecule data.
      - If `fmt` is None and `text_or_path` is a file path with a suffix,
        infer the format from the suffix.
    Supported formats: 'sdf', 'pdb', 'mol2', 'xyz'.
    """
    s = str(text_or_path)

    # robust path detection: Path.exists() can raise for very long strings (e.g. multi-line mol blocks).
    is_path = False
    try:
        is_path = Path(s).exists()
    except Exception:
        is_path = False

    if is_path:
        text = Path(s).read_text()
        if fmt is None and Path(s).suffix:
            fmt = Path(s).suffix.lstrip(".").lower()
    else:
        text = s
        fmt = (fmt or "sdf").lower()

    fmt = (fmt or "sdf").lower()

    # try project parser first (if available)
    if _parse_with_project is not None:
        try:
            mol = _parse_with_project(text, fmt)
            if mol is not None:
                return mol
        except Exception:
            # project parser may raise for unsupported formats/inputs; fall back to RDKit
            pass

    # fallback to RDKit parsing
    return _parse_with_rdkit(text, fmt)
