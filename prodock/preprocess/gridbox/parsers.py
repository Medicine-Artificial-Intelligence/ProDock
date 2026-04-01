# prodock/process/gridbox/parsers.py
"""
Parsing helpers for converting molecular text or structure files into RDKit molecules.

This module provides lightweight utilities for reading molecular content from
either raw text or filesystem paths and converting that content into an RDKit
``Mol`` object.

Supported formats
-----------------
The following formats are supported:

- ``"sdf"``
- ``"pdb"``
- ``"mol2"``
- ``"xyz"``

Parsing strategy
----------------
Parsing follows a two-stage strategy:

1. Attempt parsing through project-specific helpers from ``prodock.io.parser``
   when available.
2. Fall back to direct RDKit parsing if project-level parsing is unavailable or
   unsuccessful.

The main public entry point is :func:`parse_text_to_mol`.

Example
-------
.. code-block:: python

    from prodock.process.gridbox.parsers import parse_text_to_mol

    mol = parse_text_to_mol("ligand.sdf")
    mol2 = parse_text_to_mol(raw_pdb_text, fmt="pdb")
"""

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
        """
        Parse molecular text using project-level parser helpers.

        This helper dispatches to format-specific parsing functions from
        ``prodock.io.parser`` when those functions are available.

        :param text:
            Raw molecular text to parse.
        :type text: str
        :param fmt:
            Input format name such as ``"sdf"``, ``"pdb"``, ``"mol2"``, or
            ``"xyz"``.
        :type fmt: str

        :returns:
            Parsed RDKit molecule, or ``None`` if the format is unsupported or
            parsing fails at the dispatch level.
        :rtype: Chem.Mol | None

        Example
        -------
        .. code-block:: python

            mol = _parse_with_project(raw_sdf_text, "sdf")
        """
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
    """
    Parse molecular text directly with RDKit.

    This function provides a fallback parsing path when project-specific parsers
    are unavailable or fail. Format matching is case-insensitive.

    :param text:
        Raw molecular text to parse.
    :type text: str
    :param fmt:
        Input format name such as ``"sdf"``, ``"pdb"``, ``"mol2"``, or
        ``"xyz"``.
    :type fmt: str

    :returns:
        Parsed RDKit molecule, or ``None`` if parsing fails.
    :rtype: Chem.Mol | None

    Example
    -------
    .. code-block:: python

        mol = _parse_with_rdkit(raw_pdb_text, "pdb")
    """
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
    Parse molecular content or a structure file path into an RDKit molecule.

    The input may be either raw molecular text or a path to an existing file.
    If the input resolves to an existing file, the file is read from disk.
    Otherwise, the input is treated as in-memory molecular text.

    If ``fmt`` is not provided and the input is a file path with a recognized
    suffix, the format is inferred from that suffix. When parsing raw text and
    no format is provided, the default format is ``"sdf"``.

    Supported formats are:

    - ``"sdf"``
    - ``"pdb"``
    - ``"mol2"``
    - ``"xyz"``

    Parsing first attempts project-level parser utilities when available, then
    falls back to direct RDKit parsing.

    :param text_or_path:
        Either raw molecular text or a filesystem path to a structure file.
    :type text_or_path: str | Path
    :param fmt:
        Optional explicit format specifier. If omitted, the format is inferred
        from the input path suffix when possible, otherwise defaults to
        ``"sdf"`` for raw text.
    :type fmt: Optional[str]

    :returns:
        Parsed RDKit molecule, or ``None`` if parsing fails.
    :rtype: Chem.Mol | None

    Notes
    -----
    Path detection is handled defensively. Calling :meth:`Path.exists` on large
    multiline strings such as MolBlocks may raise exceptions on some platforms,
    so this function guards that step and falls back to text interpretation.

    Example
    -------
    .. code-block:: python

        from prodock.process.gridbox.parsers import parse_text_to_mol

        mol1 = parse_text_to_mol("ligand.sdf")
        mol2 = parse_text_to_mol(raw_mol2_text, fmt="mol2")
        mol3 = parse_text_to_mol(raw_pdb_text, fmt="pdb")
    """
    s = str(text_or_path)

    # Robust path detection: Path.exists() can raise for very long strings
    # such as multiline MolBlocks or other raw structure text.
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

    if _parse_with_project is not None:
        try:
            mol = _parse_with_project(text, fmt)
            if mol is not None:
                return mol
        except Exception:
            pass

    return _parse_with_rdkit(text, fmt)
