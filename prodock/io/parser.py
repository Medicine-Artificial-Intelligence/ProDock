"""
Small text-to-RDKit parsers used by gridbox and other modules.

Each helper returns the first successfully parsed :class:`rdkit.Chem.rdchem.Mol`
or ``None`` when parsing failed. These functions intentionally swallow parser
exceptions and return ``None`` so callers can decide how to handle
format-specific failures.

They are small, single-responsibility helpers suitable for unit testing.

Module API
----------
Helpers provided:
- :func:`_parse_sdf_text` — robust first-molecule-from-SDF parsing (string input).
- :func:`_parse_pdb_text` — parse a PDB block.
- :func:`_parse_mol2_text` — parse a MOL2 block.
- :func:`_parse_xyz_text` — parse an XYZ block (if RDKit supports it).

All functions return either a :class:`rdkit.Chem.rdchem.Mol` instance on success
or ``None`` on failure.

Examples
--------
Simple usage::

    from prodock.process import parser
    sdf_content = open("example.sdf").read()
    mol = parser._parse_sdf_text(sdf_content)
    if mol is None:
        print("Failed to parse SDF")
    else:
        print("Parsed atoms:", mol.GetNumAtoms())

Notes
-----
- These helpers are defensive: they try multiple strategies (direct block parsing,
  light sanitization, and SDMolSupplier via a temporary file) to maximize the
  chance of successfully parsing real-world SDF artifacts.
- RDKit C/C++ warnings are silenced at import time for cleaner logs.
"""

from typing import Optional, List
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger
import tempfile
import os
import re

#  Quiet RDKit C/C++ warnings for parsing helpers
RDLogger.DisableLog("rdApp.*")


def _parse_block(block: str) -> Optional[Mol]:
    """
    Try to parse a single MDL Mol block using :func:`rdkit.Chem.MolFromMolBlock`.

    This is a thin wrapper that catches exceptions and returns ``None`` on failure.

    :param str block: Text containing a single mol block (MDL Mol format).
    :returns: Parsed RDKit Mol on success or ``None`` on failure.
    :rtype: Optional[:class:`rdkit.Chem.rdchem.Mol`]
    """
    try:
        m = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
    except Exception:
        m = None
    return m


def _try_blocks(blocks: List[str]) -> Optional[Mol]:
    """
    Iterate candidate mol-block strings and return the first successfully parsed Mol.

    Skips empty/whitespace-only blocks.

    :param List[str] blocks: List of mol-block strings to try.
    :returns: First parsed RDKit Mol or ``None`` if none could be parsed.
    :rtype: Optional[:class:`rdkit.Chem.rdchem.Mol`]
    """
    for block in blocks:
        if not block or not block.strip():
            continue
        m = _parse_block(block)
        if m is not None:
            return m
    return None


def _sanitize_text(text: str) -> str:
    """
    Apply lightweight sanitization for common SDF formatting issues.

    Current rules:
      - replace occurrences of ``-0.`` with a conservative ``0.000`` which is
        a frequent fragile formatting artifact in coordinate lines.

    :param str text: Raw SDF text to sanitize.
    :returns: Sanitized text (string).
    :rtype: str
    """
    # Replace substrings like " -0." and "-0." conservatively
    sanitized = text.replace(" -0.", " 0.000").replace("-0.", "0.000")
    # Additionally handle isolated patterns with regex (defensive)
    sanitized = re.sub(r"(?<=\s)-0\.(?=\s)", "0.000", sanitized)
    return sanitized


def _supplier_first_mol(text: str) -> Optional[Mol]:
    """
    Write text to a temporary ``.sdf`` file and return the first molecule from
    :class:`rdkit.Chem.SDMolSupplier`.

    This path is more robust for severely malformed SDFs where SDMolSupplier
    implements additional parsing heuristics.

    :param str text: Full SDF content (may contain multiple records).
    :returns: First parsed RDKit Mol or ``None`` if the supplier yields nothing.
    :rtype: Optional[:class:`rdkit.Chem.rdchem.Mol`]
    """
    tf = None
    try:
        # create a named temp file to allow SDMolSupplier to open it
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False)
        tf.write(text)
        tf.flush()
        tf.close()
        supplier = Chem.SDMolSupplier(tf.name, sanitize=True, removeHs=False)
        for mol in supplier:
            if mol is not None:
                return mol
    except Exception:
        # swallow supplier exceptions and return None
        return None
    finally:
        if tf is not None:
            try:
                os.unlink(tf.name)
            except Exception:
                pass
    return None


def _parse_sdf_text(text: str) -> Optional[Mol]:
    """
    Parse SDF-like text and return the first valid RDKit :class:`rdkit.Chem.rdchem.Mol`.

    Flow:
      1. Quick-fail on empty input.
      2. Split by ``$$$$`` and attempt per-block :func:`rdkit.Chem.MolFromMolBlock`.
      3. If that fails, apply light sanitization and retry per-block parsing.
      4. If still failing, write the text to a temporary .sdf and use
         :class:`rdkit.Chem.SDMolSupplier`.

    The functions intentionally swallow exceptions and return ``None`` rather than
    raising so callers can implement fallback behavior.

    :param str text: SDF-style content (possibly containing multiple records).
    :returns: First parsed RDKit Mol or ``None`` if none parsed.
    :rtype: Optional[:class:`rdkit.Chem.rdchem.Mol`]

    Example
    -------

    >>> mol = _parse_sdf_text(open("example.sdf").read())
    >>> if mol is None:
    ...     print("No molecule parsed")
    ... else:
    ...     print(mol.GetNumAtoms())
    """
    if not text or not text.strip():
        return None

    # preserve blocks (do not aggressively strip inner whitespace)
    blocks = [b for b in text.split("$$$$")]

    # 1) Fast path: try per-block MolFromMolBlock
    mol = _try_blocks(blocks)
    if mol is not None:
        return mol

    # 2) Try sanitized blocks (fix common '-0.' formatting)
    sanitized = _sanitize_text(text)
    mol = _try_blocks([b for b in sanitized.split("$$$$")])
    if mol is not None:
        return mol

    # 3) Final robust attempt using SDMolSupplier on a temp file
    mol = _supplier_first_mol(text)
    return mol


def _parse_pdb_text(text: str) -> Optional[Mol]:
    """
    Parse a PDB block into an RDKit Mol using :func:`rdkit.Chem.MolFromPDBBlock`.

    :param str text: PDB-format text (single model/block).
    :returns: Parsed :class:`rdkit.Chem.rdchem.Mol` or ``None`` on failure.
    :rtype: Optional[:class:`rdkit.Chem.rdchem.Mol`]
    """
    try:
        m = Chem.MolFromPDBBlock(text, removeHs=False)
    except Exception:
        m = None
    return m


def _parse_mol2_text(text: str) -> Optional[Mol]:
    """
    Parse a MOL2 block into an RDKit Mol.

    Note
    ----
    Some RDKit builds or installs may omit MOL2 support; in those cases this
    function will typically return ``None``.

    :param str text: MOL2-format text.
    :returns: Parsed :class:`rdkit.Chem.rdchem.Mol` or ``None`` on failure.
    :rtype: Optional[:class:`rdkit.Chem.rdchem.Mol`]
    """
    try:
        m = Chem.MolFromMol2Block(text, sanitize=True, removeHs=False)
    except Exception:
        m = None
    return m


def _parse_xyz_text(text: str) -> Optional[Mol]:
    """
    Parse an XYZ-format block into an RDKit Mol.

    Uses :func:`rdkit.Chem.MolFromXYZBlock` when available; older RDKit builds may
    not provide this helper and the function will return ``None``.

    :param str text: XYZ-format text.
    :returns: Parsed :class:`rdkit.Chem.rdchem.Mol` or ``None`` on failure.
    :rtype: Optional[:class:`rdkit.Chem.rdchem.Mol`]
    """
    try:
        m = Chem.MolFromXYZBlock(text)
    except Exception:
        m = None
    return m
