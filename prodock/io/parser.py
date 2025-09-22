# prodock/process/parser.py
"""
Small text-to-RDKit parsers used by gridbox and other modules.

Each helper returns the first successfully parsed :class:`rdkit.Chem.Mol`
or ``None`` when parsing failed. These functions intentionally swallow
parser exceptions and return ``None`` so callers can decide how to
handle format-specific failures.

They are small, single-responsibility helpers suitable for unit testing.
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
    Try to parse a single MDL Mol block with MolFromMolBlock.

    :param block: text containing a single mol block
    :returns: parsed Mol or None
    """
    try:
        m = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
    except Exception:
        m = None
    return m


def _try_blocks(blocks: List[str]) -> Optional[Mol]:
    """
    Iterate candidate blocks and return the first successfully parsed Mol.

    :param blocks: list of mol-block strings
    :returns: first parsed Mol or None
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
      - replace occurrences of "-0." with "0.000" (common fragile formatting).

    :param text: raw SDF text
    :returns: sanitized text
    """
    # Replace substrings like " -0." and "-0." conservatively
    sanitized = text.replace(" -0.", " 0.000").replace("-0.", "0.000")
    # Additionally handle isolated patterns with regex (defensive)
    sanitized = re.sub(r"(?<=\s)-0\.(?=\s)", "0.000", sanitized)
    return sanitized


def _supplier_first_mol(text: str) -> Optional[Mol]:
    """
    Write text to a temporary .sdf file and return first molecule from SDMolSupplier.

    :param text: full SDF content (may contain multiple records)
    :returns: first parsed Mol or None
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
    Parse SDF-like text and return the first valid RDKit ``Mol``.

    Flow:
      1. Quick-fail on empty input.
      2. Split by "$$$$" and attempt per-block MolFromMolBlock.
      3. If that fails, apply light sanitization and retry per-block parsing.
      4. If still failing, write the text to a temporary .sdf and use SDMolSupplier.

    :param text: SDF-style content (possibly containing multiple records).
    :returns: first parsed :class:`rdkit.Chem.Mol` or ``None`` if none parsed.
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
    Parse a PDB block into an RDKit Mol.

    :param text: PDB-format text (single model/block).
    :type text: str
    :returns: parsed :class:`rdkit.Chem.Mol` or ``None`` on failure.
    :rtype: rdkit.Chem.rdchem.Mol | None
    """
    try:
        m = Chem.MolFromPDBBlock(text, removeHs=False)
    except Exception:
        m = None
    return m


def _parse_mol2_text(text: str) -> Optional[Mol]:
    """
    Parse a MOL2 block into an RDKit Mol.

    Note: some RDKit builds or installs may omit MOL2 support; in those
    cases this function will typically return ``None``.

    :param text: MOL2-format text.
    :type text: str
    :returns: parsed :class:`rdkit.Chem.Mol` or ``None`` on failure.
    :rtype: rdkit.Chem.rdchem.Mol | None
    """
    try:
        m = Chem.MolFromMol2Block(text, sanitize=True, removeHs=False)
    except Exception:
        m = None
    return m


def _parse_xyz_text(text: str) -> Optional[Mol]:
    """
    Parse an XYZ-format block into an RDKit Mol.

    This uses :func:`rdkit.Chem.MolFromXYZBlock` when available; older RDKit
    builds may not provide this helper and the function will return ``None``.

    :param text: XYZ-format text.
    :type text: str
    :returns: parsed :class:`rdkit.Chem.Mol` or ``None`` on failure.
    :rtype: rdkit.Chem.rdchem.Mol | None
    """
    try:
        m = Chem.MolFromXYZBlock(text)
    except Exception:
        m = None
    return m
