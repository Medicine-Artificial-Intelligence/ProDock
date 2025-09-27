"""
prodock.io.parser
=================

Small text-to-RDKit parsers used by gridbox and other modules.

These helpers attempt to parse molecular text blocks (SDF, PDB, MOL2, XYZ)
and return the first successfully parsed :class:`rdkit.Chem.rdchem.Mol`
or ``None`` when parsing failed. They are defensive by design: parser
exceptions are swallowed and ``None`` is returned so callers can decide
fallback behavior.

The main helpers are:

- :func:`_parse_sdf_text` — robust first-molecule-from-SDF parsing (string input).
- :func:`_parse_pdb_text` — parse a PDB block.
- :func:`_parse_mol2_text` — parse a MOL2 block.
- :func:`_parse_xyz_text` — parse an XYZ block (if RDKit supports it).

These functions are intentionally small and easy to unit-test.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger

# Quiet RDKit C/C++ warnings for parsing helpers (defensive)
try:
    RDLogger.DisableLog("rdApp.*")
except Exception:
    # If RDLogger is unavailable or disabling fails, continue silently.
    pass


def _parse_block(block: str) -> Optional[Mol]:
    """
    Try to parse a single MDL mol block using RDKit.

    :param block: Text containing a single MDL Mol block.
    :type block: str
    :return: Parsed RDKit Mol on success, otherwise ``None``.
    :rtype: Optional[rdkit.Chem.rdchem.Mol]
    """
    try:
        m = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
    except Exception:
        m = None
    return m


def _try_blocks(blocks: List[str]) -> Optional[Mol]:
    """
    Iterate candidate mol-block strings and return the first successfully
    parsed molecule.

    :param blocks: List of candidate mol-block strings.
    :type blocks: List[str]
    :return: First parsed RDKit Mol or ``None`` if none parsed.
    :rtype: Optional[rdkit.Chem.rdchem.Mol]
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
    Apply light sanitization to SDF text to handle some common malformed cases.

    Current rules: conservative replacements of "-0." coordinate artifacts that
    sometimes break strict parsers.

    :param text: Raw SDF text.
    :type text: str
    :return: Sanitized text.
    :rtype: str
    """
    sanitized = text.replace(" -0.", " 0.000").replace("-0.", "0.000")
    try:
        sanitized = re.sub(r"(?<=\s)-0\.(?=\s)", "0.000", sanitized)
    except re.error:
        # If regex fails for any reason, fall back to the simple replacements.
        pass
    return sanitized


def _supplier_first_mol(text: str) -> Optional[Mol]:
    """
    Use RDKit's SDMolSupplier as a robust fallback for SDF parsing.

    Writes the provided text to a temporary .sdf file and iterates the
    SDMolSupplier collecting valid molecules. The temporary file is removed
    after supplier iteration completes to avoid races where RDKit may still
    read the file.

    :param text: Full SDF content (may contain multiple records).
    :type text: str
    :return: First parsed RDKit Mol or ``None``.
    :rtype: Optional[rdkit.Chem.rdchem.Mol]
    """
    tf_path = None
    mols: List[Mol] = []
    try:
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False)
        tf_path = tf.name
        tf.write(text)
        tf.flush()
        tf.close()
        supplier = Chem.SDMolSupplier(tf_path, sanitize=True, removeHs=False)
        for m in supplier:
            if m is not None:
                mols.append(m)
    except Exception:
        # Swallow any supplier-related exceptions and fall through to return None.
        mols = []
    finally:
        if tf_path is not None:
            try:
                os.unlink(tf_path)
            except Exception:
                # Ignore cleanup failures
                pass
    return mols[0] if mols else None


def _parse_sdf_text(text: str) -> Optional[Mol]:
    """
    Parse SDF-like text and return the first valid RDKit Mol.

    Strategy:
      1. Quick-fail on empty input.
      2. Split by ``$$$$`` and attempt per-block ``MolFromMolBlock``.
      3. If that fails, apply light sanitization and retry per-block parsing.
      4. If still failing, write the text to a temporary .sdf file and use
         ``SDMolSupplier`` as a robust fallback.

    :param text: SDF-style content (possibly multiple records).
    :type text: str
    :return: First parsed RDKit Mol or ``None`` if parsing fails.
    :rtype: Optional[rdkit.Chem.rdchem.Mol]
    """
    if not text or not text.strip():
        return None

    # Fast path: attempt per-block MolFromMolBlock parsing
    blocks = [b for b in text.split("$$$$")]
    mol = _try_blocks(blocks)
    if mol is not None:
        return mol

    # Sanitized retry
    sanitized = _sanitize_text(text)
    mol = _try_blocks([b for b in sanitized.split("$$$$")])
    if mol is not None:
        return mol

    # Last-resort: let SDMolSupplier parse the (possibly malformed) SDF
    return _supplier_first_mol(text)


def _parse_pdb_text(text: str) -> Optional[Mol]:
    """
    Parse a PDB block into an RDKit Mol.

    :param text: PDB-format text (single model/block).
    :type text: str
    :return: Parsed RDKit Mol or ``None`` on failure.
    :rtype: Optional[rdkit.Chem.rdchem.Mol]
    """
    try:
        m = Chem.MolFromPDBBlock(text, removeHs=False)
    except Exception:
        m = None
    return m


def _parse_mol2_text(text: str) -> Optional[Mol]:
    """
    Parse a MOL2 block into an RDKit Mol.

    Note:
        Some RDKit binaries may lack MOL2 parsing support. In that case this
        function typically returns ``None``.

    :param text: MOL2-format text.
    :type text: str
    :return: Parsed RDKit Mol or ``None`` on failure.
    :rtype: Optional[rdkit.Chem.rdchem.Mol]
    """
    try:
        m = Chem.MolFromMol2Block(text, sanitize=True, removeHs=False)
    except Exception:
        m = None
    return m


def _parse_xyz_text(text: str) -> Optional[Mol]:
    """
    Parse an XYZ-format block into an RDKit Mol.

    Uses ``Chem.MolFromXYZBlock`` when available; older RDKit releases may not
    provide this helper and the function will return ``None`` in that case.

    :param text: XYZ-format text.
    :type text: str
    :return: Parsed RDKit Mol or ``None`` on failure.
    :rtype: Optional[rdkit.Chem.rdchem.Mol]
    """
    try:
        m = Chem.MolFromXYZBlock(text)
    except Exception:
        m = None
    return m
