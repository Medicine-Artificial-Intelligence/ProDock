# rdkit_io.py
"""
RDKit-only molecular I/O helpers.

Provides simple, well-documented functions for common conversions:
- smiles2mol, mol2smiles
- smiles2sdf, sdf2smiles, mol2sdf, sdf2mol
- smiles2pdb, pdb2smiles, mol2pdb, pdb2mol

All functions use RDKit (rdkit.Chem and rdkit.Chem.AllChem).
If RDKit is not available an ImportError is raised immediately.

Usage examples
--------------
>>> from rdkit_io import smiles2mol, mol2smiles
>>> m = smiles2mol("CCO")
>>> mol2smiles(m)
'CCO'
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

# Ensure RDKit is available
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception as e:
    raise ImportError(
        "RDKit import failed. Please install RDKit (conda install -c conda-forge rdkit) "
        "or ensure it is on PYTHONPATH."
    ) from e


# ---------- Basic SMILES <-> Mol ----------


def smiles2mol(smiles: str, sanitize: bool = True) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit Mol.

    :param smiles: SMILES string.
    :param sanitize: If True, run RDKit sanitization on the generated molecule.
    :return: RDKit Mol object (or None if conversion fails).
    :rtype: Chem.Mol
    """
    if smiles is None:
        raise ValueError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")
    return mol


def mol2smiles(mol: Chem.Mol, canonical: bool = True, isomeric: bool = True) -> str:
    """
    Convert an RDKit Mol to a SMILES string.

    :param mol: RDKit Mol object.
    :param canonical: If True, produce canonical SMILES.
    :param isomeric: If True, include stereochemistry in SMILES.
    :return: SMILES string.
    :rtype: str
    """
    if mol is None:
        raise ValueError("mol must be an RDKit Mol")
    return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)


# ---------- SDF readers/writers ----------


def mol2sdf(mol: Chem.Mol, out_path: Union[str, Path], sanitize: bool = True) -> Path:
    """
    Write a single RDKit Mol to an SDF file.

    :param mol: RDKit Mol object.
    :param out_path: Destination file path (overwrites if exists).
    :param sanitize: If True, attempt to sanitize molecule before writing.
    :return: Path to written SDF file.
    :rtype: Path
    """
    out_path = Path(out_path)
    if mol is None:
        raise ValueError("mol must be a RDKit Mol")
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # continue even if sanitize fails; RDKit can still write
            pass
    writer = Chem.SDWriter(str(out_path))
    if writer is None:
        raise IOError(f"Unable to create SDWriter for {out_path}")
    writer.write(mol)
    writer.close()
    return out_path


def smiles2sdf(
    smiles: str,
    out_path: Union[str, Path],
    embed3d: bool = False,
    add_hs: bool = False,
    optimize: bool = True,
) -> Path:
    """
    Convert SMILES -> SDF (single molecule). Optionally embed 3D coords.

    :param smiles: Input SMILES string.
    :param out_path: Destination SDF file path.
    :param embed3d: If True, generate 3D coordinates (RDKit EmbedMolecule).
    :param add_hs: If True, add hydrogens before embedding/optimization.
    :param optimize: If True and embed3d True, run a force-field optimization.
    :return: Path to written SDF file.
    :rtype: Path
    """
    mol = smiles2mol(smiles, sanitize=True)
    if embed3d:
        working = Chem.AddHs(mol) if add_hs else Chem.Mol(mol)
        # embed
        params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else None
        if params is not None:
            try:
                AllChem.EmbedMolecule(working, params)
            except Exception:
                AllChem.EmbedMolecule(working)
        else:
            AllChem.EmbedMolecule(working)
        if optimize:
            try:
                AllChem.UFFOptimizeMolecule(working)
            except Exception:
                try:
                    AllChem.MMFFOptimizeMolecule(working)
                except Exception:
                    pass
        # remove Hs if we didn't want them in final file
        if not add_hs:
            working = Chem.RemoveHs(working)
        mol = working
    return mol2sdf(mol, out_path)


def sdf2mol(
    sdf_path: Union[str, Path], sanitize: bool = True, removeHs: bool = False
) -> Optional[Chem.Mol]:
    """
    Load the first molecule from an SDF file.

    :param sdf_path: Path to SDF file.
    :param sanitize: If True, sanitize molecules on load.
    :param removeHs: If True, remove hydrogens from returned Mol.
    :return: First RDKit Mol in file or None if file empty.
    :rtype: Chem.Mol | None
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize)
    # SDMolSupplier is lazy; iterate once
    for m in supplier:
        if m is None:
            continue
        if removeHs:
            m = Chem.RemoveHs(m)
        return m
    return None


def sdf2mols(sdf_path: Union[str, Path], sanitize: bool = True) -> List[Chem.Mol]:
    """
    Load all molecules from an SDF file.

    :param sdf_path: Path to SDF file.
    :param sanitize: If True, sanitize molecules on load.
    :return: List of RDKit Mol objects (may be empty).
    :rtype: list[Chem.Mol]
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize)
    return [m for m in supplier if m is not None]


def sdftosmiles(sdf_path: Union[str, Path], sanitize: bool = True) -> List[str]:
    """
    Read an SDF file and return a list of SMILES (one per molecule).

    :param sdf_path: Path to SDF file.
    :param sanitize: If True, sanitize molecules on load.
    :return: List of SMILES strings.
    :rtype: list[str]
    """
    mols = sdf2mols(sdf_path, sanitize=sanitize)
    return [mol2smiles(m) for m in mols]


# ---------- PDB readers/writers ----------


def mol2pdb(
    mol: Chem.Mol,
    out_path: Union[str, Path],
    add_hs: bool = False,
    embed3d: bool = False,
    optimize: bool = True,
) -> Path:
    """
    Write an RDKit Mol to a PDB file. If molecule has no conformer,
    embed3d can be used to generate coordinates.

    :param mol: RDKit Mol object.
    :param out_path: Destination PDB file path.
    :param add_hs: If True, add hydrogens before writing.
    :param embed3d: If True, embed 3D coordinates when no conformer exists.
    :param optimize: If True and embed3d True, run force-field optimization.
    :return: Path to written PDB file.
    :rtype: Path
    """
    out_path = Path(out_path)
    if mol is None:
        raise ValueError("mol must be an RDKit Mol")
    working = Chem.Mol(mol)
    # ensure coordinates
    if working.GetNumConformers() == 0 and embed3d:
        working = Chem.AddHs(working) if add_hs else Chem.Mol(working)
        params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else None
        if params is not None:
            try:
                AllChem.EmbedMolecule(working, params)
            except Exception:
                AllChem.EmbedMolecule(working)
        else:
            AllChem.EmbedMolecule(working)
        if optimize:
            try:
                AllChem.UFFOptimizeMolecule(working)
            except Exception:
                try:
                    AllChem.MMFFOptimizeMolecule(working)
                except Exception:
                    pass
        if not add_hs:
            working = Chem.RemoveHs(working)
    # write PDB
    Chem.MolToPDBFile(working, str(out_path))
    return out_path


def smiles2pdb(
    smiles: str,
    out_path: Union[str, Path],
    add_hs: bool = False,
    embed3d: bool = True,
    optimize: bool = True,
) -> Path:
    """
    Convert SMILES -> PDB file. Uses embedding + optional optimization to get 3D coords.

    :param smiles: Input SMILES string.
    :param out_path: Destination PDB file path.
    :param add_hs: If True, add hydrogens to molecule before embedding.
    :param embed3d: If True, embed 3D coordinates (recommended for PDB).
    :param optimize: If True, run force-field optimization after embedding.
    :return: Path to written PDB file.
    :rtype: Path
    """
    mol = smiles2mol(smiles, sanitize=True)
    return mol2pdb(mol, out_path, add_hs=add_hs, embed3d=embed3d, optimize=optimize)


def pdb2mol(
    pdb_path: Union[str, Path], sanitize: bool = True, removeHs: bool = False
) -> Optional[Chem.Mol]:
    """
    Load a molecule from a PDB file.

    :param pdb_path: Path to PDB file.
    :param sanitize: If True, run RDKit sanitization on load.
    :param removeHs: If True, remove explicit hydrogens from returned Mol.
    :return: RDKit Mol object or None if load fails.
    :rtype: Chem.Mol | None
    """
    m = Chem.MolFromPDBFile(str(pdb_path), sanitize=sanitize, removeHs=False)
    if m is None:
        return None
    if removeHs:
        m = Chem.RemoveHs(m)
    return m


def pdb2smiles(
    pdb_path: Union[str, Path], sanitize: bool = True, removeHs: bool = True
) -> str:
    """
    Load a PDB and return a SMILES string (tries to canonicalize).

    :param pdb_path: Path to PDB file.
    :param sanitize: If True, run RDKit sanitization on load.
    :param removeHs: If True, remove hydrogens before converting to SMILES.
    :return: SMILES string.
    :rtype: str
    """
    m = pdb2mol(pdb_path, sanitize=sanitize, removeHs=removeHs)
    if m is None:
        raise ValueError(f"Failed to read PDB file: {pdb_path}")
    return mol2smiles(m)


# ---------- Convenience wrappers ----------


def mol_from_smiles_write_all_formats(
    smiles: str,
    out_prefix: Union[str, Path],
    write_sdf: bool = True,
    write_pdb: bool = True,
    embed3d: bool = True,
    add_hs: bool = False,
) -> dict:
    """
    Convenience helper: from SMILES write SDF and/or PDB with the same prefix.

    :param smiles: Input SMILES string.
    :param out_prefix: Output filename prefix (no extension).
    :param write_sdf: If True, write <prefix>.sdf.
    :param write_pdb: If True, write <prefix>.pdb.
    :param embed3d: If True, embed 3D coordinates before writing.
    :param add_hs: If True, add hydrogens before embedding.
    :return: Dict mapping 'sdf'/'pdb' to written Path (if written).
    :rtype: dict
    """
    prefix = Path(out_prefix)
    mol = smiles2mol(smiles, sanitize=True)
    results = {}
    if write_sdf:
        sdfp = prefix.with_suffix(".sdf")
        smiles2sdf(smiles, sdfp, embed3d=embed3d, add_hs=add_hs)
        results["sdf"] = sdfp
    if write_pdb:
        pdbp = prefix.with_suffix(".pdb")
        smiles2pdb(smiles, pdbp, add_hs=add_hs, embed3d=embed3d)
        results["pdb"] = pdbp
    return results


# ---------- Small utilities ----------


def is_valid_smiles(smiles: str) -> bool:
    """
    Quick check whether a SMILES string can be parsed by RDKit.

    :param smiles: SMILES string to check.
    :return: True if parsable, False otherwise.
    :rtype: bool
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        return m is not None
    except Exception:
        return False


# Expose a simple API
__all__ = [
    "smiles2mol",
    "mol2smiles",
    "smiles2sdf",
    "sdf2mol",
    "sdf2mols",
    "sdftosmiles",
    "mol2sdf",
    "mol2pdb",
    "pdb2mol",
    "pdb2smiles",
    "smiles2pdb",
    "mol_from_smiles_write_all_formats",
    "is_valid_smiles",
]
