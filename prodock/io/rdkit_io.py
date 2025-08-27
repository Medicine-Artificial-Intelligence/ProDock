# prodock/io/file.py
"""
RDKit-only molecular I/O helpers that prefer prodock.chem.Conformer for
embedding/optimization.

Functions:
 - smiles2mol, mol2smiles
 - smiles2sdf, sdf2mol, sdf2mols, sdftosmiles, mol2sdf
 - smiles2pdb, pdb2mol, pdb2smiles, mol2pdb
 - convenience: mol_from_smiles_write_all_formats, is_valid_smiles

If prodock.chem.conformer.Conformer is available it will be used for any
operation that requires embedding 3D coordinates or force-field optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union, Dict
import logging

# RDKit imports (required)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    raise ImportError(
        "RDKit import failed. Please install RDKit (conda install -c conda-forge rdkit) "
        "or ensure it is on PYTHONPATH."
    ) from e

# prodock logging utilities (preferred)
try:
    from prodock.io.logging import get_logger, StructuredAdapter
except Exception:
    # fallback
    def get_logger(name: str):
        return logging.getLogger(name)

    class StructuredAdapter(logging.LoggerAdapter):
        def __init__(self, logger, extra):
            super().__init__(logger, extra)


logger = StructuredAdapter(get_logger("prodock.io.file"), {"component": "file"})
logger._base_logger = getattr(logger, "_base_logger", getattr(logger, "logger", None))

# Try to import internal Conformer (preferred for embedding/optimization)
try:
    from prodock.chem.conformer import Conformer  # type: ignore

    _HAS_CONFORMER = True
except Exception:
    Conformer = None  # type: ignore
    _HAS_CONFORMER = False
    logger.debug(
        "prodock.chem.conformer.Conformer not available; falling back to RDKit-only methods."
    )


# ---------- Basic SMILES <-> Mol ----------
def smiles2mol(smiles: str, sanitize: bool = True) -> Chem.Mol:
    """Convert a SMILES string to an RDKit Mol (raises ValueError on failure)."""
    if not smiles:
        raise ValueError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")
    return mol


def mol2smiles(mol: Chem.Mol, canonical: bool = True, isomeric: bool = True) -> str:
    """Convert an RDKit Mol to a SMILES string."""
    if mol is None:
        raise ValueError("mol must be an RDKit Mol")
    return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)


# ---------- SDF readers/writers ----------


def mol2sdf(
    mol: Chem.Mol,
    out_path: Union[str, Path],
    sanitize: bool = True,
    embed3d: bool = True,
    add_hs: bool = True,
    optimize: bool = True,
    embed_algorithm: Optional[str] = "ETKDGv3",
    opt_method: Optional[str] = "MMFF94",
    conformer_seed: int = 42,
    conformer_n_jobs: int = 1,
    opt_max_iters: int = 200,
) -> Path:
    """
    Write a single RDKit Mol to an SDF file. If embed3d True but the mol lacks coordinates,
    prefer using internal Conformer to generate+optimize coordinates; otherwise fall back to RDKit.
    """
    out_path = Path(out_path)
    if mol is None:
        raise ValueError("mol must be an RDKit Mol")

    # If no conformers and user requested embed3d, try using Conformer if available
    if embed3d and mol.GetNumConformers() == 0:
        # prefer Conformer when available
        if _HAS_CONFORMER:
            try:
                smiles = mol2smiles(mol)
                cm = Conformer(seed=conformer_seed)
                cm.load_smiles([smiles])
                cm.embed_all(
                    n_confs=1,
                    n_jobs=conformer_n_jobs,
                    add_hs=bool(add_hs),
                    embed_algorithm=embed_algorithm or "ETKDGv3",
                )
                if optimize:
                    cm.optimize_all(
                        method=(opt_method or "MMFF94"),
                        n_jobs=conformer_n_jobs,
                        max_iters=opt_max_iters,
                    )
                # get generated molblock
                if not cm.molblocks:
                    raise RuntimeError(
                        "Conformer failed to produce an embedded molecule"
                    )
                mb = cm.molblocks[0]
                m2 = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=(not add_hs))
                if m2 is None:
                    raise RuntimeError("Failed to parse MolBlock produced by Conformer")
                writer = Chem.SDWriter(str(out_path))
                writer.write(m2)
                writer.close()
                return out_path
            except Exception as exc:
                logger.warning(
                    "Conformer-based embedding failed, falling back to RDKit embed: %s",
                    exc,
                )

        # fallback RDKit embedding
        working = Chem.Mol(mol)
        if add_hs:
            working = Chem.AddHs(working)
        params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else None
        try:
            if params is not None:
                AllChem.EmbedMolecule(working, params)
            else:
                AllChem.EmbedMolecule(working)
        except Exception:
            try:
                AllChem.EmbedMolecule(working)
            except Exception:
                logger.exception("RDKit EmbedMolecule failed")
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
        writer = Chem.SDWriter(str(out_path))
        writer.write(working)
        writer.close()
        return out_path

    # default: write given molecule directly
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()
    return out_path


def smiles2sdf(
    smiles: str,
    out_path: Union[str, Path],
    embed3d: bool = True,
    add_hs: bool = True,
    optimize: bool = True,
    embed_algorithm: Optional[str] = "ETKDGv3",
    opt_method: Optional[str] = "MMFF94",
    conformer_seed: int = 42,
    conformer_n_jobs: int = 1,
    opt_max_iters: int = 200,
) -> Path:
    """
    Convert SMILES -> SDF (single molecule). Prefer internal Conformer for embedding/optimization.
    """
    out_path = Path(out_path)
    if not smiles:
        raise ValueError("smiles must be provided")
    # If embedding/optimization requested prefer Conformer
    if embed3d or optimize:
        if _HAS_CONFORMER:
            cm = Conformer(seed=conformer_seed)
            cm.load_smiles([smiles])
            cm.embed_all(
                n_confs=1,
                n_jobs=conformer_n_jobs,
                add_hs=bool(add_hs),
                embed_algorithm=embed_algorithm,
            )
            if optimize:
                cm.optimize_all(
                    method=(opt_method or "MMFF94"),
                    n_jobs=conformer_n_jobs,
                    max_iters=opt_max_iters,
                )
            if not cm.molblocks:
                raise RuntimeError("Conformer failed to produce an embedded molecule")
            mb = cm.molblocks[0]
            m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=(not add_hs))
            if m is None:
                raise RuntimeError("Failed to parse MolBlock produced by Conformer")
            writer = Chem.SDWriter(str(out_path))
            writer.write(m)
            writer.close()
            return out_path
        else:
            # fallback to RDKit direct (existing behavior)
            mol = smiles2mol(smiles, sanitize=True)
            working = Chem.AddHs(mol) if add_hs else Chem.Mol(mol)
            params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else None
            try:
                if params is not None:
                    AllChem.EmbedMolecule(working, params)
                else:
                    AllChem.EmbedMolecule(working)
            except Exception:
                try:
                    AllChem.EmbedMolecule(working)
                except Exception:
                    logger.exception("RDKit EmbedMolecule failed")
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
            writer = Chem.SDWriter(str(out_path))
            writer.write(working)
            writer.close()
            return out_path

    # no embedding requested -> convert SMILES to Mol and write as SDF
    mol = smiles2mol(smiles, sanitize=True)
    return mol2sdf(mol, out_path)


def sdf2mol(
    sdf_path: Union[str, Path], sanitize: bool = True, removeHs: bool = False
) -> Optional[Chem.Mol]:
    """Load the first molecule from an SDF file."""
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize)
    for m in supplier:
        if m is None:
            continue
        if removeHs:
            m = Chem.RemoveHs(m)
        return m
    return None


def sdf2mols(sdf_path: Union[str, Path], sanitize: bool = True) -> List[Chem.Mol]:
    """Load all molecules from an SDF file."""
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize)
    return [m for m in supplier if m is not None]


def sdftosmiles(sdf_path: Union[str, Path], sanitize: bool = True) -> List[str]:
    """Read an SDF file and return a list of SMILES (one per molecule)."""
    mols = sdf2mols(sdf_path, sanitize=sanitize)
    return [mol2smiles(m) for m in mols]


# ---------- PDB readers/writers ----------


def mol2pdb(
    mol: Chem.Mol,
    out_path: Union[str, Path],
    add_hs: bool = False,
    embed3d: bool = False,
    optimize: bool = True,
    embed_algorithm: Optional[str] = None,
    opt_method: Optional[str] = None,
    conformer_seed: int = 42,
    conformer_n_jobs: int = 1,
    opt_max_iters: int = 200,
) -> Path:
    """
    Write an RDKit Mol to a PDB file. If the mol lacks coordinates and embed3d True,
    prefer the internal Conformer to create coordinates; otherwise fall back to RDKit.
    """
    out_path = Path(out_path)
    if mol is None:
        raise ValueError("mol must be an RDKit Mol")

    if embed3d and mol.GetNumConformers() == 0:
        if _HAS_CONFORMER:
            try:
                s = mol2smiles(mol)
                cm = Conformer(seed=conformer_seed)
                cm.load_smiles([s])
                cm.embed_all(
                    n_confs=1,
                    n_jobs=conformer_n_jobs,
                    add_hs=bool(add_hs),
                    embed_algorithm=embed_algorithm or "ETKDGv3",
                )
                if optimize:
                    cm.optimize_all(
                        method=(opt_method or "MMFF94"),
                        n_jobs=conformer_n_jobs,
                        max_iters=opt_max_iters,
                    )
                if not cm.molblocks:
                    raise RuntimeError("Conformer failed to produce coordinates")
                mb = cm.molblocks[0]
                m2 = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=(not add_hs))
                if m2 is None:
                    raise RuntimeError("Failed to parse MolBlock produced by Conformer")
                Chem.MolToPDBFile(m2, str(out_path))
                return out_path
            except Exception as exc:
                logger.warning(
                    "Conformer-based PDB generation failed, falling back to RDKit: %s",
                    exc,
                )

        # fallback RDKit embedding
        working = Chem.Mol(mol)
        if add_hs:
            working = Chem.AddHs(working)
        params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else None
        try:
            if params is not None:
                AllChem.EmbedMolecule(working, params)
            else:
                AllChem.EmbedMolecule(working)
        except Exception:
            try:
                AllChem.EmbedMolecule(working)
            except Exception:
                logger.exception("RDKit EmbedMolecule failed")
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
        Chem.MolToPDBFile(working, str(out_path))
        return out_path

    # no embedding required; just write existing coordinates (or RDKit will write no coords)
    Chem.MolToPDBFile(mol, str(out_path))
    return out_path


def smiles2pdb(
    smiles: str,
    out_path: Union[str, Path],
    add_hs: bool = False,
    embed3d: bool = True,
    optimize: bool = True,
    embed_algorithm: Optional[str] = "ETKDGv3",
    opt_method: Optional[str] = "MMFF94",
    conformer_seed: int = 42,
    conformer_n_jobs: int = 1,
    opt_max_iters: int = 200,
) -> Path:
    """Convert SMILES -> PDB file using Conformer when embedding/optimization is needed."""
    if not smiles:
        raise ValueError("smiles must be provided")
    # Prefer Conformer for embedding/optimization
    if _HAS_CONFORMER and (embed3d or optimize):
        cm = Conformer(seed=conformer_seed)
        cm.load_smiles([smiles])
        cm.embed_all(
            n_confs=1,
            n_jobs=conformer_n_jobs,
            add_hs=bool(add_hs),
            embed_algorithm=embed_algorithm,
        )
        if optimize:
            cm.optimize_all(
                method=(opt_method or "MMFF94"),
                n_jobs=conformer_n_jobs,
                max_iters=opt_max_iters,
            )
        if not cm.molblocks:
            raise RuntimeError("Conformer failed to produce an embedded molecule")
        mb = cm.molblocks[0]
        m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=(not add_hs))
        if m is None:
            raise RuntimeError("Failed to parse MolBlock produced by Conformer")
        Chem.MolToPDBFile(m, str(out_path))
        return Path(out_path)
    # fallback to RDKit direct
    mol = smiles2mol(smiles, sanitize=True)
    return mol2pdb(mol, out_path, add_hs=add_hs, embed3d=embed3d, optimize=optimize)


def pdb2mol(
    pdb_path: Union[str, Path], sanitize: bool = True, removeHs: bool = False
) -> Optional[Chem.Mol]:
    """Load a molecule from a PDB file."""
    m = Chem.MolFromPDBFile(str(pdb_path), sanitize=sanitize, removeHs=False)
    if m is None:
        return None
    if removeHs:
        m = Chem.RemoveHs(m)
    return m


def pdb2smiles(
    pdb_path: Union[str, Path], sanitize: bool = True, removeHs: bool = True
) -> str:
    """Load a PDB and return a SMILES string."""
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
    add_hs: bool = True,
    embed_algorithm: Optional[str] = "ETKDGv3",
    opt_method: Optional[str] = "MMFF94",
) -> Dict[str, Path]:
    """
    Convenience helper: from SMILES write SDF and/or PDB with the same prefix.
    Uses Conformer for embedding/optimization when available.
    """
    prefix = Path(out_prefix)
    results: Dict[str, Path] = {}
    if write_sdf:
        sdfp = prefix.with_suffix(".sdf")
        smiles2sdf(
            smiles,
            sdfp,
            embed3d=embed3d,
            add_hs=add_hs,
            optimize=True,
            embed_algorithm=embed_algorithm,
            opt_method=opt_method,
        )
        results["sdf"] = Path(sdfp)
    if write_pdb:
        pdbp = prefix.with_suffix(".pdb")
        smiles2pdb(
            smiles,
            pdbp,
            add_hs=add_hs,
            embed3d=embed3d,
            optimize=True,
            embed_algorithm=embed_algorithm,
            opt_method=opt_method,
        )
        results["pdb"] = Path(pdbp)
    return results


# ---------- Small utilities ----------


def is_valid_smiles(smiles: str) -> bool:
    """Quick check whether a SMILES string can be parsed by RDKit."""
    try:
        m = Chem.MolFromSmiles(smiles)
        return m is not None
    except Exception:
        return False


# Expose simple API
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
