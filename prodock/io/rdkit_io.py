# prodock/io/rdkit_io.py
"""
RDKit-only molecular I/O helpers that prefer prodock.chem.Conformer for
embedding/optimization.

Provides:
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


# ---------------------
# Helpers
# ---------------------
def _write_sdf(mol: Chem.Mol, out_path: Union[str, Path]) -> Path:
    """
    Write a molecule to an SDF file (single-record). Ensures writer is closed.

    :param mol: RDKit Mol to write
    :param out_path: output path
    :returns: Path to written file
    """
    out_path = Path(out_path)
    writer = None
    try:
        writer = Chem.SDWriter(str(out_path))
        writer.write(mol)
    finally:
        if writer is not None:
            writer.close()
    return out_path


def _get_embed_params(embed_algorithm: Optional[str]):
    """
    Return an RDKit Embed parameters object if available for `embed_algorithm`,
    otherwise return None.

    :param embed_algorithm: name of an AllChem embed params factory, e.g. "ETKDGv3"
    :returns: params object or None
    """
    if not embed_algorithm:
        return None
    factory = getattr(AllChem, embed_algorithm, None)
    if callable(factory):
        try:
            return factory()
        except Exception:
            # If instantiation fails, fall back to None (RDKit default)
            logger.debug(
                "Failed to instantiate embed params '%s'",
                embed_algorithm,
                exc_info=True,
            )
            return None
    return None


def _try_embed(working: Chem.Mol, params) -> bool:
    """
    Try to embed `working` with given params, with a simple fallback.
    Returns True if embedding likely succeeded (no exception raised).
    """
    try:
        if params is not None:
            AllChem.EmbedMolecule(working, params)
        else:
            AllChem.EmbedMolecule(working)
        return True
    except Exception:
        # One more simple fallback attempt without params
        try:
            AllChem.EmbedMolecule(working)
            return True
        except Exception:
            logger.exception("RDKit EmbedMolecule failed")
            return False


def _optimize_with_method(
    working: Chem.Mol, method: Optional[str], max_iters: int
) -> bool:
    """
    Attempt optimization using the preferred `method` and sensible fallback.
    Returns True on success (no exception), False otherwise.
    """
    # Prefer MMFF unless method explicitly indicates UFF
    preferred = (method or "MMFF94").upper()
    tried = []

    def _try_mmff():
        try:
            AllChem.MMFFOptimizeMolecule(working, maxIters=max_iters)
            return True
        except Exception:
            return False

    def _try_uff():
        try:
            AllChem.UFFOptimizeMolecule(working, maxIters=max_iters)
            return True
        except Exception:
            return False

    if preferred.startswith("U"):
        tried.append("UFF")
        if _try_uff():
            return True
        tried.append("MMFF")
        if _try_mmff():
            return True
    else:
        tried.append("MMFF")
        if _try_mmff():
            return True
        tried.append("UFF")
        if _try_uff():
            return True

    logger.debug("Optimizers %s all failed", tried)
    return False


def _rdkit_embed_and_optimize(
    mol: Chem.Mol,
    add_hs: bool = True,
    embed_algorithm: Optional[str] = None,
    optimize: bool = True,
    opt_method: Optional[str] = "MMFF94",
    opt_max_iters: int = 200,
) -> Chem.Mol:
    """
    (Refactored coordinator) Embed and optionally optimize `mol` using RDKit.
    This function delegates work to small helpers to keep complexity low.

    Returns a molecule with coordinates (and hydrogens according to add_hs).
    """
    # create working copy and add Hs if requested
    working = Chem.Mol(mol)
    if add_hs:
        working = Chem.AddHs(working)

    params = _get_embed_params(embed_algorithm)

    # embedding
    embedded_ok = _try_embed(working, params)
    if not embedded_ok:
        # embedding failed but we return whatever RDKit produced (no exception)
        logger.warning("RDKit embedding did not succeed cleanly for mol; continuing")

    # optimization (best-effort)
    if optimize:
        opt_ok = _optimize_with_method(working, opt_method, opt_max_iters)
        if not opt_ok:
            logger.debug("RDKit optimization failed for all tried force fields")

    if not add_hs:
        working = Chem.RemoveHs(working)

    return working


def _use_conformer_for_smiles(
    smiles: str,
    conformer_seed: int = 42,
    conformer_n_jobs: int = 1,
    add_hs: bool = True,
    embed_algorithm: Optional[str] = None,
    optimize: bool = True,
    opt_method: Optional[str] = "MMFF94",
    opt_max_iters: int = 200,
) -> Chem.Mol:
    """
    Use the internal Conformer class to create a molecule from SMILES with coordinates.

    :param smiles: input SMILES
    :returns: RDKit Mol with coordinates (raises on failure)
    """
    assert _HAS_CONFORMER and Conformer is not None
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
    if not cm.molblocks:
        raise RuntimeError("Conformer failed to produce an embedded molecule")
    mb = cm.molblocks[0]
    m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=(not add_hs))
    if m is None:
        raise RuntimeError("Failed to parse MolBlock produced by Conformer")
    return m


def _use_conformer_for_mol(
    mol: Chem.Mol,
    conformer_seed: int = 42,
    conformer_n_jobs: int = 1,
    add_hs: bool = True,
    embed_algorithm: Optional[str] = None,
    optimize: bool = True,
    opt_method: Optional[str] = "MMFF94",
    opt_max_iters: int = 200,
) -> Chem.Mol:
    """
    Use Conformer by converting the input mol to SMILES, embedding, and returning an RDKit Mol.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return _use_conformer_for_smiles(
        smiles,
        conformer_seed=conformer_seed,
        conformer_n_jobs=conformer_n_jobs,
        add_hs=add_hs,
        embed_algorithm=embed_algorithm,
        optimize=optimize,
        opt_method=opt_method,
        opt_max_iters=opt_max_iters,
    )


def _ensure_mol_has_coords(
    mol: Chem.Mol,
    add_hs: bool,
    use_conformer: bool,
    conformer_seed: int,
    conformer_n_jobs: int,
    embed_algorithm: Optional[str],
    optimize: bool,
    opt_method: Optional[str],
    opt_max_iters: int,
) -> Chem.Mol:
    """
    Ensure the provided mol has a conformer. Prefer Conformer if requested/available,
    otherwise use RDKit embedding. Returns a mol with coordinates.
    """
    if mol.GetNumConformers() > 0:
        return mol

    if use_conformer and _HAS_CONFORMER:
        try:
            return _use_conformer_for_mol(
                mol,
                conformer_seed=conformer_seed,
                conformer_n_jobs=conformer_n_jobs,
                add_hs=add_hs,
                embed_algorithm=embed_algorithm,
                optimize=optimize,
                opt_method=opt_method,
                opt_max_iters=opt_max_iters,
            )
        except Exception as exc:
            logger.warning(
                "Conformer embedding for mol failed: %s — falling back to RDKit", exc
            )

    # fallback to RDKit
    return _rdkit_embed_and_optimize(
        mol,
        add_hs=add_hs,
        embed_algorithm=embed_algorithm,
        optimize=optimize,
        opt_method=opt_method,
        opt_max_iters=opt_max_iters,
    )


# ---------------------
# Basic SMILES <-> Mol
# ---------------------
def smiles2mol(smiles: str, sanitize: bool = True) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit Mol (raises ValueError on failure).

    :param smiles: SMILES string
    :param sanitize: whether to run RDKit sanitization
    :returns: RDKit Mol
    """
    if not smiles:
        raise ValueError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")
    return mol


def mol2smiles(mol: Chem.Mol, canonical: bool = True, isomeric: bool = True) -> str:
    """
    Convert an RDKit Mol to a SMILES string.

    :param mol: RDKit Mol
    :param canonical: produce canonical SMILES
    :param isomeric: include stereochemistry
    :returns: SMILES string
    """
    if mol is None:
        raise ValueError("mol must be an RDKit Mol")
    return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)


# ---------------------
# SDF readers/writers
# ---------------------
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

    :param mol: RDKit Mol
    :param out_path: destination SDF path
    :param sanitize: whether to sanitize molecule before writing
    :param embed3d: whether to embed coordinates if missing
    :param add_hs: whether to add hydrogens for embedding
    :param optimize: whether to optimize geometry
    :param embed_algorithm: embedding algorithm name (e.g. "ETKDGv3")
    :param opt_method: optimization method
    :param conformer_seed: seed passed to Conformer
    :param conformer_n_jobs: number of jobs for Conformer
    :param opt_max_iters: maximum iterations for optimizers
    :returns: Path to written SDF
    """
    out_path = Path(out_path)
    if mol is None:
        raise ValueError("mol must be an RDKit Mol")

    # If no conformers and user requested embed3d, attempt to create coordinates
    if embed3d and mol.GetNumConformers() == 0:
        m_with_coords = _ensure_mol_has_coords(
            mol,
            add_hs=add_hs,
            use_conformer=True,
            conformer_seed=conformer_seed,
            conformer_n_jobs=conformer_n_jobs,
            embed_algorithm=embed_algorithm,
            optimize=optimize,
            opt_method=opt_method,
            opt_max_iters=opt_max_iters,
        )
        return _write_sdf(m_with_coords, out_path)

    # default: write given molecule directly
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            logger.debug("SanitizeMol raised an exception but continuing to write")
    return _write_sdf(mol, out_path)


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

    :param smiles: SMILES string
    :param out_path: destination SDF path
    :param embed3d: whether to embed 3D coordinates
    :param add_hs: whether to add hydrogens for embedding
    :param optimize: whether to optimize geometry
    :param embed_algorithm: embedding algorithm
    :param opt_method: optimization method
    :param conformer_seed: seed for Conformer (if used)
    :param conformer_n_jobs: number of jobs for Conformer
    :param opt_max_iters: optimizer iterations
    :returns: Path to written SDF
    """
    out_path = Path(out_path)
    if not smiles:
        raise ValueError("smiles must be provided")

    # prefer Conformer when embedding/optimization is requested
    if (embed3d or optimize) and _HAS_CONFORMER:
        try:
            m = _use_conformer_for_smiles(
                smiles,
                conformer_seed=conformer_seed,
                conformer_n_jobs=conformer_n_jobs,
                add_hs=add_hs,
                embed_algorithm=embed_algorithm,
                optimize=optimize,
                opt_method=opt_method,
                opt_max_iters=opt_max_iters,
            )
            return _write_sdf(m, out_path)
        except Exception as exc:
            logger.warning(
                "Conformer-based embedding failed, falling back to RDKit embed: %s", exc
            )

    # fallback: RDKit-only flow
    mol = smiles2mol(smiles, sanitize=True)
    mol_with_coords = _rdkit_embed_and_optimize(
        mol,
        add_hs=add_hs,
        embed_algorithm=embed_algorithm,
        optimize=optimize,
        opt_method=opt_method,
        opt_max_iters=opt_max_iters,
    )
    return _write_sdf(mol_with_coords, out_path)


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


# ---------------------
# PDB readers/writers
# ---------------------
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

    :returns: Path to written PDB file
    """
    out_path = Path(out_path)
    if mol is None:
        raise ValueError("mol must be an RDKit Mol")

    if embed3d and mol.GetNumConformers() == 0:
        m_with_coords = _ensure_mol_has_coords(
            mol,
            add_hs=add_hs,
            use_conformer=True,
            conformer_seed=conformer_seed,
            conformer_n_jobs=conformer_n_jobs,
            embed_algorithm=embed_algorithm or "ETKDGv3",
            optimize=optimize,
            opt_method=opt_method or "MMFF94",
            opt_max_iters=opt_max_iters,
        )
        Chem.MolToPDBFile(m_with_coords, str(out_path))
        return out_path

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

    if (embed3d or optimize) and _HAS_CONFORMER:
        try:
            m = _use_conformer_for_smiles(
                smiles,
                conformer_seed=conformer_seed,
                conformer_n_jobs=conformer_n_jobs,
                add_hs=add_hs,
                embed_algorithm=embed_algorithm,
                optimize=optimize,
                opt_method=opt_method,
                opt_max_iters=opt_max_iters,
            )
            Chem.MolToPDBFile(m, str(out_path))
            return Path(out_path)
        except Exception as exc:
            logger.warning(
                "Conformer-based PDB generation failed, falling back to RDKit: %s", exc
            )

    mol = smiles2mol(smiles, sanitize=True)
    return mol2pdb(
        mol,
        out_path,
        add_hs=add_hs,
        embed3d=embed3d,
        optimize=optimize,
        embed_algorithm=embed_algorithm,
        opt_method=opt_method,
        conformer_seed=conformer_seed,
        conformer_n_jobs=conformer_n_jobs,
        opt_max_iters=opt_max_iters,
    )


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


# ---------------------
# Convenience wrappers
# ---------------------
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


# ---------------------
# Small utilities
# ---------------------
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
