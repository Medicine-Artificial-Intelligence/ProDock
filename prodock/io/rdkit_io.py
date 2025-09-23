# prodock/io/rdkit_io.py
"""
RDKit-only molecular I/O helpers that prefer prodock.chem.Conformer for
embedding/optimization.

This module provides convenience functions to convert between SMILES, SDF and
PDB formats, to embed 3D coordinates, and to optionally use the internal
:class:`prodock.chem.conformer.Conformer` class for higher-quality,
parallelized embedding/optimization when available.

Provided functions
------------------
 - smiles2mol, mol2smiles
 - smiles2sdf, sdf2mol, sdf2mols, sdftosmiles, mol2sdf
 - smiles2pdb, pdb2mol, pdb2smiles, mol2pdb
 - convenience: mol_from_smiles_write_all_formats, is_valid_smiles

Behavior notes
--------------
- If :class:`prodock.chem.conformer.Conformer` is importable it will be used
  as the preferred engine for embedding/optimizing 3D conformers. If it is not
  available the module falls back to RDKit embedding/force-field optimization.
- Functions that write 3D formats (SDF/PDB) will attempt to create coordinates
  when the provided RDKit ``Mol`` lacks conformers, honoring the `embed3d`
  and `optimize` flags.
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
    Write a molecule to an SDF file (single-record). Ensures the RDKit writer is closed.

    :param mol: RDKit Mol instance to write.
    :type mol: rdkit.Chem.rdchem.Mol
    :param out_path: Destination path for the SDF file.
    :type out_path: str | pathlib.Path
    :returns: Path to the written file.
    :rtype: pathlib.Path
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
    Return an RDKit Embed parameters object for the named algorithm or None.

    :param embed_algorithm: Name of an RDKit AllChem embed params factory, e.g. "ETKDGv3".
                            If None or empty, None is returned to allow RDKit defaults.
    :type embed_algorithm: Optional[str]
    :returns: An RDKit embedding parameters object (if available) or None.
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

    :param working: An RDKit Mol to embed (mutable).
    :type working: rdkit.Chem.rdchem.Mol
    :param params: Embedding parameters object returned by `_get_embed_params` or None.
    :returns: True if embedding likely succeeded (no exception), False otherwise.
    :rtype: bool
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
    Attempt geometry optimization using the preferred `method` with fallbacks.

    Preference: MMFF (default) then UFF. If `method` explicitly contains 'U' or
    'UFF' we try UFF first.

    :param working: RDKit Mol (with conformers) to optimize in-place.
    :type working: rdkit.Chem.rdchem.Mol
    :param method: Preferred optimizer name (e.g., 'MMFF94' or 'UFF'). None -> MMFF.
    :type method: Optional[str]
    :param max_iters: Maximum iterations for the optimizer.
    :type max_iters: int
    :returns: True if any optimizer succeeded, False otherwise.
    :rtype: bool
    """
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
    Embed and optionally optimize `mol` using RDKit (best-effort helper).

    Workflow:
      1. Optionally add explicit hydrogens (recommended for optimization).
      2. Attempt embedding using the provided `embed_algorithm` params (if any),
         with a fallback to RDKit defaults.
      3. Optionally run force-field optimization using MMFF/UFF fallbacks.
      4. If add_hs was False, hydrogens are removed from the returned molecule.

    :param mol: RDKit Mol object (may or may not have conformers).
    :type mol: rdkit.Chem.rdchem.Mol
    :param add_hs: Whether to add explicit hydrogens before embedding/optimization.
    :type add_hs: bool
    :param embed_algorithm: Name of embed params factory (e.g. "ETKDGv3") or None.
    :type embed_algorithm: Optional[str]
    :param optimize: Whether to attempt force-field optimization after embedding.
    :type optimize: bool
    :param opt_method: Preferred optimization method, e.g. "MMFF94" or "UFF".
    :type opt_method: Optional[str]
    :param opt_max_iters: Maximum iterations for the optimizer.
    :type opt_max_iters: int
    :returns: An RDKit Mol that (best-effort) has 3D coordinates.
    :rtype: rdkit.Chem.rdchem.Mol
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
    Use the internal Conformer class to create a 3D RDKit Mol from SMILES.

    Preferred when Conformer is available because it typically supports parallel
    embedding and higher quality defaults.

    :param smiles: Input SMILES string.
    :type smiles: str
    :param conformer_seed: RNG seed passed to Conformer.
    :type conformer_seed: int
    :param conformer_n_jobs: Number of parallel jobs for Conformer operations.
    :type conformer_n_jobs: int
    :param add_hs: Whether to add explicit hydrogens during embedding.
    :type add_hs: bool
    :param embed_algorithm: Embedding algorithm name (e.g., "ETKDGv3").
    :type embed_algorithm: Optional[str]
    :param optimize: Whether to run geometry optimization after embedding.
    :type optimize: bool
    :param opt_method: Optimization method to pass (e.g., "MMFF94").
    :type opt_method: Optional[str]
    :param opt_max_iters: Maximum iterations for optimizer.
    :type opt_max_iters: int
    :returns: RDKit Mol with coordinates on success.
    :rtype: rdkit.Chem.rdchem.Mol
    :raises RuntimeError: if Conformer fails to produce an embedded molecule or parsing fails.
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
    Use Conformer by converting the input Mol to SMILES, embedding, and returning an RDKit Mol.

    :param mol: RDKit Mol to re-embed via Conformer.
    :type mol: rdkit.Chem.rdchem.Mol
    :param conformer_seed: RNG seed for Conformer.
    :type conformer_seed: int
    :param conformer_n_jobs: Number of parallel jobs for Conformer.
    :type conformer_n_jobs: int
    :param add_hs: Whether to add hydrogens for embedding.
    :type add_hs: bool
    :param embed_algorithm: Embedding algorithm name (e.g. "ETKDGv3").
    :type embed_algorithm: Optional[str]
    :param optimize: Whether to optimize geometry after embedding.
    :type optimize: bool
    :param opt_method: Preferred optimization method (e.g., "MMFF94").
    :type opt_method: Optional[str]
    :param opt_max_iters: Maximum optimization iterations.
    :type opt_max_iters: int
    :returns: RDKit Mol with generated coordinates.
    :rtype: rdkit.Chem.rdchem.Mol
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
    Ensure the provided Mol has a conformer. Prefer Conformer if requested/available,
    otherwise use RDKit embedding and optimization.

    :param mol: RDKit Mol instance (may lack conformers).
    :type mol: rdkit.Chem.rdchem.Mol
    :param add_hs: Whether to add explicit hydrogens during embedding.
    :type add_hs: bool
    :param use_conformer: If True and Conformer is available, attempt to use it.
    :type use_conformer: bool
    :param conformer_seed: RNG seed for Conformer fallback.
    :type conformer_seed: int
    :param conformer_n_jobs: Number of parallel jobs for Conformer fallback.
    :type conformer_n_jobs: int
    :param embed_algorithm: Embedding algorithm name (e.g., 'ETKDGv3').
    :type embed_algorithm: Optional[str]
    :param optimize: Whether to optimize geometry after embedding.
    :type optimize: bool
    :param opt_method: Preferred optimizer method.
    :type opt_method: Optional[str]
    :param opt_max_iters: Maximum optimizer iterations.
    :type opt_max_iters: int
    :returns: RDKit Mol that (best-effort) has 3D coordinates.
    :rtype: rdkit.Chem.rdchem.Mol
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

    :param smiles: Input SMILES string (non-empty).
    :type smiles: str
    :param sanitize: Whether to run RDKit sanitization on the parsed Mol.
    :type sanitize: bool
    :returns: Parsed RDKit Mol.
    :rtype: rdkit.Chem.rdchem.Mol
    :raises ValueError: If parsing fails or smiles is empty.
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

    :param mol: RDKit Mol to convert.
    :type mol: rdkit.Chem.rdchem.Mol
    :param canonical: If True produce canonical SMILES.
    :type canonical: bool
    :param isomeric: If True include stereochemistry in SMILES.
    :type isomeric: bool
    :returns: SMILES string representation.
    :rtype: str
    :raises ValueError: If mol is None.
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
    Write a single RDKit Mol to an SDF file. If `embed3d` is True and the molecule
    lacks coordinates, attempt to generate coordinates (preferring Conformer if available).

    :param mol: RDKit Mol to write.
    :type mol: rdkit.Chem.rdchem.Mol
    :param out_path: Destination SDF path.
    :type out_path: str | pathlib.Path
    :param sanitize: Whether to sanitize the molecule before writing.
    :type sanitize: bool
    :param embed3d: If True and no conformer present, attempt to embed 3D coords.
    :type embed3d: bool
    :param add_hs: Whether to add explicit hydrogens prior to embedding/optimization.
    :type add_hs: bool
    :param optimize: Whether to run force-field optimization after embedding.
    :type optimize: bool
    :param embed_algorithm: Embedding algorithm name (e.g., "ETKDGv3") or None.
    :type embed_algorithm: Optional[str]
    :param opt_method: Preferred optimizer method (e.g., "MMFF94" or "UFF").
    :type opt_method: Optional[str]
    :param conformer_seed: RNG seed for Conformer or embedding fallbacks.
    :type conformer_seed: int
    :param conformer_n_jobs: Number of parallel jobs for Conformer fallback.
    :type conformer_n_jobs: int
    :param opt_max_iters: Maximum optimizer iterations.
    :type opt_max_iters: int
    :returns: Path to the written SDF file.
    :rtype: pathlib.Path
    :raises ValueError: If mol is None.
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
    Convert a SMILES string to an SDF file (single-record). Prefer the internal
    Conformer engine for embedding/optimization when available.

    :param smiles: Input SMILES string.
    :type smiles: str
    :param out_path: Destination SDF path.
    :type out_path: str | pathlib.Path
    :param embed3d: Whether to embed 3D coordinates if missing.
    :type embed3d: bool
    :param add_hs: Whether to add explicit hydrogens for embedding.
    :type add_hs: bool
    :param optimize: Whether to optimize geometry after embedding.
    :type optimize: bool
    :param embed_algorithm: Embedding algorithm name (e.g., "ETKDGv3").
    :type embed_algorithm: Optional[str]
    :param opt_method: Optimization method preference.
    :type opt_method: Optional[str]
    :param conformer_seed: RNG seed to pass to Conformer if used.
    :type conformer_seed: int
    :param conformer_n_jobs: Parallel jobs for Conformer if available.
    :type conformer_n_jobs: int
    :param opt_max_iters: Maximum optimizer iterations.
    :type opt_max_iters: int
    :returns: Path to the written SDF file.
    :rtype: pathlib.Path
    :raises ValueError: If smiles is empty.
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
    """
    Load the first molecule from an SDF file.

    :param sdf_path: Path to the SDF file.
    :type sdf_path: str | pathlib.Path
    :param sanitize: Whether to sanitize molecules while reading.
    :type sanitize: bool
    :param removeHs: If True drop explicit hydrogens from the returned Mol.
    :type removeHs: bool
    :returns: The first RDKit Mol found or None if none are readable.
    :rtype: Optional[rdkit.Chem.rdchem.Mol]
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize)
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

    :param sdf_path: Path to the SDF file.
    :type sdf_path: str | pathlib.Path
    :param sanitize: Whether to sanitize molecules while reading.
    :type sanitize: bool
    :returns: List of RDKit Mol objects (may be empty).
    :rtype: list[rdkit.Chem.rdchem.Mol]
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize)
    return [m for m in supplier if m is not None]


def sdftosmiles(sdf_path: Union[str, Path], sanitize: bool = True) -> List[str]:
    """
    Read an SDF file and return a list of SMILES (one per molecule).

    :param sdf_path: Path to the SDF file.
    :type sdf_path: str | pathlib.Path
    :param sanitize: Whether to sanitize molecules while reading.
    :type sanitize: bool
    :returns: List of SMILES strings corresponding to each molecule.
    :rtype: list[str]
    """
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
    Write an RDKit Mol to a PDB file. If the molecule lacks coordinates and
    `embed3d` is True, attempt to generate coordinates (prefer Conformer).

    :param mol: RDKit Mol to write.
    :type mol: rdkit.Chem.rdchem.Mol
    :param out_path: Destination PDB file path.
    :type out_path: str | pathlib.Path
    :param add_hs: Whether to add explicit hydrogens for embedding.
    :type add_hs: bool
    :param embed3d: If True attempt to embed 3D coordinates when missing.
    :type embed3d: bool
    :param optimize: Whether to run geometry optimization after embedding.
    :type optimize: bool
    :param embed_algorithm: Embedding algorithm name (e.g., "ETKDGv3") or None.
    :type embed_algorithm: Optional[str]
    :param opt_method: Optimization method (e.g., "MMFF94" or "UFF").
    :type opt_method: Optional[str]
    :param conformer_seed: RNG seed for Conformer fallback.
    :type conformer_seed: int
    :param conformer_n_jobs: Parallel jobs for Conformer fallback.
    :type conformer_n_jobs: int
    :param opt_max_iters: Maximum optimizer iterations.
    :type opt_max_iters: int
    :returns: Path to the written PDB file.
    :rtype: pathlib.Path
    :raises ValueError: If mol is None.
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
    """
    Convert a SMILES string to a PDB file, using Conformer when embedding/optimization
    is requested and available.

    :param smiles: Input SMILES string.
    :type smiles: str
    :param out_path: Destination PDB file path.
    :type out_path: str | pathlib.Path
    :param add_hs: Whether to add explicit hydrogens prior to embedding.
    :type add_hs: bool
    :param embed3d: Whether to embed 3D coordinates when missing.
    :type embed3d: bool
    :param optimize: Whether to optimize geometry after embedding.
    :type optimize: bool
    :param embed_algorithm: Embedding algorithm name (e.g., "ETKDGv3") or None.
    :type embed_algorithm: Optional[str]
    :param opt_method: Optimization method preference (e.g., "MMFF94").
    :type opt_method: Optional[str]
    :param conformer_seed: RNG seed for Conformer fallback.
    :type conformer_seed: int
    :param conformer_n_jobs: Number of parallel jobs for Conformer fallback.
    :type conformer_n_jobs: int
    :param opt_max_iters: Maximum optimizer iterations.
    :type opt_max_iters: int
    :returns: Path to the written PDB file.
    :rtype: pathlib.Path
    :raises ValueError: If smiles is empty.
    """
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
    """
    Load a molecule from a PDB file.

    :param pdb_path: Path to a PDB file.
    :type pdb_path: str | pathlib.Path
    :param sanitize: Whether to sanitize the molecule during parsing.
    :type sanitize: bool
    :param removeHs: If True, remove explicit hydrogens from the returned Mol.
    :type removeHs: bool
    :returns: RDKit Mol or None if reading fails.
    :rtype: Optional[rdkit.Chem.rdchem.Mol]
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
    Load a PDB file and return a SMILES string.

    :param pdb_path: Path to the PDB file.
    :type pdb_path: str | pathlib.Path
    :param sanitize: Whether to sanitize the molecule during parsing.
    :type sanitize: bool
    :param removeHs: If True remove explicit hydrogens prior to SMILES generation.
    :type removeHs: bool
    :returns: SMILES string for the first molecule in the PDB.
    :rtype: str
    :raises ValueError: If the PDB cannot be parsed into a molecule.
    """
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
    Convenience helper: from SMILES write SDF and/or PDB files sharing the same prefix.

    This function uses Conformer for embedding/optimization when available,
    otherwise falls back to RDKit.

    :param smiles: Input SMILES string.
    :type smiles: str
    :param out_prefix: Output path prefix (the function appends .sdf and/or .pdb).
    :type out_prefix: str | pathlib.Path
    :param write_sdf: If True produce an SDF file.
    :type write_sdf: bool
    :param write_pdb: If True produce a PDB file.
    :type write_pdb: bool
    :param embed3d: Whether to embed 3D coordinates for outputs that require them.
    :type embed3d: bool
    :param add_hs: Whether to add hydrogens for embedding/optimization.
    :type add_hs: bool
    :param embed_algorithm: Embedding algorithm name (e.g., "ETKDGv3") or None.
    :type embed_algorithm: Optional[str]
    :param opt_method: Preferred optimization method (e.g., "MMFF94").
    :type opt_method: Optional[str]
    :returns: Dictionary mapping format keys ("sdf","pdb") to written pathlib.Path objects.
    :rtype: dict[str, pathlib.Path]
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
    """
    Quick check whether a SMILES string can be parsed by RDKit.

    :param smiles: SMILES string to validate.
    :type smiles: str
    :returns: True if RDKit can parse the SMILES; False otherwise.
    :rtype: bool
    """
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
