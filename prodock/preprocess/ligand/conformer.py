# prodock/chem/conformer.py
"""
High-level conformer generation and optimization utilities for ``prodock.chem``.

This module provides :class:`Conformer`, a high-level orchestration class that
combines:

- :class:`prodock.chem.embed.Embedder` for conformer embedding,
- :class:`prodock.chem.optimize.Optimizer` for force-field optimization,
- optional joblib-based parallel execution for per-ligand workflows,
- SDF export with optional conformer energy annotations.

Design
------
The low-level embedding and optimization logic is intentionally delegated to
single-process worker utilities. Parallelism is handled only at this top-level
layer, which simplifies worker behavior and makes the execution model easier to
reason about.

The public class is :class:`Conformer`. For backward compatibility,
``ConformerManager`` is retained as an alias.

Example
-------
.. code-block:: python

    from prodock.chem.conformer import Conformer

    conf = Conformer(seed=42, backend="loky")
    conf.load_smiles(["CCO", "c1ccccc1"])
    conf.embed_all(n_confs=2, n_jobs=1, embed_algorithm="ETKDGv3")
    conf.optimize_all(method="MMFF94", n_jobs=1)
    conf.prune_top_k(k=1)
    conf.write_sdf("outdir", per_mol_folder=False)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    raise ImportError("RDKit is required for prodock.chem.conformer")

from .embed import Embedder
from .optimize import Optimizer
from prodock.io.logging import get_logger, StructuredAdapter

logger = StructuredAdapter(
    get_logger("prodock.chem.conformer"), {"component": "conformer"}
)
logger._base_logger = getattr(logger, "_base_logger", getattr(logger, "logger", None))

try:
    from joblib import Parallel, delayed

    _JOBLIB_AVAILABLE = True
except Exception:
    _JOBLIB_AVAILABLE = False


def _embed_worker(
    smiles: str,
    seed: int,
    n_confs: int,
    add_hs: bool,
    embed_algorithm: Optional[str],
) -> Tuple[Optional[str], int]:
    """
    Embed a single SMILES string inside a worker context.

    This helper creates a local :class:`Embedder` instance, embeds exactly one
    input SMILES string, and returns the first generated MolBlock together with
    the conformer count.

    The function is designed to run safely in a worker process and attempts to
    limit thread over-subscription by setting common BLAS/OpenMP environment
    variables to ``1``.

    :param smiles:
        SMILES string to embed.
    :type smiles: str
    :param seed:
        Random seed forwarded to :class:`Embedder`.
    :type seed: int
    :param n_confs:
        Number of conformers to generate.
    :type n_confs: int
    :param add_hs:
        Whether explicit hydrogens should be added before embedding.
    :type add_hs: bool
    :param embed_algorithm:
        RDKit embedding algorithm name such as ``"ETKDGv3"``.
    :type embed_algorithm: Optional[str]

    :returns:
        Tuple ``(molblock, conf_count)`` where ``molblock`` is ``None`` if
        embedding failed.
    :rtype: Tuple[Optional[str], int]

    Example
    -------
    .. code-block:: python

        molblock, n_conf = _embed_worker(
            "CCO",
            seed=42,
            n_confs=1,
            add_hs=True,
            embed_algorithm="ETKDGv3",
        )
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    e = Embedder(seed=seed)
    e.load_smiles_iterable([smiles])
    e.embed_all(
        n_confs=n_confs,
        add_hs=add_hs,
        embed_algorithm=embed_algorithm,
        random_seed=seed,
    )
    if not e.molblocks:
        return None, 0
    return e.molblocks[0], (e.conf_counts[0] if e.conf_counts else 0)


def _optimize_worker(
    molblock: str,
    method: str,
    max_iters: int,
) -> Tuple[Optional[str], Dict[int, float]]:
    """
    Optimize a single MolBlock inside a worker context.

    This helper creates a local :class:`Optimizer`, optimizes one MolBlock, and
    returns the optimized MolBlock together with a conformer-energy mapping.

    As with :func:`_embed_worker`, common thread-count environment variables are
    set to reduce over-subscription in worker processes.

    :param molblock:
        Input MolBlock string to optimize.
    :type molblock: str
    :param method:
        Force-field optimization method such as ``"MMFF94"`` or ``"UFF"``.
    :type method: str
    :param max_iters:
        Maximum number of optimization iterations.
    :type max_iters: int

    :returns:
        Tuple ``(optimized_molblock, energies)`` where ``optimized_molblock`` is
        ``None`` on failure and ``energies`` maps conformer id to energy.
    :rtype: Tuple[Optional[str], Dict[int, float]]

    Example
    -------
    .. code-block:: python

        optimized_block, energies = _optimize_worker(
            molblock_str,
            method="MMFF94",
            max_iters=200,
        )
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    opt = Optimizer(max_iters=max_iters)
    opt.load_molblocks([molblock])
    opt.optimize_all(method=method)
    if not opt.optimized_molblocks:
        return None, {}
    return opt.optimized_molblocks[0], (opt.energies[0] if opt.energies else {})


class Conformer:
    """
    High-level conformer embedding and optimization workflow.

    This class composes :class:`Embedder` and :class:`Optimizer` into a single
    workflow-oriented interface. It supports sequential execution and optional
    per-ligand parallel execution through joblib.

    Methods are chainable and return ``self``.

    Typical workflow
    ----------------
    1. Create an instance.
    2. Load SMILES from a file or Python list.
    3. Run :meth:`embed_all`.
    4. Run :meth:`optimize_all`.
    5. Optionally run :meth:`prune_top_k`.
    6. Export structures with :meth:`write_sdf`.

    :param seed:
        Random seed used during embedding.
    :type seed: int
    :param backend:
        Joblib backend used for parallel execution. The default is ``"loky"``.
    :type backend: str

    Example
    -------
    .. code-block:: python

        conf = Conformer(seed=7)
        conf.load_smiles(["CCO", "c1ccccc1"])
        conf.embed_all(n_confs=1, n_jobs=1)
        conf.optimize_all(method="MMFF94", n_jobs=1)
        conf.write_sdf("outdir", per_mol_folder=False)
    """

    def __init__(self, seed: int = 42, backend: str = "loky") -> None:
        self._seed = int(seed)
        self._backend = backend
        self._smiles: List[str] = []
        self._molblocks: List[str] = []
        self._conf_counts: List[int] = []
        self._energies: List[Dict[int, float]] = []

    def __repr__(self) -> str:
        return (
            f"<Conformer smiles={len(self._smiles)} "
            f"mols={len(self._molblocks)} seed={self._seed}>"
        )

    @property
    def smiles(self) -> List[str]:
        """
        Return a copy of loaded SMILES strings.

        :returns:
            Loaded SMILES strings.
        :rtype: List[str]
        """
        return list(self._smiles)

    @property
    def molblocks(self) -> List[str]:
        """
        Return the current MolBlock strings.

        These MolBlocks may correspond to embedded structures or optimized
        structures, depending on which workflow steps have been executed.

        :returns:
            Current MolBlock strings.
        :rtype: List[str]
        """
        return list(self._molblocks)

    @property
    def conf_counts(self) -> List[int]:
        """
        Return the conformer count for each molecule.

        :returns:
            Number of conformers stored for each molecule.
        :rtype: List[int]
        """
        return list(self._conf_counts)

    @property
    def energies(self) -> List[Dict[int, float]]:
        """
        Return optimization energies for each molecule.

        Each element is a dictionary mapping conformer id to energy value.

        :returns:
            Per-molecule conformer energy maps.
        :rtype: List[Dict[int, float]]
        """
        return [dict(e) for e in self._energies]

    def load_smiles_file(self, path: str) -> "Conformer":
        """
        Load SMILES from a newline-separated text file.

        The first whitespace-separated token on each non-empty line is treated as
        the SMILES string. This allows simple ``SMILES name`` input formats.

        :param path:
            Path to the SMILES file.
        :type path: str

        :returns:
            The current conformer workflow instance.
        :rtype: Conformer

        :raises FileNotFoundError:
            If ``path`` does not exist.

        Example
        -------
        .. code-block:: python

            conf = Conformer()
            conf.load_smiles_file("my_smiles.txt")
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        with p.open("r", encoding="utf-8") as fh:
            self._smiles = [ln.strip().split()[0] for ln in fh if ln.strip()]
        logger.info("Conformer: loaded %d SMILES", len(self._smiles))
        return self

    def load_smiles(self, smiles: List[str]) -> "Conformer":
        """
        Load SMILES from an in-memory list.

        For each non-empty entry, the first whitespace-separated token is used as
        the SMILES string.

        :param smiles:
            Input SMILES strings.
        :type smiles: List[str]

        :returns:
            The current conformer workflow instance.
        :rtype: Conformer

        Example
        -------
        .. code-block:: python

            conf = Conformer()
            conf.load_smiles(["CCO", "O=C=O"])
        """
        self._smiles = [s.strip().split()[0] for s in smiles if s]
        return self

    def embed_all(
        self,
        n_confs: int = 1,
        n_jobs: int = 1,
        add_hs: bool = True,
        embed_algorithm: Optional[str] = "ETKDGv3",
    ) -> "Conformer":
        """
        Embed all loaded SMILES strings.

        Embedding is performed sequentially when ``n_jobs == 1`` or when joblib
        is unavailable. Otherwise, one worker task is dispatched per molecule.

        :param n_confs:
            Number of conformers to generate per molecule.
        :type n_confs: int
        :param n_jobs:
            Number of parallel jobs. Use ``1`` for sequential execution.
        :type n_jobs: int
        :param add_hs:
            Whether explicit hydrogens should be added before embedding.
        :type add_hs: bool
        :param embed_algorithm:
            RDKit embedding algorithm name such as ``"ETKDGv3"``, ``"ETKDGv2"``,
            ``"ETKDG"``, or ``None``.
        :type embed_algorithm: Optional[str]

        :returns:
            The current conformer workflow instance.
        :rtype: Conformer

        :raises RuntimeError:
            If no SMILES have been loaded.

        Example
        -------
        .. code-block:: python

            conf = Conformer()
            conf.load_smiles(["CCO"])
            conf.embed_all(n_confs=1, n_jobs=1, embed_algorithm="ETKDGv3")
        """
        if not self._smiles:
            raise RuntimeError(
                "No SMILES loaded; call load_smiles_file() or load_smiles()"
            )

        if n_jobs == 1 or not _JOBLIB_AVAILABLE:
            results = [
                _embed_worker(
                    smi, self._seed, int(n_confs), bool(add_hs), embed_algorithm
                )
                for smi in self._smiles
            ]
        else:
            results = Parallel(n_jobs=n_jobs, backend=self._backend)(
                delayed(_embed_worker)(
                    smi, self._seed, int(n_confs), bool(add_hs), embed_algorithm
                )
                for smi in self._smiles
            )

        molblocks: List[str] = []
        conf_counts: List[int] = []
        for mb, c in results:
            if mb is None:
                continue
            molblocks.append(mb)
            conf_counts.append(c)

        self._molblocks = molblocks
        self._conf_counts = conf_counts
        logger.info(
            "Conformer: embedded %d / %d molecules",
            len(self._molblocks),
            len(self._smiles),
        )
        return self

    def optimize_all(
        self, method: str = "MMFF94", n_jobs: int = 1, max_iters: int = 200
    ) -> "Conformer":
        """
        Optimize all currently stored MolBlocks.

        Optimization is performed sequentially when ``n_jobs == 1`` or when
        joblib is unavailable. Otherwise, optimization is parallelized at the
        per-molecule level.

        :param method:
            Force-field method. Typical values include ``"UFF"``, ``"MMFF"``,
            ``"MMFF94"``, and ``"MMFF94S"``.
        :type method: str
        :param n_jobs:
            Number of parallel jobs. Use ``1`` for sequential execution.
        :type n_jobs: int
        :param max_iters:
            Maximum number of optimization iterations.
        :type max_iters: int

        :returns:
            The current conformer workflow instance.
        :rtype: Conformer

        :raises RuntimeError:
            If there are no MolBlocks available for optimization.

        Example
        -------
        .. code-block:: python

            conf = Conformer()
            conf.load_smiles(["CCO"])
            conf.embed_all()
            conf.optimize_all(method="MMFF94", n_jobs=1, max_iters=200)
        """
        if not self._molblocks:
            raise RuntimeError(
                "No embedded molecules available; call embed_all() first"
            )

        if n_jobs == 1 or not _JOBLIB_AVAILABLE:
            results = [
                _optimize_worker(mb, method, int(max_iters)) for mb in self._molblocks
            ]
        else:
            results = Parallel(n_jobs=n_jobs, backend=self._backend)(
                delayed(_optimize_worker)(mb, method, int(max_iters))
                for mb in self._molblocks
            )

        optimized_blocks: List[str] = []
        energies_list: List[Dict[int, float]] = []
        for mb, en in results:
            if mb is None:
                continue
            optimized_blocks.append(mb)
            energies_list.append(en)

        self._molblocks = optimized_blocks
        self._energies = energies_list
        logger.info("Conformer: optimized %d molecules", len(self._molblocks))
        return self

    def prune_top_k(self, k: int = 1) -> "Conformer":
        """
        Keep only the top-``k`` lowest-energy conformers for each molecule.

        Pruning is based on the most recent optimization energies. After pruning,
        conformer identifiers are reassigned to a dense range starting from zero.

        :param k:
            Number of lowest-energy conformers to keep per molecule.
        :type k: int

        :returns:
            The current conformer workflow instance.
        :rtype: Conformer

        :raises RuntimeError:
            If there are no molecules available to prune.

        Example
        -------
        .. code-block:: python

            conf.prune_top_k(k=1)
        """
        if not self._molblocks:
            raise RuntimeError("No molecules to prune")
        if not self._energies:
            logger.warning("Conformer: no energy data available; skipping prune")
            return self

        new_blocks: List[str] = []
        new_energies: List[Dict[int, float]] = []
        for block, e_map in zip(self._molblocks, self._energies):
            mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            if mol is None:
                continue
            if not e_map:
                new_blocks.append(block)
                new_energies.append({})
                continue

            keep_ids = [
                cid
                for cid, _ in sorted(e_map.items(), key=lambda kv: kv[1])[
                    : max(1, int(k))
                ]
            ]

            base = Chem.Mol(mol)
            try:
                base.RemoveAllConformers()
            except Exception:
                base = Chem.Mol(mol)
                base.RemoveAllConformers()

            for cid in keep_ids:
                try:
                    conf = mol.GetConformer(cid)
                    base.AddConformer(conf, assignId=True)
                except Exception:
                    logger.warning("Conformer: failed to copy conformer %s", cid)

            new_map = {i: e_map[cid] for i, cid in enumerate(keep_ids)}
            new_blocks.append(Chem.MolToMolBlock(base))
            new_energies.append(new_map)

        self._molblocks = new_blocks
        self._energies = new_energies
        self._conf_counts = [len(e) for e in new_energies]
        logger.info(
            "Conformer: pruned to top-%d conformers for %d molecules",
            k,
            len(self._molblocks),
        )
        return self

    def write_sdf(
        self,
        out_folder: str,
        per_mol_folder: bool = True,
        write_energy_tags: bool = True,
    ) -> "Conformer":
        """
        Write current MolBlocks to SDF files.

        Each molecule is written as one SDF file. When energy data are available
        and ``write_energy_tags`` is enabled, conformer energies are stored as
        properties named ``CONF_ENERGY_<id>``.

        If ``per_mol_folder`` is ``True``, output files are written as:

        ``out_folder/ligand_i/ligand_i.sdf``

        Otherwise files are written directly under ``out_folder`` as:

        ``out_folder/ligand_i.sdf``

        :param out_folder:
            Destination directory.
        :type out_folder: str
        :param per_mol_folder:
            Whether to create one subdirectory per molecule.
        :type per_mol_folder: bool
        :param write_energy_tags:
            Whether to write ``CONF_ENERGY_<id>`` properties into the SDF.
        :type write_energy_tags: bool

        :returns:
            The current conformer workflow instance.
        :rtype: Conformer

        Example
        -------
        .. code-block:: python

            conf.write_sdf(
                "outdir",
                per_mol_folder=False,
                write_energy_tags=True,
            )
        """
        out = Path(out_folder)
        out.mkdir(parents=True, exist_ok=True)
        for i, block in enumerate(self._molblocks):
            mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            if mol is None:
                logger.warning(
                    "Conformer.write_sdf: could not parse molblock for index %d",
                    i,
                )
                continue

            if write_energy_tags and i < len(self._energies):
                e_map = self._energies[i]
                for cid, energy in e_map.items():
                    try:
                        mol.SetProp(f"CONF_ENERGY_{cid}", str(energy))
                    except Exception:
                        logger.debug(
                            "Failed to set energy property for mol %d cid %s", i, cid
                        )

            if per_mol_folder:
                folder = out / f"ligand_{i}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{folder.name}.sdf"
            else:
                path = out / f"ligand_{i}.sdf"

            writer = Chem.SDWriter(str(path))
            writer.write(mol)
            writer.close()
            logger.debug("Conformer: wrote SDF for ligand %d -> %s", i, path)

        logger.info("Conformer: wrote %d SDF files to %s", len(self._molblocks), out)
        return self


ConformerManager = Conformer
