# prodock/chem/conformer.py
"""
Conformer manager: orchestrates embedding + optimization, exposes write_sdf.

This file provides ConformerManager (alias Conformer) which is the high-level
orchestrator. It uses joblib (loky) for parallelism (only in this module).
Large steps are logged with prodock.io.logging via log_step.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

# RDKit imports and log suppression
try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    raise ImportError("RDKit is required for prodock.chem.conformer")

# local modules
from prodock.io.logging import get_logger, StructuredAdapter
from prodock.chem.embed import Embedder
from prodock.chem.optimize import Optimizer

try:
    from joblib import Parallel, delayed

    _JOBLIB_AVAILABLE = True
except Exception:
    _JOBLIB_AVAILABLE = False


logger = StructuredAdapter(
    get_logger("prodock.chem.conformer"), {"component": "conformer"}
)
logger._base_logger = getattr(logger, "_base_logger", getattr(logger, "logger", None))


def _embed_worker(
    smiles: str, seed: int, embed_kwargs: Dict
) -> Tuple[Optional[str], int]:
    """
    Worker wrapper for embedding: creates local Embedder, embeds one SMILES,
    returns (MolBlock or None, conf_count).
    """
    # limit native threads in worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    e = Embedder(seed=seed)
    e.load_smiles_iterable([smiles])
    e.embed_all(**embed_kwargs)
    if not e.molblocks:
        return None, 0
    return e.molblocks[0], (e.conf_counts[0] if e.conf_counts else 0)


def _optimize_worker(
    molblock: str, seed: int, method: str, max_iters: int
) -> Tuple[Optional[str], Dict[int, float]]:
    """
    Worker wrapper for optimization: create local Optimizer, optimize single MolBlock,
    return optimized MolBlock and energy map.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    opt = Optimizer(max_iters=max_iters)
    opt.load_molblocks([molblock])
    opt.optimize_all(method=method)
    if not opt.optimized_molblocks:
        return None, {}
    return opt.optimized_molblocks[0], (opt.energies[0] if opt.energies else {})


class ConformerManager:
    """
    High-level manager composing Embedder + Optimizer.

    :param seed: RNG seed passed to worker embedder instances
    :param backend: joblib backend to use when parallelizing (default 'loky')
    """

    def __init__(self, seed: int = 42, backend: str = "loky") -> None:
        self._seed = int(seed)
        self._backend = backend
        self._smiles: List[str] = []
        self._molblocks: List[str] = []
        self._conf_counts: List[int] = []
        self._energies: List[Dict[int, float]] = []

    def __repr__(self) -> str:
        return f"<ConformerManager smiles={len(self._smiles)} mols={len(self._molblocks)} seed={self._seed}>"

    def help(self) -> None:
        print(self.__doc__)

    # ---------- properties ----------
    @property
    def smiles(self) -> List[str]:
        return list(self._smiles)

    @property
    def molblocks(self) -> List[str]:
        return list(self._molblocks)

    @property
    def conf_counts(self) -> List[int]:
        return list(self._conf_counts)

    @property
    def energies(self) -> List[Dict[int, float]]:
        return [dict(e) for e in self._energies]

    # ---------- loading ----------
    def load_smiles_file(self, path: str) -> "ConformerManager":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        with p.open("r", encoding="utf-8") as fh:
            self._smiles = [ln.strip().split()[0] for ln in fh if ln.strip()]
        logger.info("ConformerManager: loaded %d SMILES", len(self._smiles))
        return self

    def load_smiles(self, smiles: List[str]) -> "ConformerManager":
        self._smiles = [s.strip().split()[0] for s in smiles if s]
        return self

    # ---------- embedding ----------
    def embed_all(
        self,
        n_confs: int = 1,
        n_jobs: int = 1,
        embed_kwargs: Optional[Dict] = None,
    ) -> "ConformerManager":
        """
        Embed loaded SMILES; returns self.

        :param n_confs: number of conformers per molecule
        :param n_jobs: number of parallel jobs (-1 for all CPUs), 1 for sequential
        :param embed_kwargs: additional kwargs for Embedder.embed_all (e.g., add_hs, use_etkdg)
        """
        if not self._smiles:
            raise RuntimeError("No SMILES loaded; call load_smiles_file()")

        embed_kwargs = embed_kwargs or {}
        # ensure the embed kwargs include the number of confs and seed
        worker_kwargs = dict(embed_kwargs)
        worker_kwargs["n_confs"] = int(n_confs)
        worker_kwargs["random_seed"] = int(self._seed)

        results: List[Tuple[Optional[str], int]]
        if n_jobs == 1 or not _JOBLIB_AVAILABLE:
            results = [
                _embed_worker(smi, self._seed, worker_kwargs) for smi in self._smiles
            ]
        else:
            jobs = n_jobs
            results = Parallel(n_jobs=jobs, backend=self._backend)(
                delayed(_embed_worker)(smi, self._seed, worker_kwargs)
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
            "ConformerManager: embedded %d / %d molecules",
            len(self._molblocks),
            len(self._smiles),
        )
        return self

    # ---------- optimization ----------
    def optimize_all(
        self, method: str = "MMFF", n_jobs: int = 1, max_iters: int = 200
    ) -> "ConformerManager":
        """
        Optimize all embedded molblocks.

        :param method: 'UFF' or 'MMFF'
        :param n_jobs: parallel jobs; 1 for sequential
        :param max_iters: max iterations for optimizer
        """
        if not self._molblocks:
            raise RuntimeError(
                "No embedded molecules available; call embed_all() first"
            )

        if n_jobs == 1 or not _JOBLIB_AVAILABLE:
            results = [
                _optimize_worker(mb, self._seed, method, int(max_iters))
                for mb in self._molblocks
            ]
        else:
            jobs = n_jobs
            results = Parallel(n_jobs=jobs, backend=self._backend)(
                delayed(_optimize_worker)(mb, self._seed, method, int(max_iters))
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
        logger.info("ConformerManager: optimized %d molecules", len(self._molblocks))
        return self

    # ---------- pruning ----------
    def prune_top_k(self, k: int = 1) -> "ConformerManager":
        """
        Keep only top-k lowest-energy conformers per molecule (based on last optimization).
        """
        if not self._molblocks:
            raise RuntimeError("No molecules to prune")
        if not self._energies:
            logger.warning("ConformerManager: no energy data available; skipping prune")
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

            # sort conf ids lowest energy first
            sorted_ids = sorted(e_map.items(), key=lambda kv: kv[1])
            keep_ids = [cid for cid, _ in sorted_ids[: max(1, int(k))]]

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
                    logger.warning("ConformerManager: failed to copy conformer %s", cid)

            new_map = {i: e_map[cid] for i, cid in enumerate(keep_ids)}
            new_blocks.append(Chem.MolToMolBlock(base))
            new_energies.append(new_map)

        self._molblocks = new_blocks
        self._energies = new_energies
        self._conf_counts = [len(e) for e in new_energies]
        logger.info(
            "ConformerManager: pruned to top-%d confs for %d molecules",
            k,
            len(self._molblocks),
        )
        return self

    # ---------- write (this fixes your failing test) ----------
    def write_sdf(
        self,
        out_folder: str,
        per_mol_folder: bool = True,
        write_energy_tags: bool = True,
    ) -> "ConformerManager":
        """
        Write SDF outputs. Each molblock becomes a SDF. Optionally add
        CONF_ENERGY_<id> properties for each conformer energy.

        :param out_folder: destination folder path
        :param per_mol_folder: if True, create ligand_i/ligand_i.sdf
        :param write_energy_tags: write CONF_ENERGY_<id> properties when energies available
        :return: self
        """
        out = Path(out_folder)
        out.mkdir(parents=True, exist_ok=True)
        written_paths: List[Path] = []

        for i, block in enumerate(self._molblocks):
            mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            if mol is None:
                logger.warning(
                    "ConformerManager.write_sdf: could not parse molblock for index %d",
                    i,
                )
                continue

            # attach energy tags if available
            if write_energy_tags and i < len(self._energies):
                e_map = self._energies[i]
                for cid, energy in e_map.items():
                    # property keys are strings
                    try:
                        mol.SetProp(f"CONF_ENERGY_{cid}", str(energy))
                    except Exception:
                        # ignore failures to set props
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
            written_paths.append(path)

        logger.info(
            "ConformerManager: wrote %d SDF files to %s", len(written_paths), out
        )
        return self


# Backwards-compatible alias: your tests tried Conformer first
Conformer = ConformerManager
