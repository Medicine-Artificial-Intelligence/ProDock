# prodock/chem/embed.py
"""
Embedder: RDKit-only embedding utilities (OOP) for prodock.chem.

Single-process embedding. Designed to be called inside worker processes
(created by ConformerManager) or used sequentially. Produces RDKit Mol objects
and MolBlock strings for downstream optimization.

Logging:
    Uses prodock.io.logging StructuredAdapter to emit structured logs for long-running operations.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Iterable
from pathlib import Path
import logging

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    raise ImportError(
        "RDKit is required for prodock.chem.embed: install rdkit from conda-forge"
    ) from e

# prodock logging utilities (assume available in your environment)
try:
    from prodock.io.logging import get_logger, StructuredAdapter
except Exception:
    # minimal fallback
    def get_logger(name: str):
        return logging.getLogger(name)

    class StructuredAdapter(logging.LoggerAdapter):
        def __init__(self, logger, extra):
            super().__init__(logger, extra)


logger = StructuredAdapter(get_logger("prodock.chem.embed"), {"component": "embed"})
logger._base_logger = getattr(logger, "_base_logger", getattr(logger, "logger", None))


class Embedder:
    """
    Embedder class encapsulates RDKit embedding functionality.

    Methods are chainable (return self). Use properties to access results.

    :param seed: random seed for deterministic embeddings.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = int(seed)
        self._smiles: List[str] = []
        self._mols: List[Chem.Mol] = []  # RDKit Mol with conformers
        self._molblocks: List[str] = []  # MolBlock representation of mols
        self._conf_counts: List[int] = []  # number of conformers per mol
        self._last_params: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"<Embedder smiles={len(self._smiles)} mols={len(self._mols)} seed={self._seed}>"

    def help(self) -> None:
        """Print short usage help for the Embedder."""
        print(
            "Embedder: load_smiles_file / load_smiles_iterable -> embed_all -> check .molblocks / .mols\n"
            "Key methods:\n"
            "  - load_smiles_file(path)\n"
            "  - load_smiles_iterable(iterable)\n"
            "  - embed_all(n_confs=1, add_hs=True, embed_algorithm='ETKDGv3', random_seed=None, max_attempts=1000)\n"
            "Properties: .smiles, .mols, .molblocks, .conf_counts"
        )

    # ---------------- properties ----------------
    @property
    def seed(self) -> int:
        """Random seed used for embeddings."""
        return self._seed

    @property
    def smiles(self) -> List[str]:
        """Return list of loaded SMILES (copy)."""
        return list(self._smiles)

    @property
    def mols(self) -> List[Chem.Mol]:
        """Return RDKit Mol objects (copied)."""
        return [Chem.Mol(m) for m in self._mols]

    @property
    def molblocks(self) -> List[str]:
        """Return MolBlock strings for embedded molecules."""
        return list(self._molblocks)

    @property
    def conf_counts(self) -> List[int]:
        """Return the number of conformers embedded per molecule."""
        return list(self._conf_counts)

    @property
    def last_params(self) -> Dict[str, Any]:
        """Return the embed parameters used in the last embed_all call."""
        return dict(self._last_params)

    # ---------------- loading ----------------
    def load_smiles_file(self, path: str, sanitize: bool = True) -> "Embedder":
        """
        Load SMILES from a newline-separated file.

        :param path: Path to SMILES file (one SMILES per line; name after whitespace allowed).
        :param sanitize: If True, RDKit sanitization is applied when parsing.
        :return: self
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        with p.open("r", encoding="utf-8") as fh:
            self._smiles = [ln.strip().split()[0] for ln in fh if ln.strip()]
        logger.info("Embedder: loaded %d SMILES from %s", len(self._smiles), path)
        return self

    def load_smiles_iterable(
        self, smiles_iter: Iterable[str], sanitize: bool = True
    ) -> "Embedder":
        """
        Load SMILES from any iterable of strings.

        :param smiles_iter: Iterable yielding SMILES strings.
        :param sanitize: If True, attempt RDKit sanitization.
        :return: self
        """
        out: List[str] = []
        for s in smiles_iter:
            if not s:
                continue
            smi = s.strip().split()[0]
            out.append(smi)
        self._smiles = out
        logger.info("Embedder: loaded %d SMILES from iterable", len(self._smiles))
        return self

    def load_molblocks(self, molblocks: Iterable[str]) -> "Embedder":
        """
        Load existing MolBlock strings (they will be interpreted as RDKit Mols).

        :param molblocks: Iterable of MolBlock strings.
        :return: self
        """
        out_mols: List[Chem.Mol] = []
        out_blocks: List[str] = []
        for mb in molblocks:
            if not mb:
                continue
            m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
            if m is None:
                logger.warning("Embedder: failed to parse MolBlock; skipping")
                continue
            out_mols.append(m)
            out_blocks.append(mb)
        self._mols = out_mols
        self._molblocks = out_blocks
        self._conf_counts = [m.GetNumConformers() for m in out_mols]
        logger.info("Embedder: loaded %d MolBlocks", len(self._molblocks))
        return self

    # ---------------- embed params builder ----------------
    @staticmethod
    def _build_embed_params(
        embed_algorithm: Optional[str] = "ETKDGv3",
        random_seed: Optional[int] = 42,
        max_attempts: int = 1000,
        clear_confs: bool = True,
        num_threads: int = 1,
        **extras: Any,
    ) -> AllChem.EmbedParameters:
        """
        Build an RDKit EmbedParameters object selecting a specific algorithm.

        :param embed_algorithm: one of "ETKDGv3", "ETKDGv2", "ETKDG", or None/"STANDARD".
                                If the requested algorithm is not available in the RDKit
                                installed, falls back to AllChem.EmbedParameters().
        :param random_seed: RNG seed (set when possible).
        :param max_attempts: maxAttempts value when supported.
        :param clear_confs: clear previous conformers before embedding.
        :param num_threads: requested number of threads (set on params if supported).
        :param extras: extra params to set on the object if attributes exist.
        :return: configured EmbedParameters
        """
        alg = (embed_algorithm or "").upper() if embed_algorithm is not None else ""
        try:
            if alg == "ETKDGV3" and hasattr(AllChem, "ETKDGv3"):
                params = AllChem.ETKDGv3()
            elif alg == "ETKDGV2" and hasattr(AllChem, "ETKDGv2"):
                params = AllChem.ETKDGv2()
            elif alg == "ETKDG" and hasattr(AllChem, "ETKDG"):
                params = AllChem.ETKDG()
            else:
                # fallback to generic params
                params = AllChem.EmbedParameters()
        except Exception:
            params = AllChem.EmbedParameters()

        # set common params if attributes exist
        if random_seed is not None and hasattr(params, "randomSeed"):
            try:
                params.randomSeed = int(random_seed)
            except Exception:
                pass

        if hasattr(params, "maxAttempts"):
            try:
                params.maxAttempts = int(max_attempts)
            except Exception:
                pass

        if hasattr(params, "clearConfs"):
            try:
                params.clearConfs = bool(clear_confs)
            except Exception:
                pass

        if hasattr(params, "numThreads"):
            try:
                params.numThreads = int(num_threads)
            except Exception:
                pass

        for k, v in extras.items():
            if hasattr(params, k):
                try:
                    setattr(params, k, v)
                except Exception:
                    pass

        return params

    # ---------------- embedding ----------------
    def embed_all(
        self,
        n_confs: int = 1,
        add_hs: bool = True,
        embed_algorithm: Optional[str] = "ETKDGv3",
        random_seed: Optional[int] = None,
        max_attempts: int = 1000,
        clear_confs: bool = True,
        num_threads: int = 1,
    ) -> "Embedder":
        """
        Sequentially embed all loaded SMILES into RDKit Mol objects with conformers.

        :param n_confs: number of conformers to generate per molecule.
        :param add_hs: add explicit hydrogens before embedding (default True).
        :param embed_algorithm: exact embedding algorithm to use (e.g. "ETKDGv3",
                                "ETKDGv2", "ETKDG", or None for generic EmbedParameters).
        :param random_seed: seed used for the EmbedParameters (fallback to self._seed when None).
        :param max_attempts: EmbedParameters.maxAttempts if supported.
        :param clear_confs: clear existing conformers before embedding.
        :param num_threads: requested thread count for embedding params (best-effort).
        :return: self
        """
        if not self._smiles:
            raise RuntimeError(
                "No SMILES loaded: call load_smiles_file / load_smiles_iterable first."
            )

        rs = int(random_seed) if random_seed is not None else int(self._seed)
        params = self._build_embed_params(
            embed_algorithm=embed_algorithm,
            random_seed=rs,
            max_attempts=max_attempts,
            clear_confs=clear_confs,
            num_threads=num_threads,
        )
        self._last_params = {
            "n_confs": int(n_confs),
            "add_hs": bool(add_hs),
            "embed_algorithm": embed_algorithm,
            "random_seed": rs,
            "max_attempts": int(max_attempts),
            "clear_confs": bool(clear_confs),
            "num_threads": int(num_threads),
        }

        out_mols: List[Chem.Mol] = []
        out_blocks: List[str] = []
        out_counts: List[int] = []

        for smi in self._smiles:
            if not smi:
                out_mols.append(None)  # keep shape; will filter later
                out_blocks.append(None)
                out_counts.append(0)
                continue
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            if mol is None:
                logger.warning("Embedder: failed to parse SMILES: %s", smi)
                out_mols.append(None)
                out_blocks.append(None)
                out_counts.append(0)
                continue

            working = Chem.Mol(mol)
            if add_hs:
                working = Chem.AddHs(working)

            # clear conformers if requested
            try:
                if hasattr(working, "RemoveAllConformers"):
                    working.RemoveAllConformers()
            except Exception:
                pass

            try:
                if int(n_confs) <= 1:
                    # single-conformer embed
                    try:
                        res = AllChem.EmbedMolecule(working, params)
                    except TypeError:
                        # older RDKit signature fallback
                        res = AllChem.EmbedMolecule(working, randomSeed=rs)
                    if res == -1:
                        logger.debug("Embedder: single embed failed for %s", smi)
                        out_mols.append(None)
                        out_blocks.append(None)
                        out_counts.append(0)
                        continue
                    conf_count = 1
                else:
                    # multiple conformers
                    try:
                        cids = AllChem.EmbedMultipleConfs(
                            working, numConfs=int(n_confs), params=params
                        )
                    except TypeError:
                        cids = AllChem.EmbedMultipleConfs(
                            working, numConfs=int(n_confs)
                        )
                    conf_count = len(cids)
                    if conf_count == 0:
                        logger.debug(
                            "Embedder: EmbedMultipleConfs returned 0 for %s", smi
                        )
                        out_mols.append(None)
                        out_blocks.append(None)
                        out_counts.append(0)
                        continue
            except Exception as e:
                logger.exception("Embedder: exception embedding %s: %s", smi, e)
                out_mols.append(None)
                out_blocks.append(None)
                out_counts.append(0)
                continue

            out_mols.append(working)
            try:
                mb = Chem.MolToMolBlock(working)
            except Exception:
                mb = ""
            out_blocks.append(mb)
            out_counts.append(conf_count)

        # filter None failures
        final_mols = []
        final_blocks = []
        final_counts = []
        for m, mb, c in zip(out_mols, out_blocks, out_counts):
            if m is None:
                continue
            final_mols.append(m)
            final_blocks.append(mb)
            final_counts.append(c)

        self._mols = final_mols
        self._molblocks = final_blocks
        self._conf_counts = final_counts
        logger.info(
            "Embedder: finished embedding: %d successes / %d attempts",
            len(self._mols),
            len(self._smiles),
        )
        return self

    # ---------------- small utilities ----------------
    def mols_to_sdf(self, out_folder: str, per_mol_folder: bool = True) -> "Embedder":
        """
        Write embedded molecules to SDF files.

        :param out_folder: destination folder path.
        :param per_mol_folder: if True, write each SDF into its own folder ligand_i/ligand_i.sdf
        :return: self
        """
        out = Path(out_folder)
        out.mkdir(parents=True, exist_ok=True)
        for i, mb in enumerate(self._molblocks):
            if not mb:
                continue
            mol = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
            if mol is None:
                continue
            if per_mol_folder:
                folder = out / f"ligand_{i}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"ligand_{i}.sdf"
            else:
                path = out / f"ligand_{i}.sdf"
            writer = Chem.SDWriter(str(path))
            writer.write(mol)
            writer.close()
            logger.debug("Embedder: wrote SDF for ligand %d -> %s", i, path)
        logger.info("Embedder: mols_to_sdf completed: wrote outputs to %s", out)
        return self
