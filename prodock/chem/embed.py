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
from typing import List, Optional, Dict, Any, Iterable, Tuple
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
    :type seed: int
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = int(seed)
        self._smiles: List[str] = []
        self._mols: List[Chem.Mol] = []  #: RDKit Mol with conformers
        self._molblocks: List[str] = []  #: MolBlock representation of mols
        self._conf_counts: List[int] = []  #: number of conformers per mol
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
        :type path: str
        :param sanitize: If True, RDKit sanitization is applied when parsing.
        :type sanitize: bool
        :return: self
        :rtype: Embedder
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
        :type smiles_iter: Iterable[str]
        :param sanitize: If True, attempt RDKit sanitization.
        :type sanitize: bool
        :return: self
        :rtype: Embedder
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
        :type molblocks: Iterable[str]
        :return: self
        :rtype: Embedder
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

    # ---------------- embed params builder (split helpers) ----------------
    @staticmethod
    def _select_algorithm_params(
        embed_algorithm: Optional[str],
    ) -> AllChem.EmbedParameters:
        """
        Select RDKit EmbedParameters object for a requested algorithm.

        :param embed_algorithm: algorithm name (case-insensitive), e.g. "ETKDGv3".
        :type embed_algorithm: Optional[str]
        :return: an EmbedParameters-like object
        :rtype: AllChem.EmbedParameters
        """
        alg = (embed_algorithm or "").upper() if embed_algorithm is not None else ""
        try:
            if alg == "ETKDGV3" and hasattr(AllChem, "ETKDGv3"):
                return AllChem.ETKDGv3()
            if alg == "ETKDGV2" and hasattr(AllChem, "ETKDGv2"):
                return AllChem.ETKDGv2()
            if alg == "ETKDG" and hasattr(AllChem, "ETKDG"):
                return AllChem.ETKDG()
        except Exception:
            # fall through to generic
            pass
        return AllChem.EmbedParameters()

    @staticmethod
    def _try_set_param(params: AllChem.EmbedParameters, attr: str, value: Any) -> None:
        """
        Try to set an attribute on params if present; swallow and log failures.

        :param params: EmbedParameters object
        :type params: AllChem.EmbedParameters
        :param attr: attribute name to set
        :type attr: str
        :param value: value to set
        :type value: Any
        """
        if value is None:
            return
        if not hasattr(params, attr):
            return
        try:
            setattr(params, attr, value)
        except Exception:
            logger.debug(
                "Embedder: could not set param %s on params", attr, exc_info=False
            )

    @staticmethod
    def _configure_params(
        params: AllChem.EmbedParameters,
        random_seed: Optional[int],
        max_attempts: int,
        clear_confs: bool,
        num_threads: int,
        extras: Dict[str, Any],
    ) -> AllChem.EmbedParameters:
        """
        Configure common EmbedParameters attributes in a best-effort manner.

        This helper delegates atomic set attempts to `_try_set_param` to keep
        cyclomatic complexity low in the orchestration function.

        :param params: EmbedParameters object to configure.
        :type params: AllChem.EmbedParameters
        :param random_seed: RNG seed (or None).
        :type random_seed: Optional[int]
        :param max_attempts: requested maxAttempts.
        :type max_attempts: int
        :param clear_confs: whether to clear previous conformers.
        :type clear_confs: bool
        :param num_threads: requested thread count.
        :type num_threads: int
        :param extras: any extra params to apply if attributes exist.
        :type extras: Dict[str, Any]
        :return: configured params
        :rtype: AllChem.EmbedParameters
        """
        # single loop over candidate attributes -> minimal branching here
        candidates = {
            "randomSeed": random_seed,
            "maxAttempts": int(max_attempts) if max_attempts is not None else None,
            "clearConfs": bool(clear_confs),
            "numThreads": int(num_threads) if num_threads is not None else None,
        }

        for attr, val in candidates.items():
            Embedder._try_set_param(params, attr, val)

        # extras: set only attributes that exist
        for k, v in (extras or {}).items():
            Embedder._try_set_param(params, k, v)

        return params

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

        This method delegates to smaller helpers so complexity is kept low.

        :param embed_algorithm: algorithm name (e.g. "ETKDGv3")
        :param random_seed: RNG seed (or None)
        :param max_attempts: maxAttempts value when supported
        :param clear_confs: clear previous conformers before embedding
        :param num_threads: requested number of threads (best-effort)
        :param extras: extra params to set on the object if attributes exist
        :return: configured EmbedParameters
        """
        params = Embedder._select_algorithm_params(embed_algorithm)
        params = Embedder._configure_params(
            params,
            random_seed=random_seed,
            max_attempts=max_attempts,
            clear_confs=clear_confs,
            num_threads=num_threads,
            extras=extras,
        )
        return params

    # ---------------- embedding helpers (small focused functions) ----------------
    @staticmethod
    def _parse_smiles(smi: str) -> Optional[Chem.Mol]:
        """
        Parse a SMILES string to an RDKit Mol (sanitization enabled).

        :param smi: SMILES string
        :type smi: str
        :return: RDKit Mol or None
        :rtype: Optional[Chem.Mol]
        """
        try:
            return Chem.MolFromSmiles(smi, sanitize=True)
        except Exception:
            logger.debug("Embedder: exception parsing SMILES %s", smi, exc_info=False)
            return None

    @staticmethod
    def _add_hs_if_requested(mol: Chem.Mol, add_hs: bool) -> Chem.Mol:
        """
        Return a copy of mol with hydrogens added if requested.

        :param mol: RDKit Mol
        :type mol: Chem.Mol
        :param add_hs: whether to add hydrogens
        :type add_hs: bool
        :return: working Mol
        :rtype: Chem.Mol
        """
        working = Chem.Mol(mol)
        if add_hs:
            try:
                working = Chem.AddHs(working)
            except Exception:
                logger.debug(
                    "Embedder: AddHs failed; using original mol", exc_info=False
                )
        return working

    @staticmethod
    def _remove_conformers_safe(mol: Chem.Mol) -> None:
        """
        Remove all conformers from mol if API exists. Best-effort.

        :param mol: RDKit Mol to modify in-place
        :type mol: Chem.Mol
        """
        try:
            if hasattr(mol, "RemoveAllConformers"):
                mol.RemoveAllConformers()
        except Exception:
            logger.debug("Embedder: RemoveAllConformers failed", exc_info=False)

    @staticmethod
    def _embed_single_conf(
        mol: Chem.Mol, params: AllChem.EmbedParameters, rs: int
    ) -> bool:
        """
        Embed single conformer (best-effort). Returns True on success.

        :param mol: RDKit Mol (modified in-place)
        :type mol: Chem.Mol
        :param params: embed parameters
        :type params: AllChem.EmbedParameters
        :param rs: fallback random seed
        :type rs: int
        :return: success flag
        :rtype: bool
        """
        try:
            try:
                res = AllChem.EmbedMolecule(mol, params)
            except TypeError:
                res = AllChem.EmbedMolecule(mol, randomSeed=rs)
            return res != -1
        except Exception:
            logger.debug("Embedder: single embed exception", exc_info=False)
            return False

    @staticmethod
    def _embed_multiple_confs(
        mol: Chem.Mol, params: AllChem.EmbedParameters, n_confs: int
    ) -> int:
        """
        Embed multiple conformers and return the number of conformers created.

        :param mol: RDKit Mol (modified in-place)
        :type mol: Chem.Mol
        :param params: embed parameters
        :type params: AllChem.EmbedParameters
        :param n_confs: requested number of conformers
        :type n_confs: int
        :return: number of conformers generated (0 on failure)
        :rtype: int
        """
        try:
            try:
                cids = AllChem.EmbedMultipleConfs(
                    mol, numConfs=int(n_confs), params=params
                )
            except TypeError:
                cids = AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs))
            return len(cids) if cids is not None else 0
        except Exception:
            logger.debug("Embedder: EmbedMultipleConfs exception", exc_info=False)
            return 0

    @staticmethod
    def _molblock_safe(mol: Chem.Mol) -> str:
        """
        Return MolBlock for mol or an empty string on failure.

        :param mol: RDKit Mol
        :type mol: Chem.Mol
        :return: MolBlock string or ""
        :rtype: str
        """
        try:
            return Chem.MolToMolBlock(mol)
        except Exception:
            logger.debug("Embedder: MolToMolBlock failed", exc_info=False)
            return ""

    def _embed_smiles_one(
        self,
        smi: str,
        params: AllChem.EmbedParameters,
        n_confs: int,
        add_hs: bool,
        random_seed: int,
    ) -> Tuple[Optional[Chem.Mol], str, int]:
        """
        Embed a single SMILES string into an RDKit Mol with conformers.

        This function is an orchestrator that delegates to many tiny helpers; the
        helpers contain the branching and try/except so this method stays small.

        :param smi: SMILES string
        :type smi: str
        :param params: embed parameters
        :type params: AllChem.EmbedParameters
        :param n_confs: requested number of conformers
        :type n_confs: int
        :param add_hs: whether to add hydrogens before embedding
        :type add_hs: bool
        :param random_seed: integer random seed to pass into fallbacks
        :type random_seed: int
        :return: (Mol or None, MolBlock string or "", number_of_conformers)
        :rtype: Tuple[Optional[Chem.Mol], str, int]
        """
        mol = self._parse_smiles(smi)
        if mol is None:
            logger.warning("Embedder: failed to parse SMILES: %s", smi)
            return None, "", 0

        working = self._add_hs_if_requested(mol, add_hs)
        self._remove_conformers_safe(working)

        if int(n_confs) <= 1:
            ok = self._embed_single_conf(working, params, random_seed)
            if not ok:
                logger.debug("Embedder: single embed failed for %s", smi)
                return None, "", 0
            conf_count = 1
        else:
            conf_count = self._embed_multiple_confs(working, params, int(n_confs))
            if conf_count == 0:
                logger.debug("Embedder: EmbedMultipleConfs returned 0 for %s", smi)
                return None, "", 0

        mb = self._molblock_safe(working)
        return working, mb, conf_count

    # ---------------- embedding (orchestration) ----------------
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
        :type n_confs: int
        :param add_hs: add explicit hydrogens before embedding (default True).
        :type add_hs: bool
        :param embed_algorithm: exact embedding algorithm to use (e.g. "ETKDGv3",
                                "ETKDGv2", "ETKDG", or None for generic EmbedParameters).
        :type embed_algorithm: Optional[str]
        :param random_seed: seed used for the EmbedParameters (fallback to self._seed when None).
        :type random_seed: Optional[int]
        :param max_attempts: EmbedParameters.maxAttempts if supported.
        :type max_attempts: int
        :param clear_confs: clear existing conformers before embedding.
        :type clear_confs: bool
        :param num_threads: requested thread count for embedding params (best-effort).
        :type num_threads: int
        :return: self
        :rtype: Embedder
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

        # record last params (simple dict)
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
                logger.debug("Embedder: empty SMILES entry encountered; skipping")
                continue

            mol, mb, conf_count = self._embed_smiles_one(
                smi, params, n_confs, add_hs, rs
            )
            if mol is None:
                continue
            out_mols.append(mol)
            out_blocks.append(mb)
            out_counts.append(conf_count)

        self._mols = out_mols
        self._molblocks = out_blocks
        self._conf_counts = out_counts

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
        :type out_folder: str
        :param per_mol_folder: if True, write each SDF into its own folder ligand_i/ligand_i.sdf
        :type per_mol_folder: bool
        :return: self
        :rtype: Embedder
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
