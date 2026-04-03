"""
RDKit-based embedding utilities for ``prodock.ligand``.

This module provides a lightweight object-oriented wrapper around RDKit
conformer embedding. It is designed for single-process use, either as a
standalone sequential utility or inside worker processes launched by higher-level
workflow managers such as ``ConformerManager``.

The main entry point is :class:`Embedder`, which supports:

- loading SMILES from files or iterables,
- loading precomputed MolBlock strings,
- embedding one or multiple conformers per molecule,
- retrieving embedded molecules as RDKit ``Mol`` objects or MolBlock strings,
- exporting embedded structures to SDF.

The embedding workflow is intentionally compact and explicit so the resulting
objects can be passed downstream to geometry optimization, docking preparation,
or file export stages.

Logging
-------
This module uses :class:`prodock.io.logging.StructuredAdapter` to emit structured
log messages for potentially long-running operations.

Example
-------
.. code-block:: python

    from prodock.chem.embed import Embedder

    emb = Embedder(seed=123)
    emb.load_smiles_iterable(["CCO", "c1ccccc1"])
    emb.embed_all(n_confs=2, add_hs=True, embed_algorithm="ETKDGv3")

    print(len(emb.mols))
    print(emb.conf_counts)
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Iterable, Tuple
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    raise ImportError(
        "RDKit is required for prodock.chem.embed: install rdkit from conda-forge"
    ) from e

from prodock.io.logging import get_logger, StructuredAdapter

logger = StructuredAdapter(get_logger("prodock.chem.embed"), {"component": "embed"})
logger._base_logger = getattr(logger, "_base_logger", getattr(logger, "logger", None))


class Embedder:
    """
    RDKit-based conformer embedding utility.

    This class encapsulates loading, embedding, and exporting functionality for
    small molecules represented as SMILES or MolBlock strings. Methods are
    chainable and return ``self`` where appropriate.

    Typical workflow
    ----------------
    1. Create an instance.
    2. Load SMILES or MolBlocks.
    3. Call :meth:`embed_all`.
    4. Access results through properties such as :attr:`mols`,
       :attr:`molblocks`, and :attr:`conf_counts`, or export with
       :meth:`mols_to_sdf`.

    :param seed:
        Default random seed used for embedding whenever an explicit
        ``random_seed`` is not provided to :meth:`embed_all`.
    :type seed: int

    Example
    -------
    .. code-block:: python

        emb = Embedder(seed=123)
        emb.load_smiles_iterable(["CCO", "c1ccccc1"])
        emb.embed_all(n_confs=2, add_hs=True, embed_algorithm="ETKDGv3")

        print(len(emb.mols))
        print(emb.conf_counts)
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = int(seed)
        self._smiles: List[str] = []
        self._mols: List[Chem.Mol] = []
        self._molblocks: List[str] = []
        self._conf_counts: List[int] = []
        self._last_params: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"<Embedder smiles={len(self._smiles)} mols={len(self._mols)} seed={self._seed}>"

    def help(self) -> None:
        """
        Print a short usage summary for the embedder.

        This is a convenience helper intended for interactive sessions.

        :returns:
            ``None``.
        :rtype: None

        Example
        -------
        .. code-block:: python

            emb = Embedder()
            emb.help()
        """
        print(
            "Embedder: load_smiles_file / load_smiles_iterable -> embed_all -> check .molblocks / .mols\n"
            "Key methods:\n"
            "  - load_smiles_file(path)\n"
            "  - load_smiles_iterable(iterable)\n"
            "  - embed_all(n_confs=1, add_hs=True, embed_algorithm='ETKDGv3', random_seed=None, max_attempts=1000)\n"
            "Properties: .smiles, .mols, .molblocks, .conf_counts"
        )

    @property
    def seed(self) -> int:
        """
        Return the default random seed.

        :returns:
            Integer seed used as the default embedding seed.
        :rtype: int
        """
        return self._seed

    @property
    def smiles(self) -> List[str]:
        """
        Return a copy of the loaded SMILES list.

        :returns:
            Loaded SMILES strings.
        :rtype: List[str]
        """
        return list(self._smiles)

    @property
    def mols(self) -> List[Chem.Mol]:
        """
        Return embedded RDKit molecules.

        A defensive copy is returned for each molecule.

        :returns:
            Embedded RDKit molecules.
        :rtype: List[Chem.Mol]
        """
        return [Chem.Mol(m) for m in self._mols]

    @property
    def molblocks(self) -> List[str]:
        """
        Return MolBlock strings for embedded molecules.

        :returns:
            MolBlock representations of embedded molecules.
        :rtype: List[str]
        """
        return list(self._molblocks)

    @property
    def conf_counts(self) -> List[int]:
        """
        Return the number of conformers generated per molecule.

        :returns:
            Number of conformers per successfully embedded molecule.
        :rtype: List[int]
        """
        return list(self._conf_counts)

    @property
    def last_params(self) -> Dict[str, Any]:
        """
        Return the parameters used in the most recent :meth:`embed_all` call.

        :returns:
            Dictionary of embedding parameters from the last run.
        :rtype: Dict[str, Any]
        """
        return dict(self._last_params)

    def load_smiles_file(self, path: str, sanitize: bool = True) -> "Embedder":
        """
        Load SMILES from a newline-separated text file.

        Each non-empty line is parsed by taking the first whitespace-separated
        token as the SMILES string. This allows files of the form
        ``<smiles> <name>`` while still supporting simple one-column input.

        The ``sanitize`` argument is accepted for API consistency, although
        validation is deferred until RDKit parsing during embedding.

        :param path:
            Path to a text file containing one SMILES entry per line.
        :type path: str
        :param sanitize:
            Placeholder flag for API compatibility. Parsing and sanitization are
            performed later during molecule construction.
        :type sanitize: bool

        :returns:
            The current embedder instance.
        :rtype: Embedder

        :raises FileNotFoundError:
            If ``path`` does not exist.

        Example
        -------
        .. code-block:: python

            emb = Embedder()
            emb.load_smiles_file("my_smiles.txt")
            emb.embed_all(n_confs=1)
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
        Load SMILES from an arbitrary iterable.

        Each item is stripped and the first whitespace-separated token is kept as
        the SMILES string.

        The ``sanitize`` argument is accepted for interface consistency, though
        actual RDKit parsing occurs later.

        :param smiles_iter:
            Iterable yielding SMILES strings or lines beginning with SMILES.
        :type smiles_iter: Iterable[str]
        :param sanitize:
            Placeholder flag for API consistency. Molecule parsing is deferred.
        :type sanitize: bool

        :returns:
            The current embedder instance.
        :rtype: Embedder

        Example
        -------
        .. code-block:: python

            emb = Embedder()
            emb.load_smiles_iterable(["CCO", "O=C=O"])
            emb.embed_all()
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
        Load existing MolBlock strings into the embedder state.

        This is useful when 3D coordinates are already available and the class is
        only needed for downstream storage, export, or unified access patterns.

        Invalid MolBlocks are skipped with a warning.

        :param molblocks:
            Iterable of MolBlock strings.
        :type molblocks: Iterable[str]

        :returns:
            The current embedder instance.
        :rtype: Embedder

        Example
        -------
        .. code-block:: python

            emb = Embedder()
            emb.load_molblocks([molblock_str1, molblock_str2])
            emb.mols_to_sdf("outdir")
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

    @staticmethod
    def _select_algorithm_params(
        embed_algorithm: Optional[str],
    ) -> AllChem.EmbedParameters:
        """
        Create an RDKit ``EmbedParameters`` object for a selected algorithm.

        Supported names include ``"ETKDGv3"``, ``"ETKDGv2"``, and ``"ETKDG"``.
        If the requested algorithm is unavailable, a generic
        ``AllChem.EmbedParameters()`` object is returned.

        :param embed_algorithm:
            Embedding algorithm name, case-insensitive.
        :type embed_algorithm: Optional[str]

        :returns:
            RDKit embedding parameter object.
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
            pass
        return AllChem.EmbedParameters()

    @staticmethod
    def _try_set_param(params: AllChem.EmbedParameters, attr: str, value: Any) -> None:
        """
        Set an attribute on an ``EmbedParameters`` object if supported.

        Unsupported attributes or assignment failures are silently ignored with a
        debug log message.

        :param params:
            RDKit embedding parameters object.
        :type params: AllChem.EmbedParameters
        :param attr:
            Attribute name to assign.
        :type attr: str
        :param value:
            Value to assign.
        :type value: Any

        :returns:
            ``None``.
        :rtype: None
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
        Configure common RDKit embedding parameters in a best-effort manner.

        :param params:
            RDKit embedding parameters object to modify.
        :type params: AllChem.EmbedParameters
        :param random_seed:
            Random seed to apply, if supported.
        :type random_seed: Optional[int]
        :param max_attempts:
            Maximum number of embedding attempts, if supported.
        :type max_attempts: int
        :param clear_confs:
            Whether existing conformers should be cleared before embedding.
        :type clear_confs: bool
        :param num_threads:
            Requested thread count, if supported by the RDKit build and API.
        :type num_threads: int
        :param extras:
            Additional attributes to attempt to set on the parameter object.
        :type extras: Dict[str, Any]

        :returns:
            Configured RDKit embedding parameters object.
        :rtype: AllChem.EmbedParameters
        """
        candidates = {
            "randomSeed": random_seed,
            "maxAttempts": int(max_attempts) if max_attempts is not None else None,
            "clearConfs": bool(clear_confs),
            "numThreads": int(num_threads) if num_threads is not None else None,
        }

        for attr, val in candidates.items():
            Embedder._try_set_param(params, attr, val)

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
        Build and configure an RDKit ``EmbedParameters`` object.

        :param embed_algorithm:
            Embedding algorithm name such as ``"ETKDGv3"`` or ``"ETKDGv2"``.
        :type embed_algorithm: Optional[str]
        :param random_seed:
            Random seed used by RDKit when supported.
        :type random_seed: Optional[int]
        :param max_attempts:
            Maximum number of embedding attempts.
        :type max_attempts: int
        :param clear_confs:
            Whether existing conformers should be removed before embedding.
        :type clear_confs: bool
        :param num_threads:
            Requested number of threads.
        :type num_threads: int
        :param extras:
            Additional RDKit embedding parameters to attempt to set dynamically.
        :type extras: Any

        :returns:
            Configured embedding parameters object.
        :rtype: AllChem.EmbedParameters
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

    @staticmethod
    def _parse_smiles(smi: str) -> Optional[Chem.Mol]:
        """
        Parse a SMILES string into an RDKit molecule.

        Sanitization is enabled.

        :param smi:
            SMILES string.
        :type smi: str

        :returns:
            Parsed RDKit molecule, or ``None`` on failure.
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
        Return a working copy of a molecule, optionally with explicit hydrogens.

        :param mol:
            Input RDKit molecule.
        :type mol: Chem.Mol
        :param add_hs:
            Whether explicit hydrogens should be added.
        :type add_hs: bool

        :returns:
            Working RDKit molecule for embedding.
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
        Remove all conformers from a molecule in a best-effort manner.

        :param mol:
            RDKit molecule modified in place.
        :type mol: Chem.Mol

        :returns:
            ``None``.
        :rtype: None
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
        Embed a single conformer.

        :param mol:
            RDKit molecule modified in place.
        :type mol: Chem.Mol
        :param params:
            RDKit embedding parameters.
        :type params: AllChem.EmbedParameters
        :param rs:
            Fallback random seed if the parameter-object call path is unavailable.
        :type rs: int

        :returns:
            ``True`` if embedding succeeded, otherwise ``False``.
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
        Embed multiple conformers for a molecule.

        :param mol:
            RDKit molecule modified in place.
        :type mol: Chem.Mol
        :param params:
            RDKit embedding parameters.
        :type params: AllChem.EmbedParameters
        :param n_confs:
            Number of conformers requested.
        :type n_confs: int

        :returns:
            Number of successfully generated conformers.
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
        Convert a molecule to MolBlock format.

        :param mol:
            RDKit molecule.
        :type mol: Chem.Mol

        :returns:
            MolBlock string, or an empty string on failure.
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
        Embed one SMILES string and return the generated molecule data.

        :param smi:
            Input SMILES string.
        :type smi: str
        :param params:
            RDKit embedding parameters.
        :type params: AllChem.EmbedParameters
        :param n_confs:
            Requested number of conformers.
        :type n_confs: int
        :param add_hs:
            Whether hydrogens should be added before embedding.
        :type add_hs: bool
        :param random_seed:
            Integer random seed used by fallback embedding paths.
        :type random_seed: int

        :returns:
            Tuple ``(mol, molblock, conf_count)`` where ``mol`` may be ``None`` if
            embedding failed.
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
        Embed all loaded SMILES into RDKit molecules with conformers.

        This method iterates over the loaded SMILES collection, parses each
        string into an RDKit molecule, optionally adds hydrogens, performs
        conformer embedding, and stores both the RDKit molecules and MolBlock
        strings internally.

        :param n_confs:
            Number of conformers to generate per molecule.
        :type n_confs: int
        :param add_hs:
            Whether explicit hydrogens should be added before embedding.
        :type add_hs: bool
        :param embed_algorithm:
            Name of the RDKit embedding algorithm. Common values include
            ``"ETKDGv3"``, ``"ETKDGv2"``, ``"ETKDG"``, or ``None`` for a generic
            parameter object.
        :type embed_algorithm: Optional[str]
        :param random_seed:
            Random seed for embedding. If ``None``, the instance seed is used.
        :type random_seed: Optional[int]
        :param max_attempts:
            Maximum number of embedding attempts when supported by RDKit.
        :type max_attempts: int
        :param clear_confs:
            Whether existing conformers should be removed before embedding.
        :type clear_confs: bool
        :param num_threads:
            Requested number of embedding threads in RDKit, applied on a
            best-effort basis.
        :type num_threads: int

        :returns:
            The current embedder instance.
        :rtype: Embedder

        :raises RuntimeError:
            If no SMILES have been loaded before calling this method.

        Example
        -------
        .. code-block:: python

            emb = Embedder(seed=7)
            emb.load_smiles_iterable(["CCO", "CCN"])
            emb.embed_all(
                n_confs=1,
                add_hs=True,
                embed_algorithm="ETKDGv3",
                random_seed=7,
            )

            print(emb.conf_counts)
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

    def mols_to_sdf(self, out_folder: str, per_mol_folder: bool = True) -> "Embedder":
        """
        Write embedded molecules to SDF files.

        If ``per_mol_folder`` is ``True``, each molecule is written to a dedicated
        subdirectory of the form ``out_folder/ligand_i/ligand_i.sdf``. Otherwise,
        all SDF files are written directly into ``out_folder``.

        :param out_folder:
            Destination directory for SDF output.
        :type out_folder: str
        :param per_mol_folder:
            Whether each molecule should be placed in its own subdirectory.
        :type per_mol_folder: bool

        :returns:
            The current embedder instance.
        :rtype: Embedder

        Example
        -------
        .. code-block:: python

            emb = Embedder()
            emb.load_smiles_iterable(["CCO"])
            emb.embed_all()
            emb.mols_to_sdf("outdir", per_mol_folder=False)
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
