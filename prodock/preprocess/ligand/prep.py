"""
prodock.process.ligand.prep
==============================

Utilities for converting SMILES strings into per-ligand 3D structure files.

This module provides the :class:`LigandPrep` class, which supports:

- loading ligands from SMILES lists, dictionaries, or pandas DataFrames,
- optional 3D embedding and geometry optimization,
- writing one intermediate SDF per ligand,
- converting SDF files into final formats such as ``PDB`` or ``PDBQT``,
- keeping all generated structures in memory as MolBlock strings,
- exporting a CSV manifest summarizing processing results.

Overview
--------
Each ligand is represented internally as a record containing the input SMILES,
optional name, processing status, output path, error message, and generated
MolBlock string.

Default behavior
----------------
- final output format: ``"pdbqt"``
- conversion backend: ``"meeko"``
- explicit hydrogens are added before embedding
- 3D embedding is enabled
- geometry optimization is enabled
- intermediate SDF files are removed after conversion unless
  :meth:`LigandPrep.set_keep_intermediate` is enabled

Processing model
----------------
For each input ligand, the workflow is typically:

1. Parse the SMILES string
2. Build 3D coordinates using either :class:`Conformer` or an RDKit fallback
3. Optionally optimize the geometry
4. Store the generated MolBlock in memory
5. Write an intermediate SDF file
6. Optionally convert the SDF into ``PDB`` or ``PDBQT``

If the final output format is already ``"sdf"``, no additional structure
conversion is performed.

Examples
--------

Basic usage from a list of SMILES::

    from prodock.process.ligand import LigandPrep

    proc = (
        LigandPrep(output_dir="ligands_out")
        .from_smiles_list(
            ["CCO", "c1ccccc1"],
            names=["ethanol", "benzene"],
        )
        .process_all()
        .save_manifest("ligands_manifest.csv")
    )

    print(proc.summary)
    print(proc.output_paths)

Using in-memory mode only (no files written)::

    proc = (
        LigandPrep(output_dir=None)
        .set_output_format("sdf")
        .from_smiles_list(["CCO", "CCN"])
        .process_all()
    )

    print(proc.sdf_strings[0])

Loading from a pandas DataFrame::

    import pandas as pd
    from prodock.process.ligand import LigandPrep

    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CC(=O)O"],
            "name": ["ethanol", "acetic_acid"],
        }
    )

    proc = (
        LigandPrep(output_dir="ligands_df")
        .from_dataframe(df)
        .set_output_format("pdb")
        .set_converter_backend("obabel")
        .process_all()
    )

    print(proc.ok)
"""

from __future__ import annotations

import csv
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from prodock.io.logging import StructuredAdapter, get_logger
except Exception:  # pragma: no cover

    def get_logger(name: str):
        """
        Fallback logger factory used when the project logging helper is unavailable.

        :param name:
            Logger name.
        :type name: str

        :returns:
            Standard library logger instance.
        :rtype: logging.Logger
        """
        return logging.getLogger(name)

    class StructuredAdapter(logging.LoggerAdapter):  # type: ignore
        """
        Lightweight fallback for structured logging.

        :param logger:
            Underlying logger.
        :type logger: logging.Logger

        :param extra:
            Extra mapping injected into log records.
        :type extra: Dict[str, Any]
        """

        def __init__(self, logger, extra):
            super().__init__(logger, extra)


logger = StructuredAdapter(
    get_logger("prodock.preprocess.ligand"),
    {"component": "ligand.process"},
)
logger._base_logger = getattr(logger, "_base_logger", getattr(logger, "logger", None))

# Optional Conformer
try:
    from .conformer import Conformer  # type: ignore

    _HAS_CONFORMER = True
except Exception:  # pragma: no cover
    Conformer = None  # type: ignore
    _HAS_CONFORMER = False
    logger.debug("Conformer not available; RDKit fallback will be used where needed.")

# Function-based conversion helpers
try:
    from prodock.structure.conversion import sdf_to_pdb, sdf_to_pdbqt  # type: ignore
except Exception:  # pragma: no cover
    sdf_to_pdb = None  # type: ignore
    sdf_to_pdbqt = None  # type: ignore
    logger.debug("Structure conversion helpers not available.")

# RDKit
try:
    from rdkit import Chem  # type: ignore
    from rdkit import RDLogger  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    AllChem = None  # type: ignore


def _sanitize_filename(name: str, max_len: int = 120) -> str:
    """
    Make a filesystem-friendly filename from an arbitrary string.

    Non-alphanumeric characters are replaced by underscores and the result is
    truncated to ``max_len`` characters.

    :param name:
        Input name to sanitize.
    :type name: str

    :param max_len:
        Maximum allowed filename length.
    :type max_len: int

    :returns:
        Sanitized filename. If the result becomes empty, ``"molecule"`` is returned.
    :rtype: str

    Example
    -------
    .. code-block:: python

        safe = _sanitize_filename("Acetic acid / sample #1")
        print(safe)
        # Acetic_acid_sample_1
    """
    cleaned = re.sub(r"[^\w\-.]+", "_", str(name).strip())
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("_")
    return cleaned or "molecule"


class LigandPrep:
    """
    High-level helper to convert SMILES strings into per-ligand 3D structure files.

    This class provides a compact workflow for ligand preparation starting from
    SMILES input. It can ingest ligands from multiple input formats, generate
    3D coordinates, optimize geometries, write intermediate SDF files, and
    convert them into downstream formats such as ``PDB`` and ``PDBQT``.

    One internal record is stored for each ligand and updated in place during
    processing.

    Internal record schema
    ----------------------
    Each record has the form:

    .. code-block:: python

        {
            "index": int,
            "smiles": str,
            "name": str,
            "out_path": Optional[Path],
            "status": "pending" | "ok" | "failed",
            "error": Optional[str],
            "molblock": Optional[str],
        }

    Default behavior
    ----------------
    - output format: ``"pdbqt"``
    - conversion backend: ``"meeko"``
    - intermediate SDF files are deleted by default
    - 3D embedding, hydrogen addition, and optimization are enabled

    Notes
    -----
    - If the optional :class:`Conformer` helper is available, it is preferred
      for 3D generation.
    - Otherwise, an RDKit-based fallback is used.
    - If ``output_dir`` is ``None``, processing is performed entirely in memory
      and no files are written.
    - If the final output format is not ``"sdf"``, an intermediate SDF is
      written and converted using structure conversion helpers.

    :param output_dir:
        Directory used for writing output files. If ``None``, file output is
        disabled and structures are only stored in memory.
    :type output_dir: Optional[Union[str, Path]]

    :param smiles_key:
        Key used to locate SMILES values in dictionary rows and DataFrames.
    :type smiles_key: str

    :param name_key:
        Key used to locate ligand names in dictionary rows and DataFrames.
    :type name_key: str

    :param index_pad:
        Zero-padding width used when auto-generating names for unnamed ligands.
        For example, ``4`` produces names such as ``0000``, ``0001``, and so on.
    :type index_pad: int

    :raises OSError:
        If the output directory cannot be created.

    Example
    -------
    Create and process ligands into PDBQT files:

    .. code-block:: python

        from prodock.process.ligand import LigandPrep

        proc = (
            LigandPrep(output_dir="ligands_out")
            .set_output_format("pdbqt")
            .set_converter_backend("meeko")
            .from_smiles_list(
                ["CCO", "CCN"],
                names=["ethanol", "ethylamine"],
            )
            .process_all()
        )

        print(proc.summary)
        print(proc.output_paths)

    Example
    -------
    Process ligands without writing any files:

    .. code-block:: python

        proc = (
            LigandPrep(output_dir=None)
            .from_smiles_list(["CCO"])
            .process_all()
        )

        print(proc.sdf_strings)
        print(proc.mols)
    """

    _EXT_MAP = {
        "sdf": "sdf",
        "pdb": "pdb",
        "pdbqt": "pdbqt",
    }

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = "ligands_out",
        smiles_key: str = "smiles",
        name_key: str = "name",
        index_pad: int = 4,
    ) -> None:
        """
        Initialize a :class:`LigandPrep` instance.

        The instance starts with default ligand preparation settings suitable for
        common docking workflows: 3D embedding enabled, explicit hydrogens added,
        geometry optimization enabled, final output format set to ``"pdbqt"``,
        and conversion backend set to ``"meeko"``.

        :param output_dir:
            Directory used for output files. If a string or :class:`~pathlib.Path`
            is provided, the directory is created automatically when needed. If
            ``None``, file writing is disabled and processed structures are kept
            only in memory.
        :type output_dir: Optional[Union[str, Path]]

        :param smiles_key:
            Key used to locate SMILES values in input dictionaries and DataFrames.
        :type smiles_key: str

        :param name_key:
            Key used to locate ligand names in input dictionaries and DataFrames.
        :type name_key: str

        :param index_pad:
            Zero-padding width used when generating fallback names for unnamed
            ligands. For example, ``index_pad=4`` yields names such as ``0000``,
            ``0001``, and ``0002``.
        :type index_pad: int

        :returns:
            This constructor returns nothing.
        :rtype: None

        :raises OSError:
            If the output directory cannot be created.

        Example
        -------
        .. code-block:: python

            proc = LigandPrep(
                output_dir="ligands_out",
                smiles_key="smiles",
                name_key="name",
                index_pad=4,
            )
        """
        self.output_dir: Optional[Path] = (
            Path(output_dir) if output_dir is not None else None
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.smiles_key = str(smiles_key)
        self.name_key = str(name_key)
        self.index_pad = int(index_pad)

        self._embed3d: bool = True
        self._add_hs: bool = True
        self._optimize: bool = True

        self._embed_algorithm: Optional[str] = "ETKDGv3"
        self._opt_method: str = "MMFF94"
        self._conformer_seed: int = 42
        self._conformer_n_jobs: int = 1
        self._opt_max_iters: int = 200

        self._output_format: str = "pdbqt"
        self._converter_backend: Optional[str] = "meeko"
        self._keep_intermediate: bool = False

        self._records: List[Dict[str, Any]] = []

    # ----------------------------- configuration ----------------------------- #
    def set_options(
        self,
        embed3d: Optional[bool] = None,
        add_hs: Optional[bool] = None,
        optimize: Optional[bool] = None,
    ) -> "LigandPrep":
        """
        Set simple boolean processing options.

        Only values explicitly provided are updated. Passing ``None`` leaves the
        corresponding setting unchanged.

        :param embed3d:
            Enable or disable 3D coordinate embedding.
        :type embed3d: Optional[bool]

        :param add_hs:
            Enable or disable explicit hydrogen addition before embedding.
        :type add_hs: Optional[bool]

        :param optimize:
            Enable or disable geometry optimization after embedding.
        :type optimize: Optional[bool]

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_options(
                embed3d=True,
                add_hs=True,
                optimize=False,
            )
        """
        if embed3d is not None:
            self._embed3d = bool(embed3d)
        if add_hs is not None:
            self._add_hs = bool(add_hs)
        if optimize is not None:
            self._optimize = bool(optimize)

        logger.debug(
            "Options: embed3d=%s add_hs=%s optimize=%s",
            self._embed3d,
            self._add_hs,
            self._optimize,
        )
        return self

    def set_embed_method(self, embed_algorithm: Optional[str]) -> "LigandPrep":
        """
        Set the embedding algorithm used by :class:`Conformer` or RDKit.

        Common values include ``"ETKDGv3"``, ``"ETKDGv2"``, and ``"ETKDG"``.
        Passing ``None`` clears the explicit preference and lets the fallback
        logic choose the best available method.

        :param embed_algorithm:
            Embedding algorithm name, or ``None``.
        :type embed_algorithm: Optional[str]

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_embed_method("ETKDGv3")
        """
        self._embed_algorithm = (
            None if embed_algorithm is None else str(embed_algorithm)
        )
        logger.debug("Embed algorithm -> %r", self._embed_algorithm)
        return self

    def set_opt_method(self, method: str) -> "LigandPrep":
        """
        Set the molecular mechanics optimization method.

        Typical values include ``"MMFF94"`` and ``"UFF"``. The chosen method is
        used for geometry optimization after 3D embedding, when optimization is
        enabled.

        :param method:
            Optimizer name.
        :type method: str

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_opt_method("UFF")
        """
        self._opt_method = str(method)
        logger.debug("Opt method -> %r", self._opt_method)
        return self

    def set_conformer_seed(self, seed: int) -> "LigandPrep":
        """
        Set the random seed used for conformer generation.

        This affects deterministic behavior in supported embedding workflows.

        :param seed:
            Integer random seed.
        :type seed: int

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_conformer_seed(123)
        """
        self._conformer_seed = int(seed)
        return self

    def set_conformer_jobs(self, n_jobs: int) -> "LigandPrep":
        """
        Set the number of parallel jobs used for conformer generation.

        This setting is forwarded to the optional :class:`Conformer` helper when
        available.

        :param n_jobs:
            Number of parallel jobs.
        :type n_jobs: int

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_conformer_jobs(4)
        """
        self._conformer_n_jobs = int(n_jobs)
        return self

    def set_opt_max_iters(self, max_iters: int) -> "LigandPrep":
        """
        Set the maximum number of optimization iterations.

        This value is used by the selected force-field optimizer.

        :param max_iters:
            Maximum optimization iteration count.
        :type max_iters: int

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_opt_max_iters(500)
        """
        self._opt_max_iters = int(max_iters)
        return self

    def set_output_format(self, fmt: str) -> "LigandPrep":
        """
        Set the final output format for processed ligands.

        Supported formats are ``"sdf"``, ``"pdb"``, and ``"pdbqt"``.

        :param fmt:
            Requested output format string.
        :type fmt: str

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        :raises ValueError:
            If the requested format is unsupported.

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_output_format("pdbqt")
        """
        key = (fmt or "").lower()
        if key not in self._EXT_MAP:
            raise ValueError(
                f"Unsupported output format {fmt!r}. Supported: {sorted(self._EXT_MAP)}"
            )
        self._output_format = key
        logger.debug("Output format -> %r", key)
        return self

    def set_converter_backend(self, backend: Optional[str]) -> "LigandPrep":
        """
        Set the backend used for SDF-to-final-format conversion.

        Typical values include:

        - ``"meeko"`` for PDBQT conversion
        - ``"obabel"`` for PDB or PDBQT conversion
        - ``"rdkit"`` for supported PDB conversion paths

        :param backend:
            Backend name, or ``None`` to clear the explicit selection.
        :type backend: Optional[str]

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_converter_backend("obabel")
        """
        self._converter_backend = None if backend is None else str(backend)
        logger.debug("Converter backend -> %r", self._converter_backend)
        return self

    def set_backend(self, backend: Optional[str]) -> "LigandPrep":
        """
        Alias for :meth:`set_converter_backend`.

        This method exists as a short convenience name.

        :param backend:
            Backend name, or ``None``.
        :type backend: Optional[str]

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_backend("meeko")
        """
        return self.set_converter_backend(backend)

    def set_keep_intermediate(self, keep: bool) -> "LigandPrep":
        """
        Control whether intermediate SDF files are retained.

        When the final output format is not ``"sdf"``, each ligand is first
        written to an intermediate SDF file. By default, that file is removed
        after conversion. Setting ``keep=True`` preserves it.

        :param keep:
            Whether to keep intermediate SDF files.
        :type keep: bool

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_keep_intermediate(True)
        """
        self._keep_intermediate = bool(keep)
        return self

    def set_output_dir(self, path: Optional[Union[str, Path]]) -> "LigandPrep":
        """
        Set or clear the output directory used for file writing.

        Passing ``None`` switches the instance into in-memory mode, where
        MolBlock strings are still generated but no output files are written.

        :param path:
            New output directory path, or ``None`` to disable file writing.
        :type path: Optional[Union[str, Path]]

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        :raises OSError:
            If the new directory cannot be created.

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_output_dir("prepared_ligands")
        """
        self.output_dir = Path(path) if path is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    # ----------------------------- input ingestion --------------------------- #
    def from_smiles_list(
        self,
        smiles: Sequence[str],
        names: Optional[Sequence[str]] = None,
    ) -> "LigandPrep":
        """
        Load ligand records from a sequence of SMILES strings.

        Optional names can be supplied in parallel. If no names are provided,
        fallback names based on the record index are used when output files are
        written.

        :param smiles:
            Sequence of SMILES strings.
        :type smiles: Sequence[str]

        :param names:
            Optional sequence of ligand names with the same length as ``smiles``.
        :type names: Optional[Sequence[str]]

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        :raises ValueError:
            If ``names`` is provided but its length does not match ``smiles``.

        Example
        -------
        .. code-block:: python

            proc = (
                LigandPrep()
                .from_smiles_list(
                    ["CCO", "CCN"],
                    names=["ethanol", "ethylamine"],
                )
            )
        """
        if names is not None and len(names) != len(smiles):
            raise ValueError("names (if provided) must match smiles length")

        entries: List[Dict[str, Any]] = []
        for i, smi in enumerate(smiles):
            row: Dict[str, Any] = {self.smiles_key: smi}
            if names is not None:
                row[self.name_key] = names[i]
            entries.append(row)

        self._load_entries(entries)
        return self

    def from_list_of_dicts(self, rows: Sequence[Dict[str, Any]]) -> "LigandPrep":
        """
        Load ligand records from a sequence of dictionaries.

        Each row must contain at least the configured SMILES key. If the
        configured name key is present, it is used as the ligand name.

        :param rows:
            Sequence of dictionaries containing ligand metadata.
        :type rows: Sequence[Dict[str, Any]]

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            rows = [
                {"smiles": "CCO", "name": "ethanol"},
                {"smiles": "CCN", "name": "ethylamine"},
            ]

            proc = LigandPrep().from_list_of_dicts(rows)
        """
        self._load_entries(list(rows))
        return self

    def from_dataframe(self, df: "pd.DataFrame") -> "LigandPrep":
        """
        Load ligand records from a pandas DataFrame.

        The DataFrame must contain at least the configured SMILES column. If the
        configured name column exists, it is used to populate ligand names.

        :param df:
            DataFrame containing ligand input records.
        :type df: pandas.DataFrame

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        :raises RuntimeError:
            If pandas is unavailable.

        :raises KeyError:
            If the required SMILES column is missing.

        Example
        -------
        .. code-block:: python

            import pandas as pd

            df = pd.DataFrame(
                {
                    "smiles": ["CCO", "CCN"],
                    "name": ["ethanol", "ethylamine"],
                }
            )

            proc = LigandPrep().from_dataframe(df)
        """
        if pd is None:
            raise RuntimeError("pandas is required for from_dataframe")
        if self.smiles_key not in df.columns:
            raise KeyError(f"DataFrame missing required column '{self.smiles_key}'")
        rows = df.to_dict(orient="records")
        self._load_entries(rows)
        return self

    def _load_entries(self, entries: List[Dict[str, Any]]) -> None:
        """
        Normalize input entries into the internal record structure.

        Existing records are replaced by the normalized entries.

        :param entries:
            List of mapping-like objects containing at least the configured
            SMILES key.
        :type entries: List[Dict[str, Any]]

        :returns:
            This method returns nothing.
        :rtype: None

        :raises KeyError:
            If an entry does not contain the configured SMILES key.

        Example
        -------
        .. code-block:: python

            proc = LigandPrep()
            proc._load_entries(
                [
                    {"smiles": "CCO", "name": "ethanol"},
                    {"smiles": "CCN", "name": "ethylamine"},
                ]
            )
        """
        self._records = []
        for i, row in enumerate(entries):
            smi = row.get(self.smiles_key) or row.get(self.smiles_key.lower())
            if smi is None:
                raise KeyError(
                    f"Entry {i} missing SMILES under key '{self.smiles_key}'"
                )
            name = row.get(self.name_key) or row.get(self.name_key.lower()) or ""
            self._records.append(
                {
                    "index": i,
                    "smiles": str(smi).strip(),
                    "name": str(name).strip(),
                    "out_path": None,
                    "status": "pending",
                    "error": None,
                    "molblock": None,
                }
            )

    # ----------------------------- RDKit fallback --------------------------- #
    def _build_embed_params(self):
        """
        Build RDKit embedding parameters from the configured algorithm name.

        The method attempts to choose the requested ETKDG variant when
        available. If the requested variant is unavailable, it falls back to the
        best available RDKit embedding parameter object.

        :returns:
            RDKit embedding parameter object, or ``None`` if RDKit is unavailable
            or parameter construction fails.
        :rtype: Any

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_embed_method("ETKDGv3")
            params = proc._build_embed_params()
        """
        if AllChem is None:
            return None

        method = (self._embed_algorithm or "").strip().lower()
        try:
            if method == "etkdgv3" and hasattr(AllChem, "ETKDGv3"):
                params = AllChem.ETKDGv3()
            elif method == "etkdgv2" and hasattr(AllChem, "ETKDGv2"):
                params = AllChem.ETKDGv2()
            elif method == "etkdg" and hasattr(AllChem, "ETKDG"):
                params = AllChem.ETKDG()
            elif hasattr(AllChem, "ETKDGv3"):
                params = AllChem.ETKDGv3()
            elif hasattr(AllChem, "ETKDGv2"):
                params = AllChem.ETKDGv2()
            elif hasattr(AllChem, "ETKDG"):
                params = AllChem.ETKDG()
            else:
                params = AllChem.EmbedParameters()

            if hasattr(params, "randomSeed"):
                params.randomSeed = int(self._conformer_seed)
            return params
        except Exception:
            return None

    def _embed_with_rdkit_inmemory(self, smiles: str) -> str:
        """
        Embed a single SMILES string into 3D coordinates using RDKit.

        This method is used as a fallback when the optional :class:`Conformer`
        helper is unavailable.

        :param smiles:
            SMILES string to parse and embed.
        :type smiles: str

        :returns:
            MolBlock string containing 3D coordinates.
        :rtype: str

        :raises RuntimeError:
            If RDKit is unavailable, the SMILES cannot be parsed, or embedding
            fails.

        Example
        -------
        .. code-block:: python

            proc = LigandPrep(output_dir=None)
            molblock = proc._embed_with_rdkit_inmemory("CCO")
            print(molblock[:100])
        """
        if Chem is None or AllChem is None:
            raise RuntimeError("RDKit not available for in-memory embedding")

        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            raise RuntimeError(f"Failed to parse SMILES: {smiles!r}")

        working = Chem.Mol(mol)
        if self._add_hs:
            working = Chem.AddHs(working)

        params = self._build_embed_params()

        try:
            if params is not None:
                rc = AllChem.EmbedMolecule(working, params)
            else:
                rc = AllChem.EmbedMolecule(
                    working, randomSeed=int(self._conformer_seed)
                )
        except Exception:
            rc = -1

        if rc != 0:
            try:
                rc = AllChem.EmbedMolecule(
                    working, randomSeed=int(self._conformer_seed)
                )
            except Exception as exc:
                raise RuntimeError(f"RDKit embedding failed: {exc}") from exc

        if rc != 0:
            raise RuntimeError("RDKit embedding failed with non-zero return code")

        if self._optimize:
            method = self._opt_method.strip().upper()
            try:
                if method.startswith("UFF"):
                    AllChem.UFFOptimizeMolecule(
                        working,
                        maxIters=int(self._opt_max_iters),
                    )
                else:
                    AllChem.MMFFOptimizeMolecule(
                        working,
                        maxIters=int(self._opt_max_iters),
                    )
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(
                        working,
                        maxIters=int(self._opt_max_iters),
                    )
                except Exception:
                    try:
                        AllChem.MMFFOptimizeMolecule(
                            working,
                            maxIters=int(self._opt_max_iters),
                        )
                    except Exception:
                        logger.debug(
                            "Optimization failed with both UFF and MMFF; "
                            "continuing with embedded coordinates."
                        )

        if not self._add_hs:
            working = Chem.RemoveHs(working)

        try:
            return Chem.MolToMolBlock(working)
        except Exception as exc:
            raise RuntimeError(f"Failed to convert Mol to MolBlock: {exc}") from exc

    # ----------------------------- filename helper -------------------------- #
    def _make_unique_base(self, base: str, ext: str) -> str:
        """
        Create a filename stem that is unique within the output directory.

        Existing files on disk and output names already assigned to current
        records are both taken into account.

        :param base:
            Desired base filename without an extension.
        :type base: str

        :param ext:
            File extension without a leading dot.
        :type ext: str

        :returns:
            Unique base filename.
        :rtype: str

        Example
        -------
        .. code-block:: python

            proc = LigandPrep(output_dir="ligands_out")
            unique_name = proc._make_unique_base("ethanol", "pdbqt")
            print(unique_name)
        """
        if self.output_dir is None:
            return base

        out_dir = Path(self.output_dir)
        candidate = out_dir / f"{base}.{ext}"
        used = {Path(r["out_path"]).name for r in self._records if r.get("out_path")}

        if not candidate.exists() and f"{base}.{ext}" not in used:
            return base

        suffix = 1
        while True:
            new_base = f"{base}_{suffix}"
            if (not (out_dir / f"{new_base}.{ext}").exists()) and (
                f"{new_base}.{ext}" not in used
            ):
                return new_base
            suffix += 1

    # ----------------------------- conversion helper ------------------------ #
    def _convert_intermediate_sdf(self, sdf_path: Path, final_out: Path) -> None:
        """
        Convert an intermediate SDF file into the requested final format.

        Supported targets are determined from the suffix of ``final_out``.

        :param sdf_path:
            Path to the intermediate input SDF file.
        :type sdf_path: Path

        :param final_out:
            Path to the desired final output file.
        :type final_out: Path

        :returns:
            This method returns nothing.
        :rtype: None

        :raises RuntimeError:
            If the required conversion helper is unavailable.

        :raises ValueError:
            If the requested format/backend combination is unsupported.

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().set_output_format("pdbqt")
            proc._convert_intermediate_sdf(
                Path("tmp_ligand.sdf"),
                Path("ligand.pdbqt"),
            )
        """
        target_ext = final_out.suffix.lower().lstrip(".")

        if target_ext == "pdbqt":
            if sdf_to_pdbqt is None:
                raise RuntimeError("sdf_to_pdbqt is not available.")
            backend = self._converter_backend or "meeko"
            sdf_to_pdbqt(
                sdf_path,
                final_out,
                backend=backend,  # type: ignore[arg-type]
            )
            return

        if target_ext == "pdb":
            if sdf_to_pdb is None:
                raise RuntimeError("sdf_to_pdb is not available.")

            backend = self._converter_backend or "obabel"
            if backend not in {"rdkit", "obabel"}:
                logger.warning(
                    "PDB conversion supports only backend='rdkit' or 'obabel'; "
                    "overriding backend=%r to 'obabel'.",
                    backend,
                )
                backend = "obabel"

            sdf_to_pdb(
                sdf_path,
                final_out,
                backend=backend,  # type: ignore[arg-type]
            )
            return

        raise ValueError(f"Unsupported conversion target: {target_ext!r}")

    # ----------------------------- core processing -------------------------- #
    def process_all(
        self,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> "LigandPrep":
        """
        Process all loaded ligand records between ``start`` and ``stop``.

        Each selected record is converted into a MolBlock representation in memory.
        If file output is enabled, an intermediate SDF is written and optionally
        converted into the configured final format.

        The range follows standard Python slicing rules:
        ``start`` is inclusive and ``stop`` is exclusive.

        :param start:
            Start index of the record range to process, inclusive.
        :type start: int

        :param stop:
            Stop index of the record range to process, exclusive. If ``None``,
            processing continues to the end of the loaded records.
        :type stop: Optional[int]

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = (
                LigandPrep(output_dir="ligands_out")
                .from_smiles_list(["CCO", "CCN", "CCC"])
                .process_all(start=1, stop=3)
            )

            print(proc.summary)
        """
        if not self._records:
            logger.warning("No records to process.")
            return self

        stop_idx = stop if stop is not None else len(self._records)
        for rec in self._records[start:stop_idx]:
            self._process_one(rec)
        return self

    def _process_one(self, rec: Dict[str, Any]) -> None:
        """
        Process a single internal record in place.

        The record is updated with its generated MolBlock, final output path,
        status, and any error message.

        :param rec:
            Internal record dictionary.
        :type rec: Dict[str, Any]

        :returns:
            This method returns nothing.
        :rtype: None

        Example
        -------
        .. code-block:: python

            proc = LigandPrep(output_dir=None).from_smiles_list(["CCO"])
            record = proc.records[0]
            proc._process_one(record)
            print(record["status"])
        """
        idx = int(rec["index"])
        smi = str(rec["smiles"])
        name = str(rec.get("name", "") or "")
        index_str = str(idx).zfill(self.index_pad)

        target_ext = self._EXT_MAP[self._output_format]
        raw_base = _sanitize_filename(name) if name else index_str
        base = (
            self._make_unique_base(raw_base, target_ext)
            if self.output_dir
            else raw_base
        )

        final_out: Optional[Path] = (
            self.output_dir / f"{base}.{target_ext}"
            if self.output_dir is not None
            else None
        )

        try:
            # Build MolBlock
            if (self._embed3d or self._optimize) and _HAS_CONFORMER:
                cm = Conformer(seed=self._conformer_seed)
                cm.load_smiles([smi])
                cm.embed_all(
                    n_confs=1,
                    n_jobs=self._conformer_n_jobs,
                    add_hs=self._add_hs,
                    embed_algorithm=self._embed_algorithm,
                )
                if self._optimize:
                    cm.optimize_all(
                        method=self._opt_method,
                        n_jobs=self._conformer_n_jobs,
                        max_iters=self._opt_max_iters,
                    )
                mb_list = getattr(cm, "molblocks", None)
                if not mb_list:
                    raise RuntimeError("Conformer failed to produce molblocks")
                mb = mb_list[0]
            else:
                mb = self._embed_with_rdkit_inmemory(smi)

            rec["molblock"] = mb

            # In-memory mode only
            if final_out is None:
                rec["out_path"] = None
                rec["status"] = "ok"
                rec["error"] = None
                logger.info("Record %d (%s) processed in-memory.", idx, name or smi)
                return

            if Chem is None:
                raise RuntimeError("RDKit required to write SDF intermediates.")

            if target_ext == "sdf":
                sdf_path = final_out
            else:
                tmp = tempfile.NamedTemporaryFile(
                    prefix=f"{base}_",
                    suffix=".sdf",
                    dir=self.output_dir,
                    delete=False,
                )
                sdf_path = Path(tmp.name)
                tmp.close()

            mol = Chem.MolFromMolBlock(
                mb,
                sanitize=False,
                removeHs=(not self._add_hs),
            )
            if mol is None:
                raise RuntimeError(
                    "Failed to parse MolBlock into RDKit Mol for writing."
                )

            if name:
                try:
                    mol.SetProp("_Name", name)
                except Exception:
                    pass

            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()

            if target_ext == "sdf":
                rec["out_path"] = sdf_path
                rec["status"] = "ok"
                rec["error"] = None
                logger.info("Record %d (%s) -> %s", idx, name or smi, str(sdf_path))
                return

            self._convert_intermediate_sdf(sdf_path, final_out)

            if (
                not self._keep_intermediate
                and sdf_path.exists()
                and sdf_path != final_out
            ):
                try:
                    sdf_path.unlink()
                except Exception:
                    logger.debug("Could not remove intermediate %s", sdf_path)

            rec["out_path"] = final_out
            rec["status"] = "ok"
            rec["error"] = None
            logger.info("Record %d (%s) -> %s", idx, name or smi, str(final_out))

        except Exception as exc:
            rec["out_path"] = None
            rec["molblock"] = None
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "Failed to process SMILES [%s] (index=%d): %s",
                smi,
                idx,
                exc,
            )

    # ----------------------------- persistence ------------------------------ #
    def save_manifest(
        self,
        path: Union[str, Path] = "ligands_manifest.csv",
    ) -> "LigandPrep":
        """
        Save a CSV manifest describing all processed records.

        The manifest contains one row per internal record and includes the
        record index, input SMILES, ligand name, output path, status, and error
        message.

        :param path:
            Destination path for the CSV manifest.
        :type path: Union[str, Path]

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = (
                LigandPrep(output_dir="ligands_out")
                .from_smiles_list(["CCO"], names=["ethanol"])
                .process_all()
                .save_manifest("ligands_manifest.csv")
            )
        """
        path = Path(path)
        rows: List[Dict[str, Any]] = []
        for r in self._records:
            rows.append(
                {
                    "index": r["index"],
                    "smiles": r["smiles"],
                    "name": r.get("name", ""),
                    "out_path": str(r["out_path"]) if r["out_path"] else "",
                    "status": r.get("status", ""),
                    "error": r.get("error", ""),
                }
            )

        if pd is not None:
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
        else:
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=rows[0].keys() if rows else ["index", "smiles"],
                )
                writer.writeheader()
                writer.writerows(rows)

        logger.info("Saved manifest to %s", path)
        return self

    # ----------------------------- properties ------------------------------- #
    @property
    def records(self) -> List[Dict[str, Any]]:
        """
        Return a shallow copy of the internal record list.

        :returns:
            List of record dictionaries.
        :rtype: List[Dict[str, Any]]

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().from_smiles_list(["CCO"])
            print(proc.records)
        """
        return list(self._records)

    @property
    def output_paths(self) -> List[Optional[Path]]:
        """
        Return output paths corresponding to all records.

        Records processed in in-memory mode or failed records may contain
        ``None`` values.

        :returns:
            List of output paths or ``None``.
        :rtype: List[Optional[Path]]

        Example
        -------
        .. code-block:: python

            proc = LigandPrep(output_dir="ligands_out").from_smiles_list(["CCO"])
            proc.process_all()
            print(proc.output_paths)
        """
        return [r["out_path"] for r in self._records]

    @property
    def failed(self) -> List[Dict[str, Any]]:
        """
        Return records that failed processing.

        :returns:
            List of failed record dictionaries.
        :rtype: List[Dict[str, Any]]

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().from_smiles_list(["not_a_smiles"]).process_all()
            print(proc.failed)
        """
        return [r for r in self._records if r.get("status") == "failed"]

    @property
    def ok(self) -> List[Dict[str, Any]]:
        """
        Return records that were processed successfully.

        :returns:
            List of successful record dictionaries.
        :rtype: List[Dict[str, Any]]

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().from_smiles_list(["CCO"]).process_all()
            print(proc.ok)
        """
        return [r for r in self._records if r.get("status") == "ok"]

    @property
    def summary(self) -> Dict[str, int]:
        """
        Return summary counts for total, successful, failed, and pending records.

        :returns:
            Summary dictionary with keys ``"total"``, ``"ok"``, ``"failed"``,
            and ``"pending"``.
        :rtype: Dict[str, int]

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().from_smiles_list(["CCO", "CCN"]).process_all()
            print(proc.summary)
        """
        total = len(self._records)
        ok = len(self.ok)
        failed = len(self.failed)
        pending = total - ok - failed
        return {
            "total": total,
            "ok": ok,
            "failed": failed,
            "pending": pending,
        }

    @property
    def sdf_strings(self) -> List[str]:
        """
        Return MolBlock strings for successfully processed records.

        Despite the property name, the stored values are MolBlock strings held in
        memory and not full multi-record SDF files.

        :returns:
            List of MolBlock strings.
        :rtype: List[str]

        Example
        -------
        .. code-block:: python

            proc = LigandPrep(output_dir=None).from_smiles_list(["CCO"]).process_all()
            print(proc.sdf_strings[0])
        """
        return [
            r["molblock"]
            for r in self._records
            if r.get("status") == "ok" and r.get("molblock")
        ]

    @property
    def mols(self) -> List[Any]:
        """
        Return RDKit Mol objects parsed from stored MolBlock strings.

        :returns:
            List of RDKit Mol objects.
        :rtype: List[Any]

        :raises RuntimeError:
            If RDKit is unavailable.

        Example
        -------
        .. code-block:: python

            proc = LigandPrep(output_dir=None).from_smiles_list(["CCO"]).process_all()
            mols = proc.mols
            print(len(mols))
        """
        if Chem is None:
            raise RuntimeError("RDKit not available to build RDKit Mol objects")

        out: List[Any] = []
        for mb in self.sdf_strings:
            mol = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
            if mol is not None:
                out.append(mol)
        return out

    def clear_records(self) -> "LigandPrep":
        """
        Remove all loaded records from the instance.

        This resets the processing state but does not delete any files already
        written to disk.

        :returns:
            The current instance for method chaining.
        :rtype: LigandPrep

        Example
        -------
        .. code-block:: python

            proc = LigandPrep().from_smiles_list(["CCO"])
            proc.clear_records()
            print(len(proc))
        """
        self._records = []
        return self

    def __len__(self) -> int:
        """
        Return the number of loaded records.

        :returns:
            Number of internal records.
        :rtype: int
        """
        return len(self._records)

    def __repr__(self) -> str:
        """
        Return a compact string representation of the current processing state.

        :returns:
            Debug-friendly representation string.
        :rtype: str
        """
        return (
            f"<LigandPrep: {len(self)} entries, "
            f"ok={self.summary['ok']}, "
            f"failed={self.summary['failed']}, "
            f"fmt={self._output_format}, "
            f"backend={self._converter_backend}>"
        )
