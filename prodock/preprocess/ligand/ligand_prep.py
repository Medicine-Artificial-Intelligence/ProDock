"""
LigandProcess
=============

Convert SMILES -> per-ligand 3D files with optional embedding and optimization.

Defaults
--------
- If `prodock.io.convert.Converter` is available, default target format is "pdbqt"
  using backend "meeko" with temporary SDF handling via "rdkit".
- If Converter is not available, automatically falls back to "sdf".

Behaviour
---------
- Produces an intermediate SDF per-record, and if target != "sdf" uses Converter
  to create the final output (per-record). Intermediate SDFs are removed by default.
- MolBlock strings are retained in memory (rec["molblock"]) even when files are written.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

try:
    from prodock.io.logging import get_logger, StructuredAdapter
except Exception:  # pragma: no cover

    def get_logger(name: str):
        return logging.getLogger(name)

    class StructuredAdapter(logging.LoggerAdapter):  # type: ignore
        def __init__(self, logger, extra):
            super().__init__(logger, extra)


logger = StructuredAdapter(
    get_logger("prodock.process.ligand"), {"component": "ligand.process"}
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

# Optional Converter for non-SDF outputs
try:
    from prodock.io.convert import Converter  # type: ignore

    _HAS_CONVERTER = True
except Exception:  # pragma: no cover
    Converter = None  # type: ignore
    _HAS_CONVERTER = False
    logger.debug("Converter not available; non-SDF outputs will fall back to SDF.")

# RDKit (used for in-memory embed and writing SDF)
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    AllChem = None  # type: ignore


def _sanitize_filename(name: str, max_len: int = 120) -> str:
    """
    Make a filesystem-friendly filename from an arbitrary string.

    Non-alphanumeric characters are replaced by underscores and the result is
    truncated to ``max_len`` characters.

    :param name: input name to sanitize.
    :type name: str
    :param max_len: maximum allowed filename length (default 120).
    :type max_len: int
    :returns: sanitized filename (never empty; returns "molecule" if input becomes empty).
    :rtype: str
    :example:

    >>> _sanitize_filename("My molecule (test).smi")
    'My_molecule__test_.smi'
    """
    cleaned = re.sub(r"[^\w\-.]+", "_", str(name).strip())
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("_")
    return cleaned or "molecule"


class LigandProcess:
    """
    High-level helper to convert SMILES -> per-ligand files (SDF default fallback,
    or PDBQT/PDB/MOL2 if Converter is present).

    Records format (internal, one per-molecule):
        {
            "index": int,
            "smiles": str,
            "name": str,
            "out_path": Optional[Path],
            "status": "pending"|"ok"|"failed",
            "error": Optional[str],
            "molblock": Optional[str],
        }

    Usage example
    -------------
    Convert an in-memory list of SMILES and write SDF outputs::

        >>> lp = LigandProcess(output_dir="outdir")
        >>> lp.from_smiles_list(["CCO", "c1ccccc1"], names=["ethanol", "benzene"])
        >>> lp.process_all()
        >>> lp.save_manifest("outdir/manifest.csv")
    """

    # Supported mapping to file extensions
    _EXT_MAP = {
        "sdf": "sdf",
        "pdb": "pdb",
        "mol2": "mol2",
        "pdbqt": "pdbqt",
        "mol": "mol",
    }

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = "ligands_out",
        smiles_key: str = "smiles",
        name_key: str = "name",
        index_pad: int = 4,
    ) -> None:
        """
        Initialize a LigandProcess instance.

        :param output_dir: directory to write outputs. If None, file writing is disabled.
        :type output_dir: Optional[str | pathlib.Path]
        :param smiles_key: key name to locate SMILES in input dicts/DataFrame.
        :type smiles_key: str
        :param name_key: key name to locate molecule name in input dicts/DataFrame.
        :type name_key: str
        :param index_pad: zero-pad width when falling back to numeric names (default 4).
        :type index_pad: int
        :returns: None
        :rtype: None
        :example:

        >>> lp = LigandProcess(output_dir=None)
        >>> len(lp)
        0
        """
        self.output_dir: Optional[Path] = (
            Path(output_dir) if output_dir is not None else None
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.smiles_key = smiles_key
        self.name_key = name_key
        self.index_pad = int(index_pad)

        # embedding / optimisation defaults
        self._embed3d: bool = True
        self._add_hs: bool = True
        self._optimize: bool = True

        # Conformer options
        self._embed_algorithm: Optional[str] = "ETKDGv3"
        self._opt_method: str = "MMFF94"
        self._conformer_seed: int = 42
        self._conformer_n_jobs: int = 1
        self._opt_max_iters: int = 200

        # output/conversion defaults: prefer pdbqt via meeko if Converter available
        if _HAS_CONVERTER:
            self._output_format: str = "pdbqt"
            self._converter_backend: Optional[str] = "meeko"
            self._tmp_from_sdf_backend: Optional[str] = "rdkit"
        else:
            self._output_format = "sdf"
            self._converter_backend = None
            self._tmp_from_sdf_backend = None
            logger.warning("Converter not present: defaulting to SDF outputs.")

        self._keep_intermediate: bool = False  # remove SDF intermediate by default

        # internal records
        self._records: List[Dict] = []

    # ----------------------------- configuration ----------------------------- #
    def set_options(
        self,
        embed3d: Optional[bool] = None,
        add_hs: Optional[bool] = None,
        optimize: Optional[bool] = None,
    ) -> "LigandProcess":
        """
        Set simple boolean options.

        :param embed3d: enable/disable 3D embedding (if None leave unchanged).
        :type embed3d: Optional[bool]
        :param add_hs: add explicit hydrogens before embedding (if None leave unchanged).
        :type add_hs: Optional[bool]
        :param optimize: run optimization after embedding (if None leave unchanged).
        :type optimize: Optional[bool]
        :returns: self
        :rtype: LigandProcess
        :example:

        >>> lp = LigandProcess()
        >>> lp.set_options(embed3d=True, add_hs=False, optimize=True)
        <LigandProcess ...>
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

    def set_embed_method(self, embed_algorithm: Optional[str]) -> "LigandProcess":
        """
        Set the embedding algorithm used by Conformer / RDKit.

        :param embed_algorithm: algorithm name (e.g. "ETKDGv3") or None.
        :type embed_algorithm: Optional[str]
        :returns: self
        :rtype: LigandProcess
        """
        self._embed_algorithm = embed_algorithm
        logger.debug("Embed algorithm -> %r", embed_algorithm)
        return self

    def set_opt_method(self, method: str) -> "LigandProcess":
        """
        Set the optimization method (used by Conformer / RDKit fallback).

        :param method: optimizer name (e.g. "MMFF94" or "UFF").
        :type method: str
        :returns: self
        :rtype: LigandProcess
        """
        self._opt_method = str(method)
        logger.debug("Opt method -> %r", self._opt_method)
        return self

    def set_conformer_seed(self, seed: int) -> "LigandProcess":
        """
        Set RNG seed for conformer generation.

        :param seed: integer seed.
        :type seed: int
        :returns: self
        :rtype: LigandProcess
        """
        self._conformer_seed = int(seed)
        return self

    def set_conformer_jobs(self, n_jobs: int) -> "LigandProcess":
        """
        Set number of parallel jobs for Conformer (if available).

        :param n_jobs: number of jobs (int).
        :type n_jobs: int
        :returns: self
        :rtype: LigandProcess
        """
        self._conformer_n_jobs = int(n_jobs)
        return self

    def set_opt_max_iters(self, max_iters: int) -> "LigandProcess":
        """
        Set maximum iterations for the optimizer.

        :param max_iters: maximum number of iterations (int).
        :type max_iters: int
        :returns: self
        :rtype: LigandProcess
        """
        self._opt_max_iters = int(max_iters)
        return self

    # Output configuration (you asked default to be pdbqt/meeko; still exposed)
    def set_output_format(self, fmt: str) -> "LigandProcess":
        """
        Configure output format.

        Supported formats are: ``sdf``, ``pdb``, ``mol2``, ``pdbqt``, ``mol``.

        :param fmt: requested format string (case-insensitive).
        :type fmt: str
        :returns: self
        :rtype: LigandProcess
        :raises ValueError: if the format is unsupported.
        :example:

        >>> lp = LigandProcess()
        >>> lp.set_output_format("sdf")
        <LigandProcess ...>
        """
        key = (fmt or "").lower()
        if key not in self._EXT_MAP:
            raise ValueError(
                f"Unsupported output format {fmt!r}. Supported: {sorted(self._EXT_MAP)}"
            )
        if key != "sdf" and not _HAS_CONVERTER:
            logger.warning("Converter not available: forcing output format to 'sdf'.")
            key = "sdf"
        self._output_format = key
        logger.debug("Output format -> %r", key)
        return self

    def set_converter_backend(self, backend: Optional[str]) -> "LigandProcess":
        """
        Set backend name for Converter (if available), e.g. "meeko".

        :param backend: backend name or None.
        :type backend: Optional[str]
        :returns: self
        :rtype: LigandProcess
        """
        self._converter_backend = backend
        logger.debug("Converter backend -> %r", backend)
        return self

    def set_tmp_from_sdf_backend(self, backend: Optional[str]) -> "LigandProcess":
        """
        Set temporary-from-SDF backend used by Converter (if available).

        :param backend: backend name or None.
        :type backend: Optional[str]
        :returns: self
        :rtype: LigandProcess
        """
        self._tmp_from_sdf_backend = backend
        logger.debug("tmp_from_sdf_backend -> %r", backend)
        return self

    def set_keep_intermediate(self, keep: bool) -> "LigandProcess":
        """
        Keep intermediate SDF files instead of removing them.

        :param keep: boolean flag to keep intermediates.
        :type keep: bool
        :returns: self
        :rtype: LigandProcess
        """
        self._keep_intermediate = bool(keep)
        return self

    # ----------------------------- input ingestion --------------------------- #
    def from_smiles_list(
        self, smiles: Sequence[str], names: Optional[Sequence[str]] = None
    ) -> "LigandProcess":
        """
        Load records from a list of SMILES with optional names.

        :param smiles: sequence of SMILES strings.
        :type smiles: Sequence[str]
        :param names: optional sequence of names matching the length of smiles.
        :type names: Optional[Sequence[str]]
        :returns: self
        :rtype: LigandProcess
        :raises ValueError: if ``names`` is provided but length mismatches ``smiles``.
        :example:

        >>> lp = LigandProcess()
        >>> lp.from_smiles_list(['CCO'], names=['ethanol'])
        <LigandProcess ...>
        """
        if names is not None and len(names) != len(smiles):
            raise ValueError("names (if provided) must match smiles length")
        entries = []
        for i, s in enumerate(smiles):
            entry = {self.smiles_key: s}
            if names is not None:
                entry[self.name_key] = names[i]
            entries.append(entry)
        self._load_entries(entries)
        return self

    def from_list_of_dicts(self, rows: Sequence[Dict]) -> "LigandProcess":
        """
        Load records from a sequence of dict-like rows.

        :param rows: sequence of dictionaries containing at least the SMILES key.
        :type rows: Sequence[Dict]
        :returns: self
        :rtype: LigandProcess
        :example:

        >>> lp = LigandProcess()
        >>> lp.from_list_of_dicts([{'smiles': 'CCO', 'name': 'ethanol'}])
        <LigandProcess ...>
        """
        self._load_entries(list(rows))
        return self

    def from_dataframe(self, df: "pd.DataFrame") -> "LigandProcess":
        """
        Load records from a pandas DataFrame.

        :param df: pandas DataFrame containing the SMILES column (self.smiles_key).
        :type df: pandas.DataFrame
        :returns: self
        :rtype: LigandProcess
        :raises RuntimeError: if pandas is not available.
        :raises KeyError: if required SMILES column is missing.
        :example:

        >>> import pandas as pd
        >>> df = pd.DataFrame({'smiles': ['CCO'], 'name': ['ethanol']})
        >>> lp = LigandProcess()
        >>> lp.from_dataframe(df)
        <LigandProcess ...>
        """
        if pd is None:
            raise RuntimeError("pandas is required for from_dataframe")
        if self.smiles_key not in df.columns:
            raise KeyError(f"DataFrame missing required column '{self.smiles_key}'")
        rows = df.to_dict(orient="records")
        self._load_entries(rows)
        return self

    def _load_entries(self, entries: List[Dict]) -> None:
        """
        Internal loader that normalizes entries into the internal record structure.

        :param entries: list of mapping-like objects with at least the SMILES key.
        :type entries: List[Dict]
        :raises KeyError: if an entry lacks the SMILES key.
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

    # ----------------------------- RDKit fallback ---------------------------- #
    def _embed_with_rdkit_inmemory(self, smiles: str) -> str:
        """
        Embed a single SMILES into 3D coordinates using RDKit (in-memory).

        This fallback is used when Conformer is not available. The method will:
          - parse SMILES
          - optionally add Hs
          - embed using a best-available ETKDG variant
          - run UFF or MMFF optimization if requested
          - return a MolBlock string

        :param smiles: SMILES string to embed.
        :type smiles: str
        :returns: MolBlock string with 3D coordinates.
        :rtype: str
        :raises RuntimeError: if RDKit is not available or embedding fails.
        :example:

        >>> lp = LigandProcess()
        >>> mb = lp._embed_with_rdkit_inmemory('CCO')
        >>> isinstance(mb, str)
        True
        """
        if Chem is None or AllChem is None:
            raise RuntimeError("RDKit not available for in-memory embedding")
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            raise RuntimeError(f"Failed to parse SMILES: {smiles!r}")
        working = Chem.Mol(mol)
        if self._add_hs:
            working = Chem.AddHs(working)
        params = None
        try:
            if hasattr(AllChem, "ETKDGv3"):
                params = AllChem.ETKDGv3()
            elif hasattr(AllChem, "ETKDGv2"):
                params = AllChem.ETKDGv2()
            elif hasattr(AllChem, "ETKDG"):
                params = AllChem.ETKDG()
            else:
                params = AllChem.EmbedParameters()
        except Exception:
            params = None
        try:
            if params is not None:
                AllChem.EmbedMolecule(working, params)
            else:
                AllChem.EmbedMolecule(working)
        except Exception:
            try:
                AllChem.EmbedMolecule(working)
            except Exception as e:
                raise RuntimeError(f"RDKit embedding failed: {e}")
        if self._optimize:
            try:
                AllChem.UFFOptimizeMolecule(working)
            except Exception:
                try:
                    AllChem.MMFFOptimizeMolecule(working)
                except Exception:
                    logger.debug(
                        "Optimization failed with both UFF and MMFF; continuing with coordinates if present."
                    )
        if not self._add_hs:
            working = Chem.RemoveHs(working)
        try:
            mb = Chem.MolToMolBlock(working)
        except Exception as e:
            raise RuntimeError(f"Failed to convert Mol to MolBlock: {e}")
        return mb

    # ----------------------------- filename helper -------------------------- #
    def _make_unique_base(self, base: str, ext: str) -> str:
        """
        Make a filesystem-unique base filename in the output directory.

        :param base: desired base name (without extension).
        :type base: str
        :param ext: extension (without dot), e.g. "sdf".
        :type ext: str
        :returns: a base name unique inside the output directory and among already used records.
        :rtype: str
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

    # ----------------------------- core processing -------------------------- #
    def process_all(
        self, start: int = 0, stop: Optional[int] = None
    ) -> "LigandProcess":
        """
        Process all loaded records between ``start`` (inclusive) and ``stop`` (exclusive).

        This iterates over internal records and calls ``_process_one`` for each.

        :param start: start index (default 0).
        :type start: int
        :param stop: exclusive stop index; if None process all remaining records.
        :type stop: Optional[int]
        :returns: self
        :rtype: LigandProcess
        :example:

        >>> lp = LigandProcess(output_dir=None)
        >>> lp.from_smiles_list(['CCO'])
        >>> lp.process_all()
        <LigandProcess ...>
        """
        if not self._records:
            logger.warning("No records to process.")
            return self
        stop_idx = stop if stop is not None else len(self._records)
        for rec in self._records[start:stop_idx]:
            self._process_one(rec)
        return self

    def _process_one(self, rec: Dict) -> None:
        """
        Internal: process a single record.

        The method produces a MolBlock (via Conformer or RDKit fallback), writes
        an intermediate SDF and (optionally) converts it to the configured final format
        using Converter. It updates the record in-place with status/out_path/error.

        :param rec: record dictionary (must follow internal record schema).
        :type rec: Dict
        :returns: None
        :rtype: None
        :notes: Exceptions are caught and converted to record['status']='failed'.
        """
        idx = rec["index"]
        smi = rec["smiles"]
        name = rec.get("name", "") or ""
        index_str = str(idx).zfill(self.index_pad)

        target_ext = self._EXT_MAP[self._output_format]

        # choose base name
        raw_base = _sanitize_filename(name) if name else index_str
        base = (
            self._make_unique_base(raw_base, target_ext)
            if self.output_dir
            else raw_base
        )

        final_out: Optional[Path] = (
            (self.output_dir / f"{base}.{target_ext}")
            if self.output_dir is not None
            else None
        )

        try:
            # produce MolBlock (Conformer preferred)
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

            # if not writing files, finish here
            if final_out is None:
                rec["out_path"] = None
                rec["status"] = "ok"
                rec["error"] = None
                logger.info("Record %d (%s) processed in-memory.", idx, name or smi)
                return

            # always write an SDF intermediate first
            if Chem is None:
                raise RuntimeError("RDKit required to write SDFs.")

            if target_ext == "sdf":
                sdf_path = final_out
            else:
                tmp = tempfile.NamedTemporaryFile(
                    prefix=f"{base}_", suffix=".sdf", dir=self.output_dir, delete=False
                )
                sdf_path = Path(tmp.name)
                tmp.close()

            # write SDF
            m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=(not self._add_hs))
            if m is None:
                raise RuntimeError(
                    "Failed to parse MolBlock into RDKit Mol for writing."
                )
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(m)
            writer.close()

            # If target is SDF, done
            if target_ext == "sdf":
                rec["out_path"] = sdf_path
                rec["status"] = "ok"
                rec["error"] = None
                logger.info("Record %d (%s) -> %s", idx, name or smi, str(sdf_path))
                return

            # If Converter missing: fallback to using SDF as final output
            if not _HAS_CONVERTER:
                logger.warning(
                    "Converter not available; using SDF as final output for record %d (%s).",
                    idx,
                    name or smi,
                )
                rec["out_path"] = sdf_path
                rec["status"] = "ok"
                rec["error"] = None
                logger.info(
                    "Record %d (%s) -> %s (SDF fallback)",
                    idx,
                    name or smi,
                    str(sdf_path),
                )
                return

            # Run Converter for this single file
            conv = (
                Converter()
                .set_input(str(sdf_path))
                .set_output(str(final_out))
                .set_mode("ligand")
            )
            if self._converter_backend:
                conv = conv.set_backend(self._converter_backend)
            if self._tmp_from_sdf_backend:
                conv = conv.set_tmp_from_sdf_backend(self._tmp_from_sdf_backend)
            conv.run()

            # cleanup intermediate unless requested otherwise
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
                "Failed to process SMILES [%s] (index=%d): %s", smi, idx, exc
            )

    # ----------------------------- batch conversion helper ------------------ #
    def finalize_batch_conversion(
        self, input_glob: str, output_pattern: str, mode: str = "ligand"
    ) -> "LigandProcess":
        """
        Convert many intermediate SDFs to the configured output format in a single
        Converter invocation (bulk conversion).

        :param input_glob: glob pattern for existing SDFs (relative to output_dir if set).
        :type input_glob: str
        :param output_pattern: output pattern (Converter-specific). Example: "out/*.pdbqt"
        :type output_pattern: str
        :param mode: Converter mode, default "ligand".
        :type mode: str
        :returns: self
        :rtype: LigandProcess
        :raises RuntimeError: if Converter is not available.
        :example:

        >>> lp.finalize_batch_conversion("*.sdf", "*.pdbqt", mode="ligand")
        <LigandProcess ...>
        """
        if not _HAS_CONVERTER:
            raise RuntimeError("Converter not available for batch conversion.")
        if self.output_dir is not None:
            input_arg = str(Path(self.output_dir) / input_glob)
            output_arg = str(Path(self.output_dir) / output_pattern)
        else:
            input_arg = input_glob
            output_arg = output_pattern
        conv = Converter().set_input(input_arg).set_output(output_arg).set_mode(mode)
        if self._converter_backend:
            conv = conv.set_backend(self._converter_backend)
        if self._tmp_from_sdf_backend:
            conv = conv.set_tmp_from_sdf_backend(self._tmp_from_sdf_backend)
        conv.run()
        logger.info("Batch conversion completed: %s -> %s", input_arg, output_arg)
        return self

    # ----------------------------- persistence ------------------------------ #
    def save_manifest(
        self, path: Union[str, Path] = "ligands_manifest.csv"
    ) -> "LigandProcess":
        """
        Save a CSV manifest describing processed records.

        If pandas is available it will be used, otherwise a csv.DictWriter fallback is used.

        :param path: destination CSV path.
        :type path: str | pathlib.Path
        :returns: self
        :rtype: LigandProcess
        :example:

        >>> lp.save_manifest("outdir/manifest.csv")
        <LigandProcess ...>
        """
        path = Path(path)
        rows = []
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
            import csv

            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=rows[0].keys() if rows else ["index", "smiles"]
                )
                writer.writeheader()
                writer.writerows(rows)
        logger.info("Saved manifest to %s", path)
        return self

    # ----------------------------- properties & helpers -------------------- #
    @property
    def records(self) -> List[Dict]:
        """
        Return a shallow copy of internal records.

        :returns: list of record dicts
        :rtype: List[Dict]
        """
        return list(self._records)

    @property
    def output_paths(self) -> List[Optional[Path]]:
        """
        Return a list of output Path objects (or None) corresponding to records.

        :returns: list of Paths or None
        :rtype: List[Optional[pathlib.Path]]
        """
        return [r["out_path"] for r in self._records]

    @property
    def failed(self) -> List[Dict]:
        """
        Return records that failed processing.

        :returns: list of failed record dicts
        :rtype: List[Dict]
        """
        return [r for r in self._records if r.get("status") == "failed"]

    @property
    def ok(self) -> List[Dict]:
        """
        Return records processed successfully.

        :returns: list of successful record dicts
        :rtype: List[Dict]
        """
        return [r for r in self._records if r.get("status") == "ok"]

    @property
    def summary(self) -> Dict[str, int]:
        """
        Return a summary of total/ok/failed/pending counts.

        :returns: dict with keys 'total', 'ok', 'failed', 'pending'
        :rtype: Dict[str, int]
        """
        total = len(self._records)
        ok = len(self.ok)
        failed = len(self.failed)
        pending = total - ok - failed
        return {"total": total, "ok": ok, "failed": failed, "pending": pending}

    @property
    def sdf_strings(self) -> List[str]:
        """
        Return MolBlock strings (SDF-like) for successfully processed records.

        :returns: list of MolBlock strings
        :rtype: List[str]
        """
        return [
            r["molblock"]
            for r in self._records
            if r.get("status") == "ok" and r.get("molblock")
        ]

    @property
    def mols(self) -> List:
        """
        Return RDKit Mol objects parsed from stored MolBlock strings.

        :returns: list of RDKit Mol objects
        :rtype: List[rdkit.Chem.Mol]
        :raises RuntimeError: if RDKit is not available.
        """
        if Chem is None:
            raise RuntimeError("RDKit not available to build RDKit Mol objects")
        out = []
        for mb in self.sdf_strings:
            m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
            if m is not None:
                out.append(m)
        return out

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"<LigandProcess: {len(self)} entries,"
            + f" ok={self.summary['ok']}, failed={self.summary['failed']},"
            + f" fmt={self._output_format}>"
        )

    # convenience
    def set_output_dir(self, path: Optional[Union[str, Path]]) -> "LigandProcess":
        """
        Set or clear the output directory used for writing files.

        :param path: new output directory path or None to disable writing.
        :type path: Optional[str | pathlib.Path]
        :returns: self
        :rtype: LigandProcess
        """
        self.output_dir = Path(path) if path is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def clear_records(self) -> "LigandProcess":
        """
        Remove all loaded records.

        :returns: self
        :rtype: LigandProcess
        """
        self._records = []
        return self
