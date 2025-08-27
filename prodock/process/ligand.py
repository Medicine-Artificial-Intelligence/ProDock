# prodock/preprocess/ligand_process.py
"""
LigandProcess module
--------------------

RDKit-based utilities to produce per-ligand SDF files from SMILES with
configurable embedding and force-field optimization.

This implementation prefers :class:`prodock.chem.conformer.Conformer` for
3D embedding/optimization, but will perform an in-memory RDKit embedding
when Conformer is unavailable. If ``output_dir`` is set to ``None``, no SDF
files are written to disk — MolBlock strings are kept in memory and exposed
via :attr:`sdf_strings` and :attr:`mols`.

See :class:`LigandProcess` for examples.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

try:
    import pandas as pd  # optional dependency
except Exception:
    pd = None  # type: ignore

# prodock logging utilities (fallback to stdlib logging adapter if unavailable)
try:
    from prodock.io.logging import get_logger, StructuredAdapter
except Exception:  # pragma: no cover - fallback path

    def get_logger(name: str):
        return logging.getLogger(name)

    class StructuredAdapter(logging.LoggerAdapter):  # type: ignore
        def __init__(self, logger, extra):
            super().__init__(logger, extra)


logger = StructuredAdapter(
    get_logger("prodock.ligand.process"), {"component": "ligand.process"}
)
logger._base_logger = getattr(logger, "_base_logger", getattr(logger, "logger", None))

# Conformer (optional)
try:
    from prodock.chem.conformer import Conformer  # type: ignore

    _HAS_CONFORMER = True
except Exception:  # pragma: no cover - optional import
    Conformer = None  # type: ignore
    _HAS_CONFORMER = False
    logger.debug(
        "Conformer not available; falling back to RDKit in-memory embedding when needed."
    )

# RDKit (required for in-memory fallback). We attempt to import here.
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except (
    Exception
):  # pragma: no cover - if RDKit isn't present callers should skip tests / fail earlier
    Chem = None  # type: ignore
    AllChem = None  # type: ignore


def _sanitize_filename(name: str, max_len: int = 120) -> str:
    """
    Sanitize an arbitrary string to a safe filename.

    :param name: Original name string.
    :param max_len: Maximum number of characters to keep.
    :returns: Sanitized filename (non-empty).
    :rtype: str
    """
    cleaned = re.sub(r"[^\w\-.]+", "_", name.strip())
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("_")
    return cleaned or "molecule"


class LigandProcess:
    """
    Convert SMILES into per-ligand SDF files (or in-memory MolBlocks) with optional 3D
    embedding and optimization.

    If ``output_dir`` is provided, SDF files are written there. If ``output_dir`` is
    ``None`` no files are written and the MolBlock strings are kept in memory.

    Examples
    --------
    1) Read SMILES from a newline-separated file (one SMILES per line) and process::

        # assume 'smiles.smi' contains:
        # CCO
        # c1ccccc1
        from prodock.ligand.process import LigandProcess
        with open("smiles.smi", "r", encoding="utf-8") as fh:
            smiles = [line.strip().split()[0] for line in fh if line.strip()]
        lp = LigandProcess(output_dir="out/sdf_from_file")
        lp.from_smiles_list(smiles)
        lp.set_options(embed3d=True, add_hs=True, optimize=True)
        lp.set_embed_method("ETKDGv3").set_opt_method("MMFF94")
        lp.process_all()
        lp.save_manifest("out/sdf_from_file/manifest.csv")

    2) Provide a list of SMILES programmatically::

        lp = LigandProcess(output_dir="out/sdf_from_list")
        lp.from_smiles_list(["CCO", "CCC", "c1ccccc1"])
        lp.set_options(embed3d=True, add_hs=True, optimize=False)
        lp.process_all()

    3) Provide a list of dictionaries (useful when you have names/metadata)::

        rows = [
            {"smiles": "CCO", "name": "ethanol"},
            {"smiles": "c1ccccc1", "name": "benzene"},
        ]
        lp = LigandProcess(output_dir="out/sdf_from_dicts", name_key="name")
        lp.from_list_of_dicts(rows)
        lp.set_embed_method("ETKDGv2").set_opt_method("UFF")
        lp.process_all()
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = "ligands_sdf",
        smiles_key: str = "smiles",
        name_key: str = "name",
        index_pad: int = 4,
    ) -> None:
        """
        :param output_dir: directory to write SDFs (set to None to disable writing).
        :param smiles_key: dict/DataFrame key for SMILES.
        :param name_key: dict/DataFrame key for name/label.
        :param index_pad: zero-pad width for index-based filenames.
        """
        # output_dir may be None to indicate "do not write files"
        self.output_dir: Optional[Path] = (
            Path(output_dir) if output_dir is not None else None
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.smiles_key = smiles_key
        self.name_key = name_key
        self.index_pad = int(index_pad)

        # processing options (defaults: embed and optimize enabled)
        self._embed3d: bool = True
        self._add_hs: bool = True
        self._optimize: bool = True

        # Conformer options (used only when Conformer is available)
        self._embed_algorithm: Optional[str] = "ETKDGv3"
        self._opt_method: str = "MMFF94"
        self._conformer_seed: int = 42
        self._conformer_n_jobs: int = 1
        self._opt_max_iters: int = 200

        # internal records list: each record is a dict with keys index/smiles/name/out_path/status/error/molblock
        self._records: List[Dict] = []

    # ------------------------------------------------------------------ #
    # configuration helpers
    # ------------------------------------------------------------------ #
    def set_options(
        self,
        embed3d: Optional[bool] = None,
        add_hs: Optional[bool] = None,
        optimize: Optional[bool] = None,
    ) -> "LigandProcess":
        """
        Configure processing options.

        :param embed3d: request 3D embedding if True.
        :param add_hs: add explicit hydrogens prior to embedding/optimization.
        :param optimize: run force-field optimization after embedding.
        :returns: self
        """
        if embed3d is not None:
            self._embed3d = bool(embed3d)
        if add_hs is not None:
            self._add_hs = bool(add_hs)
        if optimize is not None:
            self._optimize = bool(optimize)
        logger.debug(
            "Options set: embed3d=%s add_hs=%s optimize=%s",
            self._embed3d,
            self._add_hs,
            self._optimize,
        )
        return self

    def set_embed_method(self, embed_algorithm: Optional[str]) -> "LigandProcess":
        """
        Select embedding algorithm used by Conformer.

        :param embed_algorithm: "ETKDGv3" | "ETKDGv2" | "ETKDG" | None
        :returns: self
        """
        self._embed_algorithm = embed_algorithm
        logger.debug("Embed algorithm set to %r", self._embed_algorithm)
        return self

    def set_opt_method(self, method: str) -> "LigandProcess":
        """
        Select optimizer method used by Conformer.

        :param method: 'UFF' | 'MMFF' | 'MMFF94' | 'MMFF94S'
        :returns: self
        """
        self._opt_method = str(method)
        logger.debug("Optimization method set to %r", self._opt_method)
        return self

    def set_conformer_seed(self, seed: int) -> "LigandProcess":
        """Set RNG seed used by Conformer."""
        self._conformer_seed = int(seed)
        return self

    def set_conformer_jobs(self, n_jobs: int) -> "LigandProcess":
        """Set number of jobs forwarded to Conformer when parallelising at that layer."""
        self._conformer_n_jobs = int(n_jobs)
        return self

    def set_opt_max_iters(self, max_iters: int) -> "LigandProcess":
        """Set maximum iterations for optimizer."""
        self._opt_max_iters = int(max_iters)
        return self

    # ------------------------------------------------------------------ #
    # input ingestion
    # ------------------------------------------------------------------ #
    def from_smiles_list(
        self, smiles: Sequence[str], names: Optional[Sequence[str]] = None
    ) -> "LigandProcess":
        """
        Load molecules from a list of SMILES strings.

        :param smiles: sequence of SMILES strings.
        :param names: optional parallel sequence of names (same length).
        :returns: self
        """
        if names is not None and len(names) != len(smiles):
            raise ValueError("`names` (if provided) must be same length as `smiles`")
        entries = []
        for i, smi in enumerate(smiles):
            entry = {self.smiles_key: smi}
            if names is not None:
                entry[self.name_key] = names[i]
            entries.append(entry)
        self._load_entries(entries)
        return self

    def from_list_of_dicts(self, rows: Sequence[Dict]) -> "LigandProcess":
        """
        Load molecules from a list of dictionaries.

        Each dict must contain the SMILES under `smiles_key`. Name is optional.
        """
        self._load_entries(list(rows))
        return self

    def from_dataframe(self, df: "pd.DataFrame") -> "LigandProcess":
        """
        Load from a pandas DataFrame.

        :raises RuntimeError: if pandas not available
        :raises KeyError: if smiles_key missing
        """
        if pd is None:
            raise RuntimeError(
                "pandas is not available; install pandas to use from_dataframe()"
            )
        if self.smiles_key not in df.columns:
            raise KeyError(f"DataFrame missing required column '{self.smiles_key}'")
        rows = df.to_dict(orient="records")
        self._load_entries(rows)
        return self

    def _load_entries(self, entries: List[Dict]) -> None:
        """Normalize input rows into internal records structure (clears previous records)."""
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

    # ------------------------------------------------------------------ #
    # embedding fallback (RDKit in-memory)
    # ------------------------------------------------------------------ #
    def _embed_with_rdkit_inmemory(self, smiles: str) -> str:
        """
        Create a MolBlock string from SMILES using RDKit embedding/optimization in-memory.

        :param smiles: input SMILES
        :returns: MolBlock string
        :raises RuntimeError: if RDKit not available or embedding fails
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
                    # ignore optimization failures — we still return coordinates if present
                    logger.debug("Both UFF and MMFF optimization failed (continuing).")

        if not self._add_hs:
            working = Chem.RemoveHs(working)

        try:
            mb = Chem.MolToMolBlock(working)
        except Exception as e:
            raise RuntimeError(f"Failed to convert Mol to MolBlock: {e}")
        return mb

    # ------------------------------------------------------------------ #
    # filename uniqueness helper
    # ------------------------------------------------------------------ #
    def _make_unique_base(self, base: str) -> str:
        """
        Ensure `base` filename (without extension) is unique within the output_dir
        and within already-produced records in this run. If not unique, append
        a numeric suffix: base -> base_1 -> base_2 -> ...
        """
        if self.output_dir is None:
            # no file writing happening; keep base unchanged
            return base

        out_dir = Path(self.output_dir)
        candidate = out_dir / f"{base}.sdf"
        used_names = {
            Path(r["out_path"]).name for r in self._records if r.get("out_path")
        }
        if not candidate.exists() and f"{base}.sdf" not in used_names:
            return base

        suffix = 1
        while True:
            new_base = f"{base}_{suffix}"
            if (
                not (out_dir / f"{new_base}.sdf").exists()
                and f"{new_base}.sdf" not in used_names
            ):
                return new_base
            suffix += 1

    # ------------------------------------------------------------------ #
    # processing
    # ------------------------------------------------------------------ #
    def process_all(
        self, start: int = 0, stop: Optional[int] = None
    ) -> "LigandProcess":
        """Process all records (or slice) and populate out_path/molblock/status/error."""
        if not self._records:
            logger.warning("No records loaded to process.")
            return self
        stop_idx = stop if stop is not None else len(self._records)
        for rec in self._records[start:stop_idx]:
            self._process_one(rec)
        return self

    def _process_one(self, rec: Dict) -> None:
        """
        Process a single record and write an SDF file if output_dir is configured.
        Always stores the MolBlock string in ``rec['molblock']`` for successful records.
        """
        idx = rec["index"]
        smi = rec["smiles"]
        name = rec.get("name", "") or ""
        index_str = str(idx).zfill(self.index_pad)

        # determine base filename
        if name:
            raw_base = _sanitize_filename(name)
            base = self._make_unique_base(raw_base)
        else:
            base = index_str

        out_path = (
            (self.output_dir / f"{base}.sdf") if self.output_dir is not None else None
        )

        try:
            # Preferred path: use Conformer to produce MolBlock when embedding/optimization requested and available
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
                if not cm.molblocks:
                    raise RuntimeError(
                        "Conformer failed to produce an embedded molecule"
                    )
                mb = cm.molblocks[0]
            else:
                # fallback: do in-memory RDKit embedding/optimization to get MolBlock
                mb = self._embed_with_rdkit_inmemory(smi)

            # store molblock into record
            rec["molblock"] = mb

            # write to disk if requested
            if out_path is not None:
                if Chem is None:
                    raise RuntimeError("RDKit not available to write MolBlock -> SDF")
                m = Chem.MolFromMolBlock(
                    mb, sanitize=False, removeHs=(not self._add_hs)
                )
                if m is None:
                    raise RuntimeError(
                        "Failed to parse MolBlock to RDKit Mol for writing"
                    )
                writer = Chem.SDWriter(str(out_path))
                writer.write(m)
                writer.close()
                rec["out_path"] = out_path
            else:
                rec["out_path"] = None

            rec["status"] = "ok"
            rec["error"] = None
            logger.info(
                "Processed record %d (%s) -> %s",
                idx,
                name or smi,
                str(out_path) if out_path is not None else "<in-memory>",
            )
        except Exception as exc:
            rec["out_path"] = None
            rec["molblock"] = None
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "Failed to process SMILES [%s] (index=%d): %s", smi, idx, exc
            )

    # ------------------------------------------------------------------ #
    # persistence & manifest helpers
    # ------------------------------------------------------------------ #
    def save_manifest(
        self, path: Union[str, Path] = "ligands_manifest.csv"
    ) -> "LigandProcess":
        """
        Save a CSV manifest summarising processing results.

        Contains fields: index, smiles, name, out_path, status, error.
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

    # ------------------------------------------------------------------ #
    # properties & helpers
    # ------------------------------------------------------------------ #
    @property
    def records(self) -> List[Dict]:
        """Return a shallow copy of internal records."""
        return list(self._records)

    @property
    def output_paths(self) -> List[Optional[Path]]:
        """Return output paths for written SDFs (None for in-memory-only records)."""
        return [r["out_path"] for r in self._records]

    @property
    def failed(self) -> List[Dict]:
        """Return records that failed processing."""
        return [r for r in self._records if r.get("status") == "failed"]

    @property
    def ok(self) -> List[Dict]:
        """Return records that completed successfully."""
        return [r for r in self._records if r.get("status") == "ok"]

    @property
    def summary(self) -> Dict[str, int]:
        """Return summary counts of processed records."""
        total = len(self._records)
        ok = len(self.ok)
        failed = len(self.failed)
        pending = total - ok - failed
        return {"total": total, "ok": ok, "failed": failed, "pending": pending}

    @property
    def sdf_strings(self) -> List[str]:
        """
        Return the MolBlock (SDF) text for successful records (in same order as records).

        :returns: list of MolBlock strings (may be empty).
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
        Return RDKit Mol objects constructed from stored MolBlocks for successful records.

        Requires RDKit to be importable; raises RuntimeError otherwise.

        :returns: list of RDKit Chem.Mol objects.
        """
        if Chem is None:
            raise RuntimeError("RDKit not available to construct Mol objects")
        out = []
        for mb in self.sdf_strings:
            m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
            if m is not None:
                out.append(m)
        return out

    def __len__(self) -> int:
        """Number of loaded records."""
        return len(self._records)

    def __repr__(self) -> str:
        s = f"<LigandProcess: {len(self)} entries, ok={self.summary['ok']}, failed={self.summary['failed']}>"
        return s

    # ------------------------------------------------------------------ #
    # convenience helpers
    # ------------------------------------------------------------------ #
    def set_output_dir(self, path: Optional[Union[str, Path]]) -> "LigandProcess":
        """
        Change the output directory. Pass ``None`` to disable writing SDFs to disk.

        :param path: new output directory or ``None``.
        """
        self.output_dir = Path(path) if path is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def clear_records(self) -> "LigandProcess":
        """Clear loaded records (does not delete files on disk)."""
        self._records = []
        return self
