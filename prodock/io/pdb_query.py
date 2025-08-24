"""
prodock.io.pdb_query
----------------------------

Robust PDBQuery for ProDock.

Features
- fetch PDB (robust to case / alternate names produced by PyMOL, auto-decompress .gz)
- filter chains, remove solvents (preserve cofactors)
- extract ligand into `reference_ligand/{ligand_code}.sdf` (if ligand_code provided)
- create `cocrystal/{pdb_id}.sdf` (via Open Babel if available, otherwise fallback copy)
- save filtered protein (no ligand inside)
- fluent API (methods return self), Sphinx-style docstrings, properties, batch helper

Notes
-----
- PyMOL's `cmd` must be importable at runtime. Open Babel (`obabel`) is optional but used if present.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Any, Dict, Union

try:
    # PyMOL's cmd is required at runtime; import here to raise an informative error if missing
    from pymol import cmd  # type: ignore
except Exception:
    cmd = None  # runtime validation will raise if missing

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PDBQuery:
    """
    Fetch, filter and export PDB-derived files using PyMOL and Open Babel.

    Produces canonical outputs when `ligand_code` is provided:

    - reference ligand: ``reference_ligand/{ligand_code}.sdf``
    - cocrystal ligand: ``cocrystal/{pdb_id}.sdf``

    The implementation guarantees that a file exists for inspection by attempting:
    obabel conversion -> temporary PDB conversion -> fallback copy.

    :param pdb_id: PDB identifier (e.g. "5N2F"). Case-insensitive.
    :param output_dir: Base output directory where subfolders will be created.
    :param chains: Sequence of chain identifiers to keep (e.g. ["A","B"]). If empty/None, keep all chains.
    :param ligand_code: Three-letter ligand residue name to extract (e.g. "LIG" or hetero code like "8HW").
    :param ligand_name: Friendly ligand name (kept for compatibility; filenames use ligand_code).
    :param cofactors: Optional list of residue names to preserve as cofactors (e.g. ["HEM","NAD"]).
    :param protein_name: Friendly protein name used when naming intermediate files (defaults to pdb_id).
    """

    DEFAULT_SOLVENTS = [
        "HOH",
        "DOD",
        "ETH",
        "IPA",
        "MEO",
        "ACT",
        "DMS",
        "DME",
        "BEN",
        "TOL",
        "DCM",
        "CCL",
        "MPG",
        "PEG",
        "PG4",
        "ACE",
        "PO4",
        "DPO",
        "SO4",
        "SUL",
        "TRS",
        "TLA",
        "HEP",
        "MES",
        "PIP",
        "CO3",
        "FMT",
        "NA",
        "K",
        "CA",
        "MG",
        "CL",
        "ZN",
        "MN",
    ]

    def __init__(
        self,
        pdb_id: str,
        output_dir: str,
        chains: Optional[Sequence[str]] = None,
        ligand_code: str = "",
        ligand_name: Optional[str] = None,
        cofactors: Optional[Sequence[str]] = None,
        protein_name: Optional[str] = None,
    ) -> None:
        """
        Initialize a PDBQuery helper.

        :param pdb_id: PDB identifier (case-insensitive).
        :param output_dir: Base directory where subfolders will be created.
        :param chains: Sequence of chain identifiers to keep (if None, all chains are kept).
        :param ligand_code: Three-letter ligand residue name to extract.
        :param ligand_name: Friendly ligand name (optional).
        :param cofactors: Optional list of residue names to preserve.
        :param protein_name: Friendly protein name (defaults to pdb_id).
        :returns: None
        """
        # core metadata
        self.pdb_id: str = str(pdb_id)
        self.output_dir: Path = Path(output_dir).expanduser().resolve()
        self.chains: List[str] = list(chains) if chains else []
        self.ligand_code: str = ligand_code
        # ligand_name is kept for compatibility but canonical filenames use ligand_code
        self.ligand_name: str = ligand_name or ligand_code or self.pdb_id
        self.cofactors: List[str] = list(cofactors) if cofactors else []
        self.protein_name: str = protein_name or self.pdb_id

        # directories (created in validate)
        self._fetch_dir: Path = self.output_dir / "fetched_protein"
        self._filtered_protein_dir: Path = self.output_dir / "filtered_protein"
        self._reference_ligand_dir: Path = self.output_dir / "reference_ligand"
        self._cocrystal_dir: Path = self.output_dir / "cocrystal"

        # derived file paths (populated in validate)
        self._pdb_path: Optional[Path] = None
        # reference ligand path (only if ligand_code provided)
        self._reference_ligand_path: Optional[Path] = None
        # cocrystal ligand path
        self._cocrystal_ligand_path: Optional[Path] = None
        self._filtered_protein_path: Optional[Path] = None

    # ----------------------------
    # Meta helpers
    # ----------------------------
    def __repr__(self) -> str:
        return (
            f"<PDBQuery pdb_id={self.pdb_id!r} output_dir={str(self.output_dir)!r} "
            f"chains={self.chains!r} ligand_code={self.ligand_code!r}>"
        )

    def help(self) -> None:
        """
        Print the class docstring / usage help.

        :returns: None
        """
        print(self.__class__.__doc__)

    @staticmethod
    def _join_selection(prefix: str, tokens: Sequence[str]) -> str:
        """
        Build a PyMOL selection string like "chain A or chain B" or "resn HOH or resn DOD".

        :param prefix: selection prefix, e.g. "chain" or "resn".
        :param tokens: sequence of tokens to join.
        :returns: selection string
        """
        return " or ".join(f"{prefix} {t}" for t in tokens)

    # ----------------------------
    # Internal utilities
    # ----------------------------
    def _decompress_gz(self, gz_path: Path) -> Path:
        """
        Decompress a .gz file next to the original file and return the decompressed Path.

        :param gz_path: Path to the .gz file.
        :returns: Path to decompressed file.
        :raises FileNotFoundError: if gz_path doesn't exist.
        """
        if not gz_path.exists():
            raise FileNotFoundError(gz_path)
        if not gz_path.name.lower().endswith(".gz"):
            return gz_path

        out_path = gz_path.with_suffix("")  # remove .gz
        if out_path.exists():
            logger.debug("Decompressed file already exists: %s", out_path)
            return out_path

        logger.info("Decompressing %s -> %s", gz_path, out_path)
        with gzip.open(gz_path, "rb") as fin, open(out_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return out_path

    def _convert_with_obabel(
        self, src: Path, dst: Path, extra_args: Optional[Sequence[str]] = None
    ) -> bool:
        """
        Convert a source file to destination using Open Babel.

        :param src: source Path (e.g. ligand PDB or SDF).
        :param dst: destination Path.
        :param extra_args: extra command-line args to obabel (e.g. ['-h']).
        :returns: True if dst exists after this call, False otherwise.
        """
        obabel_bin = shutil.which("obabel")
        if obabel_bin is None:
            logger.debug(
                "Open Babel not found in PATH; skipping conversion %s -> %s", src, dst
            )
            return False

        dst.parent.mkdir(parents=True, exist_ok=True)
        cwd = Path.cwd()
        try:
            os.chdir(src.parent)
            src_fmt = src.suffix.lstrip(".") or "pdb"
            dst_fmt = dst.suffix.lstrip(".") or "sdf"
            cmdline = [
                obabel_bin,
                "-i",
                src_fmt,
                src.name,
                "-o",
                dst_fmt,
                "-O",
                dst.name,
            ]
            if extra_args:
                cmdline += list(extra_args)
            logger.debug("Running obabel: %s", " ".join(cmdline))
            completed = subprocess.run(cmdline, check=False)
            if completed.returncode != 0:
                logger.warning(
                    "obabel exited with code %s for %s -> %s",
                    completed.returncode,
                    src,
                    dst,
                )
        except Exception as exc:
            logger.warning("obabel conversion error for %s -> %s: %s", src, dst, exc)
        finally:
            try:
                os.chdir(cwd)
            except Exception:
                pass
        return dst.exists()

    def _cleanup_reference_sdfs(self, keep: Path) -> None:
        """
        Remove any SDF files in reference_ligand that match the ligand_code prefix
        except the `keep` path. This ensures only the canonical file remains.

        :param keep: Path to the canonical file to keep.
        :returns: None
        """
        if not self._reference_ligand_dir.exists() or not self.ligand_code:
            return
        pattern = f"{self.ligand_code}*.sdf"
        for p in self._reference_ligand_dir.glob(pattern):
            try:
                if p.resolve() != keep.resolve():
                    p.unlink()
                    logger.debug("Removed stray reference SDF: %s", p)
            except Exception:
                logger.debug("Could not remove stray SDF: %s", p)

    # ----------------------------
    # Validation & setup
    # ----------------------------
    def validate(self) -> "PDBQuery":
        """
        Validate runtime preconditions (PyMOL present). Ensure output directories exist
        and set derived file paths.

        :returns: self
        :raises RuntimeError: if PyMOL cmd is not importable.
        """
        if cmd is None:
            raise RuntimeError(
                "PyMOL 'cmd' is not importable. Ensure PyMOL is installed and available to Python."
            )

        # create directories
        for d in (
            self._fetch_dir,
            self._filtered_protein_dir,
            self._reference_ligand_dir,
            self._cocrystal_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # derived file paths
        self._pdb_path = self._fetch_dir / f"{self.pdb_id}.pdb"
        self._filtered_protein_path = self._filtered_protein_dir / f"{self.pdb_id}.pdb"
        # reference ligand: named by ligand_code (explicitly avoid pdb_id here)
        self._reference_ligand_path = (
            self._reference_ligand_dir / f"{self.ligand_code}.sdf"
            if self.ligand_code
            else None
        )
        # cocrystal ligand: named by pdb id
        self._cocrystal_ligand_path = self._cocrystal_dir / f"{self.pdb_id}.sdf"

        return self

    # ----------------------------
    # Workflow steps (fluent)
    # ----------------------------
    def fetch(self) -> "PDBQuery":
        """
        Fetch the PDB using PyMOL and load it into the session.

        Behavior:
          - calls cmd.fetch(...)
          - searches fetch_dir for files containing pdb id case-insensitively
          - prefers extensions in order: .pdb, .ent, .cif, .mmcif, .pdb.gz, .ent.gz
          - decompresses .gz into the same directory (if necessary)
          - sets self._pdb_path to the discovered file and loads it via cmd.load(...)

        :returns: self
        """
        if cmd is None:
            raise RuntimeError("PyMOL cmd is not available. Cannot fetch PDB.")

        logger.info("Fetching PDB %s to %s", self.pdb_id, str(self._fetch_dir))
        self._fetch_dir.mkdir(parents=True, exist_ok=True)
        # synchronous fetch
        cmd.fetch(self.pdb_id, path=str(self._fetch_dir), type="pdb", async_=0)

        pdb_lower = self.pdb_id.lower()
        allowed_ext_order = [".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".ent.gz"]
        candidates: List[Path] = []

        # collect files that contain the pdb id (case-insensitive)
        for p in self._fetch_dir.iterdir():
            try:
                if pdb_lower in p.name.lower():
                    candidates.append(p)
            except Exception:
                continue

        # fallback heuristics
        if not candidates:
            for p in self._fetch_dir.iterdir():
                nl = p.name.lower()
                if (
                    nl.startswith(f"pdb{pdb_lower}")
                    or nl.endswith(f"{pdb_lower}.pdb")
                    or nl.endswith(f"{pdb_lower}.ent")
                ):
                    candidates.append(p)

        # scoring to prefer certain extensions
        def score_path(p: Path) -> int:
            nl = p.name.lower()
            for idx, ext in enumerate(allowed_ext_order):
                if nl.endswith(ext):
                    return idx
            if nl.startswith(f"pdb{pdb_lower}"):
                return len(allowed_ext_order)
            return len(allowed_ext_order) + 10

        if candidates:
            candidates_sorted = sorted(candidates, key=score_path)
            chosen = candidates_sorted[0]
            logger.debug(
                "Found fetched file candidates: %s", [str(x) for x in candidates_sorted]
            )
            # handle gzipped file
            if chosen.name.lower().endswith(".gz"):
                try:
                    chosen = self._decompress_gz(chosen)
                except Exception as exc:
                    logger.warning("Failed to decompress %s: %s", chosen, exc)
                    # will still attempt to load chosen (PyMOL may read gz directly)
            self._pdb_path = chosen
        else:
            dir_listing = ", ".join(sorted([p.name for p in self._fetch_dir.iterdir()]))
            raise FileNotFoundError(
                f"Fetched PDB file not found at expected path for id '{self.pdb_id}'. "
                f"Search directory: {self._fetch_dir}. Contents: [{dir_listing}]"
            )

        # finally load file into PyMOL session
        logger.debug("Loading PDB file %s into PyMOL session", self._pdb_path)
        if not self._pdb_path.exists():
            raise FileNotFoundError(
                f"Discovered PDB path does not exist: {self._pdb_path}"
            )
        cmd.load(str(self._pdb_path))
        return self

    def filter_chains(self) -> "PDBQuery":
        """
        Keep only requested chains (if provided). Removes everything else from the PyMOL session.

        :returns: self
        """
        if not self.chains:
            logger.debug("No chains provided; keeping all chains.")
            return self

        selection = self._join_selection("chain", self.chains)
        cmd.select("kept_chains", selection)
        logger.info("Keeping chains selection: %s", selection)
        cmd.select("removed_complex", "all and not kept_chains")
        cmd.remove("removed_complex")
        return self

    def extract_ligand(self) -> "PDBQuery":
        """
        Save ligand selection into reference and cocrystal destinations.

        Flow:
         - create a single temporary PDB in reference_ligand: "{ligand_code}_tmp.pdb"
         - convert tmp_pdb -> reference_ligand/{ligand_code}.sdf via obabel (preferred)
         - fallback to copy tmp_pdb -> reference_ligand/{ligand_code}.sdf if obabel missing/fails
         - convert reference -> cocrystal/{pdb_id}.sdf via obabel or copy fallback
         - remove tmp_pdb and delete any stray SDFs in reference_ligand that do not match canonical filename
         - remove ligand atoms from session

        :returns: self
        :raises RuntimeError: if ligand selection is empty or reference SDF cannot be produced
        """
        if not self.ligand_code:
            logger.debug("No ligand_code provided; skipping ligand extraction.")
            return self

        assert self._reference_ligand_path is not None
        assert self._cocrystal_ligand_path is not None

        # ensure target dirs exist
        self._reference_ligand_dir.mkdir(parents=True, exist_ok=True)
        self._cocrystal_dir.mkdir(parents=True, exist_ok=True)

        canonical_ref = self._reference_ligand_path
        canonical_cocrystal = self._cocrystal_ligand_path

        saved_ref = False
        chain_candidates = self.chains if self.chains else [None]

        for chain in chain_candidates:
            sel = f"resn {self.ligand_code}" + (f" and chain {chain}" if chain else "")
            cmd.select("ligand", sel)
            try:
                count = cmd.count_atoms("ligand")
            except Exception:
                try:
                    count = len(cmd.get_model("ligand").atom)
                except Exception:
                    count = 0
            logger.debug("Tried ligand selection '%s' -> %d atoms", sel, count)

            if count == 0:
                continue

            # Always write a single temporary PDB for conversion (deterministic name)
            tmp_pdb = self._reference_ligand_dir / f"{self.ligand_code}_tmp.pdb"
            try:
                if tmp_pdb.exists():
                    tmp_pdb.unlink()
                cmd.save(str(tmp_pdb), "ligand")
            except Exception as exc:
                logger.warning("PyMOL cmd.save to temporary PDB failed: %s", exc)

            # If PyMOL did not produce tmp_pdb, attempt direct save to canonical (last resort)
            if not tmp_pdb.exists():
                try:
                    cmd.save(str(canonical_ref), "ligand")
                except Exception as exc:
                    logger.warning("PyMOL direct save fallback failed: %s", exc)
                if canonical_ref.exists():
                    saved_ref = True
                    break
                else:
                    continue

            # Convert tmp_pdb -> canonical reference (preferred)
            converted = self._convert_with_obabel(
                tmp_pdb, canonical_ref, extra_args=("-h",)
            )
            if converted and canonical_ref.exists():
                logger.info(
                    "Converted tmp_pdb -> reference SDF via obabel: %s", canonical_ref
                )
                saved_ref = True
            else:
                # fallback: copy tmp_pdb -> canonical_ref (ensures artifact exists)
                try:
                    shutil.copy2(tmp_pdb, canonical_ref)
                    logger.info(
                        "Copied tmp_pdb -> reference SDF (fallback): %s", canonical_ref
                    )
                    saved_ref = True
                except Exception as exc:
                    logger.warning("Failed to copy tmp_pdb -> reference_sdf: %s", exc)
                    saved_ref = False

            # If we produced reference SDF, create cocrystal from it (obabel preferred, else copy)
            if saved_ref:
                converted2 = self._convert_with_obabel(
                    canonical_ref, canonical_cocrystal, extra_args=("-h",)
                )
                if converted2 and canonical_cocrystal.exists():
                    logger.info("Created cocrystal via obabel: %s", canonical_cocrystal)
                else:
                    try:
                        shutil.copy2(canonical_ref, canonical_cocrystal)
                        logger.info(
                            "Copied reference -> cocrystal (fallback): %s",
                            canonical_cocrystal,
                        )
                    except Exception as exc:
                        logger.warning("Failed to create cocrystal by copying: %s", exc)

            # Cleanup tmp_pdb
            try:
                if tmp_pdb.exists():
                    tmp_pdb.unlink()
            except Exception:
                logger.debug("Could not delete tmp_pdb: %s", tmp_pdb)

            # Remove stray SDFs in reference_ligand that do not match canonical_ref
            try:
                if self._reference_ligand_dir.exists():
                    for p in self._reference_ligand_dir.glob("*.sdf"):
                        try:
                            if p.resolve() != canonical_ref.resolve():
                                p.unlink()
                                logger.debug("Removed stray reference SDF: %s", p)
                        except Exception:
                            logger.debug("Could not remove stray SDF: %s", p)
            except Exception:
                logger.debug("Cleanup of stray SDFs failed.")

            if saved_ref:
                break

        if not saved_ref:
            msg = (
                f"Failed to save reference ligand for PDB {self.pdb_id} ligand_code={self.ligand_code}. "
                f"Expected path: {canonical_ref}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # remove ligand atoms from the protein model so filtered_protein does not include ligand
        try:
            cmd.remove(f"resn {self.ligand_code}")
        except Exception:
            logger.debug("cmd.remove might have failed or ligand already removed.")

        return self

    def clean_solvents_and_cofactors(self) -> "PDBQuery":
        """
        Remove common solvents while optionally preserving cofactors.

        :returns: self
        """
        solvent_sel = self._join_selection("resn", self.DEFAULT_SOLVENTS)
        cmd.select("solvents", solvent_sel)

        if self.cofactors:
            cof_sel = self._join_selection("resn", self.cofactors)
            cmd.select("cofactors", cof_sel)
            cmd.select("removed_solvent", f"solvents and not cofactors")
            logger.info("Preserving cofactors: %s", ", ".join(self.cofactors))
        else:
            cmd.select("removed_solvent", "solvents")
            logger.info("Removing all listed solvents (no cofactors provided).")

        cmd.remove("removed_solvent")
        return self

    def save_filtered_protein(self) -> "PDBQuery":
        """
        Save the remaining protein (no ligand) to filtered_protein path and clear the PyMOL session.

        :returns: self
        """
        self._filtered_protein_dir.mkdir(parents=True, exist_ok=True)
        try:
            cmd.save(str(self._filtered_protein_path), "all")
            logger.info("Saved filtered protein to: %s", self._filtered_protein_path)
        except Exception as exc:
            logger.warning("PyMOL cmd.save for filtered protein failed: %s", exc)
        try:
            cmd.delete("all")
        except Exception:
            pass
        return self

    def run_all(self, obabel_args: Optional[Sequence[str]] = ("-h",)) -> "PDBQuery":
        """
        Execute the full pipeline: validate -> fetch -> filter_chains -> extract_ligand ->
        clean_solvents_and_cofactors -> save_filtered_protein.

        :param obabel_args: args passed to obabel for processed output (kept for API compatibility).
        :returns: self
        """
        return (
            self.validate()
            .fetch()
            .filter_chains()
            .extract_ligand()
            .clean_solvents_and_cofactors()
            .save_filtered_protein()
        )

    # ----------------------------
    # Debug / inspection helpers
    # ----------------------------
    def list_reference_dir(self) -> Dict[str, List[str]]:
        """
        Return a dict with files currently present in reference_ligand and cocrystal directories.

        :returns: { 'reference': [...], 'cocrystal': [...] }
        """
        ref_list: List[str] = []
        coco_list: List[str] = []
        if self._reference_ligand_dir.exists():
            ref_list = [
                str(p.name) for p in sorted(self._reference_ligand_dir.iterdir())
            ]
        if self._cocrystal_dir.exists():
            coco_list = [str(p.name) for p in sorted(self._cocrystal_dir.iterdir())]
        return {"reference": ref_list, "cocrystal": coco_list}

    # ----------------------------
    # Properties
    # ----------------------------
    @property
    def pdb_path(self) -> Optional[str]:
        """
        Path to fetched PDB file (string) or None.

        :returns: path or None
        """
        return str(self._pdb_path) if self._pdb_path else None

    @property
    def filtered_protein_path(self) -> Optional[str]:
        """
        Path to the saved filtered protein PDB file (string) or None.

        :returns: path or None
        """
        return str(self._filtered_protein_path) if self._filtered_protein_path else None

    @property
    def reference_ligand_path(self) -> Optional[str]:
        """
        Path to the reference ligand SDF: reference_ligand/{ligand_code}.sdf (string) or None.

        Note: if `ligand_code` was not provided at construction, this will be None.

        :returns: path or None
        """
        return str(self._reference_ligand_path) if self._reference_ligand_path else None

    @property
    def cocrystal_ligand_path(self) -> Optional[str]:
        """
        Path to the cocrystal ligand SDF: cocrystal/{pdb_id}.sdf (string) or None.

        :returns: path or None
        """
        return str(self._cocrystal_ligand_path) if self._cocrystal_ligand_path else None

    # compatibility aliases
    @property
    def ligand_path(self) -> Optional[str]:
        """
        Alias -> reference_ligand_path

        :returns: path or None
        """
        return self.reference_ligand_path

    @property
    def ligand2_path(self) -> Optional[str]:
        """
        Alias -> cocrystal_ligand_path

        :returns: path or None
        """
        return self.cocrystal_ligand_path

    # ----------------------------
    # Batch processing helper
    # ----------------------------
    @classmethod
    def process_batch(
        cls,
        items: Union[List[Dict[str, Any]], Any],
        output_dir: str,
        keys: Optional[Dict[str, str]] = None,
        default_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process many entries (list of dicts or pandas.DataFrame-like).
        Returns a list of result dicts for each row/item with `success` and path info.

        :param items: list of dicts or pandas.DataFrame containing records to process.
        :param output_dir: base output directory for all items.
        :param keys: mapping of expected keys in each item -> {
            'pdb_id_key', 'protein_name_key', 'ligand_key', 'chains_key', 'cofactors_key'
        } (optional).
        :param default_kwargs: extra kwargs passed to each PDBQuery constructor.
        :returns: list of dicts: {
            'pdb_id', 'protein_name', 'reference', 'cocrystal', 'filtered_protein', 'success', 'error'
        }
        """
        # lazy-import pandas to avoid forcing it for simple usage
        try:
            import pandas as _pd  # type: ignore
        except Exception:
            _pd = None  # type: ignore

        keys = keys or {}
        default_kwargs = default_kwargs or {}

        pdb_id_key = keys.get("pdb_id_key", "pdb_id")
        protein_name_key = keys.get("protein_name_key", "protein_name")
        ligand_key = keys.get("ligand_key", "ligand_code")
        chains_key = keys.get("chains_key", "chains")
        cofactors_key = keys.get("cofactors_key", "cofactors")

        # normalize input to an iterable of dicts
        rows: List[Dict[str, Any]] = []
        if isinstance(items, list):
            rows = list(items)
        else:
            # try pandas-like conversion
            if _pd is not None:
                try:
                    df = _pd.DataFrame(items)
                    rows = df.to_dict(orient="records")
                except Exception:
                    rows = list(items)
            else:
                rows = list(items)

        results: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            pdb_id = row.get(pdb_id_key) or row.get("pdb") or row.get("id")
            protein_name = row.get(protein_name_key) or pdb_id
            ligand_code = row.get(ligand_key) or row.get("ligand") or ""
            chains = row.get(chains_key) or row.get("chain") or []
            cofactors = row.get(cofactors_key) or []

            try:
                # per-pdb subfolder to avoid collisions
                per_pdb_out = Path(output_dir) / str(pdb_id)
                proc = cls(
                    pdb_id=str(pdb_id),
                    output_dir=str(per_pdb_out),
                    chains=chains,
                    ligand_code=str(ligand_code),
                    ligand_name=str(ligand_code) if ligand_code else None,
                    cofactors=cofactors,
                    protein_name=str(protein_name),
                    **default_kwargs,
                )
                proc.run_all()
                results.append(
                    {
                        "pdb_id": pdb_id,
                        "protein_name": protein_name,
                        "reference": proc.reference_ligand_path,
                        "cocrystal": proc.cocrystal_ligand_path,
                        "filtered_protein": proc.filtered_protein_path,
                        "success": True,
                        "error": None,
                    }
                )
            except Exception as exc:
                logger.exception(
                    "Failed to process row %s (pdb=%s): %s", idx, pdb_id, exc
                )
                results.append(
                    {
                        "pdb_id": pdb_id,
                        "protein_name": protein_name,
                        "reference": None,
                        "cocrystal": None,
                        "filtered_protein": None,
                        "success": False,
                        "error": str(exc),
                    }
                )

        return results
