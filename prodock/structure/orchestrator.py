# prodock/structure/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from prodock.io.logging import get_logger

# selection + constants imported at top-level (stable, small)
from .selection import chain_selection, resn_selection
from .constants import DEFAULT_SOLVENTS

logger = get_logger(__name__)

# Lazy/defensive import of pymol.cmd (import at top-level but inside try block)
try:
    from pymol import cmd  # type: ignore
except Exception:
    cmd = None


class PDBOrchestrator:
    """
    Orchestrator doing the actual steps (fetch/filter/extract/clean/save).

    Keep this class small: it performs step-wise actions and stores derived paths.
    """

    def __init__(
        self,
        pdb_id: str,
        base_out: Path,
        chains: Optional[List[str]] = None,
        ligand_code: str = "",
        cofactors: Optional[List[str]] = None,
        auto_create_dirs: bool = True,
    ):
        self.pdb_id = str(pdb_id)
        self.base_out = Path(base_out)
        self.chains = list(chains) if chains else []
        self.ligand_code = ligand_code
        self.cofactors = list(cofactors) if cofactors else []
        self.auto_create_dirs = bool(auto_create_dirs)

        self.fetch_dir = self.base_out / "fetched_protein"
        self.filtered_dir = self.base_out / "filtered_protein"
        self.ref_dir = self.base_out / "reference_ligand"
        self.cocrystal_dir = self.base_out / "cocrystal"

        self.pdb_path: Optional[Path] = None
        self.ref_path: Optional[Path] = None
        self.cocrystal_path: Optional[Path] = None
        self.filtered_path: Optional[Path] = None

    def _ensure_dir(self, p: Path) -> None:
        if self.auto_create_dirs:
            p.mkdir(parents=True, exist_ok=True)

    def validate(self) -> "PDBOrchestrator":
        if cmd is None:
            raise RuntimeError(
                "PyMOL 'cmd' is not importable. Install PyMOL for runtime ops."
            )
        for d in (self.fetch_dir, self.filtered_dir, self.ref_dir, self.cocrystal_dir):
            self._ensure_dir(d)

        self.pdb_path = self.fetch_dir / f"{self.pdb_id}.pdb"
        self.filtered_path = self.filtered_dir / f"{self.pdb_id}.pdb"
        self.ref_path = (
            (self.ref_dir / f"{self.ligand_code}.sdf") if self.ligand_code else None
        )
        self.cocrystal_path = self.cocrystal_dir / f"{self.pdb_id}.sdf"
        return self

    def fetch(self) -> "PDBOrchestrator":
        """Fetch PDB into fetch_dir and load it into PyMOL session."""
        if cmd is None:
            raise RuntimeError("PyMOL cmd is not available. Cannot fetch PDB.")
        self._ensure_dir(self.fetch_dir)

        # lazy import so tests can monkeypatch prodock.structure.fetch.fetch_pdb_to_dir
        from .fetch import fetch_pdb_to_dir  # local import

        chosen = fetch_pdb_to_dir(self.pdb_id, self.fetch_dir)
        # allow string or Path
        self.pdb_path = Path(chosen)
        logger.debug("Loading %s into PyMOL session", chosen)
        cmd.load(str(chosen))
        return self

    def filter_chains(self) -> "PDBOrchestrator":
        if not self.chains:
            logger.debug("No chains provided; keeping all chains.")
            return self
        sel = chain_selection(self.chains)
        cmd.select("kept_chains", sel)
        logger.info("Keeping chains selection: %s", sel)
        cmd.select("removed_complex", "all and not kept_chains")
        cmd.remove("removed_complex")
        return self

    def extract_ligand(self) -> "PDBOrchestrator":
        if not self.ligand_code:
            logger.debug("No ligand_code provided; skipping ligand extraction.")
            return self
        assert self.ref_path is not None and self.cocrystal_path is not None

        self._ensure_dir(self.ref_dir)
        self._ensure_dir(self.cocrystal_dir)

        # lazy import so tests can monkeypatch convert module functions
        from .convert import convert_with_obabel, copy_fallback  # local import

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
            if count == 0:
                continue

            tmp_pdb = self.ref_dir / f"{self.ligand_code}_tmp.pdb"
            if tmp_pdb.exists():
                try:
                    tmp_pdb.unlink()
                except Exception:
                    pass
            try:
                cmd.save(str(tmp_pdb), "ligand")
            except Exception as exc:
                logger.warning("PyMOL cmd.save to temporary PDB failed: %s", exc)

            if not tmp_pdb.exists():
                try:
                    # try direct save to ref_path
                    cmd.save(str(self.ref_path), "ligand")
                except Exception as exc:
                    logger.warning("PyMOL direct save fallback failed: %s", exc)
                if self.ref_path.exists():
                    saved_ref = True
                    break
                else:
                    continue

            converted = convert_with_obabel(tmp_pdb, self.ref_path, extra_args=("-h",))
            if converted and self.ref_path.exists():
                saved_ref = True
            else:
                if copy_fallback(tmp_pdb, self.ref_path):
                    saved_ref = True
                else:
                    saved_ref = False

            if saved_ref:
                if not (
                    convert_with_obabel(
                        self.ref_path, self.cocrystal_path, extra_args=("-h",)
                    )
                ):
                    copy_fallback(self.ref_path, self.cocrystal_path)

            try:
                if tmp_pdb.exists():
                    tmp_pdb.unlink()
            except Exception:
                pass

            # cleanup stray sdf
            try:
                for p in self.ref_dir.glob("*.sdf"):
                    try:
                        if p.resolve() != self.ref_path.resolve():
                            p.unlink()
                    except Exception:
                        pass
            except Exception:
                pass

            if saved_ref:
                break

        if not saved_ref:
            raise RuntimeError(
                f"Failed to save reference ligand for PDB {self.pdb_id} ligand_code={self.ligand_code}"
            )

        try:
            cmd.remove(f"resn {self.ligand_code}")
        except Exception:
            pass

        return self

    def clean_solvents_and_cofactors(self) -> "PDBOrchestrator":
        solvent_sel = resn_selection(DEFAULT_SOLVENTS)
        cmd.select("solvents", solvent_sel)
        if self.cofactors:
            cof_sel = resn_selection(self.cofactors)
            cmd.select("cofactors", cof_sel)
            cmd.select("removed_solvent", "solvents and not cofactors")
            logger.info("Preserving cofactors: %s", ", ".join(self.cofactors))
        else:
            cmd.select("removed_solvent", "solvents")
            logger.info("Removing all listed solvents (no cofactors provided).")
        cmd.remove("removed_solvent")
        return self

    def save_filtered_protein(self) -> "PDBOrchestrator":
        try:
            cmd.save(str(self.filtered_path), "all")
            logger.info("Saved filtered protein to: %s", self.filtered_path)
        except Exception as exc:
            logger.warning("PyMOL cmd.save for filtered protein failed: %s", exc)
        try:
            cmd.delete("all")
        except Exception:
            pass
        return self

    def run_all(self) -> "PDBOrchestrator":
        return (
            self.validate()
            .fetch()
            .filter_chains()
            .extract_ligand()
            .clean_solvents_and_cofactors()
            .save_filtered_protein()
        )
