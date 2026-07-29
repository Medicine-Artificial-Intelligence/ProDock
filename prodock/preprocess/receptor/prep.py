"""
ReceptorPrep orchestration using small helpers (minimizers + converters).

Key behaviour
- use_meeko=True by default
- the Meeko executable name is fixed internally as "mk_prepare_receptor.py" (not provided by callers)
- prep(...) is the main orchestration method (previously fix_and_minimize_pdb)
- OpenMM minimization is attempted first. On failure we fallback to OpenBabel minimizer.
- If fallback to OpenBabel occurs we prefer OpenBabel for conversions.
- If Meeko conversion fails, we fallback to OpenBabel conversion.
- Produced receptor PDBQT is validated and sanitized to improve downstream docking robustness.

Note
By default prep appends a "_prep" suffix to output basenames to avoid in-place overwrites.
Set add_prep_suffix=False to disable that behavior.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from pymol import cmd  # type: ignore

from .minimizers import fix_pdb, minimize_with_openmm, minimize_with_obabel
from ...structure.conversion import convert_with_meeko, convert_with_obabel
from .repr_helpers import ReprMixin

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ReceptorPrep(ReprMixin):
    """
    High-level receptor preprocessor.

    :param use_meeko: If True, attempt to use Meeko for receptor PDBQT conversion first.
    :type use_meeko: bool
    :param enable_logging: Enable console logging for the instance.
    :type enable_logging: bool

    Notes
    -----
    The Meeko executable is a fixed internal constant: "mk_prepare_receptor.py".
    """

    _MEKOO_EXE = "mk_prepare_receptor.py"

    def __init__(self, use_meeko: bool = True, enable_logging: bool = False) -> None:
        self._mekoo_cmd: str = self._MEKOO_EXE
        self._use_meeko: bool = bool(use_meeko)

        self._final_artifact: Optional[Path] = None
        self._last_simulation_report: Optional[Dict[str, Any]] = None

        self._used_obabel: bool = False
        self._minimized_stage: Optional[str] = None
        self._conversion_backend: Optional[str] = None
        self._conversion_fallback: bool = False
        self._conversion_error: Optional[str] = None

        self._last_input_pdb: Optional[Path] = None
        self._last_output_dir: Optional[Path] = None
        self._last_out_fmt: Optional[str] = None
        self._last_add_prep_suffix: Optional[bool] = None
        self._last_base_name: Optional[str] = None

        if enable_logging:
            self.enable_console_logging()

    def enable_console_logging(self, level: int = logging.DEBUG) -> None:
        """
        Enable console logging for this ReceptorPrep instance.

        :param level: logging level.
        :type level: int
        """
        logger.setLevel(level)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(sh)

    def toggle_meeko(self, on_off: bool) -> None:
        """
        Enable or disable usage of mekoo for conversions.

        :param on_off: True to enable mekoo, False to disable.
        :type on_off: bool
        """
        self._use_meeko = bool(on_off)

    @property
    def use_meeko(self) -> bool:
        """Whether Meeko is enabled for this instance."""
        return self._use_meeko

    @property
    def mekoo_cmd(self) -> str:
        """Return the fixed internal Meeko command name."""
        return self._mekoo_cmd

    @property
    def final_artifact(self) -> Optional[Path]:
        """Path to the final artifact produced by the last run (or None)."""
        return self._final_artifact

    @property
    def last_simulation_report(self) -> Optional[Dict[str, Any]]:
        """Last simulation report dictionary (or None if none)."""
        return self._last_simulation_report

    @property
    def used_obabel(self) -> bool:
        """True if the last run used OpenBabel as fallback for minimization/conversion."""
        return self._used_obabel

    @property
    def minimized_stage(self) -> Optional[str]:
        """Which minimization stage succeeded ('gas', 'solvent', 'obabel', etc.)."""
        return self._minimized_stage

    @property
    def conversion_backend(self) -> Optional[str]:
        """Backend used for final conversion ('meeko', 'obabel', or None)."""
        return self._conversion_backend

    @property
    def conversion_fallback(self) -> bool:
        """True if conversion required fallback from preferred backend."""
        return self._conversion_fallback

    def to_dict(self) -> Optional[Dict[str, Any]]:
        """
        Return a copy of the last_simulation_report (or None).

        :returns: shallow copy of the report or None
        :rtype: dict or None
        """
        if self._last_simulation_report is None:
            return None
        return dict(self._last_simulation_report)

    def save_report(self, path: str | Path, *, indent: int = 2) -> Path:
        """
        Save last_simulation_report as JSON.

        :param path: Destination file path.
        :type path: str or pathlib.Path
        :param indent: JSON indent level.
        :type indent: int
        :returns: path written
        :rtype: pathlib.Path
        :raises RuntimeError: if there is no report to save.
        """
        p = Path(path)
        if self._last_simulation_report is None:
            raise RuntimeError("No simulation report available to save.")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(self._last_simulation_report, fh, indent=indent)
        return p

    def _ensure_output_dir(self, output_dir: Path) -> None:
        """Ensure output directory exists."""
        output_dir.mkdir(parents=True, exist_ok=True)

    def _postprocess_pymol(
        self, pdb_path: Path, start_at: int = 1, cofactors: Optional[List[str]] = None
    ) -> None:
        """
        PyMOL postprocessing: renumber residues, remove solvent/ions, save in-place.

        Non-fatal: exceptions are logged and suppressed.

        :param pdb_path: path to PDB file to postprocess.
        :type pdb_path: pathlib.Path
        :param start_at: residue numbering start (1-based).
        :type start_at: int
        :param cofactors: list of residue names to keep when removing solvent.
        :type cofactors: list[str] or None
        """
        if cmd is None:
            logger.debug("PyMOL not available: skipping postprocessing")
            return
        try:
            cmd.load(str(pdb_path))
            offset = int(start_at) - 1
            cmd.alter("all", f"resi=str(int(resi)+{offset})")
            if cofactors:
                cmd.select("cofactors", " or ".join([f"resn {c}" for c in cofactors]))
                cmd.select("removed_solvent", "solvent and not cofactors")
            else:
                cmd.select("removed_solvent", "solvent")
            cmd.remove("removed_solvent")
            cmd.select("nacl", "resn NA or resn CL")
            cmd.remove("nacl")
            cmd.save(str(pdb_path), "all")
            cmd.delete("all")
            logger.debug("PyMOL postprocessing finished: %s", pdb_path)
        except Exception:
            logger.exception("PyMOL postprocessing failed (non-fatal)")

    @property
    def expected_output_path(self) -> Optional[Path]:
        """
        Return the path where the final artifact will be written for the most recent
        prep() parameters, or the actual final artifact if a run completed.
        """
        if self._final_artifact:
            return self._final_artifact

        if (
            self._last_input_pdb is None
            or self._last_output_dir is None
            or self._last_out_fmt is None
        ):
            return None

        return self.expected_output_for(
            input_pdb=self._last_input_pdb,
            output_dir=self._last_output_dir,
            out_fmt=self._last_out_fmt,
            add_prep_suffix=bool(self._last_add_prep_suffix),
            basename=self._last_base_name,
        )

    def expected_output_for(
        self,
        input_pdb: Path | str,
        output_dir: Path | str,
        out_fmt: str = "pdb",
        add_prep_suffix: bool = True,
        basename: Optional[str] = None,
    ) -> Path:
        """
        Compute the expected output Path for the provided arguments without changing instance state.
        """
        inp = Path(input_pdb)
        outd = Path(output_dir)
        base = (
            basename
            if basename
            else (f"{inp.stem}_prep" if add_prep_suffix else inp.stem)
        )

        fmt = out_fmt.lstrip(".").lower()
        return outd / (f"{base}.pdbqt" if fmt == "pdbqt" else f"{base}.pdb")

    def _convert_receptor_with_obabel(
        self,
        input_pdb: Path,
        out_pdbqt: Path,
        obabel_convert_args: Optional[List[str]] = None,
    ) -> Path:
        """
        Convert receptor PDB -> PDBQT with Open Babel, forcing rigid receptor output.

        :param input_pdb: minimized receptor PDB
        :type input_pdb: pathlib.Path
        :param out_pdbqt: output receptor PDBQT path
        :type out_pdbqt: pathlib.Path
        :param obabel_convert_args: optional extra Open Babel arguments
        :type obabel_convert_args: list[str] or None
        :returns: output receptor PDBQT path
        :rtype: pathlib.Path
        """
        convert_with_obabel(
            input_pdb,
            out_pdbqt,
            extra_args=obabel_convert_args,
            sanitize_rebuild=False,
            sanitize_aggressive=False,
            sanitize_backup=False,
            validate_receptor=True,
        )
        self._conversion_backend = "obabel"
        return out_pdbqt

    def _convert_receptor_with_meeko_then_fallback(
        self,
        input_pdb: Path,
        out_dir: Path,
        base_name: str,
        obabel_convert_args: Optional[List[str]] = None,
    ) -> tuple[Path, Dict[str, Any]]:
        """
        Try Meeko first, then fallback to Open Babel if Meeko fails or does not produce usable PDBQT.

        :param input_pdb: minimized receptor PDB
        :type input_pdb: pathlib.Path
        :param out_dir: output directory
        :type out_dir: pathlib.Path
        :param base_name: basename for output files
        :type base_name: str
        :param obabel_convert_args: optional extra Open Babel arguments for fallback
        :type obabel_convert_args: list[str] or None
        :returns: tuple of final artifact path and meeko info dictionary
        :rtype: tuple[pathlib.Path, dict[str, Any]]
        :raises RuntimeError: if both Meeko and Open Babel conversion fail
        """
        out_pdbqt = out_dir / f"{base_name}.pdbqt"
        mk_info: Dict[str, Any] = {}

        try:
            mk_info = convert_with_meeko(
                self._mekoo_cmd,
                input_pdb=input_pdb,
                out_basename=out_dir / base_name,
                write_pdbqt=out_pdbqt,
                sanitize_rebuild=True,
                sanitize_aggressive=False,
                sanitize_backup=False,
            )

            produced_pdbqt = None
            if mk_info.get("produced"):
                produced_pdbqt = next(
                    (
                        p
                        for p in mk_info["produced"]
                        if str(p).lower().endswith(".pdbqt")
                    ),
                    None,
                )

            rc = mk_info.get("rc", None)
            if rc == 0 and produced_pdbqt and Path(produced_pdbqt).exists():
                self._conversion_backend = "meeko"
                return Path(produced_pdbqt), mk_info

            msg = (
                f"Meeko conversion did not produce usable PDBQT "
                f"(rc={rc}, produced={mk_info.get('produced')})"
            )
            logger.warning(msg)
            self._conversion_fallback = True
            self._conversion_error = msg

        except Exception as exc:
            logger.exception(
                "Meeko conversion failed; falling back to OpenBabel: %s", exc
            )
            self._conversion_fallback = True
            self._conversion_error = str(exc)

        logger.info(
            "Falling back to OpenBabel receptor conversion: %s -> %s",
            input_pdb,
            out_pdbqt,
        )
        final_artifact = self._convert_receptor_with_obabel(
            input_pdb=input_pdb,
            out_pdbqt=out_pdbqt,
            obabel_convert_args=obabel_convert_args,
        )
        return final_artifact, mk_info

    def prep(
        self,
        input_pdb: str,
        output_dir: str,
        out_fmt: str = "pdb",
        energy_diff: float = 10.0,
        max_minimization_steps: int = 5000,
        start_at: int = 1,
        ion_conc: float = 0.15,
        cofactors: Optional[List[str]] = None,
        minimize_in_water: bool = False,
        backbone_k_kcal_per_A2: float = 5.0,
        enable_logging: bool = False,
        obabel_steps: int = 500,
        obabel_convert_args: Optional[List[str]] = None,
        add_prep_suffix: bool = True,
    ) -> "ReceptorPrep":
        """
        High-level orchestration for preparing a receptor.

        The workflow runs PDBFixer, attempts OpenMM minimization, and falls back
        to OpenBabel minimization if needed. For ``pdbqt`` output it prefers
        Meeko unless minimization already required OpenBabel, and it falls back
        to OpenBabel if Meeko conversion fails. The resulting receptor PDBQT is
        sanitized and validated before PyMOL post-processing.

        :raises RuntimeError:
            If minimization fails, or if ``out_fmt="pdbqt"`` and no valid
            PDBQT file is produced.
        """
        if enable_logging:
            self.enable_console_logging()

        out_dir_p = Path(output_dir)
        self._ensure_output_dir(out_dir_p)

        self._last_input_pdb = Path(input_pdb)
        self._last_output_dir = out_dir_p
        self._last_out_fmt = out_fmt
        self._last_add_prep_suffix = add_prep_suffix

        base_name = (
            f"{self._last_input_pdb.stem}_prep"
            if add_prep_suffix
            else self._last_input_pdb.stem
        )
        self._last_base_name = base_name

        final_pdb = out_dir_p / f"{base_name}.pdb"
        tmp_gas = out_dir_p / f"{base_name}_gas_tmp.pdb"

        logger.info(
            "ReceptorPrep.prep: %s -> %s (out_fmt=%s) basename=%s",
            input_pdb,
            out_dir_p,
            out_fmt,
            base_name,
        )

        self._used_obabel = False
        self._minimized_stage = None
        self._conversion_backend = None
        self._conversion_fallback = False
        self._conversion_error = None
        self._final_artifact = None
        self._last_simulation_report = None

        modeller = fix_pdb(input_pdb)

        try:
            final_path, minimized_stage = minimize_with_openmm(
                modeller,
                out_pdb=final_pdb,
                tmp_gas=tmp_gas,
                backbone_k_kcal_per_A2=backbone_k_kcal_per_A2,
                energy_diff=energy_diff,
                max_minimization_steps=max_minimization_steps,
                minimize_in_water=minimize_in_water,
                ion_conc=ion_conc,
            )
            self._minimized_stage = minimized_stage
        except Exception as exc:
            logger.exception(
                "OpenMM minimization failed; attempting OpenBabel fallback: %s", exc
            )
            try:
                final_path = minimize_with_obabel(
                    Path(input_pdb), final_pdb, steps=obabel_steps
                )
                self._minimized_stage = "obabel"
                self._used_obabel = True
            except Exception as exc2:
                logger.exception("OpenBabel fallback also failed: %s", exc2)
                raise RuntimeError(
                    "Both OpenMM and OpenBabel minimization failed"
                ) from exc2

        mk_info: Dict[str, Any] = {}
        final_artifact = Path(final_path)
        requested_fmt = out_fmt.lower().lstrip(".")

        if requested_fmt == "pdbqt":
            out_pdbqt = out_dir_p / f"{base_name}.pdbqt"

            try:
                if self._used_obabel:
                    logger.info(
                        "Using OpenBabel for receptor conversion because minimization fallback was used"
                    )
                    final_artifact = self._convert_receptor_with_obabel(
                        input_pdb=Path(final_path),
                        out_pdbqt=out_pdbqt,
                        obabel_convert_args=obabel_convert_args,
                    )
                elif self._use_meeko:
                    final_artifact, mk_info = (
                        self._convert_receptor_with_meeko_then_fallback(
                            input_pdb=Path(final_path),
                            out_dir=out_dir_p,
                            base_name=base_name,
                            obabel_convert_args=obabel_convert_args,
                        )
                    )
                else:
                    final_artifact = self._convert_receptor_with_obabel(
                        input_pdb=Path(final_path),
                        out_pdbqt=out_pdbqt,
                        obabel_convert_args=obabel_convert_args,
                    )
            except Exception as exc:
                self._conversion_error = str(exc)
                logger.exception("Receptor conversion to PDBQT failed: %s", exc)
                raise RuntimeError(
                    f"Failed to produce valid receptor PDBQT for docking: {exc}"
                ) from exc

            if final_artifact.suffix.lower() != ".pdbqt" or not final_artifact.exists():
                raise RuntimeError(
                    f"Requested out_fmt='pdbqt' but valid PDBQT was not produced: {final_artifact}"
                )

        if final_artifact.suffix.lower() == ".pdb":
            try:
                self._postprocess_pymol(
                    final_artifact, start_at=start_at, cofactors=cofactors
                )
            except Exception:
                pass

        self._final_artifact = final_artifact
        self._last_simulation_report = {
            "final_artifact": str(self._final_artifact),
            "out_fmt": out_fmt,
            "mekoo_info": mk_info,
            "minimized_stage": self._minimized_stage,
            "used_obabel": self._used_obabel,
            "conversion_backend": self._conversion_backend,
            "conversion_fallback": self._conversion_fallback,
            "conversion_error": self._conversion_error,
            "basename": base_name,
            "add_prep_suffix": add_prep_suffix,
        }

        logger.info(
            "ReceptorPrep.prep finished. Artifact: %s "
            "(used_obabel=%s, conversion_backend=%s, conversion_fallback=%s)",
            self._final_artifact,
            self._used_obabel,
            self._conversion_backend,
            self._conversion_fallback,
        )
        return self
