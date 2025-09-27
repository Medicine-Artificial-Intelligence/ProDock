"""
ReceptorPrep orchestration using small helpers (minimizers + converters).

Key behaviour
- use_meeko=True by default
- the mekoo executable name is fixed internally as "mk_prepare_receptor.py" (not provided by callers)
- prep(...) is the main orchestration method (previously fix_and_minimize_pdb)
- OpenMM minimization is attempted first. On failure we fallback to OpenBabel minimizer.
- If fallback to OpenBabel occurs we **do not** call mekoo; instead we use OpenBabel for conversions.

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
from .convert import convert_with_mekoo, convert_with_obabel
from .repr_helpers import ReprMixin

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ReceptorPrep(ReprMixin):
    """
    High-level receptor preprocessor.

    :param use_meeko: If True, attempt to use mekoo for conversions when appropriate.
    :type use_meeko: bool
    :param enable_logging: Enable console logging for the instance.
    :type enable_logging: bool

    Notes
    -----
    The mekoo executable is a fixed internal constant: "mk_prepare_receptor.py".
    """

    # hard-coded mekoo executable name (not configurable via __init__)
    _MEKOO_EXE = "mk_prepare_receptor.py"

    def __init__(self, use_meeko: bool = True, enable_logging: bool = False) -> None:
        self._mekoo_cmd: str = self._MEKOO_EXE
        self._use_meeko: bool = bool(use_meeko)

        # results / report
        self._final_artifact: Optional[Path] = None
        self._last_simulation_report: Optional[Dict[str, Any]] = None

        # transient state set by prep()
        self._used_obabel: bool = False
        self._minimized_stage: Optional[str] = None

        # stored parameters for prediction/inspection
        self._last_input_pdb: Optional[Path] = None
        self._last_output_dir: Optional[Path] = None
        self._last_out_fmt: Optional[str] = None
        self._last_add_prep_suffix: Optional[bool] = None
        self._last_base_name: Optional[str] = None

        if enable_logging:
            self.enable_console_logging()

    # -------------------------
    # Basic config helpers
    # -------------------------
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

    # -------------------------
    # Properties & utilities
    # -------------------------
    @property
    def use_meeko(self) -> bool:
        """Whether mekoo is enabled for this instance."""
        return self._use_meeko

    @property
    def mekoo_cmd(self) -> str:
        """Internal mekoo command (fixed): returns 'mk_prepare_receptor.py'."""
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
        """Ensure output directory exists (internal helper)."""
        output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Internal PyMOL postprocessing
    # -------------------------
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

    # -------------------------
    # Expected output helpers (property + pure helper)
    # -------------------------
    @property
    def expected_output_path(self) -> Optional[Path]:
        """
        Return the path where the final artifact *will be written* for the most
        recent prep() call parameters — or the actual final artifact if the
        last run produced one.

        Behaviour:
          - If a run has completed and `self._final_artifact` is set, that Path is returned.
          - Otherwise, if `prep()` was called (or `expected_output_for()` used) and
            we have stored parameters, return the predicted output Path using the
            same basename/out_fmt/add_prep_suffix logic as `prep()`.
          - If no parameters are known, return None.

        :returns: pathlib.Path or None
        """
        # prefer actual final produced artifact if available
        if self._final_artifact:
            return self._final_artifact

        # otherwise attempt to predict from stored parameters
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
        Compute the expected output Path for the provided arguments without changing
        instance state.

        :param input_pdb: path or str of the input file
        :type input_pdb: pathlib.Path or str
        :param output_dir: path or str of the output directory
        :type output_dir: pathlib.Path or str
        :param out_fmt: 'pdb' or 'pdbqt' ('.' prefix allowed)
        :type out_fmt: str
        :param add_prep_suffix: whether '_prep' would be appended to the basename
        :type add_prep_suffix: bool
        :param basename: optional explicit basename (if provided, `input_pdb` stem is ignored)
        :type basename: str or None
        :returns: pathlib.Path of the predicted artifact
        """
        inp = Path(input_pdb)
        outd = Path(output_dir)
        if basename:
            base = basename
        else:
            base_stem = inp.stem
            base = f"{base_stem}_prep" if add_prep_suffix else base_stem

        fmt = out_fmt.lstrip(".").lower()
        if fmt == "pdbqt":
            fname = f"{base}.pdbqt"
        else:
            # default to pdb for any other value
            fname = f"{base}.pdb"
        return outd / fname

    # -------------------------
    # Orchestration
    # -------------------------
    def prep(
        self,
        input_pdb: str,
        output_dir: str,
        out_fmt: str = "pdb",  # 'pdb' | 'pdbqt'
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

        Behaviour summary:
        - run PDBFixer (fix_pdb)
        - attempt OpenMM minimization (minimize_with_openmm)
            - if OpenMM fails, fall back to OpenBabel minimization (minimize_with_obabel)
        - if out_fmt == 'pdbqt' and OpenMM succeeded -> optionally call mekoo (if use_meeko=True)
        - if out_fmt == 'pdbqt' and fallback to OpenBabel occurred -> use OpenBabel for conversion (convert_with_obabel)
        - run PyMOL postprocessing for PDB artifacts

        :param input_pdb: Path to input PDB file to prepare. (basename for outputs is derived from this filename)
        :type input_pdb: str
        :param output_dir: Directory to write outputs.
        :type output_dir: str
        :param out_fmt: Desired output format: 'pdb' or 'pdbqt'.
        :type out_fmt: str
        :param energy_diff: Minimizer tolerance for OpenMM.
        :type energy_diff: float
        :param max_minimization_steps: Maximizer iterations for OpenMM minimizer.
        :type max_minimization_steps: int
        :param start_at: Residue renumber start for PyMOL postprocessing.
        :type start_at: int
        :param ion_conc: Ionic strength for explicit solvent (molar).
        :type ion_conc: float
        :param cofactors: Residue names that should not be removed as solvent.
        :type cofactors: list[str] or None
        :param minimize_in_water: If True, perform explicit TIP3P minimization after gas-phase.
        :type minimize_in_water: bool
        :param backbone_k_kcal_per_A2: Backbone restraint for minimizer in kcal/A^2.
        :type backbone_k_kcal_per_A2: float
        :param enable_logging: Enable console logging for the duration of this call.
        :type enable_logging: bool
        :param obabel_steps: Steps used for OpenBabel minimization fallback.
        :type obabel_steps: int
        :param obabel_convert_args: Extra args passed to obabel for conversion (if used).
        :type obabel_convert_args: list[str] or None
        :param add_prep_suffix: If True (default), append "_prep" to output basenames to avoid overwriting input.
        :type add_prep_suffix: bool
        :returns: self
        :rtype: ReceptorPrep
        :raises RuntimeError: on unrecoverable failures (both OpenMM and OpenBabel fail).
        """
        if enable_logging:
            self.enable_console_logging()

        out_dir_p = Path(output_dir)
        self._ensure_output_dir(out_dir_p)

        # record last-call parameters so expected_output_path can be queried later
        self._last_input_pdb = Path(input_pdb)
        self._last_output_dir = out_dir_p
        self._last_out_fmt = out_fmt
        self._last_add_prep_suffix = add_prep_suffix

        # compute basename from input stem and optionally append suffix
        base_stem = self._last_input_pdb.stem
        base_name = f"{base_stem}_prep" if add_prep_suffix else base_stem
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

        # reset transient state
        self._used_obabel = False
        self._minimized_stage = None
        self._final_artifact = None
        self._last_simulation_report = None

        # 1) Fix
        modeller = fix_pdb(input_pdb)

        # 2) Minimize (OpenMM preferred). If OpenMM fails, fallback to OpenBabel.
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
            # fallback to obabel minimization (operate on the original input PDB)
            try:
                minimized = minimize_with_obabel(
                    Path(input_pdb), final_pdb, steps=obabel_steps
                )
                final_path = minimized
                minimized_stage = "obabel"
                self._minimized_stage = minimized_stage
                self._used_obabel = True
            except Exception as exc2:
                logger.exception("OpenBabel fallback also failed: %s", exc2)
                raise RuntimeError(
                    "Both OpenMM and OpenBabel minimization failed"
                ) from exc2

        mk_info: Dict[str, Any] = {}
        final_artifact = Path(final_path)

        # 3) Conversion / produce requested out_fmt
        if out_fmt.lower() == "pdbqt":
            # If we fell back to OpenBabel, prefer OpenBabel for conversion and do NOT call mekoo.
            if self._used_obabel:
                out_pdbqt = out_dir_p / f"{base_name}.pdbqt"
                try:
                    convert_with_obabel(
                        Path(final_path), out_pdbqt, extra_args=obabel_convert_args
                    )
                    final_artifact = out_pdbqt
                except Exception as exc:
                    logger.exception("OpenBabel conversion to pdbqt failed: %s", exc)
                    # allow final artifact to remain the PDB if conversion fails
            else:
                # OpenMM succeeded. If use_meeko is True, try mekoo conversion.
                if self._use_meeko:
                    mk_info = convert_with_mekoo(
                        self._mekoo_cmd,
                        input_pdb=Path(final_path),
                        out_basename=out_dir_p / base_name,
                        write_pdbqt=out_dir_p / f"{base_name}.pdbqt",
                    )
                    if mk_info.get("produced"):
                        produced_pdbqt = next(
                            (
                                p
                                for p in mk_info["produced"]
                                if p.lower().endswith(".pdbqt")
                            ),
                            None,
                        )
                        if produced_pdbqt:
                            final_artifact = Path(produced_pdbqt)

        # 4) PyMOL postprocessing for PDB artifacts
        if final_artifact.suffix.lower() == ".pdb":
            try:
                self._postprocess_pymol(
                    final_artifact, start_at=start_at, cofactors=cofactors
                )
            except Exception:
                # _postprocess_pymol already logs; ignore
                pass

        # finalize report
        self._final_artifact = final_artifact
        self._last_simulation_report = {
            "final_artifact": str(self._final_artifact),
            "out_fmt": out_fmt,
            "mekoo_info": mk_info,
            "minimized_stage": self._minimized_stage,
            "used_obabel": self._used_obabel,
            "basename": base_name,
            "add_prep_suffix": add_prep_suffix,
        }
        logger.info(
            "ReceptorPrep.prep finished. Artifact: %s (used_obabel=%s)",
            self._final_artifact,
            self._used_obabel,
        )
        return self

    # -------------------------
    # Representations (inherited from ReprMixin)
    # -------------------------
