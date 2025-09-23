# protein_process.py
"""
ProteinProcess
===============

Fix/minimize PDB files and prepare docking artifacts (PDB / PDBQT / GPF).

This version is adapted to the GridBox implementation at
`prodock.preprocess.gridbox.GridBox` (uses .load_ligand(), .preset(), .from_ligand_pad*, etc.)

Key points
- If out_fmt == 'gpf' an `input_ligand` must be provided; GridBox is used to compute the box.
- GridBox builders: .preset('safe') is attempted first, then .from_ligand_pad_adv or .from_ligand_pad as fallback.
- Clear errors are raised if GridBox cannot compute a box (prevents calling mk_prepare_receptor.py without box).
- Detailed mekoo/ADT stdout+stderr are stored in last_simulation_report.

Dependencies
------------
- pdbfixer
- openmm
- pymol (optional for postprocessing)
- prodock.process.gridbox.GridBox

Example
-------
Minimal example that fixes & prepares a receptor and requests a GPF file (requires a ligand file):

.. code-block:: python

    from protein_process import ReceptorProcess
    pp = ReceptorProcess(enable_logging=True)
    pp.fix_and_minimize_pdb(
        input_pdb="receptor_input.pdb",
        output_dir="out",
        out_fmt="gpf",
        input_ligand="ligand.sdf",
        minimize_in_water=False
    )
    print(pp.last_simulation_report)
"""

from __future__ import annotations

import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from prodock.process.gridbox import GridBox
from pymol import cmd  # type: ignore


# PDBFixer + OpenMM imports
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Modeller, ForceField, Simulation, PME, NoCutoff, HBonds
from openmm import Platform, CustomExternalForce, LangevinIntegrator
from openmm.unit import nanometer, kelvin, picosecond, molar

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ReceptorProcess:
    """
    Fix and minimize proteins and prepare docking-ready artifacts.

    The class provides a fluent primary method :meth:`fix_and_minimize_pdb` which:
      - applies PDBFixer repairs,
      - runs an OpenMM minimization (gas-phase and optionally in explicit water),
      - computes a docking grid box via :class:`prodock.process.gridbox.GridBox` when needed,
      - attempts to generate PDBQT/GPF artifacts via `mk_prepare_receptor.py` (mekoo),
        falling back to AutoDockTools (ADT) if mekoo is not available,
      - stores verbose tool output in :pyattr:`last_simulation_report`.

    :param mekoo_cmd: Path or command name for mk_prepare_receptor.py (mekoo wrapper).
    :type mekoo_cmd: str
    :param adt_cmd: Path or command name for AutoDockTools prepare_receptor4.py (fallback).
    :type adt_cmd: str
    :param enable_logging: If True, enable console logging for this instance.
    :type enable_logging: bool

    Attributes
    ----------
    final_artifact : Optional[pathlib.Path]
        Path to the final produced artifact (PDB / PDBQT / GPF) after the run.
    last_simulation_report : Optional[Dict[str, Any]]
        Dictionary containing details about the run (mekoo_info, adt_info, grid_params, ...)
    """

    def __init__(
        self,
        mekoo_cmd: str = "mk_prepare_receptor.py",
        adt_cmd: str = "prepare_receptor4.py",
        enable_logging: bool = False,
    ) -> None:
        self.mekoo_cmd = mekoo_cmd
        self.adt_cmd = adt_cmd
        self.final_artifact: Optional[Path] = None
        self.last_simulation_report: Optional[Dict[str, Any]] = None
        if enable_logging:
            self.enable_console_logging()

    # -------------------------
    # Logging helpers
    # -------------------------
    def enable_console_logging(self, level: int = logging.DEBUG) -> None:
        """
        Enable console logging for this module/instance.

        :param level: logging level to set for the instance logger.
        :type level: int
        """
        logger.setLevel(level)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(sh)

    # -------------------------
    # Platform / pos helpers
    # -------------------------
    @staticmethod
    def _choose_platform() -> Tuple[Platform, Dict[str, str]]:
        """
        Select the best-available OpenMM platform.

        Tries CUDA, then OpenCL, and falls back to CPU.

        :returns: (Platform object, platform property dict)
        :rtype: tuple
        """
        for name in ("CUDA", "OpenCL", "CPU"):
            try:
                plat = Platform.getPlatformByName(name)
                props = {"CudaPrecision": "mixed"} if name == "CUDA" else {}
                logger.debug("OpenMM platform selected: %s", name)
                return plat, props
            except Exception:
                continue
        plat = Platform.getPlatformByName("CPU")
        return plat, {}

    @staticmethod
    def _pos_to_nm(pos) -> Tuple[float, float, float]:
        """
        Return (x,y,z) in nanometers from an OpenMM position object.

        This helper normalizes multiple position representations.

        :param pos: position object or sequence from OpenMM state
        :returns: tuple of floats (x, y, z) in nanometers
        :rtype: tuple
        """
        try:
            vals = pos.value_in_unit(nanometer)
            return float(vals[0]), float(vals[1]), float(vals[2])
        except Exception:
            try:
                return float(pos[0]), float(pos[1]), float(pos[2])
            except Exception:
                return float(pos.x), float(pos.y), float(pos.z)

    # -------------------------
    # GridBox adaptation helpers
    # -------------------------
    def compute_gridbox_from_ligand(
        self,
        ligand_path: str,
        *,
        scale: float = 1.25,
        min_size: float = 22.5,
        pad: float = 4.0,
        isotropic: bool = True,
    ) -> Dict[str, float]:
        """
        Compute a vina-style grid dictionary from ligand using the project's GridBox.

        Strategy:
          1. Load ligand via GridBox.load_ligand(...)
          2. Try gb.preset('safe') (recommended)
          3. If preset missing/fails, try gb.from_ligand_pad_adv(...) or gb.from_ligand_pad(...)
          4. If still failing, try gb.from_ligand_scale(...)

        :param ligand_path: Path to ligand file (SDF/PDB/etc) used to compute box.
        :type ligand_path: str
        :param scale: multiplier used by from_ligand_scale fallback (default 1.25).
        :type scale: float
        :param min_size: minimal axis size in Å (default 22.5).
        :type min_size: float
        :param pad: padding in Å for pad builder (default 4.0).
        :type pad: float
        :param isotropic: whether to force isotropic/cubic box in pad/scale builders.
        :type isotropic: bool
        :returns: dictionary with keys center_x, center_y, center_z, size_x, size_y, size_z
        :rtype: dict
        :raises RuntimeError: if GridBox cannot compute a box after all fallbacks
        """
        try:
            gb = GridBox().load_ligand(ligand_path)
        except Exception as exc:
            logger.exception("GridBox.load_ligand failed for %s: %s", ligand_path, exc)
            raise RuntimeError(
                f"GridBox.load_ligand failed for {ligand_path}: {exc}"
            ) from exc

        # preferred quick preset
        try:
            if hasattr(gb, "preset"):
                gb.preset("safe", heavy_only=False, snap_step=0.25, round_ndigits=3)
                logger.debug("GridBox.preset('safe') succeeded")
                return gb.vina_dict
        except Exception:
            logger.debug(
                "GridBox.preset('safe') failed; trying pad builder...", exc_info=True
            )

        # try advanced pad builder if implemented
        try:
            if hasattr(gb, "from_ligand_pad_adv"):
                gb.from_ligand_pad_adv(
                    pad=pad,
                    isotropic=isotropic,
                    min_size=min_size,
                    heavy_only=False,
                    snap_step=0.25,
                    round_ndigits=3,
                )
                logger.debug("GridBox.from_ligand_pad_adv succeeded")
                return gb.vina_dict
        except Exception:
            logger.debug(
                "GridBox.from_ligand_pad_adv failed; trying simple pad...",
                exc_info=True,
            )

        # try simple pad builder
        try:
            if hasattr(gb, "from_ligand_pad"):
                gb.from_ligand_pad(pad=pad, isotropic=isotropic, min_size=min_size)
                logger.debug("GridBox.from_ligand_pad succeeded")
                return gb.vina_dict
        except Exception:
            logger.debug(
                "GridBox.from_ligand_pad failed; trying scale-based builder...",
                exc_info=True,
            )

        # try scale-based fallback
        try:
            if hasattr(gb, "from_ligand_scale"):
                gb.from_ligand_scale(scale=scale, isotropic=isotropic)
                # enforce minimum sizes if necessary
                sx, sy, sz = gb.size
                sx, sy, sz = (max(sx, min_size), max(sy, min_size), max(sz, min_size))
                gb.from_center_size(gb.center, (sx, sy, sz))
                logger.debug("GridBox.from_ligand_scale succeeded")
                return gb.vina_dict
        except Exception:
            logger.debug(
                "GridBox.from_ligand_scale failed; all attempts exhausted",
                exc_info=True,
            )

        # if reached here, fail clearly
        raise RuntimeError(
            "GridBox failed to compute a box for ligand: " + str(ligand_path)
        )

    # -------------------------
    # External tool invocation
    # -------------------------
    def _call_mekoo(
        self,
        input_pdb: Path,
        out_basename: Path,
        write_pdbqt: Optional[Path] = None,
        write_gpf: Optional[Path] = None,
        box_center: Optional[Tuple[float, float, float]] = None,
        box_size: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Call mk_prepare_receptor.py with correct flags and capture output.

        :param input_pdb: input PDB path to hand to mekoo.
        :type input_pdb: pathlib.Path
        :param out_basename: base path for outputs (without extension).
        :type out_basename: pathlib.Path
        :param write_pdbqt: optional path to request that mekoo writes a PDBQT.
        :type write_pdbqt: pathlib.Path or None
        :param write_gpf: optional path to request that mekoo writes a GPF.
        :type write_gpf: pathlib.Path or None
        :param box_center: optional (x,y,z) center to pass to mekoo.
        :type box_center: tuple or None
        :param box_size: optional (sx,sy,sz) size to pass to mekoo.
        :type box_size: tuple or None
        :return: info dict with keys 'called','rc','stdout','stderr','produced'
        :rtype: dict
        """
        exe = shutil.which(self.mekoo_cmd) or (
            self.mekoo_cmd if Path(self.mekoo_cmd).exists() else None
        )
        info: Dict[str, Any] = {
            "called": None,
            "rc": None,
            "stdout": None,
            "stderr": None,
            "produced": [],
        }
        if not exe:
            info["stderr"] = f"mekoo ({self.mekoo_cmd}) not found"
            logger.warning(info["stderr"])
            return info

        args: List[str] = [
            str(exe),
            "--read_pdb",
            str(input_pdb),
            "-o",
            str(out_basename),
        ]
        if box_center:
            args += [
                "--box_center",
                str(box_center[0]),
                str(box_center[1]),
                str(box_center[2]),
            ]
        if box_size:
            args += ["--box_size", str(box_size[0]), str(box_size[1]), str(box_size[2])]
        if write_pdbqt:
            args += ["--write_pdbqt", str(write_pdbqt)]
        if write_gpf:
            args += ["--write_gpf", str(write_gpf)]

        info["called"] = " ".join(args)
        logger.debug("Calling mekoo: %s", info["called"])
        try:
            proc = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            info["rc"] = proc.returncode
            info["stdout"] = proc.stdout
            info["stderr"] = proc.stderr
            logger.debug(
                "mekoo rc=%s stdout_len=%d stderr_len=%d",
                proc.returncode,
                len(proc.stdout or ""),
                len(proc.stderr or ""),
            )
        except Exception as exc:
            info["rc"] = -1
            info["stderr"] = str(exc)
            logger.exception("mekoo invocation failed: %s", exc)

        produced: List[str] = []
        if write_pdbqt and Path(write_pdbqt).exists():
            produced.append(str(Path(write_pdbqt).resolve()))
        if write_gpf and Path(write_gpf).exists():
            produced.append(str(Path(write_gpf).resolve()))
        for ext in (".pdbqt", ".gpf", ".json", ".box", ".vina"):
            cand = out_basename.with_suffix(ext)
            if cand.exists():
                produced.append(str(cand.resolve()))
        info["produced"] = produced
        logger.debug("mekoo produced: %s", produced)
        return info

    def _call_adt(self, input_pdb: Path, out_pdbqt: Path) -> Dict[str, Any]:
        """
        Call AutoDockTools prepare_receptor4.py as a fallback to produce PDBQT.

        :param input_pdb: input PDB path
        :type input_pdb: pathlib.Path
        :param out_pdbqt: desired output pdbqt path
        :type out_pdbqt: pathlib.Path
        :return: info dict similar to :meth:`_call_mekoo`
        :rtype: dict
        """
        exe = shutil.which(self.adt_cmd) or (
            self.adt_cmd if Path(self.adt_cmd).exists() else None
        )
        info: Dict[str, Any] = {
            "called": None,
            "rc": None,
            "stdout": None,
            "stderr": None,
            "produced": [],
        }
        if not exe:
            info["stderr"] = f"ADT ({self.adt_cmd}) not found"
            logger.warning(info["stderr"])
            return info
        args = [str(exe), "-r", str(input_pdb), "-o", str(out_pdbqt)]
        info["called"] = " ".join(args)
        logger.debug("Calling ADT: %s", info["called"])
        try:
            proc = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            info["rc"] = proc.returncode
            info["stdout"] = proc.stdout
            info["stderr"] = proc.stderr
            if out_pdbqt.exists():
                info["produced"].append(str(out_pdbqt.resolve()))
            logger.debug(
                "ADT rc=%s stdout_len=%d stderr_len=%d",
                proc.returncode,
                len(proc.stdout or ""),
                len(proc.stderr or ""),
            )
        except Exception as exc:
            info["rc"] = -1
            info["stderr"] = str(exc)
            logger.exception("ADT invocation failed: %s", exc)
        return info

    # -------------------------
    # Primary fluent method
    # -------------------------
    def fix_and_minimize_pdb(
        self,
        input_pdb: str,
        output_dir: str,
        energy_diff: float = 10.0,
        max_minimization_steps: int = 5000,
        start_at: int = 1,
        ion_conc: float = 0.15,
        cofactors: Optional[List[str]] = None,
        pdb_id: Optional[str] = None,
        protein_name: Optional[str] = None,
        minimize_in_water: bool = False,
        backbone_k_kcal_per_A2: float = 5.0,
        out_fmt: str = "pdb",  # 'pdb' | 'pdbqt' | 'gpf'
        input_ligand: Optional[str] = None,  # REQUIRED if out_fmt == 'gpf'
        grid_scale: float = 1.25,
        grid_min_size: float = 22.5,
        pad_angstrom: float = 4.0,
        enable_logging: bool = False,
    ) -> "ReceptorProcess":
        """
        Fix a PDB (PDBFixer), minimize (OpenMM), and optionally prepare docking files.

        The method performs the following high-level steps:
          - run PDBFixer repairs and add missing atoms/hydrogens,
          - run a gas-phase OpenMM minimization with backbone restraints,
          - optionally add solvent and minimize in water if minimize_in_water=True,
          - compute docking grid (if out_fmt == 'gpf' or to pass to mekoo),
          - call mekoo (mk_prepare_receptor.py) to create PDBQT/GPF, fallback to ADT.

        :param input_pdb: path to input PDB file.
        :type input_pdb: str
        :param output_dir: directory to write output artifacts.
        :type output_dir: str
        :param energy_diff: convergence tolerance for minimizer (OpenMM units).
        :type energy_diff: float
        :param max_minimization_steps: maximum iterations for minimization.
        :type max_minimization_steps: int
        :param start_at: residue renumbering start index (used in PyMOL postprocessing).
        :type start_at: int
        :param ion_conc: ionic strength (molar) used when solvating (if requested).
        :type ion_conc: float
        :param cofactors: list of residue names to treat as cofactors (kept in selection).
        :type cofactors: list[str] or None
        :param pdb_id: optional PDB id used for naming output.
        :type pdb_id: str or None
        :param protein_name: optional name used for output basename (overrides pdb_id).
        :type protein_name: str or None
        :param minimize_in_water: if True, perform an explicit-water minimization after gas-phase.
        :type minimize_in_water: bool
        :param backbone_k_kcal_per_A2: backbone restraint force constant (kcal / A^2).
        :type backbone_k_kcal_per_A2: float
        :param out_fmt: desired final artifact format: 'pdb', 'pdbqt', or 'gpf'.
        :type out_fmt: str
        :param input_ligand: required when out_fmt == 'gpf' to compute box via GridBox.
        :type input_ligand: str or None
        :param grid_scale: scale fallback for grid computation if pad-based builders fail.
        :type grid_scale: float
        :param grid_min_size: minimum grid axis size in Å when constructing GPF fallback.
        :type grid_min_size: float
        :param pad_angstrom: padding used for pad builders in Å.
        :type pad_angstrom: float
        :param enable_logging: enable console logging for the duration of this call.
        :type enable_logging: bool

        :returns: self (fluent)
        :rtype: ReceptorProcess

        :raises RuntimeError:
            - if out_fmt == 'gpf' but input_ligand is not provided
            - or if GridBox cannot compute a box when required
            - or if gas/solvent minimization fails (OpenMM exceptions are raised as RuntimeError)

        Example
        -------

        .. code-block:: python

            rp = ReceptorProcess(enable_logging=True)
            rp.fix_and_minimize_pdb(
                "receptor.pdb", "outdir", out_fmt="pdbqt"
            )
            print(rp.last_simulation_report)
        """
        if enable_logging:
            self.enable_console_logging()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = protein_name or pdb_id or Path(input_pdb).stem
        final_pdb = out_dir / f"{base_name}.pdb"
        tmp_gas = out_dir / f"{base_name}_gas_tmp.pdb"

        logger.info(
            "ReceptorProcess: fixing %s -> %s (out_fmt=%s)", input_pdb, out_dir, out_fmt
        )

        # Step 1: fix PDB
        fixer = PDBFixer(filename=input_pdb)
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.4)
        modeller = Modeller(fixer.topology, fixer.positions)

        # Step 2: gas-phase minimization with backbone restraints
        ff = ForceField("amber14-all.xml")
        system = ff.createSystem(
            modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds
        )

        k_kj_per_mol_nm2 = backbone_k_kcal_per_A2 * 4.184 * 100.0
        restr = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        restr.addPerParticleParameter("k")
        restr.addPerParticleParameter("x0")
        restr.addPerParticleParameter("y0")
        restr.addPerParticleParameter("z0")

        for atom in modeller.topology.atoms():
            if atom.name in ("N", "CA", "C"):
                idx = atom.index
                x0, y0, z0 = self._pos_to_nm(modeller.positions[idx])
                restr.addParticle(idx, [k_kj_per_mol_nm2, x0, y0, z0])

        system.addForce(restr)

        platform, plat_props = self._choose_platform()
        integrator = LangevinIntegrator(
            300 * kelvin, 1.0 / picosecond, 0.002 * picosecond
        )
        integrator.setRandomNumberSeed(42)

        simulation = Simulation(
            modeller.topology, system, integrator, platform, plat_props
        )
        simulation.context.setPositions(modeller.positions)

        try:
            logger.info(
                "Minimizing (gas phase)... tol=%s maxIter=%s",
                energy_diff,
                max_minimization_steps,
            )
            simulation.minimizeEnergy(
                tolerance=energy_diff, maxIterations=max_minimization_steps
            )
        except Exception as exc:
            logger.exception("Gas minimization failed: %s", exc)
            raise RuntimeError("Gas-phase minimization failed") from exc

        state = simulation.context.getState(getPositions=True)
        with open(tmp_gas, "w") as fh:
            PDBFile.writeFile(simulation.topology, state.getPositions(), fh)

        minimized_stage = "gas"

        # Step 3: optional solvent minimization
        if minimize_in_water:
            logger.info("Adding solvent and minimizing in explicit water (TIP3P).")
            pdb_in = PDBFile(str(tmp_gas))
            modeller = Modeller(pdb_in.topology, pdb_in.positions)
            ff_water = ForceField("amber14-all.xml", "amber14/tip3p.xml")
            modeller.addSolvent(
                ff_water,
                model="tip3p",
                padding=1.0 * nanometer,
                ionicStrength=ion_conc * molar,
            )
            system = ff_water.createSystem(
                modeller.topology, nonbondedMethod=PME, constraints=HBonds
            )
            simulation = Simulation(
                modeller.topology, system, integrator, platform, plat_props
            )
            simulation.context.setPositions(modeller.positions)
            try:
                simulation.minimizeEnergy(
                    tolerance=energy_diff, maxIterations=max_minimization_steps
                )
            except Exception as exc:
                logger.exception("Solvent minimization failed: %s", exc)
                raise RuntimeError("Solvent minimization failed") from exc
            state = simulation.context.getState(getPositions=True)
            with open(final_pdb, "w") as fh:
                PDBFile.writeFile(simulation.topology, state.getPositions(), fh)
            minimized_stage = "solvent"
        else:
            shutil.copy2(tmp_gas, final_pdb)

        try:
            tmp_gas.unlink()
        except Exception:
            pass

        # -------------------------
        # If GPF requested, compute grid_params from ligand (required)
        # -------------------------
        grid_params: Optional[Dict[str, float]] = None
        if out_fmt.lower() == "gpf":
            if not input_ligand:
                raise RuntimeError(
                    "out_fmt='gpf' requires input_ligand (SDF/PDB) to compute box via GridBox."
                )
            try:
                grid_params = self.compute_gridbox_from_ligand(
                    input_ligand,
                    scale=grid_scale,
                    min_size=grid_min_size,
                    pad=pad_angstrom,
                    isotropic=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to compute grid box from ligand; cannot create .gpf without explicit box. "
                    "Ensure the ligand file is valid and GridBox is functional. "
                    f"Original error: {exc}"
                ) from exc

        # -------------------------
        # Attempt to produce PDBQT / GPF via mekoo (with fallback to ADT)
        # -------------------------
        final_artifact: Path = final_pdb
        mk_info: Dict[str, Any] = {}
        adt_info: Dict[str, Any] = {}
        gpf_path: Optional[Path] = None

        if out_fmt.lower() in ("pdbqt", "gpf"):
            out_basename = Path(out_dir) / base_name
            write_pdbqt = out_dir / f"{base_name}.pdbqt"
            write_gpf = (
                out_dir / f"{base_name}.gpf" if out_fmt.lower() == "gpf" else None
            )

            box_center = None
            box_size = None
            if grid_params:
                box_center = (
                    grid_params["center_x"],
                    grid_params["center_y"],
                    grid_params["center_z"],
                )
                box_size = (
                    grid_params["size_x"],
                    grid_params["size_y"],
                    grid_params["size_z"],
                )
            else:
                # fallback: attempt to compute box from final_pdb coords (protein) via GridBox
                try:
                    gb_fb = GridBox().load_ligand(str(final_pdb)).preset("safe")
                    gp_fb = gb_fb.vina_dict
                    box_center = (
                        gp_fb["center_x"],
                        gp_fb["center_y"],
                        gp_fb["center_z"],
                    )
                    box_size = (gp_fb["size_x"], gp_fb["size_y"], gp_fb["size_z"])
                    logger.debug("Fallback box from protein coords computed.")
                except Exception:
                    box_center = None
                    box_size = None

            mk_info = self._call_mekoo(
                input_pdb=Path(final_pdb),
                out_basename=out_basename,
                write_pdbqt=(
                    write_pdbqt if out_fmt.lower() in ("pdbqt", "gpf") else None
                ),
                write_gpf=write_gpf if out_fmt.lower() == "gpf" else None,
                box_center=box_center,
                box_size=box_size,
            )

            if mk_info.get("produced"):
                produced = mk_info["produced"]
                produced_pdbqt = next(
                    (p for p in produced if p.lower().endswith(".pdbqt")), None
                )
                produced_gpf = next(
                    (p for p in produced if p.lower().endswith(".gpf")), None
                )
                if out_fmt.lower() == "pdbqt" and produced_pdbqt:
                    final_artifact = Path(produced_pdbqt)
                elif out_fmt.lower() == "gpf":
                    if produced_gpf:
                        gpf_path = Path(produced_gpf)
                        if produced_pdbqt:
                            final_artifact = Path(produced_pdbqt)
                    elif produced_pdbqt:
                        final_artifact = Path(produced_pdbqt)
            else:
                # mekoo failed or not found: try ADT fallback to produce pdbqt
                adt_info = self._call_adt(
                    Path(final_pdb), Path(out_dir / f"{base_name}.pdbqt")
                )
                if adt_info.get("produced"):
                    final_artifact = Path(adt_info["produced"][0])
                else:
                    final_artifact = final_pdb

        # If GPF requested and mekoo didn't produce gpf, generate a minimal local GPF using grid_params
        if out_fmt.lower() == "gpf" and gpf_path is None:
            gp = grid_params
            if gp is None:
                try:
                    gp = GridBox().load_ligand(str(final_pdb)).preset("safe").vina_dict
                except Exception:
                    gp = {
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "center_z": 0.0,
                        "size_x": 22.5,
                        "size_y": 22.5,
                        "size_z": 22.5,
                    }
            spacing = 0.375

            def _odd(n: int) -> int:
                ni = int(round(n))
                return ni if (ni % 2 == 1) else ni + 1

            nx = max(5, _odd(round(gp["size_x"] / spacing)))
            ny = max(5, _odd(round(gp["size_y"] / spacing)))
            nz = max(5, _odd(round(gp["size_z"] / spacing)))
            gpf_lines = [
                f"# Auto-generated minimal GPF for {base_name}",
                f"npts {nx} {ny} {nz}",
                f"spacing {spacing}",
                f"gridcenter {gp['center_x']:.6f} {gp['center_y']:.6f} {gp['center_z']:.6f}",
                f"receptor_file {Path(final_pdb).name}",
                "mapdir .",
                "end",
            ]
            gpf_path = out_dir / f"{base_name}.gpf"
            with open(gpf_path, "w") as fh:
                fh.write("\n".join(gpf_lines))
            logger.info("Wrote fallback local GPF: %s", gpf_path)

        # PyMOL postprocessing (renumber/cleanup) for PDB artifacts
        if cmd is not None and final_artifact.suffix.lower() == ".pdb":
            try:
                cmd.load(str(final_artifact))
                offset = int(start_at) - 1
                cmd.alter("all", f"resi=str(int(resi)+{offset})")
                if cofactors:
                    cmd.select(
                        "cofactors", " or ".join([f"resn {c}" for c in cofactors])
                    )
                    cmd.select("removed_solvent", "solvent and not cofactors")
                else:
                    cmd.select("removed_solvent", "solvent")
                cmd.remove("removed_solvent")
                cmd.select("nacl", "resn NA or resn CL")
                cmd.remove("nacl")
                cmd.save(str(final_artifact), "all")
                cmd.delete("all")
            except Exception:
                logger.exception("PyMOL postprocessing failed (non-fatal).")

        artifact_path = (
            gpf_path
            if (out_fmt.lower() == "gpf" and gpf_path is not None)
            else final_artifact
        )
        self.final_artifact = Path(artifact_path)
        self.last_simulation_report = {
            "final_artifact": str(self.final_artifact),
            "out_fmt": out_fmt,
            "mekoo_info": mk_info,
            "adt_info": adt_info,
            "minimized_stage": minimized_stage,
            "grid_params": grid_params,
        }
        logger.info("ReceptorProcess finished. Artifact: %s", self.final_artifact)
        return self

    def __repr__(self) -> str:
        return f"<ReceptorProcess final_artifact={self.final_artifact}>"
