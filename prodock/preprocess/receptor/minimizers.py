"""
Minimizer helpers for ReceptorProcess.

Contains:
- fix_pdb
- minimize_with_openmm
- minimize_with_obabel
"""

from __future__ import annotations

import shutil
import logging
import subprocess
from pathlib import Path

from typing import Tuple

from pdbfixer import PDBFixer
from openmm.app import Modeller, PDBFile, ForceField, Simulation, NoCutoff, HBonds, PME
from openmm import Platform, CustomExternalForce, LangevinIntegrator
from openmm.unit import nanometer, kelvin, picosecond, molar

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _choose_platform():
    """Select best available OpenMM platform (CUDA, OpenCL, CPU)."""
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            plat = Platform.getPlatformByName(name)
            props = {"CudaPrecision": "mixed"} if name == "CUDA" else {}
            logger.debug("OpenMM platform selected: %s", name)
            return plat, props
        except Exception:
            continue
    return Platform.getPlatformByName("CPU"), {}


def _pos_to_nm(pos):
    """Return (x,y,z) floats in nanometers from an OpenMM position-like object."""
    try:
        vals = pos.value_in_unit(nanometer)
        return float(vals[0]), float(vals[1]), float(vals[2])
    except Exception:
        try:
            return float(pos[0]), float(pos[1]), float(pos[2])
        except Exception:
            return float(pos.x), float(pos.y), float(pos.z)


def fix_pdb(input_pdb: str) -> Modeller:
    """
    Run PDBFixer repairs and return an OpenMM Modeller ready for minimization.

    :param input_pdb: Path to input PDB file.
    :type input_pdb: str
    :returns: OpenMM Modeller object with fixed topology and positions.
    :rtype: openmm.app.Modeller
    """
    logger.info("fix_pdb: fixing %s", input_pdb)
    fixer = PDBFixer(filename=input_pdb)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)
    modeller = Modeller(fixer.topology, fixer.positions)
    logger.debug("fix_pdb: done")
    return modeller


def minimize_with_openmm(
    modeller: Modeller,
    out_pdb: Path,
    tmp_gas: Path,
    backbone_k_kcal_per_A2: float = 5.0,
    energy_diff: float = 10.0,
    max_minimization_steps: int = 5000,
    minimize_in_water: bool = False,
    ion_conc: float = 0.15,
) -> Tuple[Path, str]:
    """
    Minimize a Modeller with OpenMM and write the final PDB.

    :param modeller: OpenMM Modeller produced by fix_pdb.
    :type modeller: openmm.app.Modeller
    :param out_pdb: Destination PDB path to write final coordinates.
    :type out_pdb: pathlib.Path
    :param tmp_gas: Temporary gas-phase PDB path used between stages.
    :type tmp_gas: pathlib.Path
    :param backbone_k_kcal_per_A2: Restraint force constant on backbone atoms (kcal/A^2).
    :type backbone_k_kcal_per_A2: float
    :param energy_diff: Minimizer tolerance.
    :type energy_diff: float
    :param max_minimization_steps: Max number of minimization iterations.
    :type max_minimization_steps: int
    :param minimize_in_water: If True, add explicit TIP3P solvent and minimize again.
    :type minimize_in_water: bool
    :param ion_conc: Ionic strength for solvation (molar).
    :type ion_conc: float
    :returns: Tuple of (written_out_pdb_path, minimized_stage) where stage in {"gas","solvent"}.
    :rtype: (pathlib.Path, str)
    :raises RuntimeError: on OpenMM failures.
    """
    logger.info("minimize_with_openmm: starting gas-phase minimization")
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
            x0, y0, z0 = _pos_to_nm(modeller.positions[idx])
            restr.addParticle(idx, [k_kj_per_mol_nm2, x0, y0, z0])

    system.addForce(restr)
    platform, plat_props = _choose_platform()
    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picosecond)
    integrator.setRandomNumberSeed(42)

    simulation = Simulation(modeller.topology, system, integrator, platform, plat_props)
    simulation.context.setPositions(modeller.positions)

    try:
        simulation.minimizeEnergy(
            tolerance=energy_diff, maxIterations=max_minimization_steps
        )
    except Exception as exc:
        logger.exception("minimize_with_openmm: gas minimization failed")
        raise RuntimeError("OpenMM gas-phase minimization failed") from exc

    state = simulation.context.getState(getPositions=True)
    with open(tmp_gas, "w") as fh:
        PDBFile.writeFile(simulation.topology, state.getPositions(), fh)

    minimized_stage = "gas"

    if minimize_in_water:
        logger.info("minimize_with_openmm: adding solvent and minimizing")
        pdb_in = PDBFile(str(tmp_gas))
        modeller_w = Modeller(pdb_in.topology, pdb_in.positions)
        ff_water = ForceField("amber14-all.xml", "amber14/tip3p.xml")
        modeller_w.addSolvent(
            ff_water,
            model="tip3p",
            padding=1.0 * nanometer,
            ionicStrength=ion_conc * molar,
        )
        system_w = ff_water.createSystem(
            modeller_w.topology, nonbondedMethod=PME, constraints=HBonds
        )
        simulation_w = Simulation(
            modeller_w.topology, system_w, integrator, platform, plat_props
        )
        simulation_w.context.setPositions(modeller_w.positions)
        try:
            simulation_w.minimizeEnergy(
                tolerance=energy_diff, maxIterations=max_minimization_steps
            )
        except Exception as exc:
            logger.exception("minimize_with_openmm: solvent minimization failed")
            raise RuntimeError("OpenMM solvent minimization failed") from exc
        state = simulation_w.context.getState(getPositions=True)
        with open(out_pdb, "w") as fh:
            PDBFile.writeFile(simulation_w.topology, state.getPositions(), fh)
        minimized_stage = "solvent"
    else:
        shutil.copy2(tmp_gas, out_pdb)

    try:
        tmp_gas.unlink()
    except Exception:
        pass

    logger.info(
        "minimize_with_openmm: completed stage=%s -> %s", minimized_stage, out_pdb
    )
    return out_pdb, minimized_stage


def minimize_with_obabel(input_pdb: Path, out_pdb: Path, steps: int = 500) -> Path:
    """
    Minimize using OpenBabel (fallback when OpenMM is unavailable/failed).

    :param input_pdb: Input PDB path to minimize.
    :type input_pdb: pathlib.Path
    :param out_pdb: Output PDB path to write minimized coordinates.
    :type out_pdb: pathlib.Path
    :param steps: Number of minimization steps for OB.
    :type steps: int
    :returns: out_pdb path once written.
    :rtype: pathlib.Path
    :raises RuntimeError: if OpenBabel is not found or invocation fails.
    """
    exe = shutil.which("obabel") or shutil.which("babel")
    if not exe:
        raise RuntimeError("OpenBabel (obabel/babel) not found in PATH")

    args = [
        exe,
        str(input_pdb),
        "-O",
        str(out_pdb),
        "--minimize",
        "--steps",
        str(steps),
    ]
    logger.debug("minimize_with_obabel: calling: %s", " ".join(args))
    try:
        proc = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
    except Exception as exc:
        logger.exception("minimize_with_obabel: invocation failed")
        raise RuntimeError("OpenBabel invocation failed") from exc

    if proc.returncode != 0:
        logger.error(
            "minimize_with_obabel: obabel rc=%s stderr=%s", proc.returncode, proc.stderr
        )
        raise RuntimeError(f"OpenBabel minimization failed: {proc.stderr.strip()}")

    if not out_pdb.exists():
        raise RuntimeError("OpenBabel reported success but output file missing")

    logger.info("minimize_with_obabel: finished -> %s", out_pdb)
    return out_pdb
