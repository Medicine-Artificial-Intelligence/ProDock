import os
import argparse
import pandas as pd
from pymol import cmd
from pdbfixer import PDBFixer
from openmm.app import *
from openmm import *
from openmm.unit import *
import sys
import subprocess


def fetch_and_process_pdb(
    pdb_id, output_dir, chains, ligand_code, ligand_name, cofactors, protein_name
):
    """
    Fetch a PDB file using PyMOL, filter chains, extract ligands based on ligand_code,
    keep cofactors, and delete water. Save ligand as named in ligand_name.
    """
    print(f"Processing PDB ID {pdb_id}...")
    fetch_output_dir = os.path.join(output_dir, "fetched_protein")
    pdb_path = os.path.join(fetch_output_dir, f"{pdb_id}.pdb")
    filtered_protein_dir = os.path.join(output_dir, "filtered_protein")
    filtered_protein_path = os.path.join(filtered_protein_dir, f"{pdb_id}.pdb")
    ligand_path = os.path.join(output_dir, "reference_ligand", f"{protein_name}.sdf")
    ligand2_path = os.path.join(output_dir, "cocrystal", f"{ligand_name}.sdf")

    # Fetch and load PDB
    cmd.fetch(pdb_id, path=fetch_output_dir, type="pdb")
    cmd.load(pdb_path)

    # Filter chains
    if chains:
        cmd.select("kept_chains", f"chain {' or chain '.join(chains)}")
        print(f"chain {' or chain '.join(chains)}")
    cmd.select("removed_complex", "all and not kept_chains")
    cmd.remove("removed_complex")

    # Extract ligand and save
    for i in range(0, len(chains)):
        cmd.select("ligand", f"resn {ligand_code} and chain {chains[i]}")
        if cmd.count_atoms("ligand") != 0:
            break
    cmd.save(ligand_path, "ligand")
    cmd.save(ligand2_path, "ligand")
    cmd.remove(f"resn {ligand_code}")

    # Keep cofactors and delete water

    solvent = [
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
    cmd.select("solvents", f"resn {' or resn '.join(solvent)}")
    if cofactors:
        cmd.select("cofactors", f"resn {'or resn '.join(cofactors)}")
        cmd.select("removed_solvent", "solvents and not cofactors")
    else:
        cmd.select("removed_solvent", "solvents")
    cmd.remove("removed_solvent")

    # Save filtered protein
    cmd.save(filtered_protein_path, "all")
    cmd.delete("all")

    current_working_dir = os.getcwd()

    for i in [ligand_path, ligand2_path]:
        os.chdir(os.path.dirname(i))
        print(os.path.dirname(i))
        command = [
            "obabel",
            "-isdf",
            os.path.basename(i),
            "-O",
            os.path.basename(i),
        ]
        result = subprocess.run(command)
        os.chdir(current_working_dir)

    return filtered_protein_path, ligand_path


def fix_and_minimize_pdb(
    input_pdb,
    output_dir,
    energy_diff,
    max_minimization_steps,
    start_at,
    ion_conc,
    cofactors,
    pdb_id=None,
    protein_name=None,
    minimize_in_water=False,
):
    """
    Fix and perform energy minimization in two steps:
    1. Gas-phase minimization with backbone restraints.s
    2. (Optional) Solvent minimization in water with ions.
    """

    os.makedirs(output_dir, exist_ok=True)

    if protein_name:
        base_name = protein_name
    else:
        base_name = pdb_id if pdb_id else "output"

    final_pdb = os.path.join(output_dir, f"{base_name}.pdb")

    print(f"Fixing and minimizing {input_pdb}...")

    ### **Step 1: Fix PDB (Add Missing Atoms & Hydrogens)**
    fixer = PDBFixer(filename=input_pdb)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    ### **Step 2: Gas-Phase Energy Minimization (With Backbone Constraints)**
    print("Performing gas-phase minimization with backbone restraints...")

    forcefield = ForceField("amber14-all.xml")  # No water force field in gas phase
    modeller = Modeller(fixer.topology, fixer.positions)
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds
    )

    # **Apply Backbone Restraints (N, CA, C)**
    force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for atom in modeller.topology.atoms():
        if atom.name in ["N", "CA", "C"]:  # Restrain backbone atoms
            idx = atom.index
            pos = modeller.positions[idx]
            force.addParticle(
                idx, [5.0 * kilocalories_per_mole / angstroms**2, pos.x, pos.y, pos.z]
            )

    system.addForce(force)

    # **Use CUDA for GPU Acceleration**
    platform = Platform.getPlatformByName("CUDA")
    platform_properties = {"CudaPrecision": "mixed"}

    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    integrator.setRandomNumberSeed(42)
    simulation = Simulation(
        modeller.topology, system, integrator, platform, platform_properties
    )
    simulation.context.setPositions(modeller.positions)

    # **Minimize Energy in Gas Phase**
    simulation.minimizeEnergy(
        tolerance=energy_diff, maxIterations=max_minimization_steps
    )
    minimized_positions = simulation.context.getState(getPositions=True).getPositions()

    # **Save Gas-Minimized Structure**
    with open(final_pdb, "w") as f:
        PDBFile.writeFile(simulation.topology, minimized_positions, f)

    print(f"Gas-phase minimization complete. Saved minimized PDB to {final_pdb}")

    ### **Step 3: Solvent Energy Minimization (Optional)**
    if minimize_in_water:
        print("Adding water and performing solvent minimization...")

        # **Reload Gas-Minimized PDB**
        modeller = Modeller(PDBFile(final_pdb).topology, PDBFile(final_pdb).positions)

        # **Add Water & Ions**
        forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=1.0 * nanometers,
            ionicStrength=ion_conc * molar,
        )

        # **Recreate System & Integrator**
        system = forcefield.createSystem(
            modeller.topology, nonbondedMethod=PME, constraints=HBonds
        )
        integrator = LangevinIntegrator(
            300 * kelvin, 1 / picosecond, 0.002 * picoseconds
        )
        integrator.setRandomNumberSeed(42)
        simulation = Simulation(
            modeller.topology, system, integrator, platform, platform_properties
        )
        simulation.context.setPositions(modeller.positions)

        # **Minimize Energy in Solvent**
        simulation.minimizeEnergy(
            tolerance=energy_diff, maxIterations=max_minimization_steps
        )
        minimized_positions = simulation.context.getState(
            getPositions=True
        ).getPositions()

        # **Save Final Minimized Structure**
        with open(final_pdb, "w") as f:
            PDBFile.writeFile(simulation.topology, minimized_positions, f)

        print(f"Solvent minimization complete. Saved minimized PDB to {final_pdb}")

    ### **Step 4: Final Cleanup & Processing with PyMOL**
    cmd.load(final_pdb)
    cmd.alter("all", f"resi=str(int(resi)+{start_at-1})")

    if cofactors:
        cmd.select("cofactors", f"resn {'or resn '.join(cofactors)}")
        cmd.select("removed_solvent", "solvent and not cofactors")
    else:
        cmd.select("removed_solvent", "solvent")

    cmd.remove("removed_solvent")
    cmd.select("nacl", "resn NA or resn CL")
    cmd.remove("nacl")

    cmd.save(final_pdb, "all")
    cmd.delete("all")

    print(f"Final minimized PDB saved to {final_pdb}")


def main(args):
    """
    Main function to process PDB files based on input CSV.
    """
    # Read the input CSV file
    print("Python Executable:", sys.executable)
    df = pd.read_csv(args.csv_file)

    # Normalize headers to lowercase
    df.columns = df.columns.str.lower()

    # Create output directories
    fetched_dir = os.path.join(args.output_dir, "fetched_protein")
    filtered_dir = os.path.join(args.output_dir, "filtered_protein")
    processed_dir = os.path.join(args.output_dir, "processed_protein")
    ligand_dir = os.path.join(args.output_dir, "reference_ligand")
    ligand2_dir = os.path.join(args.output_dir, "cocrystal")
    os.makedirs(fetched_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(ligand_dir, exist_ok=True)
    os.makedirs(ligand2_dir, exist_ok=True)

    for index, row in df.iterrows():
        try:
            pdb_id = str(row["pdb_id"].lower())
            chains = (
                [chain.strip().upper() for chain in row["chains"].split("+")]
                if "chains" in row and pd.notna(row["chains"])
                else None
            )
            cofactors = (
                [cofactor.strip().upper() for cofactor in row["cofactors"].split("+")]
                if "cofactors" in row and pd.notna(row["cofactors"])
                else None
            )
            ligand_code = str(row["ligand_code"].upper())
            ligand_name = str(row["ligand"]).lower()
            protein_name = str(row["protein"])
            if row["start_at"] != None:
                start_at = int(row["start_at"])
            if start_at == None:
                start_at = 1
            print(start_at)
            filtered_protein_path, ligand_path = fetch_and_process_pdb(
                pdb_id,
                args.output_dir,
                chains,
                ligand_code,
                ligand_name,
                cofactors,
                protein_name,
            )
            fix_and_minimize_pdb(
                input_pdb=filtered_protein_path,
                output_dir=processed_dir,
                energy_diff=args.energy,
                max_minimization_steps=args.steps,
                start_at=start_at,
                ion_conc=args.ion_conc,
                cofactors=cofactors,
                pdb_id=pdb_id.upper(),
                protein_name=protein_name,
                minimize_in_water=args.water_em,
            )

        except Exception as e:
            print(f"An error occurred: {e}")

    print("All operations completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDB files from a CSV input.")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the CSV file with input data.",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=1,
        help="Energy difference to stop energy minimization.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum step to stop energy minimization.",
    )
    parser.add_argument(
        "--ion_conc",
        type=str,
        default=0.15,
        help="Ion concentration for energy minimization in water (if applicable).",
    )
    parser.add_argument(
        "--water_em", type=str, default=False, help="Energy minimization in water"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files."
    )
    args = parser.parse_args()
    main(args)
