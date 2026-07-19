import os
import argparse
import time
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform energy minimization on ligands."
    )
    parser.add_argument(
        "--ligand_dir",
        type=str,
        required=True,
        help="Directory or path to the SDF or CSV file containing ligands.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to place the output files. Defaults to 'minimized_ligands'.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for the output SDF files when a single SDF file is provided.",
    )
    return parser.parse_args()


def minimize_molecule(mol, mol_id):
    """Perform energy minimization on the molecule, with error handling."""
    try:
        mol = Chem.AddHs(mol)  # Add hydrogens
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
            raise RuntimeError(f"Embedding failed for molecule {mol_id}")

        if mol.GetNumConformers() == 0:
            raise RuntimeError(f"No conformer found for molecule {mol_id}")

        AllChem.MMFFOptimizeMolecule(mol)  # Energy minimization
        mol = Chem.RemoveHs(mol)
        return mol

    except Exception as e:
        logging.error(f"Skipping {mol_id}: {e}")
        print(f"Error: {e} (Skipping {mol_id})")
        return None


def process_sdf_molecules(supplier, output_dir, prefix, is_single_file):
    success_count, failure_count = 0, 0
    print("Starting SDF molecule processing...\n")

    for i, mol in enumerate(supplier):
        mol_id = f"{prefix}_{i + 1}" if is_single_file else f"{prefix}"
        if mol is None:
            logging.error(f"Skipping {mol_id}: Invalid molecule in SDF file.")
            print(f"Skipping {mol_id}: Invalid molecule.")
            failure_count += 1
            continue

        print(f"Processing molecule {mol_id}...")
        minimized_mol = minimize_molecule(mol, mol_id)

        if minimized_mol:
            try:
                output_filename = os.path.join(output_dir, f"{mol_id}.sdf")
                with Chem.SDWriter(output_filename) as writer:
                    writer.write(minimized_mol)
                print(f"Successfully processed and wrote {output_filename}")
                success_count += 1
            except Exception as e:
                logging.error(f"Failed to write {mol_id}: {e}")
                print(f"Error writing {mol_id}: {e}")
                failure_count += 1
        else:
            print(f"Skipping {mol_id} due to optimization failure.")
            failure_count += 1

    return success_count, failure_count


def process_csv_molecules(csv_file, output_dir):
    success_count, failure_count = 0, 0
    print("Starting CSV molecule processing...\n")

    try:
        df = pd.read_csv(csv_file)
        if "Smiles" not in df.columns or "Compounds" not in df.columns:
            raise ValueError("CSV file must contain 'Smiles' and 'Compounds' columns.")

        for _, row in df.iterrows():
            smiles = row["Smiles"]
            compound_name = row["Compounds"]
            mol_id = compound_name

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES for {mol_id}")

                print(f"Processing molecule {mol_id}...")
                minimized_mol = minimize_molecule(mol, mol_id)

                if minimized_mol:
                    output_filename = os.path.join(output_dir, f"{mol_id}.sdf")
                    with Chem.SDWriter(output_filename) as writer:
                        writer.write(minimized_mol)
                    print(f"Successfully processed and wrote {output_filename}")
                    success_count += 1
                else:
                    print(f"Skipping {mol_id} due to optimization failure.")
                    failure_count += 1

            except Exception as e:
                logging.error(f"Skipping {mol_id}: {e}")
                print(f"Error processing {mol_id}: {e}")
                failure_count += 1

    except Exception as e:
        logging.error(f"Failed to process CSV file {csv_file}: {e}")
        print(f"Error processing CSV file {csv_file}: {e}")
        failure_count += 1

    return success_count, failure_count


def main():
    start_time = time.time()
    args = parse_arguments()

    is_file = os.path.isfile(args.ligand_dir)
    if is_file and not args.prefix and args.ligand_dir.endswith(".sdf"):
        raise ValueError("Prefix must be provided for a single SDF file input.")

    if not args.output_dir:
        args.output_dir = os.path.join(os.getcwd(), "minimized_ligands")
    os.makedirs(args.output_dir, exist_ok=True)

    log_filename = os.path.join(args.output_dir, "log.txt")
    logging.basicConfig(
        filename=log_filename,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    success, failure = 0, 0

    try:
        if is_file:
            if args.ligand_dir.endswith(".sdf"):
                supplier = Chem.SDMolSupplier(args.ligand_dir)
                success, failure = process_sdf_molecules(
                    supplier, args.output_dir, args.prefix, is_single_file=True
                )
            elif args.ligand_dir.endswith(".csv"):
                success, failure = process_csv_molecules(
                    args.ligand_dir, args.output_dir
                )
            else:
                raise ValueError("Input file must be a .sdf or .csv file.")
        else:
            for file_name in os.listdir(args.ligand_dir):
                file_path = os.path.join(args.ligand_dir, file_name)
                if file_name.endswith(".sdf"):
                    supplier = Chem.SDMolSupplier(file_path)
                    prefix = os.path.splitext(file_name)[0]
                    s, f = process_sdf_molecules(
                        supplier, args.output_dir, prefix, is_single_file=False
                    )
                    success += s
                    failure += f
                elif file_name.endswith(".csv"):
                    s, f = process_csv_molecules(file_path, args.output_dir)
                    success += s
                    failure += f

    except Exception as e:
        logging.error(f"Critical Error: {e}")
        print(f"Critical Error: {e}")

    end_time = time.time()
    total_time = end_time - start_time

    logging.error(
        f"Run Summary: {success} molecules processed successfully, {failure} failed. Total runtime: {total_time:.2f} seconds."
    )
    print(
        f"\nRun Summary: {success} molecules processed successfully, {failure} failed."
    )
    print(f"Total runtime: {total_time:.2f} seconds.")
    print(f"\nProcessing complete. See {log_filename} for details on failed molecules.")


if __name__ == "__main__":
    main()
