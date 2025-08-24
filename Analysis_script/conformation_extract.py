import os
import argparse
from rdkit import Chem
import warnings
import logging


# Suppress specific warning
logging.getLogger("rdkit").setLevel(logging.ERROR)  # Suppress all warnings from RDKit


def extract_molecules(sdf_file, conformation, output_dir):
    # Get the base name of the file (without extension) to create the folder
    base_name = os.path.splitext(os.path.basename(sdf_file))[0]
    output_folder = os.path.join(output_dir or os.getcwd(), base_name)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the SDF file
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]
    if not mols:
        print("Error: No valid molecules found in the SDF file.")
        return

    # Determine the number of molecules to export
    conformation = min(conformation, len(mols))
    print(
        f"Found {len(mols)} molecules in the SDF file. Exporting the first {conformation} molecules."
    )

    # Export the molecules
    for i in range(conformation):
        output_file = os.path.join(output_folder, f"rank{i + 1}_confidence.sdf")
        try:
            writer = Chem.SDWriter(output_file)
            writer.write(mols[i])
        except Exception as e:
            print(f"Error writing molecule {i + 1}: {e}")
        finally:
            writer.close()

    print(f"Exported {conformation} molecules to folder: {output_folder}")


def list_primary_files(directory):
    # List all items (files and directories) in the specified directory
    all_items = os.listdir(directory)

    # Filter out and return only primary directories (directly in the specified directory)
    primary_files = [
        item for item in all_items if os.path.isfile(os.path.join(directory, item))
    ]

    return primary_files


def list_primary_folders(directory):
    # List all items (files and directories) in the specified directory
    all_items = os.listdir(directory)

    # Filter out and return only primary directories (directly in the specified directory)
    primary_folders = [
        item for item in all_items if os.path.isdir(os.path.join(directory, item))
    ]

    return primary_folders


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Extract the first N molecules from an SDF file."
    )

    # Create a mutually exclusive group for sdf_file and source_dir
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sdf_file", type=str, help="Path to the input SDF file.")
    group.add_argument(
        "--source_dir", type=str, help="Directory to the result of all targets"
    )
    parser.add_argument(
        "--conformation", type=int, default=10, help="Number of molecules to extract."
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save the output files."
    )

    args = parser.parse_args()

    # Extract molecules if sdf_file is provided
    if args.sdf_file:
        if not args.output_dir:
            output_dir = os.path.dirname(args.sdf_file)
        else:
            output_dir = args.output_dir
        extract_molecules(args.sdf_file, args.conformation, output_dir)
    elif args.source_dir:
        target_dir = list_primary_folders(args.source_dir)
        for i in target_dir:
            if i != "all":
                raw_dir = os.path.join(args.source_dir, i, "gnina_output", "raw")
                files = list_primary_files(raw_dir)
            for j in files:
                file = os.path.join(raw_dir, j)
                if not args.output_dir:
                    output_dir = os.path.join(args.source_dir, i)
                else:
                    output_dir = os.path.join(args.output_dir, i)
                extract_molecules(file, args.conformation, output_dir)


if __name__ == "__main__":
    main()
