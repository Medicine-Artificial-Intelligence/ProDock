import argparse
from pathlib import Path
import subprocess
import prolif as plf
import MDAnalysis as mda
from rdkit import Chem
import logging
from pymol import cmd, CmdException
from IPython.core.display import HTML


# Custom logging filter to add processing percentage as a progress bar
class PercentCompleteFilter(logging.Filter):
    def __init__(self, total):
        super().__init__()
        self.total = total
        self.current = 0

    def filter(self, record):
        if self.total > 0:
            percent = (self.current / self.total) * 100
            bar_length = 20  # Adjust the length of the progress bar
            filled_length = int(round(bar_length * self.current / float(self.total)))
            bar = "#" * filled_length + "-" * (bar_length - filled_length)
            record.percent = f"[{bar}] {percent:.2f}% Complete"
        else:
            record.percent = "[--------------------] 0% Complete"
        return True


def ensure_directory_exists(directory_path):
    """
    Ensures that the specified directory exists.
    If the directory does not exist, it will be created.

    Args:
    directory_path (str): The path to the directory to check and create if necessary.
    """
    # Convert the directory path to a Path object
    directory = Path(directory_path)

    # Check if the directory exists
    if not directory.exists():
        # Create the directory if it does not exist
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def setup_logging(output_dir, protein_path, ligands_path, total_ligands):
    protein_parent = protein_path.stem
    ligands_parent = ligands_path.name
    log_file = output_dir / "prolif_output" / f"{protein_parent}_{ligands_parent}.txt"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(percent)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    percent_filter = PercentCompleteFilter(total_ligands)
    logger.addFilter(percent_filter)
    return logger, percent_filter


def extract_first_pose_obabel(filepath, output_path):
    command = f"obabel -i sdf {filepath} -o sdf -O {output_path} -at 1"
    subprocess.run(command, shell=True, check=True)


def add_hydrogens_pymol(filepath, new_filename):
    try:
        cmd.reinitialize()
        cmd.load(filepath, "molecule")
        new_path = Path(filepath).parent / new_filename
        cmd.h_add("molecule")
        cmd.save(str(new_path), "molecule")
        cmd.delete("all")
        return new_path
    except CmdException as e:
        logging.error(f"Failed to process file {filepath} with PyMOL: {e}")
        return None


def process_ligand(protein_path, ligand_file, output_dir, logger, percent_filter):
    try:
        new_ligand_filename = f"{ligand_file.stem}_hydro{ligand_file.suffix}"
        new_ligand_path = add_hydrogens_pymol(str(ligand_file), new_ligand_filename)
        if not new_ligand_path:
            raise Exception("Failed to add hydrogens to ligand.")

        u = mda.Universe(str(protein_path))
        protein = plf.Molecule.from_mda(u)
        ligand = Chem.MolFromMolFile(str(new_ligand_path), removeHs=False)
        prolif_ligand = plf.Molecule.from_rdkit(ligand)
        fp = plf.Fingerprint()
        fp.run_from_iterable([prolif_ligand], protein)
        df = fp.to_dataframe()

        output_csv = (
            output_dir / "prolif_output/csv_files" / f"prolif_{ligand_file.stem}.csv"
        )
        df.to_csv(output_csv)

        # Generate the HTML interaction diagram
        html_fig = fp.plot_lignetwork(prolif_ligand)
        if isinstance(html_fig, HTML):
            html_content = html_fig.data  # Access the HTML content from the HTML object
            html_output_path = (
                output_dir
                / "prolif_output/html_files"
                / f"prolif_{ligand_file.stem}.html"
            )
            with open(html_output_path, "w") as file:
                file.write(html_content)
        else:
            logger.error(
                f"Unexpected output type from plot_lignetwork: {type(html_fig)}"
            )

        percent_filter.current += 1
        logger.info(
            f"Successfully processed and saved fingerprint and HTML diagram for {output_csv.name}"
        )
    except Exception as e:
        logger.error(f"Error processing {ligand_file.name}: {str(e)}")
    finally:
        Path(new_ligand_path).unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Generate interaction fingerprints for multiple ligands."
    )
    parser.add_argument(
        "-p", "--protein", required=True, help="Path to the original protein PDB file."
    )
    parser.add_argument(
        "-l",
        "--ligands",
        required=True,
        help="Path to the folder containing ligand files.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default="prolif_csv",
        help="Optional: Output folder to save the CSV files.",
    )
    args = parser.parse_args()
    extensions = ["pdbqt", "sdf", "mol2"]
    output_dir = Path(args.output_folder)
    ensure_directory_exists(f"{args.output_folder}/prolif_output")
    ensure_directory_exists(f"{args.output_folder}/prolif_output/csv_files")
    ensure_directory_exists(f"{args.output_folder}/prolif_output/html_files")
    ligand_files = [
        file for ext in extensions for file in Path(args.ligands).glob(f"*.{ext}")
    ]
    total_ligands = len(ligand_files)
    logger, percent_filter = setup_logging(
        output_dir, Path(args.protein), Path(args.ligands), total_ligands
    )
    original_protein_path = Path(args.protein)
    new_protein_filename = (
        f"{original_protein_path.stem}_hydro{original_protein_path.suffix}"
    )
    new_protein_path = add_hydrogens_pymol(
        str(original_protein_path), new_protein_filename
    )
    if not new_protein_path:
        logger.error("Failed to add hydrogens to protein.")
        return
    for index, ligand_file in enumerate(ligand_files):
        process_ligand(
            original_protein_path, ligand_file, output_dir, logger, percent_filter
        )
    new_protein_path.unlink()


if __name__ == "__main__":
    main()
