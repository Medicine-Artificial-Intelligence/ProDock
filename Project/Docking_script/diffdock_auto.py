import os
import argparse
import subprocess
import time
from tqdm import tqdm
import re
from datetime import datetime
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automate DiffDock on multiple ligands with .sdf extension."
    )
    parser.add_argument(
        "--protein_dir",
        required=True,
        help="Path to the folder containing protein PDB files",
    )
    parser.add_argument(
        "--ligand_dir", required=True, help="Directory containing the ligand files"
    )
    parser.add_argument(
        "--output_dir",
        default=os.getcwd(),
        help="Directory to save the docking output.",
    )
    parser.add_argument(
        "--model_dir", default=None, help="Directory to the model folder"
    )
    parser.add_argument(
        "--save_visualisation",
        choices=["yes", "no"],
        default="no",
        help="Whether to save visualisation",
    )
    parser.add_argument(
        "--samples_per_complex",
        type=int,
        default=None,
        help="Number of samples per complex",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for processing"
    )
    return parser.parse_args()


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_diffdock(args, protein_file, ligand_file, complex_name, output_dir):
    current_dir = os.getcwd()
    diffdock_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DiffDock"
    )
    os.chdir(diffdock_dir)
    diffdock_script = os.path.join(diffdock_dir, "inference.py")
    config_dir = os.path.join(diffdock_dir, "default_inference_args.yaml")
    command = [
        "python",
        diffdock_script,
        "--config",
        config_dir,
        "--protein_path",
        str(protein_file),
        "--ligand",
        str(ligand_file),
        "--out_dir",
        output_dir,
        "--complex_name",
        complex_name,
    ]
    if args.model_dir:
        command.extend(["--model_dir", args.model_dir])
    if args.samples_per_complex is not None:
        command.extend(["--samples_per_complex", str(args.samples_per_complex)])
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.save_visualisation == "yes":
        command.extend(["--save_visualisation", ""])

    start_time = time.time()
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    duration = time.time() - start_time
    os.chdir(current_dir)
    return duration, result.stdout, result.stderr


def setup_logging(output_dir, log_name_date):
    # Configure the loggers
    logger_time = logging.getLogger("time_logger")
    logger_out = logging.getLogger("output_logger")
    logger_error = logging.getLogger("error_logger")

    # Configure log levels
    logger_time.setLevel(logging.INFO)
    logger_out.setLevel(logging.INFO)
    logger_error.setLevel(logging.ERROR)

    # Set file handlers
    file_handler_time = logging.FileHandler(os.path.join(output_dir, "log_time.txt"))
    file_handler_out = logging.FileHandler(os.path.join(output_dir, "log_out.txt"))
    file_handler_error = logging.FileHandler(os.path.join(output_dir, "log_error.txt"))

    # Set formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler_time.setFormatter(formatter)
    file_handler_out.setFormatter(formatter)
    file_handler_error.setFormatter(formatter)

    # Add handlers to the loggers
    logger_time.addHandler(file_handler_time)
    logger_out.addHandler(file_handler_out)
    logger_error.addHandler(file_handler_error)

    return logger_time, logger_out, logger_error


def main():
    args = parse_args()
    start_time = datetime.now()
    log_name_date = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    ligands = [f for f in sorted(os.listdir(args.ligand_dir)) if f.endswith(".sdf")]
    protein_files = [f for f in os.listdir(args.protein_dir) if f.endswith(".pdb")]
    output_dir = os.path.join(args.output_dir, "result_diffdock")
    create_directory(output_dir)
    log_dir = os.path.join(os.path.dirname(output_dir), "log", f"log_{log_name_date}")
    create_directory(log_dir)
    logger_time, logger_out, logger_error = setup_logging(log_dir, log_name_date)

    protein_bar = tqdm(total=len(protein_files), desc="Proteins", position=0)
    ligand_bar = tqdm(total=len(ligands), desc="Ligands/current protein", position=1)
    for protein in protein_files:
        ligand_bar.reset()
        protein_path = os.path.join(args.protein_dir, protein)
        output_temp_dir = re.sub(r"_protein.*", "", protein)
        output_dir_individual = os.path.splitext(
            os.path.join(output_dir, output_temp_dir)
        )[0]
        create_directory(output_dir_individual)

        for ligand in ligands:
            ligand_path = os.path.join(args.ligand_dir, ligand)
            complex_name = os.path.splitext(ligand)[0]
            try:
                duration, stdout, stderr = run_diffdock(
                    args,
                    str(protein_path),
                    str(ligand_path),
                    str(complex_name),
                    str(output_dir_individual),
                )
                logger_time.info(
                    f"{complex_name} docked with {output_temp_dir}_protein in {duration:.2f} seconds."
                )
                if stdout:
                    logger_out.info(stdout)
                if stderr:
                    logger_error.error(stderr)
            except Exception as e:
                logger_error.error(
                    f"An error occurred while processing {complex_name}: {str(e)}"
                )
            ligand_bar.update(1)
        protein_bar.update(1)
    protein_bar.close()
    ligand_bar.close()


if __name__ == "__main__":
    main()
