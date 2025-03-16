import os
import argparse
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automate GNINA on multiple ligands with .sdf extension."
    )
    parser.add_argument(
        "--protein_dir", required=True, help="Directory containing the protein files"
    )
    parser.add_argument(
        "--ligand_dir", required=True, help="Directory containing the ligand files"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save the docking output (default uses the protein file name)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for docking runs (default: 42)",
    )
    parser.add_argument(
        "--cnn_scoring",
        choices=["none", "rescore", "refinement", "metrorescore", "metrorefine", "all"],
        default="rescore",
        help="CNN scoring option for docking (default: rescore)",
    )
    parser.add_argument(
        "--autobox_ligand",
        default=None,
        help="Path to reference ligand folder (not file, same name as protein) (optional)",
    )
    parser.add_argument(
        "--autobox_add",
        type=float,
        default=5,
        help="Buffer space for auto-generated box (default: 10, used with --autobox_ligand)",
    )
    parser.add_argument(
        "--autobox_extend",
        type=int,
        default=None,
        help="Expand the autobox if needed so docked ligands can freely rotate",
    )
    parser.add_argument(
        "--center_x",
        type=float,
        help="X coordinate of the center (required if --autobox_ligand is not provided)",
    )
    parser.add_argument(
        "--center_y",
        type=float,
        help="Y coordinate of the center (required if --autobox_ligand is not provided)",
    )
    parser.add_argument(
        "--center_z",
        type=float,
        help="Z coordinate of the center (required if --autobox_ligand is not provided)",
    )
    parser.add_argument(
        "--size_x",
        type=float,
        default=30,
        help="Size in X (default: 30 if --autobox_add not provided)",
    )
    parser.add_argument(
        "--size_y",
        type=float,
        default=30,
        help="Size in Y (default: 30 if --autobox_add not provided)",
    )
    parser.add_argument(
        "--size_z",
        type=float,
        default=30,
        help="Size in Z (default: 30 if --autobox_add not provided)",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=32,
        help="Exhaustiveness level (default: 32)",
    )
    parser.add_argument(
        "--num_modes",
        type=int,
        default=10,
        help="Number of modes to generate (default: 100)",
    )
    parser.add_argument(
        "--flexres", help="File with flexible residues for flexible docking (optional)"
    )
    parser.add_argument(
        "--quiet",
        choices=["Y", "y", "Yes", "yes", "N", "n", "No", "no"],
        help="Quiet verbose output?",
    )
    parser.add_argument(
        "--start_at",
        type=int,
        default=1,
        help="Start at which molecule (default at compound 0)",
    )
    parser.add_argument(
        "--end_at",
        required=False,
        type=int,
        default=None,
        help="Start at which molecule (default at compound 0)",
    )
    parser.add_argument(
        "--device", default=0, help="Which GPU should be used (default: first GPU)"
    )
    parser.add_argument(
        "--flexdist_ligand",
        default=None,
        required=False,
        help="Ligand to use for flexdist",
    )
    parser.add_argument(
        "--flexdist",
        default=5,
        help="Set all side chains within this specified distance to flexdist_ligand to flexible (default = 5 A)",
    )
    parser.add_argument(
        "--out_flex",
        default=False,
        help="True or False, in the same folder of docking output",
    )
    parser.add_argument("--cpu", help="Number of CPU to be used")

    return parser.parse_args()


def create_directory_structure(base_dir, protein_name, out_flex):
    result_dir = os.path.join(base_dir, "result_gnina", protein_name)
    result_gnina_dir = os.path.join(result_dir, "gnina_output")
    subfolders = ["log", "raw"]
    for folder in subfolders:
        os.makedirs(os.path.join(result_gnina_dir, folder), exist_ok=True)
    if out_flex:
        os.makedirs(os.path.join(result_gnina_dir, "flexres"), exist_ok=True)
    return result_dir


def run_docking(args, protein_path, ligand_path, result_dir):
    protein_name = os.path.splitext(os.path.basename(protein_path))[0]
    ligand_name = os.path.splitext(os.path.basename(ligand_path))[0]
    output_file = os.path.join(result_dir, "gnina_output", "raw", ligand_name + ".sdf")
    log_file = os.path.join(result_dir, "gnina_output", "log", ligand_name + ".txt")
    gnina_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gnina")
    gnina_script = os.path.join(gnina_dir, "gnina")
    print(gnina_script)
    command = [
        gnina_script,
        "-r",
        protein_path,
        "-l",
        ligand_path,
        "--out",
        output_file,
        "--seed",
        str(args.random_seed),
        "--cnn_scoring",
        args.cnn_scoring,
        "--exhaustiveness",
        str(args.exhaustiveness),
        "--num_modes",
        str(args.num_modes),
        "--log",
        str(log_file),
        "--device",
        str(args.device),
    ]

    if args.flexres is True:
        command.extend(["--flexres", args.flexres])

    if args.autobox_ligand:
        reference_ligand = os.path.join(args.autobox_ligand, f"{protein_name}.sdf")
        command.extend(
            [
                "--autobox_ligand",
                reference_ligand,
                "--autobox_add",
                str(args.autobox_add),
            ]
        )
    else:
        command.extend(
            [
                "--center_x",
                str(args.center_x),
                "--center_y",
                str(args.center_y),
                "--center_z",
                str(args.center_z),
                "--size_x",
                str(args.size_x),
                "--size_y",
                str(args.size_y),
                "--size_z",
                str(args.size_z),
            ]
        )
    if args.quiet in ["Yes", "Y", "yes", "y"]:
        command.extend(["--q"])
    if args.autobox_add:
        if args.autobox_extend:
            command.extend(["--autobox_extend", str(args.autobox_extend)])
    if args.flexdist_ligand:
        reference_ligand_flex = os.path.join(
            args.flexdist_ligand, f"{protein_name}.sdf"
        )
        command.extend(["--flexdist_ligand", str(reference_ligand_flex)])
        command.extend(["--flexdist", str(args.flexdist)])
    if args.out_flex is not False:
        out_flex_file = os.path.join(
            os.path.join(result_dir, "gnina_output", "flexres", ligand_name + ".pdb")
        )
        command.extend(["--out_flex", str(out_flex_file)])
        command.extend(["--full_flex_output"])
    if args.cpu:
        command.extend(["--cpu", str(args.cpu)])
    start_time = time.time()
    subprocess.run(command)
    duration = time.time() - start_time
    return duration


def main():
    args = parse_args()

    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = os.path.dirname(args.protein_dir)
    os.makedirs(os.path.join(base_dir, "result_gnina"), exist_ok=True)
    ligands = [f for f in sorted(os.listdir(args.ligand_dir)) if f.endswith(".sdf")]

    protein_files = [
        f
        for f in os.listdir(args.protein_dir)
        if f.endswith(".pdb") or f.endswith(".pdbqt")
    ]
    print(protein_files)
    if not args.end_at:
        args.end_at = len(ligands) + 1
    for protein_file in protein_files:
        protein_path = os.path.join(args.protein_dir, protein_file)
        protein_name = os.path.splitext(protein_file)[0]
        result_dir = create_directory_structure(base_dir, protein_name, args.out_flex)
        for i, ligand_file in enumerate(ligands):
            ligand_path = os.path.join(args.ligand_dir, ligand_file)
            if i < args.start_at - 1:
                continue
            if i > args.end_at - 1:
                break
            print(f"  Docking ligand {ligand_file} with protein {protein_file}...")
            run_docking(args, protein_path=protein_path, ligand_path=ligand_path, result_dir=result_dir)
    conformation_extract_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Analysis_script")
    print(str(os.path.dirname(result_dir)))
    conformation_extract_script = os.path.join(conformation_extract_dir,"conformation_extract.py")
    command_extract = ["python", str(conformation_extract_script),
    "--source_dir", str(os.path.dirname(result_dir)),
    "--conformation", str(args.num_modes)
    ]
    print(command_extract)
    subprocess.run(command_extract)


if __name__ == "__main__":
    main()
