import argparse
import csv
import os
import re
import pymol
from pymol import cmd
import pandas as pd
import glob
from rdkit import Chem  # RDKit for ligand validation
import MDAnalysis as mda
import prolif as plf
import rdkit
from rdkit import Chem
import pandas as pd
import gc
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from Bio import PDB
import subprocess
from memory_profiler import profile


# Initialize PyMOL in command-line mode
pymol.finish_launching(["pymol", "-c", "-q"])


def validate_ligand(file_path):
    """
    Validate a ligand file using RDKit.
    Returns True if valid, False otherwise.
    """
    try:
        mol = Chem.MolFromMolFile(file_path)
        if mol is None or mol.GetNumAtoms() == 0:
            return False
        return True
    except Exception:
        return False


def calculate_contact_surface_area(ligand_selection, binding_site_selection):
    """
    Calculate the surface area of the ligand that is in contact with the binding site.
    """
    cmd.select("isolated_ligand", ligand_selection)
    isolated_sasa = cmd.get_area("isolated_ligand")
    cmd.select("isolated_binding_site", binding_site_selection)
    binding_site_area = cmd.get_area("isolated_binding_site")

    # Create a temporary object that includes both the ligand and the binding site
    try:
        cmd.create(
            "temp_complex", f"({ligand_selection}) or ({binding_site_selection})"
        )
        complex_sasa = cmd.get_area("temp_complex")
    finally:
        cmd.delete("temp_complex")

    # Calculate the contact area
    if binding_site_area + isolated_sasa == complex_sasa:
        contact_area = binding_site_area
    else:
        contact_area = abs(isolated_sasa - complex_sasa)
    occupation_percent = (
        abs((binding_site_area - contact_area)) / binding_site_area
    ) * 100
    return contact_area, occupation_percent


def calculate_total_sasa(binding_site_selection):
    """
    Calculate the total solvent-accessible surface area (SASA) of the binding site.
    """
    cmd.select("binding_site_sasa", binding_site_selection)
    return cmd.get_area("binding_site_sasa")


def percentage_atoms_in_site(ligand_selection, binding_site_selection, threshold=5.0):
    """
    Calculate the percentage of ligand atoms within a specified distance of the binding site.
    """
    cmd.select(
        "temp_near",
        f"byres ({ligand_selection} within {threshold} of {binding_site_selection})",
    )
    in_site_atoms = cmd.count_atoms("temp_near")
    total_atoms = cmd.count_atoms(ligand_selection)
    cmd.delete("temp_near")
    return (in_site_atoms / total_atoms) * 100 if total_atoms else 0


def natural_sort_key(s):
    """
    Sort strings containing numbers by converting numeric parts into integers.
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def merge_confidence_contact(
    directory, output_file, atom_threshold, occupation_threshold, confidence_threshold
):
    """
    Merge confidence score and contact files into a single output CSV.
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    csv_files.sort(key=natural_sort_key)

    # Load the relevant CSV files
    df1 = pd.read_csv(glob.glob(os.path.join(directory, f"*confidence_score.csv"))[0])
    df2 = pd.read_csv(glob.glob(os.path.join(directory, f"*contact.csv"))[0])
    df1 = pd.merge(df1, df2, on="Compounds", how="outer")

    atom_column = [col for col in df1.columns if re.match(r"%atoms.*", col)]
    occupation_column = [col for col in df1.columns if re.match(r"%Occupation.*", col)]
    final_name = os.path.basename(directory)

    # Identify the relevant columns for criteriae
    confidence_column = [
        col for col in df1.columns if re.match(r"Confidence_score.*", col)
    ]

    if atom_column and confidence_column:
        # Calculate 'Final' column
        df1[f"Final_{final_name}"] = (
            (df1[atom_column[0]] >= atom_threshold)
            & (df1[confidence_column[0]] >= confidence_threshold)
            & (df1[occupation_column[0]] >= occupation_threshold)
        ).astype(int)
        final_column = df1.pop(f"Final_{final_name}")
        df1[f"Final_{final_name}"] = final_column
    else:
        print("Required columns not found in the data.")

    # Save the merged DataFrame to a new CSV file
    output_path = os.path.join(directory, f"{os.path.basename(directory)}_final.csv")
    df1.to_csv(output_path, index=False)
    print(f"Merged file saved to {output_path}")


def interaction_fingerprint(protein_file, ligand_file, distance, clashing_dist):
    u = mda.Universe(protein_file)

    protein = plf.Molecule.from_mda(u, NoImplicit=False)
    ligand = plf.sdf_supplier(ligand_file)

    # use default interactions
    fp = plf.Fingerprint(vicinity_cutoff=distance)
    # run on your poses
    fp.run_from_iterable(ligand, protein, n_jobs=1)
    fp_df = fp.ifp

    fp_df = pd.DataFrame.from_dict(fp.ifp[0], orient="index")
    if fp_df.empty == False:
        fp_df.iloc[0:, 0:] = fp_df.iloc[0:, 0:].applymap(
            lambda x: 0 if pd.isna(x) else 1
        )

        def melt_type1(df):
            df = df.reset_index()
            df.columns = ["Ligand", "Residue"] + list(df.columns[2:])

            df_melted = df.melt(
                id_vars=[
                    "Ligand",
                    "Residue",
                ],  # Use the first two columns as identifiers
                var_name="Interaction",  # Rename the column for interaction types
                value_name="Presence",  # Rename the column for binary values
            )
            df_melted = df_melted[df_melted["Presence"] == 1]
            df_melted
            df_melted["Residue"] = df_melted["Residue"].astype(str)
            df_melted["Interaction"] = df_melted["Interaction"].astype(str)
            df_melted["Residue-Interaction"] = (
                df_melted["Residue"].str.strip() + ":" + df_melted["Interaction"]
            )
            df_pivot = df_melted.pivot_table(
                index="Ligand",  # Rows correspond to unique ligands/molecules
                columns="Residue-Interaction",  # Columns are unique residue-interaction pairs
                values="Presence",  # The binary interaction data (1 or 0) fills the table
                fill_value=int(0),  # Missing interactions are filled with 0
            )
            df = df_pivot.drop(columns="Ligand", errors="ignore").reset_index(drop=True)
            df = df.astype(int)
            return df

        def melt_type2(df):
            df = df.reset_index()
            df.columns = ["Ligand", "Residue"] + list(df.columns[2:])
            df_res = df[["Residue"]]
            df_res["Fingerprint"] = 1
            df_res = df_res.T
            df_res.columns = df_res.iloc[0]  # Set the first row as the header
            df_res = df_res[1:]  # Remove the first row from the data
            # df = df.reset_index(drop=True)
            return df_res

        fp_df = pd.DataFrame.from_dict(fp.ifp[0], orient="index")
        fp_df.iloc[0:, 0:] = fp_df.iloc[0:, 0:].applymap(
            lambda x: 0 if pd.isna(x) else 1
        )
        fp_df_type1 = melt_type1(fp_df)
        fp_df_type2 = melt_type2(fp_df)

        # molecular clashing calculation
        parser = PDB.PDBParser(QUIET=True)

        # Load protein atoms
        protein_atoms = [
            atom for atom in parser.get_structure("protein", protein_file).get_atoms()
        ]

        # Load ligand atoms from SDF (already 3D)
        suppl = Chem.SDMolSupplier(ligand_file)
        ligand = suppl[0]

        ligand_atoms = []
        conf = ligand.GetConformer()
        for atom in ligand.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            ligand_atoms.append((atom.GetSymbol(), pos.x, pos.y, pos.z))

        clashes = []

        # Calculate atomic distances and detect clashes
        for p_atom in protein_atoms:
            p_pos = p_atom.get_coord()  # Protein atom coordinates (numpy array)
            for l_atom in ligand_atoms:
                l_pos = [l_atom[1], l_atom[2], l_atom[3]]  # Ligand atom coordinates
                distance = (
                    (p_pos[0] - l_pos[0]) ** 2
                    + (p_pos[1] - l_pos[1]) ** 2
                    + (p_pos[2] - l_pos[2]) ** 2
                ) ** 0.5

                if distance < clashing_dist:
                    clashes.append((p_atom, l_atom, distance))
        clashes_len = len(clashes)
        del (
            protein,
            ligand,
            u,
            p_atom,
            l_atom,
            distance,
            ligand_atoms,
            suppl,
            conf,
            pos,
            protein_atoms,
            parser,
            fp_df,
        )
    else:
        fp_df_type1 = None
        fp_df_type2 = None
        clashes_len = None
        del protein, ligand, u, fp_df
    return fp_df_type1, fp_df_type2, clashes_len


def merge_interaction_affinity(
    directory,
    output_file,
    cnnpose_threshold,
    cnnaffinity_threshold,
    affinity_threshold,
    type1_threshold,
    type2_threshold,
    clashing_count,
):
    """
    Merge confidence score and contact files into a single output CSV.
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    csv_files.sort(key=natural_sort_key)

    # Load the relevant CSV files
    df1 = pd.read_csv(glob.glob(os.path.join(directory, f"*docking_score.csv"))[0])
    df2 = pd.read_csv(glob.glob(os.path.join(directory, f"*interaction.csv"))[0])
    df1 = pd.merge(df1, df2, on="Compounds", how="outer")

    type1_column = [col for col in df1.columns if re.match(r"Similarity-type1.*", col)]
    type2_column = [col for col in df1.columns if re.match(r"Similarity-type2.*", col)]
    clashing_column = [col for col in df1.columns if re.match(r"Clashing.*", col)]
    final_name = os.path.basename(directory)

    cnnpose_column = [col for col in df1.columns if re.match(r"CNNpose.*", col)]
    cnnaffinity_column = [col for col in df1.columns if re.match(r"CNNaffinity.*", col)]
    affinity_column = [col for col in df1.columns if re.match(r"Affinity.*", col)]

    if (
        type1_column
        and type2_column
        and cnnaffinity_column
        and cnnpose_column
        and affinity_column
        and clashing_column
    ):
        # Calculate 'Final' column
        df1[f"Final_{final_name}"] = (
            (df1[cnnpose_column[0]] >= cnnpose_threshold)
            & (df1[cnnaffinity_column[0]] >= cnnaffinity_threshold)
            & (df1[affinity_column[0]] <= affinity_threshold)
            & (df1[type1_column[0]] >= type1_threshold)
            & (df1[type2_column[0]] >= type2_threshold)
            & (df1[clashing_column[0]] <= clashing_count)
        ).astype(int)
    else:
        print("Required columns not found in the data.")

    # Save the merged DataFrame to a new CSV file
    output_path = os.path.join(directory, f"{os.path.basename(directory)}_final.csv")

    df1.to_csv(output_path, index=False)
    print(f"Merged file saved to {output_path}")


def interaction_similarity(row1, row2):
    intersection = np.sum(np.logical_and(row1, row2))  # Common 1's
    reference = np.count_nonzero(row2)
    simi_ratio = intersection / reference
    return simi_ratio


def process_ligand_diffdock(ligand_file, args, total_sasa):
    cmd.load(args.protein_file, "protein")
    if args.reference_ligand:
        cmd.load(args.reference_ligand, "ref_ligand")
        cmd.select(
            "binding_site", f"byres (ref_ligand within {args.distance} of protein)"
        )
    elif args.residues:
        binding_site_selection_str = f"resi {' or resi '.join(args.residues)}"
        cmd.select("binding_site", binding_site_selection_str)
    else:
        raise ValueError(
            "Either a reference ligand or residues must be provided to define the binding site."
        )

    if ligand_file.endswith(".sdf"):
        ligand_path = os.path.join(args.ligand_dir, ligand_file)

        if validate_ligand(ligand_path):
            cmd.load(ligand_path, "ligand")
            contact_area, occupation_percentage = calculate_contact_surface_area(
                "ligand", "binding_site"
            )
            atom_percentage_in_site = percentage_atoms_in_site(
                "ligand", "binding_site", args.distance
            )
            cmd.delete("ligand")
            return [
                ligand_file[:-4],
                contact_area,
                occupation_percentage,
                atom_percentage_in_site,
            ]
        else:
            print(f"Invalid ligand file: {ligand_file}")
            return [ligand_file[:-4], total_sasa, 0, 0]
    return None


def process_ligand_gnina(ligand_file, args, reference_fp_type1, reference_fp_type2):
    if ligand_file.endswith(".sdf"):
        ligand_path = os.path.join(args.ligand_dir, ligand_file)

        if validate_ligand(ligand_path):
            ligand_fp_type1, ligand_fp_type2, clash_count = interaction_fingerprint(
                protein_file=args.protein_file,
                ligand_file=ligand_path,
                distance=args.distance,
                clashing_dist=args.clashing_dist,
            )
            if clash_count != None:
                if args.reference_ligand:
                    aligned_ligand_fp_type1 = ligand_fp_type1.reindex(
                        columns=ligand_fp_type1.columns.union(
                            reference_fp_type1.columns, sort=False
                        ),
                        fill_value=0,
                    )
                    aligned_reference_fp_type1 = reference_fp_type1.reindex(
                        columns=ligand_fp_type1.columns.union(
                            reference_fp_type1.columns, sort=False
                        ),
                        fill_value=0,
                    )
                    aligned_ligand_fp_type2 = ligand_fp_type2.reindex(
                        columns=ligand_fp_type2.columns.union(
                            reference_fp_type2.columns, sort=False
                        ),
                        fill_value=0,
                    )
                    aligned_reference_fp_type2 = reference_fp_type2.reindex(
                        columns=ligand_fp_type2.columns.union(
                            reference_fp_type2.columns, sort=False
                        ),
                        fill_value=0,
                    )
                    overlap_type1 = interaction_similarity(
                        aligned_ligand_fp_type1.values[0],
                        aligned_reference_fp_type1.values[0],
                    )
                    overlap_type2 = interaction_similarity(
                        aligned_ligand_fp_type2.values[0],
                        aligned_reference_fp_type2.values[0],
                    )
                elif args.residues:
                    overlap_type1 = 0
                    overlap_type2 = len(ligand_fp_type2.values[0]) / len(args.residues)
            else:
                overlap_type1 = None
                overlap_type2 = None
            return [ligand_file[:-4], overlap_type1, overlap_type2, clash_count]
        del (
            aligned_ligand_fp_type1,
            aligned_ligand_fp_type2,
            reference_fp_type1,
            reference_fp_type2,
        )
    return None


def main(args):
    batch_size = args.batch
    cpu = args.cpu
    if args.metric == "diffdock":
        cmd.load(args.protein_file, "protein")

        if args.reference_ligand:
            cmd.load(args.reference_ligand, "ref_ligand")
            cmd.select(
                "binding_site", f"byres (ref_ligand within {args.distance} of protein)"
            )
        elif args.residues:
            binding_site_selection_str = f"resi {' or resi '.join(args.residues)}"
            cmd.select("binding_site", binding_site_selection_str)
        else:
            raise ValueError(
                "Either a reference ligand or residues must be provided to define the binding site."
            )

        total_sasa = calculate_total_sasa("binding_site")
        ligand_files = sorted(os.listdir(args.ligand_dir), key=natural_sort_key)

        # Determine output file path
        output_directory = args.output_dir if args.output_dir else os.getcwd()
        output_file_path = os.path.join(
            output_directory, f"{os.path.basename(output_directory)}_contact.csv"
        )

        # Write header (overwrite if file exists)
        with open(output_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Compounds",
                    f"Occupation_{os.path.basename(args.output_dir)}",
                    f"%Occupation_{os.path.basename(args.output_dir)}",
                    f"%atoms_{os.path.basename(args.output_dir)}",
                ]
            )

        # Process and append each batch
        for i in range(0, len(ligand_files), batch_size):
            batch = ligand_files[i : i + batch_size]

            # Process the ligands in the current batch
            results = Parallel(n_jobs=cpu, backend="loky")(
                delayed(process_ligand_diffdock)(ligand, args, total_sasa)
                for ligand in batch
            )
            results = [r for r in results if r is not None]

            # Append results directly to CSV
            with open(
                output_file_path, "a", newline=""
            ) as csvfile:  # 'a' for append mode
                writer = csv.writer(csvfile)
                writer.writerows(results)

            # Free memory after each batch
            del results
            gc.collect()
        print("Results saved to:", output_file_path)
        cmd.delete("all")
        merge_confidence_contact(
            args.ligand_dir,
            args.output_dir,
            args.atom_threshold,
            args.occupation_threshold,
            args.confidence_threshold,
        )

    if args.metric == "gnina":
        if args.reference_ligand:
            print(args.clashing_dist)
            reference_fp_type1, reference_fp_type2, reference_clash_count = (
                interaction_fingerprint(
                    protein_file=args.protein_file,
                    ligand_file=args.reference_ligand,
                    distance=args.distance,
                    clashing_dist=args.clashing_dist,
                )
            )
        else:
            raise ValueError(
                "Either a reference ligand or residues must be provided for interaction fingerprint comparison."
            )

        ligand_files = sorted(os.listdir(args.ligand_dir), key=natural_sort_key)

        # Determine output file path
        output_directory = args.output_dir if args.output_dir else os.getcwd()
        output_file_path = os.path.join(
            output_directory, f"{os.path.basename(output_directory)}_interaction.csv"
        )

        # Write header (overwrite if file exists)
        with open(output_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Compounds",
                    f"Similarity-type1_{os.path.basename(args.output_dir)}",
                    f"Similarity-type2_{os.path.basename(args.output_dir)}",
                    f"Clashing<{args.clashing_dist}A_{os.path.basename(args.output_dir)}",
                ]
            )

        # Process and append each batch
        backend = parallel_backend("loky", n_jobs=cpu)
        backend.__enter__()
        try:
            for i in range(0, len(ligand_files), batch_size):
                batch = ligand_files[i : i + batch_size]

                with Parallel(n_jobs=cpu, backend="loky") as parallel:
                    results = parallel(
                        delayed(process_ligand_gnina)(
                            ligand, args, reference_fp_type1, reference_fp_type2
                        )
                        for ligand in batch
                    )

                results = [r for r in results if r is not None]

                with open(output_file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(results)

                # Cleanup memory
                del results, batch
                gc.collect()
        finally:
            backend.__exit__(None, None, None)
            gc.collect()
        print("Results saved to:", output_file_path)
        merge_interaction_affinity(
            args.ligand_dir,
            args.output_dir,
            cnnpose_threshold=args.cnnpose_threshold,
            cnnaffinity_threshold=args.cnnaffinity_threshold,
            affinity_threshold=args.affinity_threshold,
            type1_threshold=args.type1_threshold,
            type2_threshold=args.type2_threshold,
            clashing_count=args.clashing_count,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the surface contact area and atom percentage occupancy of ligands in a protein binding site."
    )
    parser.add_argument(
        "--protein_file", required=True, help="Path to the protein PDB file"
    )
    parser.add_argument("--reference_ligand", help="Path to the reference ligand file")
    parser.add_argument(
        "--ligand_dir", required=True, help="Directory containing ligand files"
    )
    parser.add_argument(
        "--residues",
        nargs="+",
        help="List of residue numbers if no reference ligand is provided",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=5.0,
        help="Distance to define the binding site from the reference ligand",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where the output CSV file will be stored (defaults to current directory)",
    )
    parser.add_argument(
        "--metric",
        choices=["diffdock", "gnina"],
        required=True,
        help="TData nalysis for diffdock or gnina",
    )
    parser.add_argument(
        "--atom_threshold",
        type=float,
        default=80.0,
        help="Threshold for percentage of atoms column (recommeded for DiffDock)",
    )
    parser.add_argument(
        "--occupation_threshold",
        type=float,
        default=50.0,
        help="Threshold for percentage of occupation column (recommeded for DiffDock)",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=-1.5,
        help="Threshold for Confidence_score column (recommeded for DiffDock)",
    )
    parser.add_argument(
        "--cnnpose_threshold",
        type=float,
        default=0.3,
        help="Threshold for CNNpose column (recommeded for GNINA)",
    )
    parser.add_argument(
        "--cnnaffinity_threshold",
        type=float,
        default=0,
        help="Threshold for CNNaffinity column (recommeded for GNINA)",
    )
    parser.add_argument(
        "--affinity_threshold",
        type=float,
        default=0,
        help="Threshold for Affinity column (recommeded for GNINA)",
    )
    parser.add_argument(
        "--type1_threshold",
        type=float,
        default=0,
        help="Threshold for Interaction similarity type 1 - same amino acid and same interaction type (recommeded for GNINA)",
    )
    parser.add_argument(
        "--type2_threshold",
        type=float,
        default=0,
        help="Threshold for Interaction similarity type 2 - same amino acid only (recommeded for GNINA)",
    )
    parser.add_argument(
        "--clashing_dist",
        type=float,
        default=1.5,
        help="Clashing distance threshold fo be considered a molecular clashing (recommeded for GNINA)",
    )
    parser.add_argument(
        "--clashing_count",
        type=int,
        default=2,
        help="Maximum number of acceptable clashing (recommeded for GNINA)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=100,
        help="Maximum number of conformers processed each parallel cycle (default: 100)",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=-1,
        help="Number of CPU for calculation (default: all)",
    )

    args = parser.parse_args()

    main(args)
