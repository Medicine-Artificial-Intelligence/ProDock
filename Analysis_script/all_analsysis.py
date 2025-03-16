import os
import glob
import shutil
import argparse
import fnmatch
import csv
import re
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from joblib import Parallel, delayed


def natural_sort_key(s):
    normalized = re.sub(r"[-_]", "", s)  # Treat '-' and '_' equivalently
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", normalized)
    ]


def custom_sort_key(value):

    if isinstance(value, str) and value:
        first_char = value[0]
        if not first_char.isalnum():  # Special character
            priority = 0
        elif first_char.isdigit():  # Number
            priority = 1
        elif first_char.islower():  # Lowercase
            priority = 2
        elif first_char.isupper():  # Uppercase
            priority = 3
        else:
            priority = 4  # Fallback for unexpected cases
    else:
        # Non-string or empty string values get the highest priority
        priority = 5

    # Return a tuple with tshe priority and natural sort key
    return (priority, natural_sort_key(value) if isinstance(value, str) else "")


def extract_confidence_score(filename):
    match = re.search(r"confidence(.+)\.sdf", filename)
    if match:
        return match.group(1)
    return None  # Default if no confidence number or string is found


def compute_solvation_energy(sdf_file, filename, folder):
    # pdb_file = os.path.join(folder,f"{filename}.pdb")
    pqr_file = os.path.join(folder, f"{filename}.pqr")
    apbs_input = os.path.join(folder, f"{filename}.IN")
    apbs_output = os.path.join(folder, f"{filename}.OUT")
    try:
        # Step 1: Convert SDF to PQR
        subprocess.run(
            ["obabel", sdf_file, "-O", pqr_file, "--partialcharge gasteiger"],
            check=True,
        )

        # Step 2: Create APBS input file
        apbs_content = f"""
read
    mol pqr {pqr_file}
end
elec name solvated
    mg-auto
    dime 65 65 65
    cglen 40.0 40.0 40.0
    fglen 30.0 30.0 30.0
    mol 1
    cgcent mol 1  
    fgcent mol 1 
    lpbe
    bcfl sdh
    pdie 2.0
    sdie 78.5
    chgm spl2
    srfm smol
    srad 1.4
    sdens 10.0
    temp 298.15
    calcenergy total
    calcforce no
end
elec name vacuum
    mg-auto
    dime 65 65 65
    cglen 40.0 40.0 40.0
    fglen 30.0 30.0 30.0
    mol 1
    cgcent mol 1  
    fgcent mol 1  
    lpbe
    bcfl sdh
    pdie 2.0
    sdie 1.0
    chgm spl2
    srfm mol
    srad 1.4
    sdens 10.0
    temp 298.15
    calcenergy total
    calcforce no
end
"""
        with open(apbs_input, "w") as f:
            f.write(apbs_content.strip())

        # Step 4: Run APBS
        subprocess.run(["apbs", apbs_input], stdout=open(apbs_output, "w"), check=True)
        # Step 5: Extract solvation energy from APBS output
        solv_energy = None
        with open(apbs_output, "r") as file:
            for line in file:
                if "Total electrostatic energy" in line:
                    solv_energy = float(line.split()[-2])  # Extract numerical value
                    break

        # Step 6: Cleanup intermediate files
        os.remove(pqr_file)
        os.remove(apbs_input)
        os.remove(apbs_output)

        return round(solv_energy * 0.239, 2)
    except subprocess.CalledProcessError as e:
        return None
    except FileNotFoundError as e:
        return None


def process_file(root, file, all_dir, num_conform):
    results = []
    for i in range(1, num_conform + 1):  # Start from 1 if ranks are 1-indexed
        pattern = f"rank{i}_*.sdf"
        if fnmatch.fnmatch(file, pattern):
            subfolder_name = os.path.basename(root)
            source_file_path = os.path.join(root, file)
            new_file_name = f"{subfolder_name}.sdf"
            rank_dir = os.path.join(all_dir, f"rank{i}")
            destination_file_path = os.path.join(rank_dir, new_file_name)

            shutil.copy2(source_file_path, destination_file_path)
            print(f"Copied and renamed {file} to {destination_file_path}")

            confidence_score = extract_confidence_score(file)
            supplier = Chem.SDMolSupplier(destination_file_path)

            for mol in supplier:
                if mol != None:
                    minimized_affinity = (
                        mol.GetProp("minimizedAffinity")
                        if mol.HasProp("minimizedAffinity")
                        else None
                    )
                    cnn_affinity = (
                        mol.GetProp("CNNaffinity")
                        if mol.HasProp("CNNaffinity")
                        else None
                    )
                    cnn_score = (
                        mol.GetProp("CNNscore") if mol.HasProp("CNNscore") else None
                    )
                else:
                    minimized_affinity = None
                    cnn_affinity = None
                    cnn_score = None

            if minimized_affinity is not None:
                solvation_energy = compute_solvation_energy(
                    os.path.join(rank_dir, new_file_name), subfolder_name, rank_dir
                )
                results.append(
                    (
                        "score",
                        i,
                        (
                            subfolder_name,
                            float(minimized_affinity),
                            float(cnn_score),
                            float(cnn_affinity),
                            solvation_energy,
                        ),
                    )
                )

            if confidence_score is not None:
                results.append(("confidence", i, (subfolder_name, confidence_score)))

    return results


def copy_and_rename_files(source_dir, destination_dir, num_conform, n_jobs):
    final_folder_name = os.path.basename(os.path.normpath(source_dir))
    all_dir = os.path.join(destination_dir, "all", final_folder_name)

    if not os.path.exists(all_dir):
        os.makedirs(all_dir)
        print(f"Created 'all' directory: {all_dir}")

    for i in range(num_conform):
        rank_dir = os.path.join(all_dir, f"rank{i+1}")
        if not os.path.exists(rank_dir):
            os.makedirs(rank_dir)
            print(f"Created destination directory: {rank_dir}")

    compound_data = [[] for _ in range(num_conform + 1)]
    compound_data_score = [[] for _ in range(num_conform + 1)]
    confidence_dir = os.path.join(all_dir, "Confidence_score")
    os.makedirs(confidence_dir, exist_ok=True)

    file_paths = []
    for root, dir, files in os.walk(source_dir):
        if dir != "all":
            files.sort()  # Ensure files are sorted before parallel processing
            for file in files:
                file_paths.append((root, file))

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(root, file, all_dir, num_conform)
        for root, file in file_paths
    )

    # Collect results from parallel execution
    diffdock_count = 0
    gnina_count = 0
    for result_list in results:
        for result in result_list:
            category, rank, data = result
            if category == "confidence":
                compound_data[rank].append(data)
                diffdock_count += 1
            elif category == "score":
                compound_data_score[rank].append(data)
                gnina_count += 1

    # Export data based on the metric
    if diffdock_count != 0:  # Export only confidence scores
        metric_def = "diffdock"
        for rank_data in compound_data[1:]:
            rank_data.sort(key=lambda x: x[0])

        dfs = []
        for i in range(1, num_conform + 1):
            csv_path = os.path.join(
                all_dir, f"rank{i}", f"rank{i}_confidence_score.csv"
            )
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Compounds", f"Confidence_score_rank{i}"])
                writer.writerows(compound_data[i])
            df = pd.read_csv(csv_path)
            df.set_index("Compounds", inplace=True)
            dfs.append(df)

        combined_df = pd.concat(dfs, axis=1, sort=False)
        combined_csv_path = os.path.join(
            confidence_dir, f"{final_folder_name}_confidence_score.csv"
        )
        combined_df.to_csv(combined_csv_path)

    if gnina_count != 0:  # Export only docking scores
        metric_def = "gnina"
        for rank_data in compound_data_score[1:]:
            rank_data.sort(key=lambda x: x[0])

        dfs = []
        for i in range(1, num_conform + 1):
            csv_path = os.path.join(all_dir, f"rank{i}", f"rank{i}_docking_score.csv")
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Compounds",
                        f"Affinity_rank{i}",
                        f"CNNpose_rank{i}",
                        f"CNNaffinity_rank{i}",
                        f"Solvation_rank{i}",
                    ]
                )
                writer.writerows(compound_data_score[i])
            df = pd.read_csv(csv_path)
            df.set_index("Compounds", inplace=True)
            dfs.append(df)

        combined_df = pd.concat(dfs, axis=1, sort=False)
        combined_csv_path = os.path.join(
            confidence_dir, f"{final_folder_name}_docking_score.csv"
        )
        combined_df.to_csv(combined_csv_path)

    return metric_def


def run_occupation(
    metric_input,
    protein_file,
    input_dir,
    n_conf,
    distance,
    atom_threshold,
    occupation_threshold,
    confidence_threshold,
    cnnpose_threshold,
    cnnaffinity_threshold,
    affinity_threshold,
    type1_threshold,
    type2_threshold,
    clashing_dist,
    clashing_count,
    cpu,
    batch_size,
    reference_ligand=None,
    residues=None,
):
    analysis_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "single_analysis.py"
    )

    # Loop through ranks 1 to number of conformations
    for i in range(1, n_conf + 1):
        act_input_dir = os.path.join(input_dir, f"rank{i}")
        output_dir = act_input_dir  # Assuming output_dir is the same as ligand_dir

        # Prepare the command
        cmd = [
            "python",
            str(analysis_script),
            "--protein_file",
            str(protein_file),
            "--ligand_dir",
            str(act_input_dir),
            "--output_dir",
            str(output_dir),
            "--distance",
            str(distance),
            "--metric",
            str(metric_input),
            "--cpu",
            str(cpu),
            "--batch",
            str(batch_size),
        ]

        # Add conditional arguments
        if reference_ligand:
            cmd.extend(["--reference_ligand", str(reference_ligand)])
        else:
            cmd.extend(["--residues", str(residues)])
        if atom_threshold:
            cmd.extend(["--atom_threshold", str(atom_threshold)])
        if occupation_threshold:
            cmd.extend(["--occupation_threshold", str(occupation_threshold)])
        if confidence_threshold:
            cmd.extend(["--confidence_threshold", str(confidence_threshold)])
        if cnnpose_threshold:
            cmd.extend(["--cnnpose_threshold", str(cnnpose_threshold)])
        if cnnaffinity_threshold:
            cmd.extend(["--cnnaffinity_threshold", str(cnnaffinity_threshold)])
        if affinity_threshold:
            cmd.extend(["--affinity_threshold", str(affinity_threshold)])
        if type1_threshold:
            cmd.extend(["--type1_threshold", str(type1_threshold)])
        if type2_threshold:
            cmd.extend(["--type2_threshold", str(type2_threshold)])
        if clashing_dist:
            cmd.extend(["--clashing_dist", str(clashing_dist)])
        if clashing_count:
            cmd.extend(["--clashing_count", str(clashing_count)])
        # Execute the command
        # print(f"command: {cmd}") #Delete # if you want to double check
        subprocess.run(cmd)

        # Print completion of the task
        print(f"Completed analysis for rank {i}.")


def merge_diffdock_matrix(
    base_directory, atom=None, occupation=None, confidence=None, distance=None
):
    # Define the output directory
    output_directory = os.path.join(base_directory, "Visualization")
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Initialize an empty DataFrame for storing merged data
    merged_df = pd.DataFrame()

    # Walk through the directory to find relevant CSV files
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith("_matrix.csv"):
                # print(file)
                file_path = os.path.join(root, file)
                # Extract the * part of the file name
                file_identifier = file.split("_matrix.csv")[0]
                # Read the CSV file
                csv_data = pd.read_csv(file_path)
                # Select required columns and rename the occupation column
                csv_data = csv_data[["Compounds", "Best_satisfied_occupation"]]
                csv_data.rename(
                    columns={"Best_satisfied_occupation": file_identifier}, inplace=True
                )
                # Merge with the main DataFrame
                if merged_df.empty:
                    merged_df = csv_data
                else:
                    merged_df = pd.merge(
                        merged_df, csv_data, on="Compounds", how="outer"
                    )
                merged_df = merged_df.drop_duplicates()

    sorted_columns = sorted(merged_df.columns, key=custom_sort_key)
    merged_df = merged_df[sorted_columns]  # Reorder columns
    # Reorder compounds
    new_order = ["Compounds"] + [c for c in merged_df.columns if c != "Compounds"]
    merged_df = merged_df[new_order]

    # Sort rows based on the first column using the custom sorting function
    merged_df = merged_df.sort_values(
        by=merged_df.columns[0], key=lambda col: col.map(custom_sort_key)
    )

    # Identify columns containing the word 'wildtype*'
    wildtype_columns = [
        col for col in merged_df.columns if re.search(r"wildtype.*", col, re.IGNORECASE)
    ]

    if wildtype_columns:
        # Separate the columns into three groups: first column, wildtype columns, and others
        first_column = [merged_df.columns[0]]
        other_columns = [
            col
            for col in merged_df.columns
            if col not in wildtype_columns and col != merged_df.columns[0]
        ]

        # New column order: First column, then wildtype columns, then the rest
        columns = first_column + wildtype_columns + other_columns

        # Reorder the DataFrame columns
        merged_df = merged_df[columns]
    output_file_path = os.path.join(
        output_directory,
        f"all_best_satisfied_occupation_atom{atom}_occupation{occupation}_confidence{confidence}_distance{distance}.csv",
    )
    merged_df.to_csv(output_file_path, index=False)

    return merged_df


def merge_gnina_matrix(
    base_directory,
    cnnpose_threshold,
    cnnaffinity_threshold,
    affinity_threshold,
    type1_threshold,
    type2_threshold,
    distance=None,
):
    # Define the output directory
    output_directory = os.path.join(base_directory, "Visualization")
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Initialize an empty DataFrame for storing merged data
    merged_df = pd.DataFrame()

    # Walk through the directory to find relevant CSV files
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith("_affinity_matrix.csv"):
                # print(file)
                file_path = os.path.join(root, file)
                # Extract the * part of the file name
                file_identifier = file.split("_affinity_matrix.csv")[0]
                # Read the CSV file
                csv_data = pd.read_csv(file_path)
                # Select required columns and rename the occupation column
                csv_data = csv_data[["Compounds", "Best_satisfied_affinity"]]
                csv_data.rename(
                    columns={"Best_satisfied_affinity": file_identifier}, inplace=True
                )
                # Merge with the main DataFrame
                if merged_df.empty:
                    merged_df = csv_data
                else:
                    merged_df = pd.merge(
                        merged_df, csv_data, on="Compounds", how="outer"
                    )
                merged_df = merged_df.drop_duplicates()

    sorted_columns = sorted(merged_df.columns, key=custom_sort_key)
    merged_df = merged_df[sorted_columns]  # Reorder columns
    # Reorder compounds
    new_order = ["Compounds"] + [c for c in merged_df.columns if c != "Compounds"]
    merged_df = merged_df[new_order]

    # Sort rows based on the first column using the custom sorting function
    merged_df = merged_df.sort_values(
        by=merged_df.columns[0], key=lambda col: col.map(custom_sort_key)
    )

    # Identify columns containing the word 'wildtype*'
    wildtype_columns = [
        col for col in merged_df.columns if re.search(r"wildtype.*", col, re.IGNORECASE)
    ]

    if wildtype_columns:
        # Separate the columns into three groups: first column, wildtype columns, and others
        first_column = [merged_df.columns[0]]
        other_columns = [
            col
            for col in merged_df.columns
            if col not in wildtype_columns and col != merged_df.columns[0]
        ]

        # New column order: First column, then wildtype columns, then the rest
        columns = first_column + wildtype_columns + other_columns

        # Reorder the DataFrame columns
        merged_df = merged_df[columns]
    output_file_path = os.path.join(
        output_directory,
        f"all_best_satisfied_affinity_cnnpose{cnnpose_threshold}_cnnaffinity{cnnaffinity_threshold}_affinity{affinity_threshold}_type1{type1_threshold}_type2{type2_threshold}_distance{distance}.csv",
    )
    merged_df.to_csv(output_file_path, index=False)

    return merged_df, "Affinity"


def visualization_gnina(
    base_directory,
    merged_df,
    name_df,
    cnnpose_threshold,
    cnnaffinity_threshold,
    affinity_threshold,
    type1_threshold,
    type2_threshold,
    distance=None,
):
    # Draw the visualization matrix

    # Extract numeric data only (excluding the first column)

    numeric_data = merged_df.iloc[:, 1:]

    # Replace blank cells with NaN
    numeric_data.replace("", np.nan, inplace=True)

    # Convert all numeric data to float and round to 2 decimal places
    numeric_data = numeric_data.astype(float).round(2)

    # Create a figure and axis
    rows, cols = numeric_data.shape
    cell_width = 2.3  # Adjust this to make cells narrower
    cell_height = 0.75  # Adjust this to make cells taller
    fig, ax = plt.subplots(figsize=(cols * cell_width, rows * cell_height))

    font_size = rows * 0.8

    # Create a heatmap with an inverted colormap
    heatmap = ax.imshow(
        numeric_data,
        cmap="viridis_r",  # Reversed colormap to indicate higher values are worse
        aspect="auto",
        interpolation="nearest",
    )

    # Set axis labels
    ax.set_xticks(np.arange(len(numeric_data.columns)))
    ax.set_yticks(np.arange(len(merged_df["Compounds"])))
    ax.set_xticklabels(
        numeric_data.columns, rotation=45, ha="right", fontsize=font_size
    )
    ax.set_yticklabels(merged_df["Compounds"], fontsize=font_size)

    # Add white patches for NaN values and display rounded values
    for i in range(numeric_data.shape[0]):
        for j in range(numeric_data.shape[1]):
            value = numeric_data.iloc[i, j]
            if np.isnan(value):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="white"))
            else:
                # Get the background color for the cell
                color = heatmap.cmap(heatmap.norm(value))
                # Compute the text color based on luminance
                text_color = "white" if np.mean(color[:3]) < 0.5 else "black"
                # Add text to the cell
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=font_size * 0.9,
                    color=text_color,
                )

    # Add a colorbar
    cbar = plt.colorbar(heatmap, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Affinity", fontsize=font_size * 1.2, fontweight="bold")
    cbar.ax.tick_params(labelsize=font_size * 0.9)
    cbar.ax.invert_yaxis()
    ax.set_xlabel("Proteins", fontsize=font_size * 1.2, fontweight="bold", labelpad=15)
    ax.set_ylabel("Compounds", fontsize=font_size * 1.2, fontweight="bold", labelpad=15)
    # Set title and layout
    original_title = "Affinity of the screening compounds (y axis) and the protein targets (x axis) heatmap"
    # Dynamically wrap the title into two lines
    wrapped_title = "\n".join(textwrap.wrap(original_title, width=cols * 7))

    plt.title(wrapped_title, fontsize=font_size * 1.5, pad=40, fontweight="bold")
    plt.tight_layout()

    # Save the visualization as an image
    output_path = os.path.join(
        base_directory,
        "Visualization",
        f"all_best_satisfied_affinity_cnnpose{cnnpose_threshold}_cnnaffinity{cnnaffinity_threshold}_affinity{affinity_threshold}_type1{type1_threshold}_type2{type2_threshold}_distance{distance}.tiff",
    )
    plt.savefig(output_path, dpi=300, format="tiff")
    output_path = os.path.join(
        base_directory,
        "Visualization",
        f"all_best_satisfied_affinity_cnnpose{cnnpose_threshold}_cnnaffinity{cnnaffinity_threshold}_affinity{affinity_threshold}_type1{type1_threshold}_type2{type2_threshold}_distance{distance}.png",
    )
    plt.savefig(output_path, dpi=300, format="png")
    return os.path.join(base_directory, "Visualization")


def visualization_diffdock(
    base_directory,
    merged_df,
    atom=None,
    occupation=None,
    confidence=None,
    distance=None,
):
    # Draw the visualization matrix

    # Extract numeric data only (excluding the first column)

    numeric_data = merged_df.iloc[:, 1:]

    # Replace blank cells with NaN
    numeric_data.replace("", np.nan, inplace=True)

    # Convert all numeric data to float and round to 2 decimal places
    numeric_data = numeric_data.astype(float).round(2)

    # Create a figure and axis
    rows, cols = numeric_data.shape
    cell_width = 2.3  # Adjust this to make cells narrower
    cell_height = 0.75  # Adjust this to make cells taller
    fig, ax = plt.subplots(figsize=(cols * cell_width, rows * cell_height))

    font_size = rows * 0.8

    # Create a heatmap with an inverted colormap
    heatmap = ax.imshow(
        numeric_data,
        cmap="viridis_r",  # Reversed colormap to indicate higher values are worse
        aspect="auto",
        interpolation="nearest",
    )

    # Set axis labels
    ax.set_xticks(np.arange(len(numeric_data.columns)))
    ax.set_yticks(np.arange(len(merged_df["Compounds"])))
    ax.set_xticklabels(
        numeric_data.columns, rotation=45, ha="right", fontsize=font_size
    )
    ax.set_yticklabels(merged_df["Compounds"], fontsize=font_size)

    # Add white patches for NaN values and display rounded values
    for i in range(numeric_data.shape[0]):
        for j in range(numeric_data.shape[1]):
            value = numeric_data.iloc[i, j]
            if np.isnan(value):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="white"))
            else:
                # Get the background color for the cell
                color = heatmap.cmap(heatmap.norm(value))
                # Compute the text color based on luminance
                text_color = "white" if np.mean(color[:3]) < 0.5 else "black"
                # Add text to the cell
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=font_size * 0.9,
                    color=text_color,
                )

    # Add a colorbar
    cbar = plt.colorbar(heatmap, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Occupation", fontsize=font_size * 1.2, fontweight="bold")
    cbar.ax.tick_params(labelsize=font_size * 0.9)
    cbar.ax.invert_yaxis()
    ax.set_xlabel("Proteins", fontsize=font_size * 1.2, fontweight="bold", labelpad=15)
    ax.set_ylabel("Compounds", fontsize=font_size * 1.2, fontweight="bold", labelpad=15)
    # Set title and layout
    original_title = "Occupation of the screening compounds (y axis) and the protein targets (x axis) heatmap"
    # Dynamically wrap the title into two lines
    wrapped_title = "\n".join(textwrap.wrap(original_title, width=cols * 7))

    plt.title(wrapped_title, fontsize=font_size * 1.5, pad=40, fontweight="bold")
    plt.tight_layout()

    # Save the visualization as an image
    output_path = os.path.join(
        base_directory,
        "Visualization",
        f"all_best_satisfied_occupation_atom{atom}_occupation{occupation}_confidence{confidence}_distance{distance}.tiff",
    )
    plt.savefig(output_path, dpi=300, format="tiff")
    output_path = os.path.join(
        base_directory,
        "Visualization",
        f"all_best_satisfied_occupation__atom{atom}_occupation{occupation}_confidence{confidence}_distance{distance}.png",
    )
    plt.savefig(output_path, dpi=300, format="png")
    return os.path.join(base_directory, "Visualization")


def merge_final_gnina(base_directory):
    merged_interaction_df = (
        pd.DataFrame()
    )  # Initialize an empty DataFrame for merging contact files
    merged_final_df = (
        pd.DataFrame()
    )  # Initialize an empty DataFrame for merging final files

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_directory):
        # Sort directories in natural order
        dirs.sort(key=natural_sort_key)
        dirs = [d for d in dirs if d.startswith("rank")]
        for dir_name in dirs:
            csv_path = os.path.join(root, dir_name)
            interaction_file = next(
                (f for f in os.listdir(csv_path) if f.endswith("_interaction.csv")),
                None,
            )
            final_file = next(
                (f for f in os.listdir(csv_path) if f.endswith("_final.csv")), None
            )

            # Process contact CSV files
            if interaction_file:
                full_interaction_path = os.path.join(csv_path, interaction_file)
                df_interaction = pd.read_csv(full_interaction_path)
                if merged_interaction_df.empty:
                    merged_interaction_df = df_interaction
                else:
                    merged_interaction_df = pd.merge(
                        merged_interaction_df,
                        df_interaction,
                        on="Compounds",
                        how="outer",
                    )

            # Process final CSV filess
            if final_file:
                full_final_path = os.path.join(csv_path, final_file)
                df_final = pd.read_csv(full_final_path)

                if merged_final_df.empty:
                    merged_final_df = df_final
                else:
                    merged_final_df = pd.merge(
                        merged_final_df, df_final, on="Compounds", how="outer"
                    )
    final_columns = [col for col in merged_final_df.columns if col.startswith("Final_")]
    affinity_columns = [
        col for col in merged_final_df.columns if col.startswith("Affinity_")
    ]
    cnnpose_columns = [
        col for col in merged_final_df.columns if col.startswith("CNNpose_")
    ]
    cnnaffinity_columns = [
        col for col in merged_final_df.columns if col.startswith("CNNaffinity_")
    ]
    type1_columns = [
        col for col in merged_final_df.columns if col.startswith("Similarity-type1_")
    ]
    type2_columns = [
        col for col in merged_final_df.columns if col.startswith("Similarity-type2_")
    ]
    solvation_columns = [
        col for col in merged_final_df.columns if col.startswith("Solvation_")
    ]

    merged_final_df["Satisfied_count"] = merged_final_df[final_columns].sum(axis=1)
    screened_df = merged_final_df[merged_final_df["Satisfied_count"] > 0]["Compounds"]
    screened_withscore_df = []
    matrix_df_affinity = merged_final_df[["Compounds"]].copy()
    matrix_df_affinity[["Best_satisfied_affinity", "Best_satisfied_rank"]] = ""
    matrix_df_cnnpose = merged_final_df[["Compounds"]].copy()
    matrix_df_cnnpose[["Best_satisfied_cnnpose", "Best_satisfied_rank"]] = ""
    matrix_df_cnnaffinity = merged_final_df[["Compounds"]].copy()
    matrix_df_cnnaffinity[["Best_satisfied_cnnaffinity", "Best_satisfied_rank"]] = ""

    for index, row in merged_final_df.iterrows():
        if row["Satisfied_count"] > 0:
            for i in range(len(final_columns)):
                if row[final_columns[i]] != 0:
                    screened_withscore_df.append(
                        [
                            row["Compounds"],
                            f"rank{i+1}",
                            row[affinity_columns[i]],
                            row[cnnpose_columns[i]],
                            row[cnnaffinity_columns[i]],
                            row[type1_columns[i]],
                            row[type2_columns[i]],
                            row[solvation_columns[i]],
                        ]
                    )
                    matrix_df_affinity.at[index, "Best_satisfied_affinity"] = row[
                        affinity_columns[i]
                    ]
                    matrix_df_affinity.at[index, "Best_satisfied_rank"] = f"rank{i+1}"
                    matrix_df_cnnpose.at[index, "Best_satisfied_cnnpose"] = row[
                        cnnpose_columns[i]
                    ]
                    matrix_df_cnnpose.at[index, "Best_satisfied_rank"] = f"rank{i+1}"
                    matrix_df_cnnaffinity.at[index, "Best_satisfied_cnnaffinity"] = row[
                        cnnaffinity_columns[i]
                    ]
                    matrix_df_cnnaffinity.at[index, "Best_satisfied_rank"] = (
                        f"rank{i+1}"
                    )
                    break
    with open(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_screened_withscore.csv",
        ),
        "w",
        newline="",
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Compounds",
                "Best_satisfied_rank",
                "Affinity",
                "CNNpose",
                "CNNaffinity",
                "Similarity-type1",
                "Similarity-type2",
                "Solvation",
            ]
        )
        writer.writerows(screened_withscore_df)
    # Save merged contact DataFrame
    merged_interaction_df.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_interaction.csv",
        ),
        index=False,
    )
    # Save merged final DataFrame
    merged_final_df.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_final.csv",
        ),
        index=False,
    )
    # Save screened DataFrame
    screened_df.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_screened.csv",
        ),
        index=False,
    )
    # Save matrix DataFrame
    matrix_df_affinity.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_affinity_matrix.csv",
        ),
        index=False,
    )
    matrix_df_cnnaffinity.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_cnnaffinity_matrix.csv",
        ),
        index=False,
    )
    matrix_df_cnnpose.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_cnnpose_matrix.csv",
        ),
        index=False,
    )

    return (
        merged_interaction_df,
        merged_final_df,
        screened_df,
        matrix_df_affinity,
        matrix_df_cnnaffinity,
        matrix_df_cnnpose,
    )


def merge_final_diffdock(base_directory):
    merged_contact_df = (
        pd.DataFrame()
    )  # Initialize an empty DataFrame for merging contact files
    merged_final_df = (
        pd.DataFrame()
    )  # Initialize an empty DataFrame for merging final files

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_directory):
        # Sort directories in natural order
        dirs.sort(key=natural_sort_key)
        dirs = [d for d in dirs if d.startswith("rank")]

        for dir_name in dirs:
            csv_path = os.path.join(root, dir_name)
            contact_file = next(
                (f for f in os.listdir(csv_path) if f.endswith("_contact.csv")), None
            )
            final_file = next(
                (f for f in os.listdir(csv_path) if f.endswith("_final.csv")), None
            )

            # Process contact CSV files
            if contact_file:
                full_contact_path = os.path.join(csv_path, contact_file)
                df_contact = pd.read_csv(full_contact_path)
                if merged_contact_df.empty:
                    merged_contact_df = df_contact
                else:
                    merged_contact_df = pd.merge(
                        merged_contact_df, df_contact, on="Compounds", how="outer"
                    )

            # Process final CSV filess
            if final_file:
                full_final_path = os.path.join(csv_path, final_file)
                df_final = pd.read_csv(full_final_path)

                if merged_final_df.empty:
                    merged_final_df = df_final
                else:
                    merged_final_df = pd.merge(
                        merged_final_df, df_final, on="Compounds", how="outer"
                    )
    final_columns = [col for col in merged_final_df.columns if col.startswith("Final_")]
    occupation_columns = [
        col for col in merged_final_df.columns if col.startswith("Occupation_")
    ]
    percent_occupation_columns = [
        col for col in merged_final_df.columns if col.startswith("%Occupation_")
    ]
    atom_columns = [
        col for col in merged_final_df.columns if col.startswith(r"%atoms_")
    ]

    merged_final_df["Satisfied_count"] = merged_final_df[final_columns].sum(axis=1)
    screened_df = merged_final_df[merged_final_df["Satisfied_count"] > 0]["Compounds"]
    screened_withscore_df = []

    matrix_df = merged_final_df[["Compounds"]].copy()
    matrix_df[["Best_satisfied_occupation", "Best_satisfied_rank"]] = ""
    for index, row in merged_final_df.iterrows():
        if row["Satisfied_count"] > 0:
            for i in range(len(final_columns)):
                if row[final_columns[i]] != 0:
                    screened_withscore_df.append(
                        [
                            row["Compounds"],
                            f"rank{i+1}",
                            row[occupation_columns[i]],
                            row[percent_occupation_columns[i]],
                            row[atom_columns[i]],
                        ]
                    )
                    matrix_df.at[index, "Best_satisfied_occupation"] = row[
                        occupation_columns[i]
                    ]
                    matrix_df.at[index, "Best_satisfied_rank"] = f"rank{i+1}"
                    break
    with open(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_screened_withscore.csv",
        ),
        "w",
        newline="",
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Compounds", "Best_satisfied_rank", "Occupation", "%Occupation", "%Atoms"]
        )
        writer.writerows(screened_withscore_df)
    # Save merged contact DataFrame
    merged_contact_df.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_contact.csv",
        ),
        index=False,
    )
    # Save merged final DataFrame
    merged_final_df.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_final.csv",
        ),
        index=False,
    )
    # Save screened DataFrame
    screened_df.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_screened.csv",
        ),
        index=False,
    )
    # Save matrix DataFrame
    matrix_df.to_csv(
        os.path.join(
            base_directory,
            "Confidence_score",
            f"{os.path.basename(base_directory)}_matrix.csv",
        ),
        index=False,
    )
    return merged_contact_df, merged_final_df, screened_df, matrix_df


def main():
    parser = argparse.ArgumentParser(
        description="Copy and rename files based on their rank and subfolder names."
    )
    parser.add_argument(
        "--source_dir",
        required=True,
        help="Specify the source directory path (to 'result' folder only)",
    )
    parser.add_argument(
        "--dest_dir",
        help="Specify the destination directory path, defaults to parent of source_dir/'all'",
    )
    parser.add_argument(
        "--num_conform",
        type=int,
        help="Number of conformations for analysis",
        default=10,
    )
    parser.add_argument(
        "--protein_dir", required=True, help="Path to the protein folder"
    )
    parser.add_argument(
        "--reference_ligand",
        default=None,
        help="Path to the reference foler of ligand sdf or the residue txt files",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=5.0,
        help="Distance to define the binding site from the reference ligand",
    )
    parser.add_argument(
        "--atom_threshold",
        type=float,
        default=80.0,
        help="Threshold for percentage of atoms column",
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
        help="Threshold for Confidence_score column",
    )
    parser.add_argument(
        "--visualization",
        choices=["yes", "y", "no", "n"],
        default="y",
        help="Export diffdock visualization of all targets? [Yes (Y) or No (N)])",
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
    protein_files = []
    if args.dest_dir is None:
        args.dest_dir = args.source_dir
    result_all_dir = []
    for entry in os.listdir(args.source_dir):
        full_entry_path = os.path.join(args.source_dir, entry)
        if (
            entry != "all"
            and entry != "log"
            and os.path.isdir(full_entry_path)
            and entry != "Visualization"
            and entry != "gnina_output"
        ):
            result_each_dir = os.path.join(args.source_dir, entry)
            result_all_dir.append(os.path.join(args.source_dir, entry, "all", entry))

            metric = copy_and_rename_files(
                result_each_dir, args.dest_dir, args.num_conform, args.cpu
            )
            if args.protein_dir:
                try:
                    protein_file = glob.glob(
                        os.path.join(args.protein_dir, f"*{entry}*")
                    )[0]
                    if glob.glob(os.path.join(args.reference_ligand, f"*{entry}*.sdf"))[
                        0
                    ]:
                        ref_ligand_file = glob.glob(
                            os.path.join(args.reference_ligand, f"*{entry}*.sdf")
                        )[0]
                        resi_file = None
                        # print(str(ref_ligand_file))
                    else:
                        with open(
                            glob.glob(
                                os.path.join(args.reference_ligand, f"*{entry}*.txt")
                            )[0],
                            "r",
                        ) as file:
                            resi_file = file.readline().strip()
                        ref_ligand_file = None
                        # print(str(resi_file))

                except IndexError:
                    print(f"No matching file found for {entry} target. Skipping.")
                    continue

                all_dir = os.path.join(args.dest_dir, "all")
                input_dir = os.path.join(all_dir, entry)

                run_occupation(
                    metric_input=metric,
                    protein_file=str(protein_file),
                    input_dir=input_dir,
                    n_conf=args.num_conform,
                    distance=str(args.distance),
                    atom_threshold=str(args.atom_threshold),
                    occupation_threshold=str(args.occupation_threshold),
                    confidence_threshold=str(args.confidence_threshold),
                    reference_ligand=str(ref_ligand_file),
                    residues=str(resi_file),
                    cnnpose_threshold=str(args.cnnpose_threshold),
                    cnnaffinity_threshold=str(args.cnnaffinity_threshold),
                    affinity_threshold=str(args.affinity_threshold),
                    type1_threshold=str(args.type1_threshold),
                    type2_threshold=str(args.type2_threshold),
                    clashing_dist=args.clashing_dist,
                    clashing_count=args.clashing_count,
                    cpu=args.cpu,
                    batch_size=args.batch,
                )
                if metric == "diffdock":
                    merge_final_diffdock(input_dir)
                if metric == "gnina":
                    merge_final_gnina(input_dir)
    if metric == "diffdock":
        merged_df = merge_diffdock_matrix(
            os.path.join(args.dest_dir, "all"),
            atom=args.atom_threshold,
            occupation=args.occupation_threshold,
            confidence=args.confidence_threshold,
            distance=args.distance,
        )
        if args.visualization.lower() in ["yes", "y"]:
            visualization_diffdock(
                os.path.join(args.dest_dir, "all"),
                merged_df,
                atom=args.atom_threshold,
                occupation=args.occupation_threshold,
                confidence=args.confidence_threshold,
                distance=args.distance,
            )
    if metric == "gnina":
        merged_df, name_df = merge_gnina_matrix(
            base_directory=os.path.join(args.dest_dir, "all"),
            cnnpose_threshold=args.cnnpose_threshold,
            cnnaffinity_threshold=args.cnnaffinity_threshold,
            affinity_threshold=args.affinity_threshold,
            type1_threshold=args.type1_threshold,
            type2_threshold=args.type2_threshold,
            distance=args.distance,
        )
        os.path.join(args.dest_dir, "all"),
        if args.visualization.lower() in ["yes", "y"]:
            visualization_gnina(
                os.path.join(args.dest_dir, "all"),
                merged_df,
                name_df,
                cnnpose_threshold=args.cnnpose_threshold,
                cnnaffinity_threshold=args.cnnaffinity_threshold,
                affinity_threshold=args.affinity_threshold,
                type1_threshold=args.type1_threshold,
                type2_threshold=args.type2_threshold,
                distance=args.distance,
            )


if __name__ == "__main__":
    main()
