import pandas as pd
import re
import os
import argparse


# Define a function to parse the required information from each file
def parse_file_content(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Extract mode 1 data using regular expression
    match = re.search(
        r"^\s*1\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)",
        content,
        re.MULTILINE,
    )
    if match:
        affinity = float(match.group(1))
        intramol = float(match.group(2))
        cnn_pose_score = float(match.group(3))
        cnn_affinity = float(match.group(4))
        return affinity, intramol, cnn_pose_score, cnn_affinity
    return None, None, None, None


# Set up argument parsing
parser = argparse.ArgumentParser(description="Extract and save ligand data to a CSV.")
parser.add_argument(
    "--input_dir", type=str, required=True, help="Directory containing all log files"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=".",
    help="Directory to save the output CSV (defaults to current directory)",
)
parser.add_argument(
    "--output_name",
    type=str,
    default="best_ligand_score",
    help="Name of the csv file of docking score of the best conformations",
)

args = parser.parse_args()

input_directory = args.input_dir
output_directory = args.output_dir

# Collect file data
data = []

# Iterate over each file in the input directory and parse its content
for file_name in os.listdir(input_directory):
    if file_name.endswith(".txt"):
        file_path = os.path.join(input_directory, file_name)
        ligand_name = os.path.splitext(file_name)[0]
        affinity, intramol, cnn_pose_score, cnn_affinity = parse_file_content(file_path)
        data.append(
            {
                "Ligand name": ligand_name,
                "Affinity (kcal/mol)": affinity,
                "Intramol (kcal/mol)": intramol,
                "CNN Pose Score": cnn_pose_score,
                "CNN Affinity": cnn_affinity,
            }
        )

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Save the DataFrame to CSV in the output directory
output_file_path = os.path.join(output_directory, f"{args.output_name}.csv")
df.to_csv(output_file_path, index=False)
# cd ..
print(f"Data saved to {output_file_path}")
