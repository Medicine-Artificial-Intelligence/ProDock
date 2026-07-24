import pandas as pd
import os
import argparse


def merge_and_sort_csv(gnina_file, diffdock_file, output_dir):
    # Load both CSV files
    df1 = pd.read_csv(gnina_file)
    df1 = df1.rename(columns={"Best_satisfied_rank": "Best_satisfied_rank_GNINA"})
    df2 = pd.read_csv(diffdock_file)
    df2 = df2.rename(columns={"Best_satisfied_rank": "Best_satisfied_rank_DiffDock"})

    # Merge using INNER JOIN (only common columns)
    merged_df = pd.merge(
        df1, df2, how="inner", on="Compounds"
    )  # Keeps only matching rows

    # Ensure "Affinity" and "CNNpose" columns are numeric for sorting
    merged_df["Affinity"] = pd.to_numeric(merged_df["Affinity"], errors="coerce")
    merged_df["CNNpose"] = pd.to_numeric(merged_df["CNNpose"], errors="coerce")
    merged_df["CNNaffinity"] = pd.to_numeric(merged_df["CNNaffinity"], errors="coerce")
    merged_df["Similarity-type1"] = pd.to_numeric(
        merged_df["Similarity-type1"], errors="coerce"
    )
    merged_df["Similarity-type2"] = pd.to_numeric(
        merged_df["Similarity-type2"], errors="coerce"
    )
    merged_df["Solvation"] = pd.to_numeric(merged_df["Solvation"], errors="coerce")
    merged_df["%Occupation"] = pd.to_numeric(merged_df["%Occupation"], errors="coerce")

    merged_df = merged_df.sort_values(
        by=[
            "Affinity",
            "CNNpose",
            "CNNaffinity",
            "Similarity-type1",
            "Similarity-type2",
            "%Occupation",
            "Solvation",
        ],
        ascending=[True, False, False, False, False, False, False],
        na_position="last",
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the sorted dataframe to output direcstory
    output_path = os.path.join(output_dir, f"{args.name}.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"Merged CSV saved at: {output_path}")


if __name__ == "__main__":
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(
        description="Merge two CSV files using an inner join and sort the results."
    )
    parser.add_argument("--gnina_file", type=str, help="Path to the first CSV file")
    parser.add_argument("--diffdock_file", type=str, help="Path to the second CSV file")
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save the merged CSV"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Merge_screened_GNINA_DiffDock",
        help="Name of the output file",
    )

    args = parser.parse_args()

    merge_and_sort_csv(args.gnina_file, args.diffdock_file, args.output_dir)
