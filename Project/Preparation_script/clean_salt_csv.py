import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strip salts/counter-ions from a SMILES column in a CSV file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing a SMILES column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the cleaned CSV file.",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="Smiles",
        help="Name of the SMILES column to clean. Defaults to 'Smiles'.",
    )
    return parser.parse_args()


# Assuming your SMILES column is named 'Smiles', modify accordingly
def clean_smiles(smiles):
    if isinstance(smiles, str):  # Check if the value is a string
        parts = smiles.split(".")

        # Remove empty strings or irrelevant parts and keep the largest one
        # You could also use other logic here, like length of SMILES or first non-ion part
        cleaned = max(parts, key=len)  # Keep the longest part as the main structure

        return cleaned
    return smiles


def main():
    args = parse_args()

    # Load your CSV file into a DataFrame
    df = pd.read_csv(args.input)

    # Apply the function to the SMILES column and overwrite it with cleaned data
    df[args.smiles_column] = df[args.smiles_column].apply(clean_smiles)

    # Save the cleaned data back to a CSV file
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
