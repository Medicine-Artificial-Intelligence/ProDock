import pandas as pd

# Load your CSV file into a DataFrame
df = pd.read_csv(
    "/home/labhhc3/Documents/Workspace/D18/Son/gnina/project/YEAST4_HTS/Ligand/HTS_ligand.csv"
)


# Assuming your SMILES column is named 'Smiles', modify accordingly
def clean_smiles(smiles):
    if isinstance(smiles, str):  # Check if the value is a string
        parts = smiles.split(".")

        # Remove empty strings or irrelevant parts and keep the largest one
        # You could also use other logic here, like length of SMILES or first non-ion part
        cleaned = max(parts, key=len)  # Keep the longest part as the main structure

        return cleaned
    return smiles


# Apply the function to the 'Smiles' column and overwrite it with cleaned data
df["Smiles"] = df["Smiles"].apply(clean_smiles)

# Save the cleaned data back to the same CSV file or a new one
df.to_csv(
    "/home/labhhc3/Documents/Workspace/D18/Son/gnina/project/YEAST4_HTS/Ligand/HTS_ligand_cleaned.csv",
    index=False,
)
