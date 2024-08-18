from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from typing import Optional


def mol_from_smiles(smiles: str) -> Optional[Mol]:
    """
    Converts a SMILES string to an RDKit Mol object.

    Parameters:
    ----------
    smiles : str
        The SMILES string representation of the molecule.

    Returns:
    -------
    Optional[Mol]
        An RDKit Mol object if the SMILES string is valid and conversion is successful; otherwise, None.

    Notes:
    -----
    SMILES (Simplified Molecular Input Line Entry System) is a notation that allows a user to specify a chemical
    structure using a series of printable characters.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol if mol else None
    except Exception as e:
        print(f"Error converting SMILES to Mol: {e}")  # Optionally log the error
        return None
