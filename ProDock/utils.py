from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from typing import Optional
from rdkit.Chem.MolStandardize import rdMolStandardize


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


def smiles_from_mol(mol: Mol) -> str:
    """
    Converts an RDKit Mol object to a SMILES string.

    Parameters:
    ----------
    mol : Mol
        An RDKit Mol object.

    Returns:
    -------
    str
        The SMILES string representation of the molecule.

    Notes:
    -----
    SMILES (Simplified Molecular Input Line Entry System) is a notation that allows a user to specify a chemical
    structure using a series of printable characters.
    """
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles if smiles else None
    except Exception as e:
        print(f"Error converting Mol to SMILES: {e}")  # Optionally log the error
        return None


def mol_from_pdb(pdb_file: str) -> str:
    """
    Converts a PDB file to an RDKit Mol object.

    Parameters:
    ----------
    pdb_file : str
        The path to the PDB file.

    Returns:
    -------
    str
        The SMILES string representation of the molecule.
    """
    try:
        mol = Chem.MolFromPDBFile(pdb_file)
        return mol if mol else None
    except Exception as e:
        print(f"Error converting PDB to Mol: {e}")  # Optionally log the error
        return None


def mol_from_mol(mol_file: str) -> str:
    """
    Converts a PDB file to an RDKit Mol object.

    Parameters:
    ----------
    pdb_file : str
        The path to the PDB file.

    Returns:
    -------
    str
        The SMILES string representation of the molecule.
    """
    try:
        mol = Chem.MolFromMolFile(mol_file)
        return mol if mol else None
    except Exception as e:
        print(f"Error converting PDB to Mol: {e}")  # Optionally log the error
        return None


def mol_from_mol2(mol2_file: str) -> str:
    """
    Converts a mol2 file to an RDKit Mol object.

    Parameters:
    ----------
    mol2_file : str
        The path to the mol2 file.

    Returns:
    -------
    str
        The SMILES string representation of the molecule.
    """
    try:
        mol = Chem.MolFromMol2File(mol2_file)
        return mol if mol else None
    except Exception as e:
        print(f"Error converting mol2 to Mol: {e}")  # Optionally log the error
        return None


def mol_from_sdf(sdf_file: str) -> str:
    """
    Converts a SDF file to an RDKit Mol object.

    Parameters:
    ----------
    sdf_file : str
        The path to the SDF file.

    Returns:
    -------
    str
        The SMILES string representation of the molecule.
    """
    try:
        mol = Chem.SDMolSupplier(sdf_file)
        return mol if mol else None
    except Exception as e:
        print(f"Error converting SDF to Mol: {e}")  # Optionally log the error
        return None


def mol_from_xyz(xyz_file: str) -> str:
    """
    Converts an XYZ file to an RDKit Mol object.

    Parameters:
    ----------
    xyz_file : str
        The path to the XYZ file.

    Returns:
    -------
    str
        The SMILES string representation of the molecule.
    """
    try:
        mol = Chem.MolFromXYZFile(xyz_file)
        return mol if mol else None
    except Exception as e:
        print(f"Error converting XYZ to Mol: {e}")  # Optionally log the error
        return None


def canonicalize_smiles(smiles: str) -> str:
    while True:
        try:
            canon_smi = Chem.CanonSmiles(smiles)
            break
        except Exception:
            canon_smi = None
            break
    return canon_smi


def standardize(smiles: str) -> str:
    # Code borrowed from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    while True:
        try:
            mol = Chem.MolFromSmiles(smiles)
            clean_mol = rdMolStandardize.Cleanup(mol)
            # if many fragments, get the "parent" (the actual mol we are interested in)
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
            # try to neutralize molecule
            uncharger = (
                rdMolStandardize.Uncharger()
            )  # annoying, but necessary as no convenience method exists
            uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
            # note that no attempt is made at reionization at this step
            # nor at ionization at some pH (rdkit has no pKa caculator)
            # the main aim to to represent all molecules from different sources
            # in a (single) standard way, for use in ML, catalogue, etc.
            te = rdMolStandardize.TautomerEnumerator()  # idem
            taut_uncharged_parent_clean_mol = te.Canonicalize(
                uncharged_parent_clean_mol
            )
            break
        except Exception:
            taut_uncharged_parent_clean_mol = None
            break
    return taut_uncharged_parent_clean_mol


def standardize_smi(mol):
    while True:
        try:
            std_smi = Chem.MolToSmiles(mol)
            break
        except Exception:
            std_smi = None
            break
    return std_smi
