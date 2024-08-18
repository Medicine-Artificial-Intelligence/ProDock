import unittest
from rdkit.Chem.rdchem import Mol
from ProDock.utils import mol_from_smiles


class TestMolFromSmiles(unittest.TestCase):
    def test_valid_smiles(self):
        smiles = "CCO"  # Ethanol
        result = mol_from_smiles(smiles)
        self.assertIsInstance(result, Mol, "The result should be an instance of Mol")

    def test_invalid_smiles(self):
        smiles = "XYZ"  # Invalid SMILES
        result = mol_from_smiles(smiles)
        self.assertIsNone(result, "The result should be None for invalid SMILES")

    def test_empty_mol(self):
        smiles = "cC"  # Empty string
        result = mol_from_smiles(smiles)
        self.assertIsNone(result, "The result should be None for empty SMILES string")


if __name__ == "__main__":
    unittest.main()
