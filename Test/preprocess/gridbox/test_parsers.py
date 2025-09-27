# tests/test_parsers.py
import unittest
import tempfile
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - skip if RDKit missing
    Chem = None
    AllChem = None

from prodock.preprocess.gridbox.parsers import parse_text_to_mol


@unittest.skipIf(Chem is None, "RDKit is required for these tests")
class TestParsers(unittest.TestCase):
    def setUp(self):
        self.m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(self.m, randomSeed=0xC0FFEE)
        try:
            AllChem.UFFOptimizeMolecule(self.m)
        except Exception:
            pass
        self.pdbblock = Chem.MolToPDBBlock(self.m)
        molblock = Chem.MolToMolBlock(self.m)
        # create valid SDF-like text (single mol + $$$$)
        self.sdf_text = molblock + "\n$$$$\n"

    def test_parse_pdb_block_from_file(self):
        # write PDB block to a temporary file and parse by path
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mol.pdb"
            p.write_text(self.pdbblock)
            mol = parse_text_to_mol(str(p), fmt=None)
            self.assertIsNotNone(mol)
            self.assertTrue(mol.GetNumAtoms() > 0)

    def test_parse_pdb_block_from_text(self):
        # also test parsing from raw PDB text (pass the block directly)
        mol = parse_text_to_mol(self.pdbblock, fmt="pdb")
        self.assertIsNotNone(mol)
        self.assertTrue(mol.GetNumAtoms() > 0)

    def test_parse_sdf_text_from_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mol.sdf"
            p.write_text(self.sdf_text)
            mol = parse_text_to_mol(str(p), fmt=None)
            self.assertIsNotNone(mol)
            self.assertTrue(mol.GetNumAtoms() > 0)

    def test_parse_sdf_text_direct(self):
        mol = parse_text_to_mol(self.sdf_text, fmt="sdf")
        self.assertIsNotNone(mol)
        self.assertTrue(mol.GetNumAtoms() > 0)


if __name__ == "__main__":
    unittest.main()
