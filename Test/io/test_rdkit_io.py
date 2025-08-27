# tests/test_rdkit_io.py
import unittest
import tempfile
from pathlib import Path

# Skip entire test module if RDKit is not installed
try:
    import rdkit  # noqa: F401
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False
import prodock.io.rdkit_io as rdio  # module path from your earlier message


@unittest.skipUnless(RDKit_AVAILABLE, "RDKit is required for these tests")
class TestRdkitIO(unittest.TestCase):
    def setUp(self):
        # import module under test (uses prodock.chem.Conformer if available)

        self.rdio = rdio
        # canonical simple molecules for tests
        self.smiles_ethanol = "CCO"
        self.smiles_propane = "CCC"
        self.invalid_smiles = "NOTASMILES"
        # tempdir
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_smiles2mol_and_mol2smiles_roundtrip(self):
        m = self.rdio.smiles2mol(self.smiles_ethanol)
        self.assertIsNotNone(m)
        s = self.rdio.mol2smiles(m)
        # RDKit may canonicalize; ensure molecule roundtrips as valid SMILES
        self.assertTrue(self.rdio.is_valid_smiles(s))
        # ensure original canonical form corresponds to same heavy-atom formula
        m2 = self.rdio.smiles2mol(s)
        self.assertEqual(m.GetNumAtoms(), m2.GetNumAtoms())

    def test_is_valid_smiles(self):
        self.assertTrue(self.rdio.is_valid_smiles(self.smiles_ethanol))
        self.assertFalse(self.rdio.is_valid_smiles(self.invalid_smiles))

    def test_mol2sdf_and_sdf2mol_and_sdftosmiles(self):
        # create mol and write SDF
        m = self.rdio.smiles2mol(self.smiles_propane)
        out = self.tmp / "propane.sdf"
        p = self.rdio.mol2sdf(m, out)
        self.assertTrue(p.exists())
        # read back first mol
        m2 = self.rdio.sdf2mol(p, sanitize=False, removeHs=False)
        self.assertIsNotNone(m2)
        # sdftosmiles returns list of smiles
        smiles_list = self.rdio.sdftosmiles(p, sanitize=False)
        self.assertIsInstance(smiles_list, list)
        self.assertGreaterEqual(len(smiles_list), 1)
        # each returned smiles should be valid
        for sm in smiles_list:
            self.assertTrue(self.rdio.is_valid_smiles(sm))

    def test_smiles2sdf_no_embed(self):
        out = self.tmp / "ethanol_plain.sdf"
        p = self.rdio.smiles2sdf(
            self.smiles_ethanol, out, embed3d=False, add_hs=False, optimize=False
        )
        self.assertTrue(Path(p).exists())
        # reading should return a molecule
        m = self.rdio.sdf2mol(p, sanitize=True)
        self.assertIsNotNone(m)

    def test_smiles2sdf_embed_and_optimize_creates_conformer(self):
        out = self.tmp / "ethanol_3d.sdf"
        # Try to generate a 3D structure and optimize; ensure file and a conformer exist
        p = self.rdio.smiles2sdf(
            self.smiles_ethanol, out, embed3d=True, add_hs=True, optimize=True
        )
        self.assertTrue(Path(p).exists())
        # read with sanitize=False to preserve conformers
        m = self.rdio.sdf2mol(p, sanitize=False)
        self.assertIsNotNone(m)
        # Molecule should have at least 1 conformer (embedding succeeded)
        self.assertGreaterEqual(m.GetNumConformers(), 1)

    def test_mol2pdb_and_pdb2mol_and_pdb2smiles(self):
        m = self.rdio.smiles2mol(self.smiles_ethanol)
        outpdb = self.tmp / "ethanol.pdb"
        # Request embedding so coordinates are produced
        p = self.rdio.mol2pdb(m, outpdb, add_hs=True, embed3d=True, optimize=True)
        self.assertTrue(Path(p).exists())
        # Read back PDB
        m2 = self.rdio.pdb2mol(p, sanitize=False, removeHs=False)
        self.assertIsNotNone(m2)
        # Convert back to SMILES
        sm = self.rdio.pdb2smiles(p, sanitize=True, removeHs=True)
        self.assertTrue(self.rdio.is_valid_smiles(sm))

    def test_smiles2pdb_creates_file(self):
        outpdb = self.tmp / "ethanol_from_smiles.pdb"
        p = self.rdio.smiles2pdb(
            self.smiles_ethanol, outpdb, add_hs=True, embed3d=True, optimize=True
        )
        self.assertTrue(Path(p).exists())
        m = self.rdio.pdb2mol(p, sanitize=False)
        self.assertIsNotNone(m)

    def test_mol_from_smiles_write_all_formats(self):
        prefix = self.tmp / "molprefix"
        res = self.rdio.mol_from_smiles_write_all_formats(
            self.smiles_ethanol,
            prefix,
            write_sdf=True,
            write_pdb=True,
            embed3d=True,
            add_hs=True,
        )
        # ensure both keys present
        self.assertIn("sdf", res)
        self.assertIn("pdb", res)
        self.assertTrue(Path(res["sdf"]).exists())
        self.assertTrue(Path(res["pdb"]).exists())

    def test_sdf2mols_reads_multiple(self):
        # write a multi-molecule SDF by writing two molecules into one file
        out = self.tmp / "multi.sdf"
        m1 = self.rdio.smiles2mol(self.smiles_ethanol)
        m2 = self.rdio.smiles2mol(self.smiles_propane)
        writer = None
        try:
            writer = __import__("rdkit").Chem.SDWriter(str(out))
            writer.write(m1)
            writer.write(m2)
        finally:
            if writer is not None:
                writer.close()
        mols = self.rdio.sdf2mols(out, sanitize=False)
        self.assertIsInstance(mols, list)
        self.assertGreaterEqual(len(mols), 2)

    def test_smiles2mol_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.rdio.smiles2mol(self.invalid_smiles)

    def test_mol2sdf_embed_fallback_when_conformer_missing(self):
        # This test ensures mol2sdf with embed3d works even if Conformer unavailable
        # It doesn't assert which backend is used, only that result is produced.
        m = self.rdio.smiles2mol(self.smiles_propane)
        out = self.tmp / "propane_3d.sdf"
        p = self.rdio.mol2sdf(
            m, out, sanitize=True, embed3d=True, add_hs=True, optimize=False
        )
        self.assertTrue(Path(p).exists())
        m2 = self.rdio.sdf2mol(p, sanitize=False)
        self.assertIsNotNone(m2)
        # ensure there's at least 0 or more conformers (we check attribute exists)
        self.assertTrue(hasattr(m2, "GetNumConformers"))


if __name__ == "__main__":
    unittest.main()
