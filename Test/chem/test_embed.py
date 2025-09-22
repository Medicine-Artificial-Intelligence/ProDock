import unittest
import tempfile
from pathlib import Path

# Skip if RDKit missing
try:
    import rdkit  # noqa: F401
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


@unittest.skipUnless(RDKit_AVAILABLE, "RDKit is required for these tests")
class TestEmbedder(unittest.TestCase):
    def setUp(self):
        from prodock.chem.embed import Embedder

        self.Embedder = Embedder
        self.smiles = ["CCO", "CCC", "c1ccccc1"]

    def test_load_smiles_iterable_and_properties(self):
        e = self.Embedder(seed=123)
        e.load_smiles_iterable(self.smiles)
        self.assertIsInstance(e.smiles, list)
        self.assertEqual(len(e.smiles), len(self.smiles))
        _ = repr(e)

    def test_embed_all_single_conf_default_alg(self):
        e = self.Embedder(seed=42)
        e.load_smiles_iterable(self.smiles)
        # default add_hs=True, embed_algorithm='ETKDGv3'
        e.embed_all(n_confs=1)
        self.assertGreaterEqual(len(e.molblocks), 1)
        self.assertEqual(len(e.mols), len(e.molblocks))
        self.assertTrue(all(c >= 1 for c in e.conf_counts))

    def test_embed_all_multiple_confs_with_alg_choice(self):
        e = self.Embedder(seed=7)
        e.load_smiles_iterable(self.smiles)
        # request ETKDGv2 explicitly (falls back if not available)
        e.embed_all(n_confs=3, add_hs=True, embed_algorithm="ETKDGv2")
        self.assertGreaterEqual(len(e.molblocks), 1)
        self.assertTrue(any(c >= 1 for c in e.conf_counts))
        self.assertTrue(all(isinstance(mb, str) and mb.strip() for mb in e.molblocks))

    def test_embed_handles_invalid_smiles(self):
        e = self.Embedder()
        e.load_smiles_iterable(["CCO", "NOTASMILES", "CCC"])
        e.embed_all(n_confs=1)
        self.assertGreaterEqual(len(e.molblocks), 1)

    def test_mols_to_sdf_writes_files(self):
        e = self.Embedder(seed=101)
        e.load_smiles_iterable(self.smiles)
        e.embed_all(n_confs=2, embed_algorithm="ETKDG")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            e.mols_to_sdf(str(out), per_mol_folder=True)
            folders = list(out.glob("ligand_*"))
            self.assertGreaterEqual(len(folders), 1)
            self.assertTrue(any((f / f"{f.name}.sdf").exists() for f in folders))


if __name__ == "__main__":
    unittest.main()
