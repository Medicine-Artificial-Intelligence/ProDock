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


@unittest.skipUnless(RDKit_AVAILABLE, "RDKit is required for these tests")
class TestEmbedder(unittest.TestCase):
    def setUp(self):
        # import from prodock package
        from prodock.chem.embed import Embedder

        self.Embedder = Embedder
        self.smiles = ["CCO", "CCC", "c1ccccc1"]

    def test_load_smiles_iterable_and_properties(self):
        e = self.Embedder(seed=123)
        e.load_smiles_iterable(self.smiles)
        self.assertIsInstance(e.smiles, list)
        self.assertEqual(len(e.smiles), len(self.smiles))
        _ = repr(e)  # ensure __repr__ works

    def test_embed_all_single_conf(self):
        e = self.Embedder(seed=42)
        e.load_smiles_iterable(self.smiles)
        e.embed_all(n_confs=1, add_hs=True, random_seed=42)
        self.assertGreaterEqual(len(e.molblocks), 1)
        self.assertEqual(len(e.mols), len(e.molblocks))
        self.assertTrue(all(c >= 1 for c in e.conf_counts))

    def test_embed_all_multiple_confs(self):
        e = self.Embedder(seed=7)
        e.load_smiles_iterable(self.smiles)
        e.embed_all(n_confs=3, add_hs=False, random_seed=7)
        self.assertGreaterEqual(len(e.molblocks), 1)
        self.assertTrue(any(c >= 1 for c in e.conf_counts))
        self.assertTrue(all(isinstance(mb, str) and mb.strip() for mb in e.molblocks))

    def test_embed_handles_invalid_smiles(self):
        e = self.Embedder()
        e.load_smiles_iterable(["CCO", "NOTASMILES", "CCC"])
        e.embed_all(n_confs=1)
        # invalid entry should be skipped; ensure at least one success
        self.assertGreaterEqual(len(e.molblocks), 1)

    def test_mols_to_sdf_writes_files(self):
        e = self.Embedder(seed=101)
        e.load_smiles_iterable(self.smiles)
        e.embed_all(n_confs=2, add_hs=True, random_seed=101)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            e.mols_to_sdf(str(out), per_mol_folder=True)
            folders = list(out.glob("ligand_*"))
            self.assertGreaterEqual(len(folders), 1)
            found = False
            for f in folders:
                sdf = f / f"{f.name}.sdf"
                if sdf.exists():
                    found = True
                    break
            self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
