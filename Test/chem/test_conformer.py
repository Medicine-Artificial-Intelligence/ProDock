import unittest
import tempfile
from pathlib import Path
import os

try:
    import rdkit  # noqa: F401
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

try:
    from joblib import Parallel  # just to check availability

    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


@unittest.skipUnless(RDKit_AVAILABLE, "RDKit is required for these tests")
class TestConformerManager(unittest.TestCase):
    def setUp(self):
        # try to import Conformer (preferred) otherwise ConformerManager
        try:
            from prodock.chem.conformer import Conformer as ConformerClass
        except Exception:
            # fallback to ConformerManager if your module exposes that name
            from prodock.chem.conformer import ConformerManager as ConformerClass  # type: ignore
        self.ConformerClass = ConformerClass

        # create temp smiles file
        self.tmpdir = tempfile.TemporaryDirectory()
        p = Path(self.tmpdir.name) / "smiles.smi"
        p.write_text("CCO\nCCC\nc1ccccc1\n")
        self.smiles_file = str(p)
        self.cm = ConformerClass(seed=42)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _mol_get_props(self, mol):
        if hasattr(mol, "GetPropsAsDict"):
            return mol.GetPropsAsDict()
        if hasattr(mol, "GetPropsAsDictionary"):
            return mol.GetPropsAsDictionary()
        if hasattr(mol, "GetPropNames"):
            return {k: mol.GetProp(k) for k in mol.GetPropNames()}
        return {}

    def test_sequence_workflow(self):
        from rdkit import Chem

        cm = self.cm
        # If your high-level class uses different method names, these are expected:
        # load_smiles_file, embed_all, optimize_all, prune_top_k, write_sdf
        cm.load_smiles_file(self.smiles_file)
        cm.embed_all(n_confs=3, n_jobs=1)
        self.assertGreaterEqual(len(cm.molblocks), 1)
        cm.optimize_all(method="MMFF", n_jobs=1, max_iters=50)
        self.assertEqual(len(cm.energies), len(cm.molblocks))
        cm.prune_top_k(1)
        for mb in cm.molblocks:
            mol = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
            self.assertIsNotNone(mol)
            self.assertEqual(mol.GetNumConformers(), 1)
        out = Path(self.tmpdir.name) / "conformer_out"
        cm.write_sdf(str(out), per_mol_folder=True, write_energy_tags=True)
        dirs = list(out.iterdir())
        self.assertGreaterEqual(len(dirs), 1)
        sdf = dirs[0] / f"{dirs[0].name}.sdf"
        self.assertTrue(sdf.exists())
        supplier = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=False)
        ms = [m for m in supplier if m is not None]
        self.assertGreaterEqual(len(ms), 1)
        props = self._mol_get_props(ms[0])
        self.assertTrue(any(k.startswith("CONF_ENERGY_") for k in props.keys()))

    @unittest.skipUnless(JOBLIB_AVAILABLE, "joblib required for parallel tests")
    def test_parallel_embedding_and_optimization(self):
        cm = self.cm
        cm.load_smiles_file(self.smiles_file)
        cm.embed_all(n_confs=2, n_jobs=2)
        self.assertGreaterEqual(len(cm.molblocks), 1)
        cm.optimize_all(method="UFF", n_jobs=2, max_iters=50)
        self.assertEqual(len(cm.energies), len(cm.molblocks))


if __name__ == "__main__":
    unittest.main()
