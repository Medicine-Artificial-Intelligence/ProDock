import unittest
import tempfile
from pathlib import Path

try:
    import rdkit  # noqa: F401
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


@unittest.skipUnless(RDKit_AVAILABLE, "RDKit is required for these tests")
class TestOptimizer(unittest.TestCase):
    def setUp(self):
        # create a small multi-conformer MolBlock using Embedder from prodock
        from prodock.chem.embed import Embedder

        self.embedder = Embedder(seed=101)
        self.embedder.load_smiles_iterable(["CCO"])  # ethanol
        self.embedder.embed_all(n_confs=2, add_hs=True, random_seed=101)
        self.assertGreaterEqual(len(self.embedder.molblocks), 1)
        self.molblock = self.embedder.molblocks[0]

    def _mol_get_props(self, mol):
        """
        Robust property extraction helper for RDKit Mol across versions.
        """
        if hasattr(mol, "GetPropsAsDict"):
            return mol.GetPropsAsDict()
        if hasattr(mol, "GetPropsAsDictionary"):
            return mol.GetPropsAsDictionary()
        if hasattr(mol, "GetPropNames"):
            return {k: mol.GetProp(k) for k in mol.GetPropNames()}
        return {}

    def test_optimize_uff_molblock(self):
        from prodock.chem.optimize import Optimizer

        opt = Optimizer(max_iters=50)
        opt.load_molblocks([self.molblock])
        opt.optimize_all(method="UFF")
        self.assertEqual(len(opt.optimized_molblocks), 1)
        energies = opt.energies
        self.assertIsInstance(energies, list)
        self.assertEqual(len(energies), 1)
        e_map = energies[0]
        self.assertIsInstance(e_map, dict)
        self.assertGreaterEqual(len(e_map), 1)
        for v in e_map.values():
            self.assertIsInstance(v, float)

    def test_optimize_mmff_molblock(self):
        from prodock.chem.optimize import Optimizer

        opt = Optimizer(max_iters=50)
        opt.load_molblocks([self.molblock])
        opt.optimize_all(method="MMFF")
        self.assertEqual(len(opt.optimized_molblocks), 1)
        e_map = opt.energies[0]
        self.assertIsInstance(e_map, dict)
        self.assertGreaterEqual(len(e_map), 1)

    def test_write_sdf_with_energy_tags(self):
        from prodock.chem.optimize import Optimizer
        from rdkit import Chem

        opt = Optimizer(max_iters=20)
        opt.load_molblocks([self.molblock])
        opt.optimize_all(method="UFF")
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "opt_out"
            opt.write_sdf(str(outdir), per_mol_folder=True, write_energy_tags=True)
            folder = outdir / "ligand_0"
            self.assertTrue(folder.exists())
            sdf_file = folder / "ligand_0.sdf"
            self.assertTrue(sdf_file.exists())

            # use sanitize=True so RDKit exposes property API consistently
            supplier = Chem.SDMolSupplier(str(sdf_file), sanitize=True, removeHs=False)
            mols = [m for m in supplier if m is not None]
            self.assertGreaterEqual(len(mols), 1)
            m = mols[0]
            props = self._mol_get_props(m)
            energy_keys = [k for k in props.keys() if k.startswith("CONF_ENERGY_")]
            self.assertGreaterEqual(len(energy_keys), 1)


if __name__ == "__main__":
    unittest.main()
