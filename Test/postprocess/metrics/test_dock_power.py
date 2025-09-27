import tempfile
import unittest
from pathlib import Path
import importlib
import importlib.util
from rdkit import Chem
from rdkit.Chem import AllChem

from prodock.postprocess.metrics.dock_power import DockEvaluator
from prodock.io.utils import shutdown_pymol


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_OPENBABEL = _module_available("openbabel")
# pymol may expose submodules; check the submodule name used at runtime:
HAS_PYMOL = _module_available("pymol.cmd") or _module_available("pymol")


class TestDockEvaluator(unittest.TestCase):
    """Unit tests for DockEvaluator with expanded cases."""

    def _embed_mol(self, smiles: str):
        """Create an RDKit Mol with a 3D conformer."""
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(m, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(m)
        return m

    def test_rdkit_rmsd_identical_mols(self):
        de = DockEvaluator(engine="rdkit")
        m1 = self._embed_mol("CCO")
        # make a true independent copy of geometry
        m2 = Chem.Mol(m1)
        rmsd = de.rmsd(m1, m2)
        self.assertIsInstance(rmsd, float)
        self.assertAlmostEqual(rmsd, 0.0, places=3)

    def test_rdkit_rmsd_different_molecules_nonzero(self):
        de = DockEvaluator(engine="rdkit")
        m_ethane = self._embed_mol("CC")
        m_ethanol = self._embed_mol("CCO")
        # RMSD between molecules with different atom counts should raise or be nonzero; ensure non-negative
        with self.assertRaises(ValueError):
            # RDKit AlignMol expects same atom counts or meaningful mapping; this should raise
            _ = de.rmsd(m_ethane, m_ethanol)

    def test_rdkit_load_from_sdf_file(self):
        de = DockEvaluator(engine="rdkit")
        m = self._embed_mol("CCO")
        tmpdir = Path(tempfile.mkdtemp())
        try:
            sdf_path = tmpdir / "mol.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(m)
            writer.close()

            rmsd = de.rmsd(str(sdf_path), m)
            self.assertIsInstance(rmsd, float)
            self.assertAlmostEqual(rmsd, 0.0, places=3)
        finally:
            try:
                for f in tmpdir.iterdir():
                    f.unlink(missing_ok=True)
                tmpdir.rmdir()
            except Exception:
                pass

    def test_invalid_engine_raises_value_error(self):
        with self.assertRaises(ValueError):
            DockEvaluator(engine="this_engine_does_not_exist")

    def test_nonexistent_path_raises(self):
        de = DockEvaluator(engine="rdkit")
        with self.assertRaises(ValueError):
            de.rmsd("this_file_does_not_exist.sdf", "also_missing.sdf")

    @unittest.skipUnless(HAS_OPENBABEL, "openbabel not available in this environment")
    def test_openbabel_rmsd_from_files(self):
        de = DockEvaluator(engine="openbabel")
        m = self._embed_mol("CCO")
        tmpdir = Path(tempfile.mkdtemp())
        try:
            sdf1 = tmpdir / "mol1.sdf"
            sdf2 = tmpdir / "mol2.sdf"
            w1 = Chem.SDWriter(str(sdf1))
            w1.write(m)
            w1.close()
            w2 = Chem.SDWriter(str(sdf2))
            w2.write(m)
            w2.close()

            rmsd = de.rmsd(str(sdf1), str(sdf2))
            self.assertIsInstance(rmsd, float)
            self.assertAlmostEqual(rmsd, 0.0, places=2)
        finally:
            try:
                for f in tmpdir.iterdir():
                    f.unlink(missing_ok=True)
                tmpdir.rmdir()
            except Exception:
                pass

    @unittest.skipUnless(HAS_PYMOL, "PyMOL not available in this environment")
    def test_pymol_rmsd_from_files(self):
        shutdown_pymol(quiet=False)
        de = DockEvaluator(engine="pymol")
        m = self._embed_mol("CCO")
        tmpdir = Path(tempfile.mkdtemp())
        try:
            pdb1 = tmpdir / "mol1.pdb"
            pdb2 = tmpdir / "mol2.pdb"
            Chem.MolToPDBFile(m, str(pdb1))
            Chem.MolToPDBFile(m, str(pdb2))
            rmsd = de.rmsd(str(pdb1), str(pdb2))
            self.assertIsInstance(rmsd, float)
            self.assertAlmostEqual(rmsd, 0.0, places=2)
        finally:
            try:
                for f in tmpdir.iterdir():
                    f.unlink(missing_ok=True)
                tmpdir.rmdir()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
