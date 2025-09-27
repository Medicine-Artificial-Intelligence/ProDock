# tests/test_utils.py
import os
import tempfile
import unittest
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - skip tests if RDKit missing
    Chem = None

from prodock.preprocess.gridbox.utils import (
    is_pathlike,
    round_tuple,
    snap_tuple,
    ensure_pos_size,
    coords_from_mol,
    gb_coords_from_mol,
    center_and_span,
)


@unittest.skipIf(Chem is None, "RDKit is required for these tests")
class TestUtils(unittest.TestCase):
    def setUp(self):
        self.mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(self.mol, randomSeed=0x1234)

    def test_is_pathlike_and_tempfile(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tfname = tf.name
            tf.write(b"hello")
        try:
            self.assertTrue(is_pathlike(tfname))
            self.assertTrue(is_pathlike(Path(tfname)))
        finally:
            os.unlink(tfname)
        self.assertFalse(is_pathlike("/unlikely/to/exist/12345"))

    def test_round_and_snap_tuple(self):
        t = (1.2345, 2.3456, 3.4567)
        self.assertEqual(round_tuple(t, 2), (1.23, 2.35, 3.46))
        s = snap_tuple(t, 0.25)
        self.assertEqual(len(s), 3)
        # each snapped value should be a multiple of 0.25 (within tolerance)
        for v in s:
            self.assertAlmostEqual((v / 0.25) - round(v / 0.25), 0.0, places=6)

    def test_ensure_pos_size(self):
        self.assertEqual(ensure_pos_size((1, 2.5, 3)), (1.0, 2.5, 3.0))
        with self.assertRaises(ValueError):
            ensure_pos_size((0, 1, 1))
        with self.assertRaises(ValueError):
            ensure_pos_size((-1, 1, 1))

    def test_coords_helpers(self):
        coords = coords_from_mol(self.mol)
        self.assertEqual(coords.shape[1], 3)
        gb_all = gb_coords_from_mol(self.mol, heavy_only=False)
        gb_heavy = gb_coords_from_mol(self.mol, heavy_only=True)
        self.assertGreaterEqual(gb_all.shape[0], gb_heavy.shape[0])
        self.assertGreater(gb_all.shape[0], 0)

    def test_center_and_span(self):
        coords = coords_from_mol(self.mol)
        center, span = center_and_span(coords)
        self.assertEqual(center.shape, (3,))
        self.assertEqual(span.shape, (3,))
        self.assertTrue((span >= 0).all())


if __name__ == "__main__":
    unittest.main()
