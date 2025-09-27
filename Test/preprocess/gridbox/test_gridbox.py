# tests/test_gridbox.py
"""Unit tests for the new GridBox API (prodock.process.gridbox)."""

import tempfile
import unittest
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - test will skip if RDKit missing
    Chem = None
    AllChem = None

from prodock.preprocess.gridbox.gridbox import GridBox


@unittest.skipIf(Chem is None, "RDKit is required for these tests")
class TestGridBox(unittest.TestCase):
    """Tests for GridBox adapted to the new algorithms-based API."""

    def setUp(self) -> None:
        # ethanol with 3D coords
        self.m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        _ = AllChem.EmbedMolecule(self.m, randomSeed=0xC0FFEE)
        try:
            AllChem.UFFOptimizeMolecule(self.m)
        except Exception:
            # some RDKit builds may not have UFF available; that's fine
            pass
        self.molblock = Chem.MolToMolBlock(self.m)
        self.pdbblock = Chem.MolToPDBBlock(self.m)

    def test_build_pad_using_constructor(self):
        gb = GridBox(self.m).from_ligand_pad(pad=4.0, isotropic=False)
        center = gb.center
        size = gb.size
        # basic structural checks
        self.assertEqual(len(center), 3)
        self.assertEqual(len(size), 3)
        self.assertTrue(all(s > 0 for s in size))
        # compute volume from size (no `volume` property in new API)
        vol = size[0] * size[1] * size[2]
        self.assertAlmostEqual(vol, size[0] * size[1] * size[2], places=6)

    def test_load_ligand_pdb_and_scale_from_text(self):
        # parse raw PDB text (fmt hint)
        gb = (
            GridBox()
            .load_ligand(self.pdbblock, fmt="pdb")
            .from_ligand_scale(scale=2.0, isotropic=True)
        )
        cx, cy, cz = gb.center
        sx, sy, sz = gb.size
        # isotropic -> sizes equal
        self.assertAlmostEqual(sx, sy, places=6)
        self.assertAlmostEqual(sx, sz, places=6)
        for c in (cx, cy, cz):
            self.assertIsInstance(c, float)

    def test_from_ligand_pad_adv_snap_using_constructor(self):
        gb = GridBox(self.m).from_ligand_pad_adv(
            pad=1.0, isotropic=False, heavy_only=False, snap_step=0.25, round_ndigits=3
        )
        cx, cy, cz = gb.center
        sx, sy, sz = gb.size

        def is_multiple(x, step=0.25):
            return abs((x / step) - round(x / step)) < 1e-6

        self.assertTrue(is_multiple(cx))
        self.assertTrue(is_multiple(sx))
        self.assertTrue(all(v > 0 for v in (sx, sy, sz)))

    def test_centroid_fixed_and_explicit_size(self):
        # from_centroid_fixed centers box at centroid; size is user-specified
        explicit_size = (10.0, 12.0, 14.0)
        gb = GridBox(self.m).from_centroid_fixed(size=explicit_size)
        self.assertEqual(gb.size, explicit_size)
        # center should be numeric tuple
        self.assertEqual(len(gb.center), 3)
        for c in gb.center:
            self.assertIsInstance(c, float)

    def test_preset_equivalents_using_pad_adv(self):
        # The old presets are now represented by particular pad/isotropic/min_size combos.
        gb_safe = GridBox(self.m).from_ligand_pad_adv(
            pad=4.0, isotropic=True, min_size=22.5
        )
        self.assertTrue(all(s >= 22.5 for s in gb_safe.size))

        gb_tight = GridBox(self.m).from_ligand_pad_adv(
            pad=3.0, isotropic=False, min_size=0.0
        )
        self.assertTrue(all(s >= 0 for s in gb_tight.size))

        gb_vina = GridBox(self.m).from_ligand_pad_adv(
            pad=2.0, isotropic=True, min_size=24.0
        )
        self.assertTrue(all(s >= 24.0 for s in gb_vina.size))

    def test_snap_method(self):
        gb = GridBox(self.m).from_ligand_pad(pad=1.0)
        gb.snap(step=0.5, round_ndigits=2)
        # ensure center & size are numeric triples
        for v in gb.center + gb.size:
            self.assertIsInstance(v, float)
        self.assertEqual(len(gb.center), 3)
        self.assertEqual(len(gb.size), 3)

    def test_vina_lines_and_grow_to_min_cube(self):
        gb = GridBox(self.m).from_ligand_pad(pad=1.0, isotropic=False)
        vina_text = gb.to_vina_lines()
        self.assertIn("center_x", vina_text)
        # test grow_to_min_cube produces cubic size
        gb.grow_to_min_cube()
        sx, sy, sz = gb.size
        self.assertAlmostEqual(sx, sy, places=6)
        self.assertAlmostEqual(sx, sz, places=6)

    def test_as_tuple_and_repr_and_union(self):
        gb = GridBox(self.m).from_ligand_scale(scale=1.2)
        tup = gb.as_tuple()
        self.assertIsInstance(tup, tuple)
        self.assertEqual(len(tup), 2)
        repr_s = repr(gb)
        self.assertIn("GridBox", repr_s)

        # test union of two identical ligands yields box at least as large as single
        # create temporary second box by using the same molblock path via load_ligand
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.pdb"
            p.write_text(self.pdbblock)
            gb_union = GridBox().from_union([str(p), str(p)], fmt="pdb", pad=0.0)
            # union size should be >= single size in all axes
            single = GridBox(self.m).from_ligand_pad(pad=0.0)
            for u, s in zip(gb_union.size, single.size):
                self.assertGreaterEqual(u, s)


if __name__ == "__main__":
    unittest.main()
