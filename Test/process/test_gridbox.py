# test_gridbox.py
"""Unit tests for gridbox.py"""

import os
import tempfile
import unittest
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - test will skip if RDKit missing
    Chem = None
    AllChem = None

from prodock.process.gridbox import (
    _is_pathlike,
    _snap_val,
    _snap_tuple,
    _round_tuple,
    _ensure_pos_size,
    _coords_from_mol,
    _gb_coords_from_mol,
    GridBox,
)


@unittest.skipIf(Chem is None, "RDKit is required for these tests")
class TestGridBox(unittest.TestCase):
    """Tests for GridBox and helper functions."""

    def setUp(self) -> None:
        """Create a simple 3D molecule (ethanol) with an embedded conformer."""
        # create ethanol from SMILES and embed 3D coords
        self.mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        # embed and optimize to get non-trivial coordinates
        _ = AllChem.EmbedMolecule(self.mol, randomSeed=0xC0FFEE)
        try:
            AllChem.UFFOptimizeMolecule(self.mol)
        except Exception:
            # UFF might fail in some RDKit builds, ignore if so (coords still present)
            pass

        # Mol blocks/blocks for IO tests
        # Keep these for testing PDB path / explicit parsing (we won't rely on SDF parsing)
        self.molblock = Chem.MolToMolBlock(self.mol)  # mol block (MDL)
        self.pdbblock = Chem.MolToPDBBlock(self.mol)
        # we won't depend on parsing XYZ in tests (some RDKit builds lack MolFromXYZBlock)
        try:
            self.xyzblock = Chem.MolToXYZBlock(self.mol)
        except Exception:
            self.xyzblock = None

    # -------------------------
    # helper tests
    # -------------------------
    def test_is_pathlike_and_tempfile(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tfname = tf.name
            tf.write(b"hello")
        try:
            self.assertTrue(_is_pathlike(tfname))
            self.assertTrue(_is_pathlike(Path(tfname)))
        finally:
            os.unlink(tfname)

        # non-existing path
        self.assertFalse(_is_pathlike("/unlikely/to/exist/12345"))

    def test_snap_and_round_helpers(self):
        self.assertEqual(_snap_val(1.23, 0.5), 1.0)
        self.assertEqual(_snap_val(1.26, 0.5), 1.5)
        t = (1.2345, 2.3456, 3.4567)
        self.assertEqual(_round_tuple(t, 2), (1.23, 2.35, 3.46))
        self.assertEqual(
            _snap_tuple(t, 0.25),
            (_snap_val(1.2345, 0.25), _snap_val(2.3456, 0.25), _snap_val(3.4567, 0.25)),
        )

    def test_ensure_pos_size(self):
        self.assertEqual(_ensure_pos_size((1, 2.5, 3)), (1.0, 2.5, 3.0))
        with self.assertRaises(ValueError):
            _ensure_pos_size((0, 1, 1))
        with self.assertRaises(ValueError):
            _ensure_pos_size((-1, 1, 1))

    def test_coords_helpers(self):
        coords = _coords_from_mol(self.mol)
        self.assertEqual(coords.shape[1], 3)
        # gb coords heavy_only should produce <= rows than all atoms
        gb_all = _gb_coords_from_mol(self.mol, heavy_only=False)
        gb_heavy = _gb_coords_from_mol(self.mol, heavy_only=True)
        self.assertGreaterEqual(gb_all.shape[0], gb_heavy.shape[0])
        # ensure fallback doesn't produce empty result
        self.assertGreater(gb_all.shape[0], 0)

    # -------------------------
    # GridBox core tests (use constructor with mol to avoid SDF re-parsing issues)
    # -------------------------
    def test_build_pad_using_constructor(self):
        # Use constructor-injected mol rather than loading/parsing SDF text (avoids RDKit parse fragility)
        gb = GridBox(self.mol).from_ligand_pad(pad=4.0, isotropic=False)
        center = gb.center
        size = gb.size
        # center should be 3-tuple, size > 0
        self.assertEqual(len(center), 3)
        self.assertEqual(len(size), 3)
        self.assertTrue(all(s > 0 for s in size))
        # volume property
        vol = gb.volume
        self.assertAlmostEqual(vol, size[0] * size[1] * size[2], places=6)

    def test_load_ligand_pdb_and_scale(self):
        # test PDB parsing path (keep this to exercise parsing branch)
        gb = (
            GridBox()
            .load_ligand(self.pdbblock, fmt="pdb")
            .from_ligand_scale(scale=2.0, isotropic=True)
        )
        cx, cy, cz = gb.center
        sx, sy, sz = gb.size
        # isotropic true -> all sizes equal (rounded to 3 decimals)
        self.assertAlmostEqual(sx, sy, places=6)
        self.assertAlmostEqual(sx, sz, places=6)
        # center numeric
        for c in (cx, cy, cz):
            self.assertIsInstance(c, float)

    def test_from_ligand_pad_adv_snap_using_constructor(self):
        # avoid XYZ parsing path (not reliable across RDKit builds) and use constructor-injected mol
        gb = GridBox(self.mol).from_ligand_pad_adv(
            pad=1.0, isotropic=False, heavy_only=False, snap_step=0.25, round_ndigits=3
        )
        cx, cy, cz = gb.center
        sx, sy, sz = gb.size

        def is_multiple(x, step=0.25):
            return abs((x / step) - round(x / step)) < 1e-6

        self.assertTrue(is_multiple(cx))
        self.assertTrue(is_multiple(sx))
        self.assertTrue(all(v > 0 for v in (sx, sy, sz)))

    def test_from_center_size_and_invalid(self):
        gb = GridBox().from_center_size((1.0, 2.0, 3.0), (5.0, 6.0, 7.0))
        self.assertEqual(gb.center, (1.0, 2.0, 3.0))
        self.assertEqual(gb.size, (5.0, 6.0, 7.0))
        # invalid sizes
        with self.assertRaises(ValueError):
            GridBox().from_center_size((0, 0, 0), (0.0, 1.0, 1.0))

    def test_preset_modes_and_unknown(self):
        # Use constructor-injected mol for presets requiring a molecule
        gb_safe = GridBox(self.mol).preset("safe")
        # safe preset is isotropic True and min_size 22.5 -> sizes at least 22.5
        self.assertTrue(all(s >= 22.5 for s in gb_safe.size))
        gb_tight = GridBox(self.mol).preset("tight")
        self.assertTrue(all(s >= 0 for s in gb_tight.size))
        gb_vina = GridBox(self.mol).preset("vina")
        self.assertTrue(all(s >= 24.0 for s in gb_vina.size))
        # unknown preset triggers ValueError about unknown preset (doesn't require molecule)
        with self.assertRaises(ValueError):
            GridBox().preset("this_must_not_exist")

    def test_snap_method(self):
        gb = GridBox(self.mol).from_ligand_pad(pad=1.0)
        # intentionally use odd step
        gb.snap(step=0.5, round_ndigits=2)
        for v in gb.center + gb.size:
            self.assertIsInstance(v, float)
        self.assertEqual(len(gb.center), 3)
        self.assertEqual(len(gb.size), 3)

    def test_vina_io_and_parse(self):
        gb = GridBox(self.mol).from_ligand_pad(pad=1.0, isotropic=True)
        vina_text = gb.to_vina_lines()
        # parse back and create GridBox
        parsed = GridBox.parse_vina_cfg(vina_text)
        self.assertSetEqual(
            set(parsed.keys()),
            {"center_x", "center_y", "center_z", "size_x", "size_y", "size_z"},
        )
        gb2 = GridBox.from_vina_cfg(vina_text)
        self.assertEqual(gb2.center, gb.center)
        # to_vina_file writes file
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test_vina.cfg"
            out = gb.to_vina_file(p)
            self.assertTrue(out.exists())
            txt = out.read_text()
            self.assertIn("center_x", txt)

        # parse should raise on missing keys
        with self.assertRaises(ValueError):
            GridBox.parse_vina_cfg("center_x = 1\nsize_x = 2")

    def test_as_tuple_and_repr_and_summary(self):
        gb = GridBox(self.mol).from_ligand_scale(scale=1.2)
        tup = gb.as_tuple()
        self.assertIsInstance(tup, tuple)
        self.assertEqual(len(tup), 2)
        repr_s = repr(gb)
        self.assertIn("GridBox", repr_s)
        s = gb.summary()
        self.assertIsInstance(s, str)
        self.assertIn("center=", s)
        self.assertIn("size=", s)
        self.assertIn("volume=", s)


if __name__ == "__main__":
    unittest.main()
