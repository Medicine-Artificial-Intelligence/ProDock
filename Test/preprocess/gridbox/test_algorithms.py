# tests/test_algorithms.py
import unittest

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    Chem = None
    AllChem = None

import numpy as np

from prodock.preprocess.gridbox.algorithms import (
    expand_by_pad,
    expand_by_scale,
    expand_by_advanced,
    expand_by_percentile,
    expand_by_pca_aabb,
    centroid_fixed,
    pad_for_scale,
    scale_for_pad,
    union_boxes,
)


@unittest.skipIf(Chem is None, "RDKit is required for these tests")
class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(self.m, randomSeed=0xC0FFEE)
        try:
            AllChem.UFFOptimizeMolecule(self.m)
        except Exception:
            pass

    def _span_from_coords(self, coords: np.ndarray) -> np.ndarray:
        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        return mx - mn

    def test_expand_by_scale_matches_formula(self):
        center, size = expand_by_scale(self.m, scale=2.0, isotropic=False)
        coords = self.m.GetConformer().GetPositions()
        import numpy as np

        span = self._span_from_coords(np.array(coords))
        # size should equal span * scale (rounded to 3 decimals in implementation)
        expected = tuple(float(round(x * 2.0, 3)) for x in span)
        self.assertEqual(size, expected)

    def test_expand_by_pad_relation(self):
        pad = 1.5
        center, size = expand_by_pad(self.m, pad=pad, isotropic=False)
        coords = self.m.GetConformer().GetPositions()
        span = self._span_from_coords(np.array(coords))
        expected = tuple(float(round(x + 2.0 * pad, 3)) for x in span)
        self.assertEqual(size, expected)

    def test_expand_by_advanced_snapping_and_heavy(self):
        center, size = expand_by_advanced(
            self.m, pad=1.0, heavy_only=False, snap_step=0.25
        )
        # sizes and center multiples of 0.25
        for v in size + center:
            self.assertAlmostEqual((v / 0.25) - round(v / 0.25), 0.0, places=6)

        # heavy_only should produce sizes >= or <= depending on hydrogens removed
        c_h, s_h = expand_by_advanced(self.m, pad=1.0, heavy_only=True)
        self.assertTrue(len(s_h) == 3)

    def test_percentile_and_pca(self):
        c_p, s_p = expand_by_percentile(self.m, low=5.0, high=95.0, pad=0.0)
        self.assertEqual(len(c_p), 3)
        self.assertEqual(len(s_p), 3)

        c_pca, s_pca = expand_by_pca_aabb(self.m, scale=1.1, pad=0.5)
        self.assertEqual(len(c_pca), 3)
        self.assertEqual(len(s_pca), 3)
        # PCA-AABB size should be >= 0 and real numbers
        self.assertTrue(all(v >= 0 for v in s_pca))

    def test_centroid_fixed_and_conversions(self):
        size = (10.0, 12.0, 14.0)
        c, s = centroid_fixed(self.m, size=size)
        self.assertEqual(s, size)
        # pad_for_scale vs scale_for_pad consistency (approx)
        coords = self.m.GetConformer().GetPositions()
        import numpy as np

        span = self._span_from_coords(np.array(coords))
        p = pad_for_scale(span, 2.0)
        sc = scale_for_pad(span, p)
        # scale_for_pad returns per-axis scale ~ 2.0 (rounded to float)
        self.assertTrue(all(abs(x - 2.0) < 1e-6 for x in sc))

    def test_union_boxes(self):
        c1 = (0.0, 0.0, 0.0)
        s1 = (4.0, 4.0, 4.0)
        c2 = (3.0, 0.0, 0.0)
        s2 = (4.0, 4.0, 4.0)
        c_u, s_u = union_boxes(c1, s1, c2, s2)
        # union should cover extremes: min x = -2, max x = 5 => size_x = 7
        self.assertAlmostEqual(s_u[0], 7.0, places=6)


if __name__ == "__main__":
    unittest.main()
