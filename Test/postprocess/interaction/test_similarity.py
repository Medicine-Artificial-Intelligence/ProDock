from __future__ import annotations

import unittest

import pandas as pd

from prodock.postprocess.interaction.exceptions import MissingDependencyError
from prodock.postprocess.interaction.similarity import tanimoto_similarity_matrix


class _FakeBitVector:
    """
    Minimal stand-in object for similarity testing.

    :param bits:
        Iterable of integer bit positions set to 1.
    :type bits: iterable[int]
    """

    def __init__(self, bits) -> None:
        self.bits = set(bits)


def _fake_tanimoto_similarity(a: _FakeBitVector, b: _FakeBitVector) -> float:
    """
    Compute Tanimoto similarity between two fake bit vectors.

    :param a:
        First fake vector.
    :type a: _FakeBitVector
    :param b:
        Second fake vector.
    :type b: _FakeBitVector

    :returns:
        Tanimoto similarity value.
    :rtype: float
    """
    union = a.bits | b.bits
    if not union:
        return 1.0
    return len(a.bits & b.bits) / len(union)


class TestTanimotoSimilarityMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self._original_import = __import__

    def tearDown(self) -> None:
        import builtins

        builtins.__import__ = self._original_import

    def _install_fake_rdkit(self) -> None:
        import builtins

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "rdkit.DataStructs" and "TanimotoSimilarity" in fromlist:
                class _FakeDataStructsModule:
                    @staticmethod
                    def TanimotoSimilarity(a, b):
                        return _fake_tanimoto_similarity(a, b)

                return _FakeDataStructsModule()
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = fake_import

    def _install_missing_rdkit(self) -> None:
        import builtins

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "rdkit.DataStructs":
                raise ImportError("No module named 'rdkit'")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = fake_import

    def test_similarity_matrix_with_explicit_names(self) -> None:
        self._install_fake_rdkit()

        vectors = [
            _FakeBitVector([1, 2, 3]),
            _FakeBitVector([1, 2]),
            _FakeBitVector([4, 5]),
        ]
        names = ["pose_a", "pose_b", "pose_c"]

        result = tanimoto_similarity_matrix(vectors, names=names)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(list(result.index), names)
        self.assertEqual(list(result.columns), names)
        self.assertEqual(result.loc["pose_a", "pose_a"], 1.0)
        self.assertEqual(result.loc["pose_b", "pose_b"], 1.0)
        self.assertEqual(result.loc["pose_c", "pose_c"], 1.0)
        self.assertAlmostEqual(result.loc["pose_a", "pose_b"], 2 / 3)
        self.assertAlmostEqual(result.loc["pose_b", "pose_a"], 2 / 3)
        self.assertEqual(result.loc["pose_a", "pose_c"], 0.0)

    def test_similarity_matrix_uses_default_names(self) -> None:
        self._install_fake_rdkit()

        vectors = [
            _FakeBitVector([1]),
            _FakeBitVector([1, 2]),
        ]

        result = tanimoto_similarity_matrix(vectors)

        self.assertEqual(list(result.index), ["mol_0000", "mol_0001"])
        self.assertEqual(list(result.columns), ["mol_0000", "mol_0001"])
        self.assertEqual(result.loc["mol_0000", "mol_0000"], 1.0)
        self.assertAlmostEqual(result.loc["mol_0000", "mol_0001"], 0.5)

    def test_similarity_matrix_empty_input(self) -> None:
        self._install_fake_rdkit()

        result = tanimoto_similarity_matrix([])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (0, 0))
        self.assertEqual(list(result.index), [])
        self.assertEqual(list(result.columns), [])

    def test_similarity_matrix_name_length_mismatch_raises_value_error(self) -> None:
        self._install_fake_rdkit()

        vectors = [
            _FakeBitVector([1]),
            _FakeBitVector([2]),
        ]
        names = ["only_one_name"]

        with self.assertRaises(ValueError):
            tanimoto_similarity_matrix(vectors, names=names)

    def test_missing_rdkit_raises_custom_error(self) -> None:
        self._install_missing_rdkit()

        with self.assertRaises(MissingDependencyError) as ctx:
            tanimoto_similarity_matrix([_FakeBitVector([1])])

        self.assertIn("RDKit is required", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, ImportError)


if __name__ == "__main__":
    unittest.main()