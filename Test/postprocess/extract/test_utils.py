import unittest
from prodock.postprocess.extract.utils import (
    normalize_engine_token,
    build_engine_pattern,
    engine_matches,
)


class TestExtractUtils(unittest.TestCase):
    def test_normalize_engine_token(self):
        self.assertEqual(normalize_engine_token(" Vina "), "vina")
        self.assertEqual(normalize_engine_token("QViNa"), "qvina")

    def test_build_engine_pattern_and_matches(self):
        pat = build_engine_pattern(["vina", "smina"])
        # pattern should be non-empty and match expected substrings
        self.assertTrue(isinstance(pat, str) and len(pat) > 0)
        self.assertTrue(engine_matches("vina", pat))
        self.assertTrue(engine_matches("smina-1.0", pat))
        self.assertFalse(engine_matches("gnina", pat))

    def test_engine_matches_empty_pattern(self):
        self.assertTrue(engine_matches("anything", ""))  # empty pattern means allow-all
        self.assertFalse(engine_matches("", "vina"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
