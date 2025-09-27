import unittest
import pandas as pd
from prodock.postprocess.extract.core import Extractor


class TestExtractorMatchingModes(unittest.TestCase):
    def setUp(self):
        # Create a simple DataFrame that mimics crawl_scores output
        self.df = pd.DataFrame(
            {
                "ligand_id": ["A", "B", "C", "D"],
                "score": [-10.0, -9.0, -8.0, -7.0],
                "engine": ["vina", "vina-gpu", "smina", "qvina"],
                "label": [1, 0, 1, 0],
            }
        )

        # simple crawl_func that ignores roots and returns self.df (copy)
        self.fake_crawl = lambda roots, **kwargs: self.df.copy()

    def test_substring_match_default(self):
        ex = Extractor(crawl_func=self.fake_crawl, match_mode="substring")
        out = ex.extract_scores(["unused"], engines=["vina"])
        vals = set(out["engine"].str.lower().unique().tolist())
        # substring mode: "vina" should match both "vina" and "vina-gpu"
        self.assertIn("vina", vals)
        self.assertIn("vina-gpu", vals)
        self.assertNotIn("smina", vals)

    def test_exact_match(self):
        ex = Extractor(crawl_func=self.fake_crawl, match_mode="exact")
        out = ex.extract_scores(["unused"], engines=["vina"])
        vals = set(out["engine"].str.lower().unique().tolist())
        # exact mode: only "vina" (not vina-gpu)
        self.assertEqual(vals, {"vina"})

    def test_regex_match(self):
        ex = Extractor(crawl_func=self.fake_crawl, match_mode="regex")
        # regex to match engines that start with 'q' or end with 'gpu'
        out = ex.extract_scores(["unused"], engines=[r"^qvina$", r"gpu$"])
        vals = set(out["engine"].str.lower().unique().tolist())
        self.assertTrue({"qvina", "vina-gpu"}.issubset(vals))

    def test_list_engines_returns_set(self):
        ex = Extractor(crawl_func=self.fake_crawl)
        engs = ex.list_engines(["unused"])
        self.assertIsInstance(engs, set)
        self.assertTrue(len(engs) >= 1)
        self.assertIn("vina", engs)

    def test_engine_map_expansion(self):
        # map "vina-family" to both 'vina' and 'vina-gpu'
        engine_map = {"vina-family": ["vina", "vina-gpu"]}
        ex = Extractor(
            crawl_func=self.fake_crawl, match_mode="exact", engine_map=engine_map
        )
        out = ex.extract_scores(["unused"], engines=["vina-family"])
        vals = set(out["engine"].str.lower().unique().tolist())
        self.assertEqual(vals, {"vina", "vina-gpu"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
