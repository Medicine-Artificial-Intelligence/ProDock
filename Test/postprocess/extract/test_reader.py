import unittest
from pathlib import Path
from prodock.postprocess.extract.reader import parse_log_text
from prodock.postprocess.extract.normalize import read_text_flexible

BASE = Path("Data/testcase/extract")


class TestExtractReader(unittest.TestCase):
    def _read(self, fname: str) -> str:
        p = BASE / fname
        if not p.exists():
            alt = Path("/mnt/data") / fname
            if alt.exists():
                p = alt
            else:
                raise FileNotFoundError(f"Missing test file: {p} (also tried {alt})")
        text, _enc = read_text_flexible(p)
        return text

    def test_parse_vina_family_files(self):
        for fname in (
            "vina.txt",
            "smina.txt",
            "qvina.txt",
            "vinagpu.txt",
            "qvinagpu.txt",
            "vinagpu2.txt",
        ):
            txt = self._read(fname)
            rows = parse_log_text(txt)
            # Expect at least one parsed row for vina-family files
            self.assertIsInstance(rows, list)
            self.assertGreater(len(rows), 0, msg=f"No rows parsed from {fname}")
            first = rows[0]
            self.assertIn("affinity_kcal_mol", first)
            self.assertIn("rmsd_lb", first)
            self.assertIn("rmsd_ub", first)

    def test_parse_gnina_file(self):
        txt = self._read("gnina.txt")
        rows = parse_log_text(txt)
        self.assertGreater(len(rows), 0)
        first = rows[0]
        self.assertIn("affinity_kcal_mol", first)
        # GNINA-specific keys
        self.assertIn("cnn_pose", first)
        self.assertIn("cnn_affinity", first)

    def test_custom_regex_used_when_provided(self):
        # create a vina-like simple snippet and supply custom regex
        custom_txt = "mode | affinity | dist from best mode\n1  -12.34  0.0  0.0\n"
        custom = {
            "vina_row": r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
        }
        rows = parse_log_text(custom_txt, engine=None, regex=custom)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["affinity_kcal_mol"], -12.34, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
