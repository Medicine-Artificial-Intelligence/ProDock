# tests/test_extract_files_unittest.py
import unittest
from pathlib import Path
from prodock.postprocess.extract import detect_engine, parse_log_text


class TestExtractFilesFromData(unittest.TestCase):
    """
    Run separate tests on each text file in Data/testcase/extract that were
    provided. Each file gets its own test method so failures show individually.
    """

    BASE = Path("Data/testcase/extract")

    def _read(self, fname: str) -> str:
        p = self.BASE / fname
        if not p.exists():
            # fallback to /mnt/data if CI places uploaded files there
            p_alt = Path("/mnt/data") / fname
            if p_alt.exists():
                p = p_alt
            else:
                raise FileNotFoundError(
                    f"Test input not found: {p} (also tried {p_alt})"
                )

        # Try several encodings to avoid UnicodeDecodeError on non-UTF-8 samples.
        # Order: utf-8 (preferred), then latin-1, then cp1252. Finally fall back to
        # latin-1 with replacement to guarantee string return.
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return p.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return p.read_text(encoding="latin-1", errors="replace")

    def _common_assertions(
        self, text: str, expected_engine: str, expect_gnina: bool = False
    ):
        eng = detect_engine(text)
        # engine detection may return None for ambiguous logs; require it at least to match expected
        self.assertIsNotNone(
            eng, msg=f"detect_engine returned None for expected {expected_engine}"
        )
        self.assertEqual(eng, expected_engine)
        rows = parse_log_text(text, engine=eng)
        # ensure parse returned at least one row
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0, msg="No rows parsed from the log")
        # confirm presence of numeric keys
        first = rows[0]
        self.assertIn("affinity_kcal_mol", first)
        if expect_gnina:
            self.assertIn("cnn_pose", first)
            self.assertIn("cnn_affinity", first)
        else:
            # vina-style should have rmsd fields
            self.assertIn("rmsd_lb", first)
            self.assertIn("rmsd_ub", first)

    def test_vina_txt(self):
        txt = self._read("vina.txt")
        self._common_assertions(txt, expected_engine="vina", expect_gnina=False)

    def test_smina_txt(self):
        txt = self._read("smina.txt")
        self._common_assertions(txt, expected_engine="smina", expect_gnina=False)

    def test_gnina_txt(self):
        txt = self._read("gnina.txt")
        self._common_assertions(txt, expected_engine="gnina", expect_gnina=True)

    def test_qvina_txt(self):
        txt = self._read("qvina.txt")
        # quickvina2 / qvina mapping expected to 'qvina'
        self._common_assertions(txt, expected_engine="qvina", expect_gnina=False)

    def test_qvina_gpu_txt(self):
        txt = self._read("qvinagpu.txt")
        # quickvina-gpu expected canonical name 'qvina-gpu'
        self._common_assertions(txt, expected_engine="qvina-gpu", expect_gnina=False)

    def test_vina_gpu_txt(self):
        txt = self._read("vinagpu.txt")
        # Vina-GPU -> 'vina-gpu'
        self._common_assertions(txt, expected_engine="vina-gpu", expect_gnina=False)

    def test_vinagpu2_txt(self):
        txt = self._read("vinagpu2.txt")
        # another variant of vina gpu (vinagpu2) — we expect detection to return vina-gpu or vina
        eng = detect_engine(txt)
        self.assertIsNotNone(eng, msg="detect_engine returned None for vinagpu2.txt")
        # Accept either 'vina-gpu' or 'vina' as canonical acceptable answers for this variant
        self.assertIn(eng, {"vina-gpu", "vina"})
        rows = parse_log_text(txt, engine=eng)
        self.assertGreater(len(rows), 0)
        self.assertIn("affinity_kcal_mol", rows[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
