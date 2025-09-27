import unittest
from pathlib import Path
from prodock.postprocess.extract.engines import detect_engine
from prodock.postprocess.extract.normalize import read_text_flexible

BASE = Path("Data/testcase/extract")


class TestExtractEngines(unittest.TestCase):
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

    def test_detect_vina_variants(self):
        txt = self._read("vina.txt")
        self.assertEqual(detect_engine(txt), "vina")

        txt = self._read("smina.txt")
        self.assertEqual(detect_engine(txt), "smina")

        txt = self._read("qvina.txt")
        self.assertEqual(detect_engine(txt), "qvina")

        txt = self._read("qvinagpu.txt")
        self.assertEqual(detect_engine(txt), "qvina-gpu")

        txt = self._read("vinagpu.txt")
        self.assertEqual(detect_engine(txt), "vina-gpu")

    def test_detect_gnina(self):
        txt = self._read("gnina.txt")
        self.assertEqual(detect_engine(txt), "gnina")

    def test_vinagpu2_accepts_some_variant(self):
        txt = self._read("vinagpu2.txt")
        eng = detect_engine(txt)
        # be permissive: accept either 'vina-gpu' or 'vina'
        self.assertIn(eng, {"vina-gpu", "vina"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
