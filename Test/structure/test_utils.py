import gzip
import tempfile
import shutil
from pathlib import Path
import unittest

from prodock.structure import utils


class TestUtils(unittest.TestCase):
    def test_decompress_gz(self):
        td = Path(tempfile.mkdtemp())
        try:
            gz = td / "file.pdb.gz"
            content = b"HELLO"
            with gzip.open(gz, "wb") as f:
                f.write(content)

            out = utils.decompress_gz(gz)
            self.assertTrue(out.exists())
            self.assertEqual(out.read_bytes(), content)

            # idempotent
            out2 = utils.decompress_gz(gz)
            self.assertEqual(out2, out)
        finally:
            shutil.rmtree(str(td))

    def test_score_path_for_pdb_search(self):
        td = Path(tempfile.mkdtemp())
        try:
            p = td / "5n2f.pdb"
            p.write_text("")
            allowed = [".pdb", ".ent", ".cif"]
            score = utils.score_path_for_pdb_search(p, "5n2f", allowed)
            self.assertIsInstance(score, int)
        finally:
            shutil.rmtree(str(td))


if __name__ == "__main__":
    unittest.main()
