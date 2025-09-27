import unittest
import tempfile
import shutil
import sys
import types
from pathlib import Path
from unittest import mock

from prodock.structure.fetch import fetch_pdb_to_dir


class FakeCmd:
    def __init__(self, fetch_dir):
        self._fetch_dir = Path(fetch_dir)

    def fetch(self, pdb_id, path=None, type=None, async_=0):
        p = Path(path) / f"{pdb_id.lower()}.pdb"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("HEADER")
        return p

    def load(self, *args, **kwargs):
        return None


class NoCreate:
    def fetch(self, pdb_id, path=None, type=None, async_=0):
        return None

    def load(self, *args, **kwargs):
        return None


class TestFetch(unittest.TestCase):
    def test_fetch_success(self):
        td = Path(tempfile.mkdtemp())
        try:
            fake_mod = types.ModuleType("pymol")
            fake_mod.cmd = FakeCmd(td)
            with mock.patch.dict(sys.modules, {"pymol": fake_mod}):
                fetch_dir = td / "fetchme"
                path = fetch_pdb_to_dir("5N2F", fetch_dir)
                self.assertTrue(path.exists())
                self.assertTrue(path.name.lower().endswith(".pdb"))
                self.assertIn("5n2f", path.name.lower())
        finally:
            shutil.rmtree(str(td))

    def test_fetch_not_found(self):
        td = Path(tempfile.mkdtemp())
        try:
            fake_mod = types.ModuleType("pymol")
            fake_mod.cmd = NoCreate()
            with mock.patch.dict(sys.modules, {"pymol": fake_mod}):
                fetch_dir = td / "out"
                with self.assertRaises(FileNotFoundError):
                    fetch_pdb_to_dir("ZZZZ", fetch_dir)
        finally:
            shutil.rmtree(str(td))


if __name__ == "__main__":
    unittest.main()
