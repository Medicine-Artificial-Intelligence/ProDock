import shutil
import tempfile
import unittest
from pathlib import Path

from prodock.dock.engine.common_binary import BaseBinaryEngine


class EngineNoAutobox(BaseBinaryEngine):
    exe_name = "___missing___"
    supports_autobox = False


class EngineWithAutobox(BaseBinaryEngine):
    exe_name = "___missing___"
    supports_autobox = True


class TestAutoboxBehavior(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_enable_autobox_raises_when_not_supported(self):
        e = EngineNoAutobox()
        with self.assertRaises(RuntimeError):
            e.enable_autobox("lig.sdf", padding=2.0)

    def test_build_cmd_contains_autobox_flags_when_supported(self):
        e = EngineWithAutobox()
        e.set_receptor("rec.pdbqt").set_ligand("lig.pdbqt")
        e.enable_autobox("lig.pdbqt", padding=3.5)
        e.set_out(str(self.td / "o.pdbqt")).set_log(str(self.td / "l.log"))
        cmd = e._build_cmd(override_exhaustiveness=None, override_nposes=None)
        # autobox flag name from flag_map should be present if autobox_ref set
        # (we only assert presence of the flag strings)
        self.assertIn(e.flag_map["autobox_ligand"], cmd)
        self.assertIn(e.flag_map["autobox_add"], cmd)


if __name__ == "__main__":
    unittest.main()
