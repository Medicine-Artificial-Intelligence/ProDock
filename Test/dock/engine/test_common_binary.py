import tempfile
import shutil
import unittest
from pathlib import Path

from prodock.dock.engine.common_binary import BaseBinaryEngine


class _MiniEngine(BaseBinaryEngine):
    # Pick a name that won't exist on PATH to force FileNotFoundError in tests.
    exe_name = "___definitely_missing_binary___"
    supports_autobox = True


class TestCommonBinary(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        # Arrange minimal files
        (self.td / "rec.pdbqt").write_text("RECEPTOR")
        (self.td / "lig.pdbqt").write_text("LIGAND")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_build_cmd_and_missing_executable(self):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt", validate=True)
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((12, 8, 5), (20, 20, 20))
            .set_exhaustiveness(8)
            .set_num_modes(9)
            .set_cpu(4)
            .set_seed(42)
            .set_out(self.td / "out/lig_docked.pdbqt")
            .set_log(self.td / "out/lig.log")
        )
        # The un-resolved command (first token is placeholder exe_name string)
        cmd = eng._build_cmd(override_exhaustiveness=None, override_nposes=None)
        self.assertEqual(cmd[0], _MiniEngine.exe_name)
        self.assertIn("--receptor", cmd)
        self.assertIn("--ligand", cmd)
        self.assertIn("--exhaustiveness", cmd)
        self.assertIn("--num_modes", cmd)
        self.assertIn("--out", cmd)
        self.assertIn("--log", cmd)

        # Running should raise because the binary cannot be resolved
        with self.assertRaises(FileNotFoundError):
            eng.run()


if __name__ == "__main__":
    unittest.main()
