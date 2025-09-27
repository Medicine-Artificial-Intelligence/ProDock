import os
import stat
import tempfile
import shutil
import unittest
from pathlib import Path

from prodock.dock.engine.common_binary import BaseBinaryEngine


class MiniEngine(BaseBinaryEngine):
    # will override exe_name at runtime in tests
    exe_name = "mini_fake"


class TestExecutableResolutionAndRun(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        # make a small script that will create the --out and --log files we pass
        self.script = self.td / "fake_engine.py"
        self.out_file = self.td / "out_pose.pdbqt"
        self.log_file = self.td / "run.log"
        self.script.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "for i,a in enumerate(sys.argv):\n"
            "    if a in ('--out','--log') and i+1 < len(sys.argv):\n"
            "        open(sys.argv[i+1],'w').write('ok')\n"
            "sys.exit(0)\n"
        )
        # make executable
        st = os.stat(str(self.script))
        os.chmod(str(self.script), st.st_mode | stat.S_IEXEC)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_run_with_explicit_executable_path(self):
        eng = MiniEngine()
        # explicitly set exe_name to the script path
        eng.set_receptor(str(self.td / "rec.pdbqt"))
        eng.set_ligand(str(self.td / "lig.pdbqt"))
        eng.set_box((1, 2, 3), (10, 10, 10))
        eng.set_exhaustiveness(4).set_num_modes(3).set_cpu(1).set_seed(7)
        eng.set_out(str(self.out_file)).set_log(str(self.log_file))
        eng.set_executable(str(self.script))  # will set exe_name to absolute path

        # Build command (first token is placeholder exe_name before resolution)
        cmd = eng._build_cmd(override_exhaustiveness=None, override_nposes=None)
        # placeholder should be the string we set (absolute path)
        self.assertTrue(str(self.script) in cmd[0] or cmd[0] == str(self.script))

        # Running should succeed (script exits 0) and create both files
        eng.run()
        self.assertTrue(self.out_file.exists())
        self.assertTrue(self.log_file.exists())
        # last_called should include the script path
        self.assertIn(str(self.script), eng._last_called)


if __name__ == "__main__":
    unittest.main()
