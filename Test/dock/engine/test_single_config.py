import os
import stat
import json
import tempfile
import shutil
import unittest
from pathlib import Path

from prodock.dock.engine.registry import register
from prodock.dock.engine.single import SingleDock
from prodock.dock.engine.common_binary import BaseBinaryEngine
from prodock.dock.engine.config import SingleConfig


class FakeCliEngine(BaseBinaryEngine):
    exe_name = "fakecli"
    supports_autobox = True


class TestSingleConfigHelpers(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        # create fake executable script that will touch out/log
        self.script = self.td / "fake_cli.py"
        self.out_path = self.td / "out_pose.pdbqt"
        self.log_path = self.td / "run.log"
        self.script.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "for i,a in enumerate(sys.argv):\n"
            "    if a in ('--out','--log') and i+1 < len(sys.argv):\n"
            "        open(sys.argv[i+1],'w').write('ok')\n"
            "sys.exit(0)\n"
        )
        os.chmod(self.script, os.stat(self.script).st_mode | stat.S_IEXEC)

        # register factory for engine key "smina" that returns an instance pointing to our script
        def factory():
            inst = FakeCliEngine()
            inst.set_executable(str(self.script))
            return inst

        register("smina", factory)

        # create Data/testcase layout similar to your snippet
        (self.td / "Data/testcase/dock/receptor").mkdir(parents=True, exist_ok=True)
        (self.td / "Data/testcase/dock/ligand").mkdir(parents=True, exist_ok=True)
        (self.td / "Data/testcase/dock/receptor/5N2F.pdbqt").write_text("R")
        (self.td / "Data/testcase/dock/ligand/8HW.pdbqt").write_text("L")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_from_config_and_run_from_config(self):
        cfg = SingleConfig(
            engine="smina",
            receptor=str(self.td / "Data/testcase/dock/receptor/5N2F.pdbqt"),
            ligand=str(self.td / "Data/testcase/dock/ligand/8HW.pdbqt"),
            box=None,
            exhaustiveness=5,
            n_poses=3,
            cpu=1,
            seed=7,
            out=str(self.out_path),
            log=str(self.log_path),
            validate_receptor=False,
        )
        # create instance from dataclass and run
        sd = SingleDock.from_config(cfg)
        res = sd.run()
        # artifacts should point to the paths we passed (script created the files)
        self.assertTrue(self.out_path.exists())
        self.assertTrue(self.log_path.exists())
        self.assertEqual(res.artifacts.out_path, Path(self.out_path))
        self.assertEqual(res.artifacts.log_path, Path(self.log_path))

        # also test file-based workflow (dump json)
        cfg_file = self.td / "single.json"
        cfg_file.write_text(json.dumps(cfg.to_dict()))
        # run_from_config should construct and execute
        SingleDock.run_from_config(str(cfg_file))
        self.assertTrue(self.out_path.exists())
        self.assertTrue(self.log_path.exists())


if __name__ == "__main__":
    unittest.main()
