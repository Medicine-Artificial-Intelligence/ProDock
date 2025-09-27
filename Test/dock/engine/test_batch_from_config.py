import os
import stat
import tempfile
import shutil
import unittest
from pathlib import Path

from prodock.dock.engine.registry import register
from prodock.dock.engine.common_binary import BaseBinaryEngine
from prodock.dock.engine.batch import worker_process_job_using_singledock, BatchDock
from prodock.dock.engine.config import BatchConfig, LigandTask


class FakeCliEngine(BaseBinaryEngine):
    exe_name = "fakecli"
    supports_autobox = True


class TestBatchWorkerAndFromConfig(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        # fake script that creates out/log
        self.script = self.td / "fake_cli.py"
        self.script.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "for i,a in enumerate(sys.argv):\n"
            "    if a in ('--out','--log') and i+1 < len(sys.argv):\n"
            "        open(sys.argv[i+1],'w').write('ok')\n"
            "sys.exit(0)\n"
        )
        os.chmod(self.script, os.stat(self.script).st_mode | stat.S_IEXEC)

        # register engine factory that returns instance pointing to script
        def factory():
            e = FakeCliEngine()
            e.set_executable(str(self.script))
            return e

        register("smina", factory)

        # create receptor/ligand files
        rec = self.td / "rec.pdbqt"
        lig = self.td / "lig.pdbqt"
        rec.write_text("R")
        lig.write_text("L")
        self.rec = rec
        self.lig = lig

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_worker_success_returns_success_true(self):
        out_path = str(self.td / "out" / "pose.pdbqt")
        log_path = str(self.td / "out" / "run.log")
        task = {
            "job_id": "L1",
            "receptor": str(self.rec),
            "ligand": str(self.lig),
            "center": (1, 2, 3),
            "size": (10, 10, 10),
            "engine_name": "smina",
            "engine_mode": None,
            "engine_options": {},
            "exhaustiveness": 4,
            "n_poses": 3,
            "cpu": 1,
            "seed": 7,
            "autobox_ref": None,
            "autobox_pad": None,
            "out_path": out_path,
            "log_path": log_path,
            "retries": 1,
            "timeout": None,
            "tmp_dir": None,
        }
        res = worker_process_job_using_singledock(task)
        self.assertEqual(res["job_id"], "L1")
        self.assertTrue(res["out_path"].endswith("pose.pdbqt"))
        self.assertTrue(res["log_path"].endswith("run.log"))
        # success should be True because the fake script exits 0
        self.assertTrue(res["success"])
        # output files should have been written by the script
        self.assertTrue(Path(out_path).exists())
        self.assertTrue(Path(log_path).exists())

    def test_batch_from_config_creates_proper_instance(self):
        # minimal BatchConfig -> ensure from_config sets engine & n_jobs correctly
        cfg = BatchConfig(
            engine="smina",
            n_jobs=2,
            rows=[LigandTask(id="L1", receptor=str(self.rec), ligand=str(self.lig))],
        )
        bd = BatchDock.from_config(cfg)
        self.assertEqual(bd._engine_name, "smina")
        self.assertEqual(bd.n_jobs, 2)


if __name__ == "__main__":
    unittest.main()
