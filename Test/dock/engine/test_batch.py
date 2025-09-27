import tempfile
import shutil
import unittest
from pathlib import Path

from prodock.dock.engine.registry import register
from prodock.dock.engine.common_binary import BaseBinaryEngine
from prodock.dock.engine.batch import BatchDock


class _SminaFake(BaseBinaryEngine):
    exe_name = "___definitely_missing_binary___"
    supports_autobox = True


class TestBatchDock(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        # Register engine key the BatchDock workers will look up
        register("smina", lambda: _SminaFake())

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_create_tasks_defaults(self):
        rows = [
            {
                "id": "L1",
                "receptor": "rec1.pdbqt",
                "ligand": "lig1.pdbqt",
                "center": [12, 8, 5],
                "size": [20, 20, 20],
            },
            {
                "id": "L2",
                "receptor": "rec2.pdbqt",
                "ligand": "lig2.pdbqt",
            },
        ]
        bd = BatchDock(engine="smina", n_jobs=1, progress=False)
        tasks = bd.create_tasks(
            rows,
            out_dir=self.td / "docked",
            log_dir=self.td / "logs",
            exhaustiveness=8,
            n_poses=9,
            cpu=4,
            seed=42,
        )
        self.assertEqual(len(tasks), 2)
        self.assertTrue(tasks[0].out_path.endswith("L1_docked.pdbqt"))
        self.assertTrue(tasks[0].log_path.endswith("L1.log"))
        self.assertEqual(tasks[0].exhaustiveness, 8)
        self.assertEqual(tasks[0].n_poses, 9)
        self.assertEqual(tasks[0].cpu, 4)
        self.assertEqual(tasks[0].seed, 42)
        self.assertEqual(tasks[0].center, (12, 8, 5))
        self.assertEqual(tasks[0].size, (20, 20, 20))

    def test_run_tasks_fails_cleanly_without_binary(self):
        # Minimal file setup so workers can try to run; we won't actually spawn workers here.
        # Instead we directly invoke worker function on a single dict to avoid forking in tests.
        from prodock.dock.engine.batch import worker_process_job_using_singledock

        # create inputs
        rec = self.td / "rec.pdbqt"
        lig = self.td / "lig.pdbqt"
        rec.write_text("RECEPTOR")
        lig.write_text("LIGAND")

        d = {
            "job_id": "L1",
            "receptor": str(rec),
            "ligand": str(lig),
            "center": (12, 8, 5),
            "size": (20, 20, 20),
            "engine_name": "smina",
            "engine_mode": None,
            "engine_options": {},
            "exhaustiveness": 8,
            "n_poses": 9,
            "cpu": 1,
            "seed": 42,
            "autobox_ref": None,
            "autobox_pad": None,
            "out_path": str(self.td / "out/pose.pdbqt"),
            "log_path": str(self.td / "out/run.log"),
            "retries": 1,
            "timeout": None,
            "tmp_dir": None,
        }

        res = worker_process_job_using_singledock(d)
        self.assertEqual(res["job_id"], "L1")
        self.assertFalse(res["success"])
        # should carry the out/log paths through even on failure
        self.assertTrue(res["out_path"].endswith("pose.pdbqt"))
        self.assertTrue(res["log_path"].endswith("run.log"))
        self.assertIsNotNone(res["error"])


if __name__ == "__main__":
    unittest.main()
