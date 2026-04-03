import json
import tempfile
import unittest
from pathlib import Path

import prodock.dock.batch as batch_mod
from prodock.dock import BatchDock, Box, LigandSpec, ReceptorSpec, SoftwareSpec
from prodock.dock.config import BatchConfig, DockRow


class TestBatchDock(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

        self._orig_worker = batch_mod.worker_process_job_using_singledock

        def fake_worker(task_dict):
            return {
                "job_id": task_dict["job_id"],
                "receptor_id": task_dict["receptor_id"],
                "engine_name": task_dict["engine_name"],
                "ligand_id": task_dict["ligand_id"],
                "success": True,
                "out_path": task_dict.get("out_path"),
                "log_path": task_dict.get("log_path"),
                "called": f"FAKE {task_dict['engine_name']}",
                "error": None,
                "traceback": None,
                "elapsed": 0.01,
                "metadata": {"mocked": True},
            }

        batch_mod.worker_process_job_using_singledock = fake_worker

    def tearDown(self) -> None:
        batch_mod.worker_process_job_using_singledock = self._orig_worker
        self.tmpdir.cleanup()

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(payload, indent=2))
        return path

    def test_init_defaults(self) -> None:
        runner = BatchDock()
        self.assertEqual(runner.default_engine, "vina")
        self.assertEqual(runner.n_jobs, 1)
        self.assertEqual(runner.default_retries, 1)

    def test_default_out_and_log(self) -> None:
        runner = BatchDock(engine="qvina")
        out_path = runner._default_out_for("4WKQ", "qvina", "erlotinib")
        log_path = runner._default_log_for("4WKQ", "qvina", "erlotinib")
        self.assertEqual(
            out_path,
            Path("docked") / "4WKQ" / "qvina" / "erlotinib_docked.pdbqt",
        )
        self.assertEqual(
            log_path,
            Path("logs") / "4WKQ" / "qvina" / "erlotinib.log",
        )

    def test_first_not_none(self) -> None:
        self.assertEqual(BatchDock._first_not_none(None, None, 5, 6), 5)
        self.assertIsNone(BatchDock._first_not_none(None, None))

    def test_resolve_output_path_with_base_dir(self) -> None:
        runner = BatchDock()
        path = runner._resolve_output_path(
            base_dir="custom_out",
            global_dir=None,
            receptor_id="4WKQ",
            engine_name="qvina",
            ligand_id="erlotinib",
            suffix="_docked.pdbqt",
        )
        self.assertEqual(path, str(Path("custom_out") / "erlotinib_docked.pdbqt"))

    def test_resolve_output_path_with_base_dir_append_engine(self) -> None:
        runner = BatchDock()
        path = runner._resolve_output_path(
            base_dir="custom_out",
            global_dir=None,
            receptor_id="4WKQ",
            engine_name="qvina",
            ligand_id="erlotinib",
            suffix="_docked.pdbqt",
            append_engine_to_base=True,
        )
        self.assertEqual(
            path,
            str(Path("custom_out") / "qvina" / "erlotinib_docked.pdbqt"),
        )

    def test_resolve_output_path_with_global_dir(self) -> None:
        runner = BatchDock()
        path = runner._resolve_output_path(
            base_dir=None,
            global_dir=Path("global_out"),
            receptor_id="4WKQ",
            engine_name="qvina",
            ligand_id="erlotinib",
            suffix=".log",
        )
        self.assertEqual(
            path,
            str(Path("global_out") / "4WKQ" / "qvina" / "erlotinib.log"),
        )

    def test_create_tasks_from_rows_dict(self) -> None:
        runner = BatchDock(engine="qvina", default_retries=2)
        tasks = runner.create_tasks(
            [
                {
                    "id": "erlotinib",
                    "receptor": "Data/testcase/4WKQ/receptor/4WKQ.pdbqt",
                    "ligand": "Data/testcase/4WKQ/ligand/erlotinib.pdbqt",
                    "center": [2.865, 193.257, 21.367],
                    "size": [27.091, 27.091, 27.091],
                    "engine_options": {"foo": "bar"},
                }
            ],
            out_dir="out",
            log_dir="log",
            exhaustiveness=8,
            n_poses=5,
            cpu=1,
            seed=42,
            retries=3,
            engine_options={"global_opt": True},
            executable="/usr/bin/qvina",
        )

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.engine_name, "qvina")
        self.assertEqual(task.ligand_id, "erlotinib")
        self.assertEqual(task.center, (2.865, 193.257, 21.367))
        self.assertEqual(task.size, (27.091, 27.091, 27.091))
        self.assertEqual(task.exhaustiveness, 8)
        self.assertEqual(task.n_poses, 5)
        self.assertEqual(task.cpu, 1)
        self.assertEqual(task.seed, 42)
        self.assertEqual(task.retries, 3)
        self.assertEqual(task.executable, "/usr/bin/qvina")
        self.assertTrue(task.engine_options["global_opt"])
        self.assertEqual(task.engine_options["foo"], "bar")
        self.assertEqual(task.out_path, str(Path("out") / "erlotinib_docked.pdbqt"))
        self.assertEqual(task.log_path, str(Path("log") / "erlotinib.log"))

    def test_create_tasks_row_specific_overrides(self) -> None:
        runner = BatchDock(engine="qvina", default_retries=1)
        row = DockRow(
            id="erlotinib",
            receptor="Data/testcase/4WKQ/receptor/4WKQ.pdbqt",
            ligand="Data/testcase/4WKQ/ligand/erlotinib.pdbqt",
            box=Box(center=(1.0, 2.0, 3.0), size=(10.0, 10.0, 10.0)),
            exhaustiveness=12,
            n_poses=9,
            cpu=4,
            seed=99,
            retries=7,
            out="my_out.pdbqt",
            log="my_log.log",
        )
        tasks = runner.create_tasks(
            [row],
            exhaustiveness=8,
            n_poses=5,
            cpu=1,
            seed=42,
            retries=3,
        )
        task = tasks[0]
        self.assertEqual(task.center, (1.0, 2.0, 3.0))
        self.assertEqual(task.size, (10.0, 10.0, 10.0))
        self.assertEqual(task.exhaustiveness, 12)
        self.assertEqual(task.n_poses, 9)
        self.assertEqual(task.cpu, 4)
        self.assertEqual(task.seed, 99)
        self.assertEqual(task.retries, 7)
        self.assertEqual(task.out_path, "my_out.pdbqt")
        self.assertEqual(task.log_path, "my_log.log")

    def test_create_tasks_autobox(self) -> None:
        runner = BatchDock(engine="qvina")
        tasks = runner.create_tasks(
            [
                {
                    "id": "erlotinib",
                    "receptor": "protein.pdbqt",
                    "ligand": "ligand.pdbqt",
                    "autobox_ref": "ref.pdbqt",
                    "autobox_pad": 4.0,
                }
            ]
        )
        task = tasks[0]
        self.assertEqual(task.autobox_ref, "ref.pdbqt")
        self.assertEqual(task.autobox_pad, 4.0)
        self.assertIsNone(task.center)
        self.assertIsNone(task.size)

    def test_create_tasks_from_receptors(self) -> None:
        runner = BatchDock(default_retries=2)
        receptors = [
            ReceptorSpec(
                id="4WKQ",
                receptor="Data/testcase/4WKQ/receptor/4WKQ.pdbqt",
                box=Box(center=(2.865, 193.257, 21.367), size=(27.091, 27.091, 27.091)),
                softwares=[
                    SoftwareSpec(
                        name="qvina",
                        exhaustiveness=8,
                        n_poses=5,
                        cpu=1,
                        seed=42,
                        executable="/usr/bin/qvina",
                        engine_options={"soft_opt": True},
                        ligands=[
                            LigandSpec(
                                id="erlotinib",
                                ligand="Data/testcase/4WKQ/ligand/erlotinib.pdbqt",
                                engine_options={"lig_opt": "x"},
                            )
                        ],
                    )
                ],
            )
        ]

        tasks = runner.create_tasks_from_receptors(
            receptors,
            out_dir="docked",
            log_dir="logs",
        )
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.job_id, "4WKQ:qvina:erlotinib")
        self.assertEqual(task.receptor_id, "4WKQ")
        self.assertEqual(task.engine_name, "qvina")
        self.assertEqual(task.ligand_id, "erlotinib")
        self.assertEqual(task.center, (2.865, 193.257, 21.367))
        self.assertEqual(task.size, (27.091, 27.091, 27.091))
        self.assertEqual(task.exhaustiveness, 8)
        self.assertEqual(task.n_poses, 5)
        self.assertEqual(task.cpu, 1)
        self.assertEqual(task.seed, 42)
        self.assertEqual(task.executable, "/usr/bin/qvina")
        self.assertTrue(task.engine_options["soft_opt"])
        self.assertEqual(task.engine_options["lig_opt"], "x")
        self.assertEqual(
            task.out_path,
            str(Path("docked") / "4WKQ" / "qvina" / "erlotinib_docked.pdbqt"),
        )
        self.assertEqual(
            task.log_path,
            str(Path("logs") / "4WKQ" / "qvina" / "erlotinib.log"),
        )
        self.assertEqual(task.retries, 2)

    def test_create_tasks_from_receptors_with_dir_precedence(self) -> None:
        runner = BatchDock(default_retries=1)
        receptors = [
            ReceptorSpec(
                id="4WKQ",
                receptor="protein.pdbqt",
                out_dir="rec_out",
                log_dir="rec_log",
                softwares=[
                    SoftwareSpec(
                        name="qvina",
                        ligands=[LigandSpec(id="erlotinib", ligand="ligand.pdbqt")],
                    )
                ],
            )
        ]
        tasks = runner.create_tasks_from_receptors(receptors)
        task = tasks[0]
        self.assertEqual(
            task.out_path,
            str(Path("rec_out") / "qvina" / "erlotinib_docked.pdbqt"),
        )
        self.assertEqual(
            task.log_path,
            str(Path("rec_log") / "qvina" / "erlotinib.log"),
        )

    def test_create_tasks_from_receptors_software_dir_overrides_receptor_dir(
        self,
    ) -> None:
        runner = BatchDock()
        receptors = [
            ReceptorSpec(
                id="4WKQ",
                receptor="protein.pdbqt",
                out_dir="rec_out",
                log_dir="rec_log",
                softwares=[
                    SoftwareSpec(
                        name="qvina",
                        out_dir="soft_out",
                        log_dir="soft_log",
                        ligands=[LigandSpec(id="erlotinib", ligand="ligand.pdbqt")],
                    )
                ],
            )
        ]
        tasks = runner.create_tasks_from_receptors(receptors)
        task = tasks[0]
        self.assertEqual(
            task.out_path,
            str(Path("soft_out") / "erlotinib_docked.pdbqt"),
        )
        self.assertEqual(
            task.log_path,
            str(Path("soft_log") / "erlotinib.log"),
        )

    def test_run_tasks_serial(self) -> None:
        runner = BatchDock(n_jobs=1, progress=False)
        tasks = [
            batch_mod.DockTask(
                job_id="4WKQ:qvina:erlotinib",
                receptor_id="4WKQ",
                engine_name="qvina",
                ligand_id="erlotinib",
                receptor="protein.pdbqt",
                ligand="ligand.pdbqt",
            )
        ]
        results = runner.run_tasks(tasks)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].called, "FAKE qvina")
        self.assertTrue(results[0].metadata["mocked"])

    def test_run_tasks_empty(self) -> None:
        runner = BatchDock(n_jobs=1)
        results = runner.run_tasks([])
        self.assertEqual(results, [])

    def test_run_flat_wrapper(self) -> None:
        runner = BatchDock(engine="qvina", n_jobs=1, progress=False)
        results = runner.run(
            [
                {
                    "id": "erlotinib",
                    "receptor": "protein.pdbqt",
                    "ligand": "ligand.pdbqt",
                }
            ]
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].job_id, "protein:qvina:erlotinib")

    def test_run_receptors_wrapper(self) -> None:
        runner = BatchDock(n_jobs=1, progress=False)
        receptors = [
            {
                "id": "4WKQ",
                "receptor": "protein.pdbqt",
                "softwares": [
                    {
                        "name": "qvina",
                        "ligands": [{"id": "erlotinib", "ligand": "ligand.pdbqt"}],
                    }
                ],
            }
        ]
        results = runner.run_receptors(receptors)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].job_id, "4WKQ:qvina:erlotinib")

    def test_from_config_with_dict(self) -> None:
        runner = BatchDock.from_config(
            {
                "engine": "qvina",
                "n_jobs": 3,
                "progress": False,
                "default_retries": 5,
                "timeout": 12.5,
                "tmp_root": "tmp_root_dir",
            }
        )
        self.assertEqual(runner.default_engine, "qvina")
        self.assertEqual(runner.n_jobs, 3)
        self.assertFalse(runner.progress)
        self.assertEqual(runner.default_retries, 5)
        self.assertEqual(runner.timeout, 12.5)
        self.assertEqual(runner.tmp_root, Path("tmp_root_dir"))
        self.assertIsInstance(runner._config, BatchConfig)

    def test_from_config_with_file(self) -> None:
        path = self.write_json(
            "batch.json",
            {
                "engine": "qvina",
                "n_jobs": 2,
                "progress": False,
                "default_retries": 2,
            },
        )
        runner = BatchDock.from_config(path)
        self.assertEqual(runner.default_engine, "qvina")
        self.assertEqual(runner.n_jobs, 2)
        self.assertFalse(runner.progress)

    def test_run_from_config_rows(self) -> None:
        results = BatchDock.run_from_config(
            {
                "engine": "qvina",
                "n_jobs": 1,
                "progress": False,
                "rows": [
                    {
                        "id": "erlotinib",
                        "receptor": "protein.pdbqt",
                        "ligand": "ligand.pdbqt",
                    }
                ],
                "out_dir": "out",
                "log_dir": "log",
                "exhaustiveness": 8,
                "n_poses": 5,
                "cpu": 1,
                "seed": 42,
                "retries": 2,
                "engine_options": {"x": 1},
            }
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(
            results[0].out_path, str(Path("out") / "erlotinib_docked.pdbqt")
        )
        self.assertEqual(results[0].log_path, str(Path("log") / "erlotinib.log"))

    def test_run_from_config_receptors(self) -> None:
        results = BatchDock.run_from_config(
            {
                "n_jobs": 1,
                "progress": False,
                "receptors": [
                    {
                        "id": "4WKQ",
                        "receptor": "protein.pdbqt",
                        "softwares": [
                            {
                                "name": "qvina",
                                "ligands": [
                                    {"id": "erlotinib", "ligand": "ligand.pdbqt"}
                                ],
                            }
                        ],
                    }
                ],
                "out_dir": "docked",
                "log_dir": "logs",
            }
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].job_id, "4WKQ:qvina:erlotinib")


if __name__ == "__main__":
    unittest.main()
