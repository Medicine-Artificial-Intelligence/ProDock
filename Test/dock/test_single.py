import json
import tempfile
import unittest
from pathlib import Path

import prodock.dock.single as single_mod
from prodock.dock.base import RunArtifacts
from prodock.dock.config import Box, SingleConfig
from prodock.dock.single import SingleDock, SingleResult


class DummyBackend:
    def __init__(self):
        self.receptor = None
        self.receptor_validate = False
        self.ligand = None
        self.center = None
        self.size = None
        self.autobox_reference = None
        self.autobox_padding = None
        self.exhaustiveness = None
        self.num_modes = None
        self.cpu = None
        self.seed = None
        self.out = None
        self.log = None
        self.exe_name = "qvina"
        self.called = None
        self.metadata = {"backend": "dummy-qvina"}

    def set_receptor(self, path, validate=False):
        self.receptor = str(path)
        self.receptor_validate = validate

    def set_ligand(self, path):
        self.ligand = str(path)

    def set_box(self, center, size):
        self.center = tuple(center)
        self.size = tuple(size)

    def enable_autobox(self, reference_file, padding=None):
        self.autobox_reference = str(reference_file)
        self.autobox_padding = padding

    def set_exhaustiveness(self, value):
        self.exhaustiveness = value

    def set_num_modes(self, value):
        self.num_modes = value

    def set_cpu(self, value):
        self.cpu = value

    def set_seed(self, value):
        self.seed = value

    def set_out(self, value):
        self.out = Path(value)

    def set_log(self, value):
        self.log = Path(value)

    def set_executable(self, exe_path):
        self.exe_name = str(exe_path)

    def run(self, *, exhaustiveness=None, n_poses=None):
        self.called = {
            "exhaustiveness": exhaustiveness,
            "n_poses": n_poses,
            "receptor": self.receptor,
            "ligand": self.ligand,
            "out": str(self.out) if self.out is not None else None,
            "log": str(self.log) if self.log is not None else None,
        }


class DummyBackendNoSetExecutable(DummyBackend):
    def set_executable(self, exe_path):
        raise AttributeError("set_executable intentionally unsupported")


class TestSingleDock(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

        self._orig_factory = single_mod.get_factory

        def fake_factory(name):
            self.assertEqual(name, "qvina")
            return DummyBackend

        single_mod.get_factory = fake_factory

    def tearDown(self) -> None:
        single_mod.get_factory = self._orig_factory
        self.tmpdir.cleanup()

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(payload, indent=2))
        return path

    def test_init_uses_lowercase_engine(self) -> None:
        dock = SingleDock("QVina")
        self.assertEqual(dock.engine, "qvina")
        self.assertIsInstance(dock._backend, DummyBackend)

    def test_chainable_setters(self) -> None:
        dock = SingleDock("qvina")
        returned = (
            dock.set_receptor("protein.pdbqt", validate=True)
            .set_ligand("ligand.pdbqt")
            .set_box((1.0, 2.0, 3.0), (20.0, 20.0, 20.0))
            .enable_autobox("ref.pdbqt", padding=4.0)
            .set_exhaustiveness(8)
            .set_num_modes(10)
            .set_cpu(2)
            .set_seed(42)
            .set_out("out.pdbqt")
            .set_log("dock.log")
        )
        self.assertIs(returned, dock)
        self.assertEqual(dock._backend.receptor, "protein.pdbqt")
        self.assertTrue(dock._backend.receptor_validate)
        self.assertEqual(dock._backend.ligand, "ligand.pdbqt")
        self.assertEqual(dock._backend.center, (1.0, 2.0, 3.0))
        self.assertEqual(dock._backend.size, (20.0, 20.0, 20.0))
        self.assertEqual(dock._backend.autobox_reference, "ref.pdbqt")
        self.assertEqual(dock._backend.autobox_padding, 4.0)
        self.assertEqual(dock._backend.exhaustiveness, 8)
        self.assertEqual(dock._backend.num_modes, 10)
        self.assertEqual(dock._backend.cpu, 2)
        self.assertEqual(dock._backend.seed, 42)
        self.assertEqual(dock._out, Path("out.pdbqt"))
        self.assertEqual(dock._log, Path("dock.log"))

    def test_set_executable_calls_backend_method(self) -> None:
        dock = SingleDock("qvina")
        dock.set_executable("/usr/bin/qvina2")
        self.assertEqual(dock._backend.exe_name, "/usr/bin/qvina2")

    def test_apply_engine_options_uses_facade_and_backend_setters(self) -> None:
        dock = SingleDock("qvina")
        dock.apply_engine_options(
            {
                "cpu": 4,
                "seed": 123,
                "num_modes": 7,
                "custom_flag": True,
            }
        )
        self.assertEqual(dock._backend.cpu, 4)
        self.assertEqual(dock._backend.seed, 123)
        self.assertEqual(dock._backend.num_modes, 7)
        self.assertTrue(getattr(dock._backend, "custom_flag"))

    def test_run_returns_single_result_and_artifacts(self) -> None:
        dock = (
            SingleDock("qvina")
            .set_receptor("protein.pdbqt")
            .set_ligand("ligand.pdbqt")
            .set_out("out.pdbqt")
            .set_log("dock.log")
        )
        result = dock.run(exhaustiveness=6, n_poses=5)

        self.assertIsInstance(result, SingleResult)
        self.assertIsInstance(result.artifacts, RunArtifacts)
        self.assertEqual(result.artifacts.out_path, Path("out.pdbqt"))
        self.assertEqual(result.artifacts.log_path, Path("dock.log"))
        self.assertEqual(result.artifacts.called["exhaustiveness"], 6)
        self.assertEqual(result.artifacts.called["n_poses"], 5)
        self.assertEqual(result.artifacts.metadata["backend"], "dummy-qvina")

    def test_from_config_with_dict(self) -> None:
        dock = SingleDock.from_config(
            {
                "engine": "qvina",
                "receptor": "protein.pdbqt",
                "ligand": "ligand.pdbqt",
                "box": {"center": [1, 2, 3], "size": [20, 20, 20]},
                "exhaustiveness": 8,
                "n_poses": 9,
                "cpu": 2,
                "seed": 42,
                "out": "out.pdbqt",
                "log": "dock.log",
                "engine_options": {"custom_attr": "ok"},
                "validate_receptor": True,
            }
        )
        self.assertEqual(dock.engine, "qvina")
        self.assertEqual(dock._backend.receptor, "protein.pdbqt")
        self.assertTrue(dock._backend.receptor_validate)
        self.assertEqual(dock._backend.ligand, "ligand.pdbqt")
        self.assertEqual(dock._backend.center, (1.0, 2.0, 3.0))
        self.assertEqual(dock._backend.size, (20.0, 20.0, 20.0))
        self.assertEqual(dock._backend.exhaustiveness, 8)
        self.assertEqual(dock._backend.num_modes, 9)
        self.assertEqual(dock._backend.cpu, 2)
        self.assertEqual(dock._backend.seed, 42)
        self.assertEqual(dock._out, Path("out.pdbqt"))
        self.assertEqual(dock._log, Path("dock.log"))
        self.assertEqual(getattr(dock._backend, "custom_attr"), "ok")

    def test_from_config_with_single_config_object(self) -> None:
        cfg = SingleConfig(
            engine="qvina",
            receptor="protein.pdbqt",
            ligand="ligand.pdbqt",
            box=Box(center=(1.0, 2.0, 3.0), size=(10.0, 10.0, 10.0)),
            out="out.pdbqt",
            log="dock.log",
        )
        dock = SingleDock.from_config(cfg)
        self.assertEqual(dock._backend.receptor, "protein.pdbqt")
        self.assertEqual(dock._backend.ligand, "ligand.pdbqt")
        self.assertEqual(dock._backend.center, (1.0, 2.0, 3.0))
        self.assertEqual(dock._backend.size, (10.0, 10.0, 10.0))

    def test_from_config_with_file(self) -> None:
        path = self.write_json(
            "config.json",
            {
                "engine": "qvina",
                "receptor": "protein.pdbqt",
                "ligand": "ligand.pdbqt",
                "out": "out.pdbqt",
                "log": "dock.log",
            },
        )
        dock = SingleDock.from_config(path)
        self.assertEqual(dock._backend.receptor, "protein.pdbqt")
        self.assertEqual(dock._backend.ligand, "ligand.pdbqt")
        self.assertEqual(dock._out, Path("out.pdbqt"))
        self.assertEqual(dock._log, Path("dock.log"))

    def test_run_from_config(self) -> None:
        result = SingleDock.run_from_config(
            {
                "engine": "qvina",
                "receptor": "protein.pdbqt",
                "ligand": "ligand.pdbqt",
                "exhaustiveness": 11,
                "n_poses": 4,
            }
        )
        self.assertEqual(result.artifacts.called["exhaustiveness"], 11)
        self.assertEqual(result.artifacts.called["n_poses"], 4)

    def test_apply_config_mutates_existing_instance(self) -> None:
        dock = SingleDock("qvina")
        returned = dock.apply_config(
            {
                "engine": "qvina",
                "receptor": "protein.pdbqt",
                "ligand": "ligand.pdbqt",
                "cpu": 6,
            }
        )
        self.assertIs(returned, dock)
        self.assertEqual(dock._backend.receptor, "protein.pdbqt")
        self.assertEqual(dock._backend.ligand, "ligand.pdbqt")
        self.assertEqual(dock._backend.cpu, 6)

    def test_load_config_alias(self) -> None:
        path = self.write_json(
            "config.json",
            {
                "engine": "qvina",
                "receptor": "protein.pdbqt",
                "ligand": "ligand.pdbqt",
                "cpu": 3,
            },
        )
        dock = SingleDock("qvina").load_config(path)
        self.assertEqual(dock._backend.receptor, "protein.pdbqt")
        self.assertEqual(dock._backend.ligand, "ligand.pdbqt")
        self.assertEqual(dock._backend.cpu, 3)

    def test_run_with_config_prefer_config(self) -> None:
        dock = SingleDock("qvina").set_receptor("old_receptor.pdbqt")
        result = dock.run_with_config(
            {
                "engine": "qvina",
                "receptor": "new_receptor.pdbqt",
                "ligand": "ligand.pdbqt",
                "exhaustiveness": 5,
                "n_poses": 2,
            },
            prefer="config",
        )
        self.assertEqual(result.artifacts.called["receptor"], "new_receptor.pdbqt")
        self.assertEqual(result.artifacts.called["ligand"], "ligand.pdbqt")
        self.assertEqual(result.artifacts.called["exhaustiveness"], 5)
        self.assertEqual(result.artifacts.called["n_poses"], 2)
        self.assertEqual(dock._backend.receptor, "old_receptor.pdbqt")

    def test_run_with_config_prefer_instance(self) -> None:
        dock = SingleDock("qvina").set_receptor("old_receptor.pdbqt")
        result = dock.run_with_config(
            {
                "engine": "qvina",
                "receptor": "new_receptor.pdbqt",
                "ligand": "ligand.pdbqt",
                "exhaustiveness": 7,
                "n_poses": 3,
            },
            prefer="instance",
        )
        self.assertEqual(dock._backend.receptor, "new_receptor.pdbqt")
        self.assertEqual(dock._backend.ligand, "ligand.pdbqt")
        self.assertEqual(result.artifacts.called["receptor"], "new_receptor.pdbqt")
        self.assertEqual(result.artifacts.called["exhaustiveness"], 7)
        self.assertEqual(result.artifacts.called["n_poses"], 3)

    def test_run_with_config_invalid_prefer_raises(self) -> None:
        dock = SingleDock("qvina")
        with self.assertRaises(ValueError):
            dock.run_with_config({"engine": "qvina"}, prefer="bad")

    def test_repr(self) -> None:
        dock = SingleDock("qvina")
        self.assertEqual(repr(dock), "<SingleDock engine=qvina>")

    def test_requested_chain_style_example(self) -> None:
        config_path = self.write_json(
            "config.json",
            {
                "engine": "qvina",
                "box": {
                    "center": [2.865, 193.257, 21.367],
                    "size": [27.091, 27.091, 27.091],
                },
                "exhaustiveness": 8,
                "n_poses": 5,
                "cpu": 1,
                "seed": 42,
            },
        )

        dock = (
            SingleDock("qvina")
            .set_receptor("./Data/testcase/4WKQ/receptor/4WKQ.pdbqt", validate=True)
            .set_ligand("./Data/testcase/4WKQ/ligand/erlotinib.pdbqt")
            .load_config(config_path)
            .set_out("tmp/erlotinib_qvina.pdbqt")
            .set_log("tmp/erlotinib_qvina.log")
        )

        self.assertEqual(
            dock._backend.receptor,
            "./Data/testcase/4WKQ/receptor/4WKQ.pdbqt",
        )
        self.assertTrue(dock._backend.receptor_validate)
        self.assertEqual(
            dock._backend.ligand,
            "./Data/testcase/4WKQ/ligand/erlotinib.pdbqt",
        )
        self.assertEqual(dock._backend.center, (2.865, 193.257, 21.367))
        self.assertEqual(dock._backend.size, (27.091, 27.091, 27.091))
        self.assertEqual(dock._backend.exhaustiveness, 8)
        self.assertEqual(dock._backend.num_modes, 5)
        self.assertEqual(dock._backend.cpu, 1)
        self.assertEqual(dock._backend.seed, 42)
        self.assertEqual(dock._out, Path("tmp/erlotinib_qvina.pdbqt"))
        self.assertEqual(dock._log, Path("tmp/erlotinib_qvina.log"))


class TestSingleDockQVinaReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qvina_exe = True
        cls.receptor = Path("./Data/testcase/4WKQ/receptor/4WKQ.pdbqt")
        cls.ligand = Path("./Data/testcase/4WKQ/ligand/erlotinib.pdbqt")

    def setUp(self) -> None:
        if self.qvina_exe is None:
            self.skipTest("qvina executable not found in PATH")
        if not self.receptor.exists():
            self.skipTest(f"Missing receptor testcase: {self.receptor}")
        if not self.ligand.exists():
            self.skipTest(f"Missing ligand testcase: {self.ligand}")

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        if hasattr(self, "tmpdir"):
            self.tmpdir.cleanup()

    def test_real_qvina_backend_run(self) -> None:
        """
        Run one real docking job with the actual qvina backend.

        This is an integration test and should stay minimal so it remains fast.
        """
        out_path = self.tmp / "erlotinib_qvina_out.pdbqt"
        log_path = self.tmp / "erlotinib_qvina.log"

        result = (
            SingleDock("qvina")
            .set_receptor(self.receptor, validate=True)
            .set_ligand(self.ligand)
            .set_box(
                (2.865, 193.257, 21.367),
                (27.091, 27.091, 27.091),
            )
            .set_cpu(1)
            .set_seed(42)
            .set_exhaustiveness(1)
            .set_num_modes(1)
            .set_out(out_path)
            .set_log(log_path)
            .run(exhaustiveness=1, n_poses=1)
        )

        self.assertTrue(out_path.exists(), f"Expected output file: {out_path}")
        self.assertTrue(log_path.exists(), f"Expected log file: {log_path}")

        out_text = out_path.read_text(errors="ignore")
        log_text = log_path.read_text(errors="ignore")

        self.assertTrue(len(out_text.strip()) > 0, "Output PDBQT is empty")
        self.assertTrue(len(log_text.strip()) > 0, "Log file is empty")

        self.assertIn("MODEL", out_text)
        self.assertIsNotNone(result.artifacts.out_path)
        self.assertIsNotNone(result.artifacts.log_path)
        self.assertEqual(result.artifacts.out_path, out_path)
        self.assertEqual(result.artifacts.log_path, log_path)


if __name__ == "__main__":
    unittest.main()
