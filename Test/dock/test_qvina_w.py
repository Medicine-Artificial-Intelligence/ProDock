from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prodock.dock.qvina_w import QVinaWEngine


class TestQVinaWEngineUnit(unittest.TestCase):
    """Unit tests for :class:`prodock.dock.engine.qvina.QVinaWEngine`."""

    def setUp(self) -> None:
        """
        Create a temporary workspace for isolated path-based tests.
        """
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        """
        Remove the temporary workspace.
        """
        self.tmpdir.cleanup()

    def test_init_defaults(self) -> None:
        """
        Verify inherited defaults are initialized correctly.

        Example
        -------
        .. code-block:: python

            engine = QVinaWEngine()
        """
        engine = QVinaWEngine()

        self.assertEqual(engine.exe_name, "qvina-w")
        self.assertFalse(engine.supports_autobox)

        self.assertIsNone(engine._receptor)
        self.assertIsNone(engine._ligand)
        self.assertIsNone(engine._center)
        self.assertIsNone(engine._size)
        self.assertIsNone(engine._exhaustiveness)
        self.assertIsNone(engine._num_modes)
        self.assertIsNone(engine._cpu)
        self.assertIsNone(engine._seed)
        self.assertIsNone(engine._out)
        self.assertIsNone(engine._log)
        self.assertIsNone(engine._autobox_ref)
        self.assertIsNone(engine._autobox_pad)
        self.assertEqual(engine._extra_args, [])
        self.assertIsNone(engine.called)

    def test_flag_map_does_not_include_autobox_flags(self) -> None:
        """
        Verify QVina removes autobox-specific flags from the inherited mapping.
        """
        self.assertNotIn("autobox_ligand", QVinaWEngine.flag_map)
        self.assertNotIn("autobox_add", QVinaWEngine.flag_map)

    def test_chainable_setters(self) -> None:
        """
        Verify inherited setters are chainable.

        Example
        -------
        .. code-block:: python

            engine = (
                QVinaWEngine()
                .set_receptor("rec.pdbqt", validate=True)
                .set_ligand("lig.pdbqt")
                .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
                .set_cpu(4)
                .set_seed(42)
                .set_exhaustiveness(16)
                .set_num_modes(20)
                .set_out("poses.pdbqt")
                .set_log("dock.log")
            )
        """
        receptor = self.tmp / "rec.pdbqt"
        ligand = self.tmp / "lig.pdbqt"
        out = self.tmp / "poses.pdbqt"
        log = self.tmp / "dock.log"

        receptor.write_text("RECEPTOR\n", encoding="utf-8")
        ligand.write_text("LIGAND\n", encoding="utf-8")

        engine = (
            QVinaWEngine()
            .set_receptor(receptor, validate=True)
            .set_ligand(ligand)
            .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
            .set_cpu(4)
            .set_seed(42)
            .set_exhaustiveness(16)
            .set_num_modes(20)
            .set_out(out)
            .set_log(log)
            .set_timeout(60.0)
            .set_extra_args("--some-flag", "value")
        )

        self.assertEqual(engine._receptor, receptor)
        self.assertEqual(engine._ligand, ligand)
        self.assertEqual(engine._center, (1.0, 2.0, 3.0))
        self.assertEqual(engine._size, (10.0, 11.0, 12.0))
        self.assertEqual(engine._cpu, 4)
        self.assertEqual(engine._seed, 42)
        self.assertEqual(engine._exhaustiveness, 16)
        self.assertEqual(engine._num_modes, 20)
        self.assertEqual(engine._out, out)
        self.assertEqual(engine._log, log)
        self.assertEqual(engine._timeout, 60.0)
        self.assertEqual(engine._extra_args, ["--some-flag", "value"])

    def test_enable_autobox_raises(self) -> None:
        """
        Verify autoboxing is rejected for QVina.
        """
        ref = self.tmp / "ref.sdf"
        ref.write_text("REF\n", encoding="utf-8")

        engine = QVinaWEngine()
        with self.assertRaises(RuntimeError):
            engine.enable_autobox(ref, padding=4.0)

    def test_load_config_dict_applies_values(self) -> None:
        """
        Verify inherited :meth:`load_config_dict` applies shared parameters.

        Example
        -------
        .. code-block:: python

            engine = QVinaWEngine().load_config_dict(
                {
                    "box": {
                        "center": [2.865, 193.257, 21.367],
                        "size": [27.091, 27.091, 27.091],
                    },
                    "cpu": 4,
                    "seed": 42,
                    "exhaustiveness": 16,
                    "n_poses": 20,
                }
            )
        """
        config = {
            "box": {
                "center": [2.865, 193.257, 21.367],
                "size": [27.091, 27.091, 27.091],
            },
            "cpu": 4,
            "seed": 42,
            "exhaustiveness": 16,
            "n_poses": 20,
        }

        engine = QVinaWEngine().load_config_dict(config)

        self.assertEqual(engine._center, (2.865, 193.257, 21.367))
        self.assertEqual(engine._size, (27.091, 27.091, 27.091))
        self.assertEqual(engine._cpu, 4)
        self.assertEqual(engine._seed, 42)
        self.assertEqual(engine._exhaustiveness, 16)
        self.assertEqual(engine._num_modes, 20)

    def test_load_config_reads_json_file(self) -> None:
        """
        Verify inherited :meth:`load_config` reads disk JSON correctly.
        """
        config_path = self.tmp / "config.json"
        config = {
            "box": {
                "center": [2.865, 193.257, 21.367],
                "size": [27.091, 27.091, 27.091],
            },
            "cpu": 4,
            "seed": 42,
            "exhaustiveness": 16,
            "n_poses": 20,
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        engine = QVinaWEngine().load_config(config_path)

        self.assertEqual(engine._center, (2.865, 193.257, 21.367))
        self.assertEqual(engine._size, (27.091, 27.091, 27.091))
        self.assertEqual(engine._cpu, 4)
        self.assertEqual(engine._seed, 42)
        self.assertEqual(engine._exhaustiveness, 16)
        self.assertEqual(engine._num_modes, 20)

    def test_build_cmd_from_box_config(self) -> None:
        """
        Verify command generation includes box and run parameters from config.
        """
        receptor = self.tmp / "rec.pdbqt"
        ligand = self.tmp / "lig.pdbqt"
        out = self.tmp / "poses.pdbqt"
        log = self.tmp / "dock.log"

        receptor.write_text("RECEPTOR\n", encoding="utf-8")
        ligand.write_text("LIGAND\n", encoding="utf-8")

        engine = (
            QVinaWEngine()
            .set_receptor(receptor)
            .set_ligand(ligand)
            .load_config_dict(
                {
                    "box": {
                        "center": [2.865, 193.257, 21.367],
                        "size": [27.091, 27.091, 27.091],
                    },
                    "cpu": 4,
                    "seed": 42,
                    "exhaustiveness": 16,
                    "n_poses": 20,
                }
            )
            .set_out(out)
            .set_log(log)
        )

        cmd = engine._build_cmd()

        self.assertEqual(cmd[0], "qvina-w")
        self.assertIn("--receptor", cmd)
        self.assertIn(str(receptor), cmd)
        self.assertIn("--ligand", cmd)
        self.assertIn(str(ligand), cmd)
        self.assertIn("--center_x", cmd)
        self.assertIn("2.865", cmd)
        self.assertIn("--center_y", cmd)
        self.assertIn("193.257", cmd)
        self.assertIn("--center_z", cmd)
        self.assertIn("21.367", cmd)
        self.assertIn("--size_x", cmd)
        self.assertIn("--size_y", cmd)
        self.assertIn("--size_z", cmd)
        self.assertIn("--cpu", cmd)
        self.assertIn("4", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("42", cmd)
        self.assertIn("--exhaustiveness", cmd)
        self.assertIn("16", cmd)
        self.assertIn("--num_modes", cmd)
        self.assertIn("20", cmd)
        self.assertIn("--out", cmd)
        self.assertIn(str(out), cmd)
        self.assertIn("--log", cmd)
        self.assertIn(str(log), cmd)

    def test_build_cmd_does_not_include_autobox_flags(self) -> None:
        """
        Verify generated QVina commands never include autobox flags.
        """
        receptor = self.tmp / "rec.pdbqt"
        ligand = self.tmp / "lig.pdbqt"

        receptor.write_text("RECEPTOR\n", encoding="utf-8")
        ligand.write_text("LIGAND\n", encoding="utf-8")

        engine = (
            QVinaWEngine()
            .set_receptor(receptor)
            .set_ligand(ligand)
            .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
        )

        cmd = engine._build_cmd()

        self.assertNotIn("--autobox_ligand", cmd)
        self.assertNotIn("--autobox_add", cmd)

    def test_validate_ready_raises_without_box(self) -> None:
        """
        Verify missing explicit box is rejected for QVina.
        """
        receptor = self.tmp / "rec.pdbqt"
        ligand = self.tmp / "lig.pdbqt"

        receptor.write_text("RECEPTOR\n", encoding="utf-8")
        ligand.write_text("LIGAND\n", encoding="utf-8")

        engine = QVinaWEngine().set_receptor(receptor).set_ligand(ligand)

        with self.assertRaises(ValueError):
            engine._validate_ready()

    def test_validate_ready_accepts_explicit_box(self) -> None:
        """
        Verify explicit box satisfies readiness validation.
        """
        receptor = self.tmp / "rec.pdbqt"
        ligand = self.tmp / "lig.pdbqt"

        receptor.write_text("RECEPTOR\n", encoding="utf-8")
        ligand.write_text("LIGAND\n", encoding="utf-8")

        engine = (
            QVinaWEngine()
            .set_receptor(receptor)
            .set_ligand(ligand)
            .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
        )

        engine._validate_ready()

    def test_resolve_executable_accepts_explicit_executable(self) -> None:
        """
        Verify explicit executable paths can be resolved.
        """
        exe = self.tmp / "fake_qvina"
        exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        exe.chmod(0o755)

        engine = QVinaWEngine().set_executable(exe)
        resolved = engine._resolve_executable()

        self.assertEqual(Path(resolved), exe.resolve())


class TestQVinaWEngineIntegration(unittest.TestCase):
    """
    Integration tests for :class:`prodock.dock.engine.qvina.QVinaWEngine`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Resolve testcase paths and create a shared temporary output directory.
        """
        cls.case_dir = Path("./Data/testcase/4WKQ")
        cls.receptor = cls.case_dir / "receptor" / "4WKQ.pdbqt"
        cls.ligand = cls.case_dir / "ligand" / "erlotinib.pdbqt"
        cls.config = cls.case_dir / "config.json"

        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmpdir.name)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Remove the shared temporary output directory.
        """
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        """
        Skip tests if testcase files are unavailable.
        """
        required = [self.receptor, self.ligand, self.config]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            self.skipTest(f"Missing testcase files: {', '.join(missing)}")

    def _make_paths(self, stem: str = "erlotinib_qvina") -> tuple[Path, Path]:
        """
        Build per-test output paths inside the temporary directory.

        :param stem:
            Stem used to name the output files.
        :type stem: str

        :returns:
            Tuple of pose and log file paths.
        :rtype: tuple[Path, Path]
        """
        out = self.tmp / f"{stem}.pdbqt"
        log = self.tmp / f"{stem}.log"

        if out.exists():
            out.unlink()
        if log.exists():
            log.unlink()

        return out, log

    def test_config_file_matches_expected_values(self) -> None:
        """
        Verify the shared testcase JSON contains expected values.
        """
        data = json.loads(self.config.read_text(encoding="utf-8"))

        self.assertEqual(data["box"]["center"], [2.865, 193.257, 21.367])
        self.assertEqual(data["box"]["size"], [27.091, 27.091, 27.091])
        self.assertEqual(data["cpu"], 4)
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["exhaustiveness"], 16)
        self.assertEqual(data["n_poses"], 20)

    def test_real_run_with_box_config(self) -> None:
        """
        Run qvina with explicit box settings loaded from config.

        Example
        -------
        .. code-block:: python

            engine = (
                QVinaWEngine()
                .set_receptor("./Data/testcase/4WKQ/receptor/4WKQ.pdbqt", validate=True)
                .set_ligand("./Data/testcase/4WKQ/ligand/erlotinib.pdbqt")
                .load_config("./Data/testcase/4WKQ/config.json")
                .set_out("tmp/erlotinib_qvina.pdbqt")
                .set_log("tmp/erlotinib_qvina.log")
            )
            engine.run()
        """
        out, log = self._make_paths("erlotinib_qvina_box")

        engine = (
            QVinaWEngine()
            .set_receptor(self.receptor, validate=True)
            .set_ligand(self.ligand)
            .load_config(self.config)
            .set_out(out)
            .set_log(log)
        )

        returned = engine.run()

        self.assertIs(returned, engine)
        self.assertTrue(out.exists(), msg="Docked pose file was not created")
        self.assertTrue(log.exists(), msg="Docking log file was not created")
        self.assertTrue(out.is_file())
        self.assertTrue(log.is_file())

        self.assertIsNotNone(engine.called)
        self.assertIn("--receptor", engine.called)
        self.assertIn("--ligand", engine.called)
        self.assertIn("--center_x", engine.called)
        self.assertIn("--size_x", engine.called)
        self.assertIn("--exhaustiveness", engine.called)
        self.assertIn("--num_modes", engine.called)
        self.assertIn("--cpu", engine.called)
        self.assertIn("--seed", engine.called)
        self.assertIn("--out", engine.called)
        self.assertIn("--log", engine.called)

    def test_run_override_arguments_take_precedence(self) -> None:
        """
        Verify per-run overrides take precedence over config defaults.
        """
        out, log = self._make_paths("erlotinib_qvina_override")

        engine = (
            QVinaWEngine()
            .set_receptor(self.receptor, validate=True)
            .set_ligand(self.ligand)
            .load_config(self.config)
            .set_out(out)
            .set_log(log)
        )

        engine.run(exhaustiveness=8, n_poses=5)

        self.assertTrue(out.exists())
        self.assertTrue(log.exists())
        self.assertIn("--exhaustiveness 8", engine.called)
        self.assertIn("--num_modes 5", engine.called)
