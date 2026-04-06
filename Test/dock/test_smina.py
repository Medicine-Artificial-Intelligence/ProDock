from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

from prodock.dock.smina import SminaEngine

IS_LINUX = sys.platform.startswith("linux")


@unittest.skipUnless(IS_LINUX, "QVina integration tests run only on Linux")
class TestSminaEngineUnit(unittest.TestCase):
    """Unit tests for :class:`prodock.dock.smina.SminaEngine`."""

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
        """
        engine = SminaEngine()

        self.assertEqual(engine.exe_name, "smina")
        self.assertTrue(engine.supports_autobox)

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

    def test_chainable_setters(self) -> None:
        """
        Verify inherited setters are chainable.
        """
        receptor = self.tmp / "rec.pdbqt"
        ligand = self.tmp / "lig.pdbqt"
        autobox = self.tmp / "ref.sdf"
        out = self.tmp / "poses.pdbqt"
        log = self.tmp / "dock.log"

        receptor.write_text("RECEPTOR\n", encoding="utf-8")
        ligand.write_text("LIGAND\n", encoding="utf-8")
        autobox.write_text("REF\n", encoding="utf-8")

        engine = (
            SminaEngine()
            .set_receptor(receptor, validate=True)
            .set_ligand(ligand)
            .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
            .set_cpu(4)
            .set_seed(42)
            .set_exhaustiveness(16)
            .set_num_modes(20)
            .set_out(out)
            .set_log(log)
            .enable_autobox(autobox, padding=4.0)
            .set_timeout(60.0)
            .set_extra_args("--quiet")
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
        self.assertEqual(engine._autobox_ref, autobox)
        self.assertEqual(engine._autobox_pad, 4.0)
        self.assertEqual(engine._timeout, 60.0)
        self.assertEqual(engine._extra_args, ["--quiet"])

    def test_load_config_dict_applies_values(self) -> None:
        """
        Verify inherited :meth:`load_config_dict` applies shared parameters.
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

        engine = SminaEngine().load_config_dict(config)

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

        engine = SminaEngine().load_config(config_path)

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
            SminaEngine()
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

        self.assertEqual(cmd[0], "smina")
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

    def test_build_cmd_with_autobox(self) -> None:
        """
        Verify smina-specific autobox flags are included when enabled.
        """
        receptor = self.tmp / "rec.pdbqt"
        ligand = self.tmp / "lig.pdbqt"
        ref = self.tmp / "ref.sdf"

        receptor.write_text("RECEPTOR\n", encoding="utf-8")
        ligand.write_text("LIGAND\n", encoding="utf-8")
        ref.write_text("REF\n", encoding="utf-8")

        engine = (
            SminaEngine()
            .set_receptor(receptor)
            .set_ligand(ligand)
            .enable_autobox(ref, padding=4.0)
        )

        cmd = engine._build_cmd()

        self.assertIn("--autobox_ligand", cmd)
        self.assertIn(str(ref), cmd)
        self.assertIn("--autobox_add", cmd)
        self.assertIn("4.0", cmd)

    def test_validate_ready_accepts_autobox_without_explicit_box(self) -> None:
        """
        Verify autoboxing satisfies readiness validation.
        """
        receptor = self.tmp / "rec.pdbqt"
        ligand = self.tmp / "lig.pdbqt"
        ref = self.tmp / "ref.sdf"

        receptor.write_text("RECEPTOR\n", encoding="utf-8")
        ligand.write_text("LIGAND\n", encoding="utf-8")
        ref.write_text("REF\n", encoding="utf-8")

        engine = (
            SminaEngine()
            .set_receptor(receptor)
            .set_ligand(ligand)
            .enable_autobox(ref, padding=4.0)
        )

        engine._validate_ready()

    def test_validate_ready_raises_without_box_or_autobox(self) -> None:
        """
        Verify missing box and missing autobox are rejected.
        """
        receptor = self.tmp / "rec.pdbqt"
        ligand = self.tmp / "lig.pdbqt"

        receptor.write_text("RECEPTOR\n", encoding="utf-8")
        ligand.write_text("LIGAND\n", encoding="utf-8")

        engine = SminaEngine().set_receptor(receptor).set_ligand(ligand)

        with self.assertRaises(ValueError):
            engine._validate_ready()

    def test_resolve_executable_accepts_explicit_executable(self) -> None:
        """
        Verify explicit executable paths can be resolved.
        """
        exe = self.tmp / "fake_smina"
        exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        exe.chmod(0o755)

        engine = SminaEngine().set_executable(exe)
        resolved = engine._resolve_executable()

        self.assertEqual(Path(resolved), exe.resolve())


@unittest.skipUnless(IS_LINUX, "QVina integration tests run only on Linux")
class TestSminaEngineIntegration(unittest.TestCase):
    """
    Integration tests for :class:`prodock.dock.smina.SminaEngine`.
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

    def _make_paths(self, stem: str = "erlotinib_smina") -> tuple[Path, Path]:
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
        Run smina with explicit box settings loaded from config.
        """
        out, log = self._make_paths("erlotinib_smina_box")

        engine = (
            SminaEngine()
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
        out, log = self._make_paths("erlotinib_smina_override")

        engine = (
            SminaEngine()
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
