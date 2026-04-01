from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prodock.dock.vina import VinaEngine

try:
    from vina import Vina  # noqa: F401

    VINA_AVAILABLE = True
except Exception:
    VINA_AVAILABLE = False


@unittest.skipUnless(VINA_AVAILABLE, "vina package is required for integration tests")
class TestVinaEngineIntegration(unittest.TestCase):
    """
    Integration tests for :class:`prodock.dock.engine.vina.VinaEngine`.

    These tests use the provided 4WKQ testcase if it is available locally.
    Output files are written into a temporary directory rather than the
    testcase directory itself.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Resolve shared testcase paths and create a temporary output directory.

        Example
        -------
        .. code-block:: python

            cls.case_dir = Path("./Data/testcase/4WKQ")
            cls.receptor = cls.case_dir / "receptor" / "4WKQ.pdbqt"
            cls.ligand = cls.case_dir / "ligand" / "erlotinib.pdbqt"
            cls.config = cls.case_dir / "config.json"
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
        Skip integration tests if testcase files are unavailable.
        """
        required = [self.receptor, self.ligand, self.config]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            self.skipTest(f"Missing testcase files: {', '.join(missing)}")

    def _make_paths(self, stem: str = "erlotinib_docked") -> tuple[Path, Path]:
        """
        Build per-test temporary output paths.

        :param stem:
            Filename stem used for pose and log outputs.
        :type stem: str

        :returns:
            Tuple ``(out_path, log_path)`` inside the temporary directory.
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
        Verify the testcase JSON contains the expected docking settings.
        """
        data = json.loads(self.config.read_text(encoding="utf-8"))

        self.assertEqual(data["box"]["center"], [2.865, 193.257, 21.367])
        self.assertEqual(data["box"]["size"], [27.091, 27.091, 27.091])
        self.assertEqual(data["cpu"], 4)
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["exhaustiveness"], 16)
        self.assertEqual(data["n_poses"], 20)

    def test_real_run_with_4wkq_testcase(self) -> None:
        """
        Run the real 4WKQ testcase end-to-end and write outputs to a temporary
        directory.

        Example
        -------
        .. code-block:: python

            out = self.tmp / "erlotinib_docked.pdbqt"
            log = self.tmp / "erlotinib_docked.log"

            engine = (
                VinaEngine()
                .set_receptor("./Data/testcase/4WKQ/receptor/4WKQ.pdbqt", validate=True)
                .set_ligand("./Data/testcase/4WKQ/ligand/erlotinib.pdbqt")
                .load_config("./Data/testcase/4WKQ/config.json")
                .set_out(out)
                .set_log(log)
            )

            engine.run()
        """
        out, log = self._make_paths("erlotinib_docked")

        engine = (
            VinaEngine()
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
        self.assertTrue(out.is_file(), msg="Docked pose path is not a file")
        self.assertTrue(log.is_file(), msg="Docking log path is not a file")

        meta = engine.metadata
        self.assertEqual(meta["receptor"], str(self.receptor))
        self.assertEqual(meta["ligand"], str(self.ligand))
        self.assertEqual(meta["center"], [2.865, 193.257, 21.367])
        self.assertEqual(meta["size"], [27.091, 27.091, 27.091])
        self.assertEqual(meta["cpu"], 4)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["exhaustiveness"], 16)
        self.assertEqual(meta["n_poses"], 20)

        self.assertIsNotNone(engine.called)
        self.assertIn("vina.Vina(", engine.called)
        self.assertIn(".dock(exhaustiveness=16, n_poses=20)", engine.called)

        out_text = out.read_text(encoding="utf-8", errors="replace")
        self.assertTrue(len(out_text.strip()) > 0, msg="Docked pose file is empty")

        log_text = log.read_text(encoding="utf-8", errors="replace")
        self.assertTrue(len(log_text.strip()) > 0, msg="Docking log file is empty")
        self.assertIn("AutoDock Vina Python backend", log_text)
        self.assertIn("Exhaustiveness   : 16", log_text)
        self.assertIn("CPU              : 4", log_text)
        self.assertIn("Seed             : 42", log_text)
        self.assertIn("Requested poses  : 20", log_text)

    def test_run_override_arguments_take_precedence(self) -> None:
        """
        Verify per-run arguments override config-loaded defaults.
        """
        out, log = self._make_paths("erlotinib_override")

        engine = (
            VinaEngine()
            .set_receptor(self.receptor, validate=True)
            .set_ligand(self.ligand)
            .load_config(self.config)
            .set_out(out)
            .set_log(log)
        )

        engine.run(exhaustiveness=8, n_poses=5)

        self.assertTrue(out.exists())
        self.assertTrue(log.exists())

        meta = engine.metadata
        self.assertEqual(meta["exhaustiveness"], 8)
        self.assertEqual(meta["n_poses"], 5)

        log_text = log.read_text(encoding="utf-8", errors="replace")
        self.assertIn("Exhaustiveness   : 8", log_text)
        self.assertIn("Requested poses  : 5", log_text)


class TestVinaEngineUnit(unittest.TestCase):
    """Unit tests for :class:`prodock.dock.engine.vina.VinaEngine`."""

    def setUp(self) -> None:
        """
        Create a temporary workspace for isolated config and path-based tests.
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
        Verify constructor defaults are stored correctly.

        Example
        -------
        .. code-block:: python

            engine = VinaEngine()
        """
        engine = VinaEngine()

        self.assertEqual(engine.sf_name, "vina")
        self.assertEqual(engine._cpu, 0)
        self.assertIsNone(engine._seed)
        self.assertEqual(engine.verbosity, 1)
        self.assertFalse(engine.no_refine)

        self.assertIsNone(engine._receptor)
        self.assertIsNone(engine._ligand)
        self.assertIsNone(engine._center)
        self.assertIsNone(engine._size)
        self.assertIsNone(engine._exhaustiveness)
        self.assertIsNone(engine._num_modes)
        self.assertIsNone(engine._out)
        self.assertIsNone(engine._log)
        self.assertIsNone(engine.called)
        self.assertEqual(engine.metadata, {})

    def test_chainable_setters(self) -> None:
        """
        Verify setters are chainable and update internal state.

        Example
        -------
        .. code-block:: python

            engine = (
                VinaEngine()
                .set_receptor("rec.pdbqt")
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
            VinaEngine()
            .set_receptor(receptor, validate=True)
            .set_ligand(ligand)
            .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
            .set_cpu(4)
            .set_seed(42)
            .set_exhaustiveness(16)
            .set_num_modes(20)
            .set_out(out)
            .set_log(log)
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

    def test_set_receptor_validate_raises_for_missing_file(self) -> None:
        """
        Verify receptor validation fails when the file does not exist.
        """
        engine = VinaEngine()
        with self.assertRaises(FileNotFoundError):
            engine.set_receptor(self.tmp / "missing.pdbqt", validate=True)

    def test_enable_autobox_always_raises(self) -> None:
        """
        Verify autoboxing is rejected for the Python Vina backend.
        """
        engine = VinaEngine()
        with self.assertRaises(RuntimeError):
            engine.enable_autobox("ref.sdf")

    def test_load_config_dict_applies_values(self) -> None:
        """
        Verify :meth:`load_config_dict` correctly applies box and run settings.

        Example
        -------
        .. code-block:: python

            engine = VinaEngine().load_config_dict(
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

        engine = VinaEngine().load_config_dict(config)

        self.assertEqual(engine._center, (2.865, 193.257, 21.367))
        self.assertEqual(engine._size, (27.091, 27.091, 27.091))
        self.assertEqual(engine._cpu, 4)
        self.assertEqual(engine._seed, 42)
        self.assertEqual(engine._exhaustiveness, 16)
        self.assertEqual(engine._num_modes, 20)

    def test_load_config_reads_json_file(self) -> None:
        """
        Verify :meth:`load_config` reads JSON from disk and applies values.
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

        engine = VinaEngine().load_config(config_path)

        self.assertEqual(engine._center, (2.865, 193.257, 21.367))
        self.assertEqual(engine._size, (27.091, 27.091, 27.091))
        self.assertEqual(engine._cpu, 4)
        self.assertEqual(engine._seed, 42)
        self.assertEqual(engine._exhaustiveness, 16)
        self.assertEqual(engine._num_modes, 20)

    def test_load_config_raises_for_missing_file(self) -> None:
        """
        Verify :meth:`load_config` fails for a missing JSON file.
        """
        with self.assertRaises(FileNotFoundError):
            VinaEngine().load_config(self.tmp / "missing.json")

    def test_load_config_raises_for_non_object_top_level(self) -> None:
        """
        Verify :meth:`load_config` rejects non-object JSON content.
        """
        config_path = self.tmp / "config.json"
        config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        with self.assertRaises(ValueError):
            VinaEngine().load_config(config_path)

    def test_load_config_dict_raises_for_invalid_box_type(self) -> None:
        """
        Verify invalid ``box`` type is rejected.
        """
        with self.assertRaises(TypeError):
            VinaEngine().load_config_dict({"box": [1, 2, 3]})

    def test_load_config_dict_raises_for_incomplete_box(self) -> None:
        """
        Verify incomplete ``box`` blocks are rejected.
        """
        with self.assertRaises(ValueError):
            VinaEngine().load_config_dict({"box": {"center": [1, 2, 3]}})

        with self.assertRaises(ValueError):
            VinaEngine().load_config_dict({"box": {"size": [1, 2, 3]}})

    def test_load_config_dict_raises_for_invalid_center_shape(self) -> None:
        """
        Verify malformed ``box.center`` vectors are rejected.
        """
        with self.assertRaises(TypeError):
            VinaEngine().load_config_dict(
                {
                    "box": {
                        "center": [1.0, 2.0],
                        "size": [10.0, 11.0, 12.0],
                    }
                }
            )

    def test_load_config_dict_raises_for_invalid_size_shape(self) -> None:
        """
        Verify malformed ``box.size`` vectors are rejected.
        """
        with self.assertRaises(TypeError):
            VinaEngine().load_config_dict(
                {
                    "box": {
                        "center": [1.0, 2.0, 3.0],
                        "size": [10.0, 11.0],
                    }
                }
            )

    def test_validate_ready_raises_without_receptor(self) -> None:
        """
        Verify run validation fails when receptor is missing.
        """
        engine = (
            VinaEngine()
            .set_ligand(self.tmp / "lig.pdbqt")
            .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
        )

        with self.assertRaises(ValueError):
            engine._validate_ready()

    def test_validate_ready_raises_without_ligand(self) -> None:
        """
        Verify run validation fails when ligand is missing.
        """
        engine = (
            VinaEngine()
            .set_receptor(self.tmp / "rec.pdbqt")
            .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
        )

        with self.assertRaises(ValueError):
            engine._validate_ready()

    def test_validate_ready_raises_without_box(self) -> None:
        """
        Verify run validation fails when box is missing.
        """
        engine = (
            VinaEngine()
            .set_receptor(self.tmp / "rec.pdbqt")
            .set_ligand(self.tmp / "lig.pdbqt")
        )

        with self.assertRaises(ValueError):
            engine._validate_ready()

    def test_metadata_returns_copy(self) -> None:
        """
        Verify the metadata property returns a copy rather than the original dict.
        """
        engine = VinaEngine()
        engine._metadata = {"seed": 42}

        meta = engine.metadata
        meta["seed"] = 999

        self.assertEqual(engine.metadata["seed"], 42)

    def test_build_log_text_contains_expected_fields(self) -> None:
        """
        Verify log text includes metadata header and captured output body.
        """
        engine = (
            VinaEngine()
            .set_receptor("rec.pdbqt")
            .set_ligand("lig.pdbqt")
            .set_box((2.865, 193.257, 21.367), (27.091, 27.091, 27.091))
            .set_cpu(4)
            .set_seed(42)
        )

        text = engine._build_log_text(
            exhaustiveness=16,
            n_poses=20,
            captured_output="mode | affinity\n1    -6.8\n",
        )

        self.assertIn("AutoDock Vina Python backend", text)
        self.assertIn("Scoring function : vina", text)
        self.assertIn("Receptor         : rec.pdbqt", text)
        self.assertIn("Ligand           : lig.pdbqt", text)
        self.assertIn("Exhaustiveness   : 16", text)
        self.assertIn("CPU              : 4", text)
        self.assertIn("Seed             : 42", text)
        self.assertIn("Requested poses  : 20", text)
        self.assertIn("mode | affinity", text)

    def test_write_log_creates_file(self) -> None:
        """
        Verify log writing creates the configured file.
        """
        log_path = self.tmp / "dock.log"
        engine = (
            VinaEngine()
            .set_receptor("rec.pdbqt")
            .set_ligand("lig.pdbqt")
            .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
            .set_log(log_path)
        )

        engine._write_log(
            exhaustiveness=16,
            n_poses=20,
            captured_output="dummy docking output",
        )

        self.assertTrue(log_path.is_file())
        text = log_path.read_text(encoding="utf-8")
        self.assertIn("dummy docking output", text)
