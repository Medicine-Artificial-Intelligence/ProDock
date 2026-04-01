import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from prodock.dock.common_binary import (
    BaseBinaryEngine,
    _as_path,
    _ensure_parent,
)


class _MiniEngine(BaseBinaryEngine):
    """
    Small concrete engine used for unit tests.
    """

    exe_name = "___definitely_missing_binary___"
    supports_autobox = True


class _NoAutoboxEngine(BaseBinaryEngine):
    """
    Engine variant that does not support autoboxing.
    """

    exe_name = "___definitely_missing_binary___"
    supports_autobox = False


class TestHelpers(unittest.TestCase):
    def test_as_path_from_str(self):
        p = _as_path("abc.txt")
        self.assertIsInstance(p, Path)
        self.assertEqual(p, Path("abc.txt"))

    def test_as_path_from_path(self):
        p0 = Path("abc.txt")
        p = _as_path(p0)
        self.assertIs(p, p0)

    def test_ensure_parent_none(self):
        _ensure_parent(None)  # should not raise

    def test_ensure_parent_creates_directory(self):
        td = Path(tempfile.mkdtemp())
        try:
            out = td / "a" / "b" / "file.txt"
            self.assertFalse(out.parent.exists())
            _ensure_parent(out)
            self.assertTrue(out.parent.exists())
            self.assertTrue(out.parent.is_dir())
        finally:
            shutil.rmtree(td, ignore_errors=True)


class TestCommonBinary(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        (self.td / "rec.pdbqt").write_text("RECEPTOR\n")
        (self.td / "lig.pdbqt").write_text("LIGAND\n")
        (self.td / "ref.sdf").write_text("REFERENCE\n")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_init_defaults(self):
        eng = _MiniEngine()
        self.assertIsNone(eng._receptor)
        self.assertIsNone(eng._ligand)
        self.assertIsNone(eng._center)
        self.assertIsNone(eng._size)
        self.assertIsNone(eng._out)
        self.assertIsNone(eng._log)
        self.assertIsNone(eng._timeout)
        self.assertEqual(eng._extra_args, [])
        self.assertIsNone(eng.called)

    def test_set_receptor_validate_success(self):
        eng = _MiniEngine().set_receptor(self.td / "rec.pdbqt", validate=True)
        self.assertEqual(eng._receptor, self.td / "rec.pdbqt")

    def test_set_receptor_validate_missing_raises(self):
        eng = _MiniEngine()
        with self.assertRaises(FileNotFoundError):
            eng.set_receptor(self.td / "missing.pdbqt", validate=True)

    def test_setters_return_self_for_chaining(self):
        eng = _MiniEngine()

        result = (
            eng.set_receptor(self.td / "rec.pdbqt")
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((1.0, 2.0, 3.0), (10.0, 11.0, 12.0))
            .set_exhaustiveness(8)
            .set_num_modes(9)
            .set_cpu(4)
            .set_seed(42)
            .set_out(self.td / "out" / "dock.pdbqt")
            .set_log(self.td / "out" / "dock.log")
            .set_timeout(30.0)
            .set_extra_args("--quiet", "--foo", "bar")
        )

        self.assertIs(result, eng)
        self.assertEqual(eng._timeout, 30.0)
        self.assertEqual(eng._extra_args, ["--quiet", "--foo", "bar"])

    def test_enable_autobox_supported(self):
        eng = _MiniEngine().enable_autobox(self.td / "ref.sdf", padding=4.5)
        self.assertEqual(eng._autobox_ref, self.td / "ref.sdf")
        self.assertEqual(eng._autobox_pad, 4.5)

    def test_enable_autobox_not_supported_raises(self):
        eng = _NoAutoboxEngine()
        with self.assertRaises(RuntimeError):
            eng.enable_autobox(self.td / "ref.sdf", padding=4.5)

    def test_validate_ready_missing_receptor(self):
        eng = (
            _MiniEngine()
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((1, 2, 3), (10, 10, 10))
        )
        with self.assertRaisesRegex(ValueError, "Receptor was not set"):
            eng._validate_ready()

    def test_validate_ready_missing_ligand(self):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt")
            .set_box((1, 2, 3), (10, 10, 10))
        )
        with self.assertRaisesRegex(ValueError, "Ligand was not set"):
            eng._validate_ready()

    def test_validate_ready_missing_box_and_autobox(self):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt")
            .set_ligand(self.td / "lig.pdbqt")
        )
        with self.assertRaisesRegex(
            ValueError, "Docking box was not set and autobox was not enabled"
        ):
            eng._validate_ready()

    def test_validate_ready_with_explicit_box(self):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt")
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((12, 8, 5), (20, 20, 20))
        )
        eng._validate_ready()  # should not raise

    def test_validate_ready_with_autobox(self):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt")
            .set_ligand(self.td / "lig.pdbqt")
            .enable_autobox(self.td / "ref.sdf", padding=3.0)
        )
        eng._validate_ready()  # should not raise

    def test_build_cmd_explicit_box(self):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt", validate=True)
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((12, 8, 5), (20, 21, 22))
            .set_exhaustiveness(8)
            .set_num_modes(9)
            .set_cpu(4)
            .set_seed(42)
            .set_out(self.td / "out/lig_docked.pdbqt")
            .set_log(self.td / "out/lig.log")
            .set_extra_args("--quiet", "--score_only")
        )

        cmd = eng._build_cmd()

        self.assertEqual(cmd[0], _MiniEngine.exe_name)
        self.assertIn("--receptor", cmd)
        self.assertIn("--ligand", cmd)
        self.assertIn("--center_x", cmd)
        self.assertIn("--center_y", cmd)
        self.assertIn("--center_z", cmd)
        self.assertIn("--size_x", cmd)
        self.assertIn("--size_y", cmd)
        self.assertIn("--size_z", cmd)
        self.assertIn("--exhaustiveness", cmd)
        self.assertIn("--num_modes", cmd)
        self.assertIn("--cpu", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("--out", cmd)
        self.assertIn("--log", cmd)
        self.assertIn("--quiet", cmd)
        self.assertIn("--score_only", cmd)

    def test_build_cmd_autobox(self):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt")
            .set_ligand(self.td / "lig.pdbqt")
            .enable_autobox(self.td / "ref.sdf", padding=4.0)
        )

        cmd = eng._build_cmd()
        self.assertIn("--autobox_ligand", cmd)
        self.assertIn(str(self.td / "ref.sdf"), cmd)
        self.assertIn("--autobox_add", cmd)
        self.assertIn("4.0", cmd)

    def test_build_cmd_override_values(self):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt")
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((12, 8, 5), (20, 20, 20))
            .set_exhaustiveness(8)
            .set_num_modes(9)
        )

        cmd = eng._build_cmd(override_exhaustiveness=99, override_nposes=3)

        ex_idx = cmd.index("--exhaustiveness")
        nm_idx = cmd.index("--num_modes")
        self.assertEqual(cmd[ex_idx + 1], "99")
        self.assertEqual(cmd[nm_idx + 1], "3")

    def test_resolve_executable_missing_raises(self):
        eng = _MiniEngine()
        with self.assertRaises(FileNotFoundError):
            eng._resolve_executable()

    def test_run_raises_when_executable_missing(self):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt", validate=True)
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((12, 8, 5), (20, 20, 20))
            .set_out(self.td / "out/lig_docked.pdbqt")
            .set_log(self.td / "out/lig.log")
        )

        with self.assertRaises(FileNotFoundError):
            eng.run()

    @patch("prodock.dock.common_binary.subprocess.run")
    @patch.object(_MiniEngine, "_resolve_executable", return_value="/fake/bin/engine")
    def test_run_success_calls_subprocess_and_creates_parents(
        self, mock_resolve, mock_run
    ):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt", validate=True)
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((12, 8, 5), (20, 20, 20))
            .set_exhaustiveness(8)
            .set_num_modes(9)
            .set_cpu(4)
            .set_seed(42)
            .set_timeout(12.5)
            .set_out(self.td / "nested/out/lig_docked.pdbqt")
            .set_log(self.td / "nested/out/lig.log")
        )

        result = eng.run()

        self.assertIs(result, eng)
        self.assertTrue((self.td / "nested/out").exists())
        self.assertIsNotNone(eng.called)
        self.assertTrue(eng.called.startswith("/fake/bin/engine"))

        mock_resolve.assert_called_once()
        mock_run.assert_called_once()

        args, kwargs = mock_run.call_args
        self.assertEqual(args[0][0], "/fake/bin/engine")
        self.assertEqual(kwargs["check"], True)
        self.assertEqual(kwargs["timeout"], 12.5)

    @patch("prodock.dock.common_binary.subprocess.run")
    @patch.object(_MiniEngine, "_resolve_executable", return_value="/fake/bin/engine")
    def test_run_uses_override_values(self, mock_resolve, mock_run):
        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt")
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((0, 0, 0), (10, 10, 10))
            .set_exhaustiveness(8)
            .set_num_modes(9)
        )

        eng.run(exhaustiveness=77, n_poses=2)

        args, kwargs = mock_run.call_args
        cmd = args[0]

        ex_idx = cmd.index("--exhaustiveness")
        nm_idx = cmd.index("--num_modes")
        self.assertEqual(cmd[ex_idx + 1], "77")
        self.assertEqual(cmd[nm_idx + 1], "2")

    @patch("prodock.dock.common_binary.subprocess.run")
    @patch.object(_MiniEngine, "_resolve_executable", return_value="/fake/bin/engine")
    def test_called_property_contains_shell_quoted_command(
        self, mock_resolve, mock_run
    ):
        out_path = self.td / "dir with space" / "dock out.pdbqt"

        eng = (
            _MiniEngine()
            .set_receptor(self.td / "rec.pdbqt")
            .set_ligand(self.td / "lig.pdbqt")
            .set_box((1, 2, 3), (10, 10, 10))
            .set_out(out_path)
        )

        eng.run()

        self.assertIsNotNone(eng.called)
        self.assertIn("/fake/bin/engine", eng.called)
        self.assertIn("dock out.pdbqt", eng.called)


if __name__ == "__main__":
    unittest.main()
