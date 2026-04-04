from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prodock.structure.conversion import (
    convert_with_meeko,
    convert_with_obabel,
    ensure_pdbqt,
    pdb_to_pdbqt,
    pdb_to_sdf,
    pdbqt_to_pdb,
    pdbqt_to_sdf,
    sdf_to_pdb,
    sdf_to_pdbqt,
)

TEST_PDB = Path("Data/testcase/4WKQ/receptor/4WKQ.pdb")
TEST_LIG = Path("Data/testcase/4WKQ/ligand/IRE.sdf")


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve()


def _resolved_list(paths) -> list[Path]:
    return [Path(p).resolve() for p in paths]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _has_torsion_tokens(text: str) -> bool:
    forbidden = (
        "ROOT",
        "BRANCH",
        "ENDBRANCH",
        "ENDROOT",
        "TORSDOF",
        "active torsions",
    )
    lowered = text.lower()
    return any(tok.lower() in lowered for tok in forbidden)


def _has_atom_records(text: str) -> bool:
    return any(line.startswith(("ATOM", "HETATM")) for line in text.splitlines())


class _FakeProc:
    def __init__(self, rc: int = 0, stdout: str = "ok", stderr: str = "") -> None:
        self.returncode = rc
        self.stdout = stdout
        self.stderr = stderr


@unittest.skipUnless(TEST_PDB.exists(), f"Missing test input: {TEST_PDB}")
class TestConvertWithOpenBabelIntegration(unittest.TestCase):
    """
    Integration-style tests for Open Babel conversion helpers.

    These tests touch the real Open Babel executable when available.
    """

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory(prefix="prodock_test_convert_")
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    def test_convert_with_obabel_pdb_to_sdf(self) -> None:
        output_path = self.tmpdir / "protein.sdf"
        convert_with_obabel(TEST_PDB, output_path)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    @unittest.skipUnless(TEST_LIG.exists(), f"Missing test ligand: {TEST_LIG}")
    def test_convert_with_obabel_sdf_to_pdb(self) -> None:
        output_path = self.tmpdir / "ligand.pdb"
        convert_with_obabel(TEST_LIG, output_path)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    def test_convert_with_obabel_rigid_receptor_pdbqt_has_no_torsion_tokens(
        self,
    ) -> None:
        output_path = self.tmpdir / "receptor_rigid.pdbqt"

        convert_with_obabel(
            TEST_PDB,
            output_path,
            validate_receptor=True,
            flexibility=False,
        )

        self.assertTrue(output_path.exists())
        text = _read_text(output_path)
        self.assertTrue(_has_atom_records(text))
        self.assertFalse(_has_torsion_tokens(text))

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    @unittest.skipUnless(TEST_LIG.exists(), f"Missing test ligand: {TEST_LIG}")
    def test_convert_with_obabel_ligand_sdf_to_pdbqt(self) -> None:
        output_path = self.tmpdir / "ligand.pdbqt"
        convert_with_obabel(TEST_LIG, output_path)
        self.assertTrue(output_path.exists())
        text = _read_text(output_path)
        self.assertTrue(_has_atom_records(text) or len(text) > 0)

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    def test_convert_with_obabel_flexible_receptor_pdbqt_torsion_behavior(self) -> None:
        output_path = self.tmpdir / "receptor_flex.pdbqt"

        convert_with_obabel(
            TEST_PDB,
            output_path,
            validate_receptor=True,
            flexibility=True,
        )

        self.assertTrue(output_path.exists())
        text = _read_text(output_path)
        self.assertTrue(_has_atom_records(text))
        self.assertIsInstance(_has_torsion_tokens(text), bool)

    def test_convert_with_obabel_missing_output_extension_raises(self) -> None:
        output_path = self.tmpdir / "no_extension"
        with self.assertRaises(ValueError):
            convert_with_obabel(TEST_PDB, output_path)

    def test_convert_with_obabel_missing_input_extension_raises(self) -> None:
        bad_input = self.tmpdir / "input_without_ext"
        bad_input.write_text(TEST_PDB.read_text(encoding="utf-8", errors="replace"))
        output_path = self.tmpdir / "output.pdbqt"
        with self.assertRaises(ValueError):
            convert_with_obabel(bad_input, output_path)


class TestConvertWithOpenBabelUnit(unittest.TestCase):
    """
    Pure unit tests for convert_with_obabel using mocks.
    """

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory(
            prefix="prodock_test_obabel_unit_"
        )
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_sanitizes_pdbqt_unconditionally(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        output_path = self.tmpdir / "out.pdbqt"

        def _fake_run(*args, **kwargs):
            output_path.write_text(
                "ATOM      1  C   UNL A   1       0.000   0.000   0.000  1.00  0.00      A    C\n",
                encoding="utf-8",
            )
            return _FakeProc()

        mock_run.side_effect = _fake_run

        convert_with_obabel(
            TEST_PDB,
            output_path,
            sanitize_rebuild=False,
            sanitize_aggressive=True,
            sanitize_backup=True,
        )

        mock_sanitize.assert_called_once_with(
            output_path,
            out_path=None,
            backend="obabel",
            rebuild=False,
            aggressive=True,
            backup=True,
        )

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_non_pdbqt_output_does_not_sanitize(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        output_path = self.tmpdir / "out.sdf"

        def _fake_run(*args, **kwargs):
            output_path.write_text("$$$$\n", encoding="utf-8")
            return _FakeProc()

        mock_run.side_effect = _fake_run

        convert_with_obabel(TEST_PDB, output_path)
        mock_sanitize.assert_not_called()

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_injects_rigid_receptor_flag_when_needed(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        output_path = self.tmpdir / "rigid.pdbqt"
        captured = {}

        def _fake_run(args, **kwargs):
            captured["args"] = list(args)
            output_path.write_text(
                "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00      A    N\n",
                encoding="utf-8",
            )
            return _FakeProc()

        mock_run.side_effect = _fake_run

        convert_with_obabel(
            TEST_PDB,
            output_path,
            validate_receptor=True,
            flexibility=False,
        )

        self.assertIn("-xrc", captured["args"])
        self.assertNotIn("-xs", captured["args"])
        mock_sanitize.assert_called_once()

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_injects_flexible_receptor_flag_when_needed(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        output_path = self.tmpdir / "flex.pdbqt"
        captured = {}

        def _fake_run(args, **kwargs):
            captured["args"] = list(args)
            output_path.write_text(
                "ROOT\n"
                "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00      A    N\n",
                encoding="utf-8",
            )
            return _FakeProc()

        mock_run.side_effect = _fake_run

        convert_with_obabel(
            TEST_PDB,
            output_path,
            validate_receptor=True,
            flexibility=True,
        )

        self.assertIn("-xs", captured["args"])
        self.assertNotIn("-xrc", captured["args"])
        mock_sanitize.assert_called_once()

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_does_not_inject_x_flag_if_user_provided_one(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        output_path = self.tmpdir / "custom.pdbqt"
        captured = {}

        def _fake_run(args, **kwargs):
            captured["args"] = list(args)
            output_path.write_text(
                "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00      A    N\n",
                encoding="utf-8",
            )
            return _FakeProc()

        mock_run.side_effect = _fake_run

        convert_with_obabel(
            TEST_PDB,
            output_path,
            extra_args=["-xs"],
            validate_receptor=True,
            flexibility=False,
        )

        self.assertEqual(captured["args"].count("-xs"), 1)
        self.assertNotIn("-xrc", captured["args"])
        mock_sanitize.assert_called_once()

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_rigid_validation_rejects_torsion_tokens(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        output_path = self.tmpdir / "bad_rigid.pdbqt"

        def _fake_run(*args, **kwargs):
            output_path.write_text(
                "ROOT\n"
                "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00      A    N\n",
                encoding="utf-8",
            )
            return _FakeProc()

        mock_run.side_effect = _fake_run

        with self.assertRaises(RuntimeError) as ctx:
            convert_with_obabel(
                TEST_PDB,
                output_path,
                validate_receptor=True,
                flexibility=False,
            )

        self.assertIn("forbidden token", str(ctx.exception).lower())
        mock_sanitize.assert_called_once()

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_validation_requires_atom_records(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        output_path = self.tmpdir / "no_atoms.pdbqt"

        def _fake_run(*args, **kwargs):
            output_path.write_text("REMARK empty-like file\n", encoding="utf-8")
            return _FakeProc()

        mock_run.side_effect = _fake_run

        with self.assertRaises(RuntimeError) as ctx:
            convert_with_obabel(
                TEST_PDB,
                output_path,
                validate_receptor=True,
                flexibility=True,
            )

        self.assertIn("no atom/hetatm lines", str(ctx.exception).lower())
        mock_sanitize.assert_called_once()

    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_raises_when_executable_missing(
        self,
        mock_which,
        mock_run,
    ) -> None:
        mock_which.side_effect = [None, None]

        with self.assertRaises(RuntimeError) as ctx:
            convert_with_obabel(TEST_PDB, self.tmpdir / "out.sdf")

        self.assertIn("not found", str(ctx.exception).lower())
        mock_run.assert_not_called()

    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_raises_on_nonzero_return_code(
        self,
        mock_which,
        mock_run,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        mock_run.return_value = _FakeProc(rc=1, stderr="conversion failed")

        with self.assertRaises(RuntimeError) as ctx:
            convert_with_obabel(TEST_PDB, self.tmpdir / "out.sdf")

        self.assertIn("conversion failed", str(ctx.exception).lower())

    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_raises_when_output_missing_after_success(
        self,
        mock_which,
        mock_run,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        mock_run.return_value = _FakeProc(rc=0, stdout="ok", stderr="")

        with self.assertRaises(RuntimeError) as ctx:
            convert_with_obabel(TEST_PDB, self.tmpdir / "missing.sdf")

        self.assertIn("output file is missing", str(ctx.exception).lower())

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_obabel_creates_output_parent_directory(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.side_effect = ["/usr/bin/obabel"]
        output_path = self.tmpdir / "nested" / "dir" / "out.pdbqt"

        def _fake_run(*args, **kwargs):
            self.assertTrue(output_path.parent.exists())
            output_path.write_text(
                "ATOM      1  C   UNL A   1       0.0   0.0   0.0  1.00  0.00      A    C\n",
                encoding="utf-8",
            )
            return _FakeProc()

        mock_run.side_effect = _fake_run

        convert_with_obabel(TEST_PDB, output_path)
        self.assertTrue(output_path.exists())
        mock_sanitize.assert_called_once()


@unittest.skipUnless(TEST_PDB.exists(), f"Missing test input: {TEST_PDB}")
class TestConvertWithMeekoUnit(unittest.TestCase):
    """
    Unit tests for the Meeko helper.
    """

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory(prefix="prodock_test_meeko_")
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_convert_with_meeko_not_found_returns_info(self) -> None:
        out_basename = self.tmpdir / "meeko_out"
        write_pdbqt = self.tmpdir / "meeko_out.pdbqt"

        info = convert_with_meeko(
            mekoo_cmd="definitely_not_a_real_meeko_command_12345",
            input_pdb=TEST_PDB,
            out_basename=out_basename,
            write_pdbqt=write_pdbqt,
        )

        self.assertIsInstance(info, dict)
        self.assertIn("stderr", info)
        self.assertIsNotNone(info["stderr"])
        self.assertIn("not found", str(info["stderr"]).lower())
        self.assertEqual(info["produced"], [])

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_meeko_sanitizes_explicit_written_pdbqt(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.return_value = "/usr/bin/mk_prepare_receptor.py"
        write_pdbqt = self.tmpdir / "meeko_written.pdbqt"

        def _fake_run(*args, **kwargs):
            write_pdbqt.write_text(
                "ATOM      1  N   ALA A   1      11.104  13.207   9.455  1.00 20.00      A    N\n",
                encoding="utf-8",
            )
            return _FakeProc()

        mock_run.side_effect = _fake_run

        info = convert_with_meeko(
            mekoo_cmd="mk_prepare_receptor.py",
            input_pdb=TEST_PDB,
            out_basename=self.tmpdir / "meeko_out",
            write_pdbqt=write_pdbqt,
        )

        self.assertEqual(info["rc"], 0)
        self.assertTrue(write_pdbqt.exists())
        mock_sanitize.assert_called_once_with(
            write_pdbqt,
            out_path=None,
            backend="meeko",
            rebuild=True,
            aggressive=False,
            backup=False,
        )
        self.assertIn(str(write_pdbqt.resolve()), info["produced"])

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_meeko_collects_out_basename_products_and_sanitizes_pdbqt(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.return_value = "/usr/bin/mk_prepare_receptor.py"

        out_basename = self.tmpdir / "bundle" / "receptor"
        out_basename.parent.mkdir(parents=True, exist_ok=True)
        pdbqt_path = out_basename.with_suffix(".pdbqt")
        json_path = out_basename.with_suffix(".json")

        def _fake_run(*args, **kwargs):
            pdbqt_path.write_text(
                "ATOM      1  C   UNL A   1       0.000   0.000   0.000  1.00  0.00      A    C\n",
                encoding="utf-8",
            )
            json_path.write_text('{"status": "ok"}', encoding="utf-8")
            return _FakeProc()

        mock_run.side_effect = _fake_run

        info = convert_with_meeko(
            mekoo_cmd="mk_prepare_receptor.py",
            input_pdb=TEST_PDB,
            out_basename=out_basename,
            write_pdbqt=None,
        )

        self.assertEqual(info["rc"], 0)
        self.assertEqual(len(info["produced"]), 2)
        self.assertIn(str(pdbqt_path.resolve()), info["produced"])
        self.assertIn(str(json_path.resolve()), info["produced"])
        mock_sanitize.assert_called_once_with(
            pdbqt_path,
            out_path=None,
            backend="meeko",
            rebuild=True,
            aggressive=False,
            backup=False,
        )

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_meeko_explicit_and_basename_same_pdbqt_not_duplicated(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.return_value = "/usr/bin/mk_prepare_receptor.py"
        out_basename = self.tmpdir / "receptor"
        write_pdbqt = out_basename.with_suffix(".pdbqt")

        def _fake_run(*args, **kwargs):
            write_pdbqt.write_text(
                "ATOM      1  C   UNL A   1       0.0   0.0   0.0  1.00  0.00      A    C\n",
                encoding="utf-8",
            )
            return _FakeProc()

        mock_run.side_effect = _fake_run

        info = convert_with_meeko(
            mekoo_cmd="mk_prepare_receptor.py",
            input_pdb=TEST_PDB,
            out_basename=out_basename,
            write_pdbqt=write_pdbqt,
        )

        self.assertEqual(info["produced"], [str(write_pdbqt.resolve())])
        self.assertEqual(mock_sanitize.call_count, 2)

    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_meeko_invocation_failure_returns_rc_minus_one(
        self,
        mock_which,
        mock_run,
    ) -> None:
        mock_which.return_value = "/usr/bin/mk_prepare_receptor.py"
        mock_run.side_effect = OSError("boom")

        info = convert_with_meeko(
            mekoo_cmd="mk_prepare_receptor.py",
            input_pdb=TEST_PDB,
            out_basename=self.tmpdir / "out",
            write_pdbqt=self.tmpdir / "out.pdbqt",
        )

        self.assertEqual(info["rc"], -1)
        self.assertIn("boom", str(info["stderr"]).lower())

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_meeko_passes_sanitize_options_through(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ) -> None:
        mock_which.return_value = "/usr/bin/mk_prepare_receptor.py"
        write_pdbqt = self.tmpdir / "custom_opts.pdbqt"

        def _fake_run(*args, **kwargs):
            write_pdbqt.write_text(
                "ATOM      1  C   UNL A   1       0.0   0.0   0.0  1.00  0.00      A    C\n",
                encoding="utf-8",
            )
            return _FakeProc()

        mock_run.side_effect = _fake_run

        convert_with_meeko(
            mekoo_cmd="mk_prepare_receptor.py",
            input_pdb=TEST_PDB,
            out_basename=self.tmpdir / "out",
            write_pdbqt=write_pdbqt,
            sanitize_rebuild=False,
            sanitize_aggressive=True,
            sanitize_backup=True,
        )

        mock_sanitize.assert_called_once_with(
            write_pdbqt,
            out_path=None,
            backend="meeko",
            rebuild=False,
            aggressive=True,
            backup=True,
        )

    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_meeko_records_called_command_string(
        self,
        mock_which,
        mock_run,
    ) -> None:
        mock_which.return_value = "/usr/bin/mk_prepare_receptor.py"
        mock_run.return_value = _FakeProc()

        info = convert_with_meeko(
            mekoo_cmd="mk_prepare_receptor.py",
            input_pdb=TEST_PDB,
            out_basename=self.tmpdir / "outbase",
            write_pdbqt=self.tmpdir / "outbase.pdbqt",
            box_center=(1.0, 2.0, 3.0),
            box_size=(10.0, 12.0, 14.0),
        )

        self.assertIsInstance(info["called"], str)
        self.assertIn("--read_pdb", info["called"])
        self.assertIn("--box_center", info["called"])
        self.assertIn("--box_size", info["called"])
        self.assertIn("--write_pdbqt", info["called"])


class TestPublicConversionWrappersUnit(unittest.TestCase):
    """
    Unit tests for the public conversion wrapper functions.

    These verify routing, argument forwarding, and high-level behavior without
    requiring the real backend tools.
    """

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory(prefix="prodock_test_public_")
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    @patch("prodock.structure.conversion._meeko_pdb_to_pdbqt")
    def test_pdb_to_pdbqt_meeko_receptor_route(self, mock_meeko) -> None:
        out = self.tmpdir / "receptor.pdbqt"
        mock_meeko.return_value = out

        result = pdb_to_pdbqt(
            TEST_PDB,
            out,
            mode="receptor",
            backend="meeko",
        )

        self.assertEqual(_resolved(result), _resolved(out))
        mock_meeko.assert_called_once()

    @patch("prodock.structure.conversion._meeko_pdb_to_pdbqt")
    def test_pdb_to_pdbqt_meeko_ligand_route(self, mock_meeko) -> None:
        out = self.tmpdir / "ligand_from_pdb.pdbqt"
        mock_meeko.return_value = out

        result = pdb_to_pdbqt(
            TEST_PDB,
            out,
            mode="ligand",
            backend="meeko",
        )

        self.assertEqual(_resolved(result), _resolved(out))
        mock_meeko.assert_called_once()

    @patch("prodock.structure.conversion.convert_with_obabel")
    def test_pdb_to_pdbqt_obabel_receptor_route(self, mock_obabel) -> None:
        out = self.tmpdir / "rec_obabel.pdbqt"

        def _fake(*args, **kwargs):
            out.write_text("ATOM\n", encoding="utf-8")

        mock_obabel.side_effect = _fake

        result = pdb_to_pdbqt(
            TEST_PDB,
            out,
            mode="receptor",
            backend="obabel",
            flexibility=False,
        )

        self.assertEqual(_resolved(result), _resolved(out))
        mock_obabel.assert_called_once()
        _, kwargs = mock_obabel.call_args
        self.assertTrue(kwargs["validate_receptor"])
        self.assertFalse(kwargs["flexibility"])

    @patch("prodock.structure.conversion.convert_with_obabel")
    def test_pdb_to_pdbqt_obabel_ligand_route(self, mock_obabel) -> None:
        out = self.tmpdir / "lig_obabel.pdbqt"

        def _fake(*args, **kwargs):
            out.write_text("ATOM\n", encoding="utf-8")

        mock_obabel.side_effect = _fake

        result = pdb_to_pdbqt(
            TEST_PDB,
            out,
            mode="ligand",
            backend="obabel",
        )

        self.assertEqual(_resolved(result), _resolved(out))
        _, kwargs = mock_obabel.call_args
        self.assertFalse(kwargs["validate_receptor"])

    @patch("prodock.structure.conversion._mgltools_pdb_to_pdbqt")
    def test_pdb_to_pdbqt_mgltools_route(self, mock_mgl) -> None:
        out = self.tmpdir / "mgl_rec.pdbqt"
        mock_mgl.return_value = out

        result = pdb_to_pdbqt(
            TEST_PDB,
            out,
            mode="receptor",
            backend="mgltools",
        )

        self.assertEqual(_resolved(result), _resolved(out))
        mock_mgl.assert_called_once()

    @patch("prodock.structure.conversion.convert_with_obabel")
    def test_pdbqt_to_pdb_route(self, mock_obabel) -> None:
        inp = self.tmpdir / "in.pdbqt"
        out = self.tmpdir / "out.pdb"
        inp.write_text("ATOM\n", encoding="utf-8")

        def _fake(*args, **kwargs):
            out.write_text("ATOM\nEND\n", encoding="utf-8")

        mock_obabel.side_effect = _fake

        result = pdbqt_to_pdb(inp, out, backend="obabel")

        self.assertEqual(_resolved(result), _resolved(out))
        mock_obabel.assert_called_once_with(
            inp,
            out,
            extra_args=None,
            validate_receptor=False,
        )

    @patch("prodock.structure.conversion.convert_with_obabel")
    def test_pdbqt_to_pdb_non_obabel_not_supported(self, mock_obabel) -> None:
        inp = self.tmpdir / "in.pdbqt"
        out = self.tmpdir / "out.pdb"
        inp.write_text("ATOM\n", encoding="utf-8")

        with self.assertRaises(NotImplementedError):
            pdbqt_to_pdb(inp, out, backend="obabelx")  # type: ignore[arg-type]

        mock_obabel.assert_not_called()

    @unittest.skipUnless(TEST_LIG.exists(), f"Missing test ligand: {TEST_LIG}")
    @patch("prodock.structure.conversion.Chem.MolToPDBFile")
    @patch("prodock.structure.conversion.Chem.SDMolSupplier")
    def test_sdf_to_pdb_rdkit_route(self, mock_supplier, mock_mol_to_pdb) -> None:
        out = self.tmpdir / "lig.pdb"
        mol = object()
        mock_supplier.return_value = [mol]

        def _fake_mol_to_pdb_file(mol_obj, out_path):
            Path(out_path).write_text(
                "HETATM    1  C   UNL A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                "END\n",
                encoding="utf-8",
            )

        mock_mol_to_pdb.side_effect = _fake_mol_to_pdb_file

        result = sdf_to_pdb(TEST_LIG, out, backend="rdkit")

        self.assertEqual(_resolved(result), _resolved(out))
        mock_supplier.assert_called_once()
        mock_mol_to_pdb.assert_called_once_with(mol, str(out))
        self.assertTrue(out.exists())

    @unittest.skipUnless(TEST_LIG.exists(), f"Missing test ligand: {TEST_LIG}")
    @patch("prodock.structure.conversion.Chem.SDWriter")
    @patch("prodock.structure.conversion.Chem.MolFromPDBFile")
    @patch("prodock.structure.conversion.Chem.MolToPDBFile")
    @patch("prodock.structure.conversion.Chem.SDMolSupplier")
    def test_rdkit_ligand_roundtrip_sdf_to_pdb_to_sdf(
        self,
        mock_supplier,
        mock_mol_to_pdb,
        mock_mol_from_pdb,
        mock_sdwriter,
    ) -> None:
        pdb_path = self.tmpdir / "ligand_roundtrip.pdb"
        sdf_path = self.tmpdir / "ligand_roundtrip_back.sdf"

        mol_from_sdf = object()
        mol_from_pdb = object()

        mock_supplier.return_value = [mol_from_sdf]
        mock_mol_from_pdb.return_value = mol_from_pdb

        def _fake_mol_to_pdb_file(mol_obj, out_path):
            Path(out_path).write_text(
                "HETATM    1  C1  UNL A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                "END\n",
                encoding="utf-8",
            )

        mock_mol_to_pdb.side_effect = _fake_mol_to_pdb_file

        writer = mock_sdwriter.return_value

        def _fake_write(mol_obj):
            sdf_path.write_text(
                "Ligand\n"
                "  RDKit          3D\n"
                "\n"
                "  0  0  0  0  0  0            999 V2000\n"
                "M  END\n"
                "$$$$\n",
                encoding="utf-8",
            )

        writer.write.side_effect = _fake_write

        pdb_out = sdf_to_pdb(TEST_LIG, pdb_path, backend="rdkit")
        sdf_out = pdb_to_sdf(pdb_out, sdf_path, backend="rdkit")

        self.assertEqual(_resolved(pdb_out), _resolved(pdb_path))
        self.assertEqual(_resolved(sdf_out), _resolved(sdf_path))

        self.assertTrue(pdb_path.exists())
        self.assertTrue(sdf_path.exists())

        mock_supplier.assert_called_once()
        mock_mol_to_pdb.assert_called_once_with(mol_from_sdf, str(pdb_path))
        mock_mol_from_pdb.assert_called_once_with(
            str(pdb_path.resolve()), removeHs=False
        )
        mock_sdwriter.assert_called_once_with(str(sdf_path))
        writer.write.assert_called_once_with(mol_from_pdb)
        writer.close.assert_called_once()

    @patch("prodock.structure.conversion.convert_with_obabel")
    def test_sdf_to_pdb_obabel_route(self, mock_obabel) -> None:
        out = self.tmpdir / "lig_obabel.pdb"

        def _fake(*args, **kwargs):
            out.write_text("ATOM\nEND\n", encoding="utf-8")

        mock_obabel.side_effect = _fake

        result = sdf_to_pdb(TEST_LIG, out, backend="obabel")

        self.assertEqual(_resolved(result), _resolved(out))
        mock_obabel.assert_called_once_with(
            TEST_LIG.resolve(),
            out,
            extra_args=None,
            validate_receptor=False,
        )

    @patch("prodock.structure.conversion._meeko_sdf_to_pdbqt")
    def test_sdf_to_pdbqt_meeko_ligand_route(self, mock_meeko) -> None:
        out = self.tmpdir / "lig_meeko.pdbqt"
        mock_meeko.return_value = out

        result = sdf_to_pdbqt(TEST_LIG, out, backend="meeko")

        self.assertEqual(_resolved(result), _resolved(out))
        mock_meeko.assert_called_once()

    @patch("prodock.structure.conversion.convert_with_obabel")
    def test_sdf_to_pdbqt_obabel_route(self, mock_obabel) -> None:
        out = self.tmpdir / "lig_obabel.pdbqt"

        def _fake(*args, **kwargs):
            out.write_text("ATOM\n", encoding="utf-8")

        mock_obabel.side_effect = _fake

        result = sdf_to_pdbqt(TEST_LIG, out, backend="obabel")

        self.assertEqual(_resolved(result), _resolved(out))
        mock_obabel.assert_called_once()
        _, kwargs = mock_obabel.call_args
        self.assertFalse(kwargs["validate_receptor"])

    @patch("prodock.structure.conversion._mgltools_sdf_to_pdbqt")
    def test_sdf_to_pdbqt_mgltools_route(self, mock_mgl) -> None:
        out = self.tmpdir / "lig_mgl.pdbqt"
        mock_mgl.return_value = out

        result = sdf_to_pdbqt(TEST_LIG, out, backend="mgltools")

        self.assertEqual(_resolved(result), _resolved(out))
        mock_mgl.assert_called_once()

    @patch("prodock.structure.conversion.pdb_to_pdbqt")
    def test_ensure_pdbqt_routes_pdb_input(self, mock_pdb_to_pdbqt) -> None:
        out = self.tmpdir / "outdir" / "5N2F.pdbqt"
        mock_pdb_to_pdbqt.return_value = out

        result = ensure_pdbqt(
            TEST_PDB,
            self.tmpdir / "outdir",
            backend="meeko",
            mode="receptor",
        )

        self.assertEqual(_resolved(result), _resolved(out))
        mock_pdb_to_pdbqt.assert_called_once()

    @patch("prodock.structure.conversion.sdf_to_pdbqt")
    def test_ensure_pdbqt_routes_sdf_input(self, mock_sdf_to_pdbqt) -> None:
        out = self.tmpdir / "outdir" / "8HW.pdbqt"
        mock_sdf_to_pdbqt.return_value = out

        result = ensure_pdbqt(
            TEST_LIG,
            self.tmpdir / "outdir",
            backend="obabel",
            mode="ligand",
        )

        self.assertEqual(_resolved(result), _resolved(out))
        mock_sdf_to_pdbqt.assert_called_once()

    def test_ensure_pdbqt_returns_existing_pdbqt_as_is(self) -> None:
        inp = self.tmpdir / "already.pdbqt"
        inp.write_text("ATOM\n", encoding="utf-8")

        result = ensure_pdbqt(
            inp,
            self.tmpdir / "outdir",
            backend="obabel",
        )

        self.assertEqual(_resolved(result), _resolved(inp))

    @patch("prodock.structure.conversion.convert_with_obabel")
    def test_ensure_pdbqt_routes_mol2_via_obabel(self, mock_obabel) -> None:
        inp = self.tmpdir / "lig.mol2"
        inp.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        outdir = self.tmpdir / "outdir"
        expected = outdir / "lig.pdbqt"

        def _fake(*args, **kwargs):
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_text("ATOM\n", encoding="utf-8")

        mock_obabel.side_effect = _fake

        result = ensure_pdbqt(
            inp,
            outdir,
            backend="obabel",
            mode="ligand",
        )

        self.assertEqual(_resolved(result), _resolved(expected))
        mock_obabel.assert_called_once()

    def test_ensure_pdbqt_mol2_non_obabel_not_supported(self) -> None:
        inp = self.tmpdir / "lig.mol2"
        inp.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")

        with self.assertRaises(NotImplementedError):
            ensure_pdbqt(inp, self.tmpdir / "outdir", backend="meeko")

    @patch("prodock.structure.conversion.convert_with_obabel")
    def test_pdb_to_sdf_obabel_route(self, mock_obabel) -> None:
        out = self.tmpdir / "protein.sdf"

        def _fake(*args, **kwargs):
            out.write_text("$$$$\n", encoding="utf-8")

        mock_obabel.side_effect = _fake

        result = pdb_to_sdf(TEST_PDB, out, backend="obabel")

        self.assertEqual(_resolved(result), _resolved(out))
        mock_obabel.assert_called_once_with(
            TEST_PDB.resolve(),
            out,
            extra_args=None,
            validate_receptor=False,
        )

    @unittest.skipUnless(TEST_LIG.exists(), f"Missing test ligand: {TEST_LIG}")
    @patch("prodock.structure.conversion.Chem.SDWriter")
    @patch("prodock.structure.conversion.Chem.MolFromPDBFile")
    def test_pdb_to_sdf_rdkit_route_for_ligand(
        self,
        mock_mol_from_pdb,
        mock_sdwriter,
    ) -> None:
        ligand_pdb = self.tmpdir / "ligand_input.pdb"
        out = self.tmpdir / "ligand_rdkit.sdf"

        ligand_pdb.write_text(
            "HETATM    1  C1  UNL A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "END\n",
            encoding="utf-8",
        )

        mol = object()
        mock_mol_from_pdb.return_value = mol

        writer = mock_sdwriter.return_value

        def _fake_write(mol_obj):
            out.write_text(
                "Ligand\n"
                "  RDKit          3D\n"
                "\n"
                "  0  0  0  0  0  0            999 V2000\n"
                "M  END\n"
                "$$$$\n",
                encoding="utf-8",
            )

        writer.write.side_effect = _fake_write

        result = pdb_to_sdf(ligand_pdb, out, backend="rdkit")

        self.assertEqual(_resolved(result), _resolved(out))
        mock_mol_from_pdb.assert_called_once_with(
            str(ligand_pdb.resolve()), removeHs=False
        )
        mock_sdwriter.assert_called_once_with(str(out))
        writer.write.assert_called_once_with(mol)
        writer.close.assert_called_once()
        self.assertTrue(out.exists())

    @patch("prodock.structure.conversion.convert_with_obabel")
    def test_pdbqt_to_sdf_route(self, mock_obabel) -> None:
        inp = self.tmpdir / "in.pdbqt"
        out = self.tmpdir / "out.sdf"
        inp.write_text("ATOM\n", encoding="utf-8")

        def _fake(*args, **kwargs):
            out.write_text("$$$$\n", encoding="utf-8")

        mock_obabel.side_effect = _fake

        result = pdbqt_to_sdf(inp, out, backend="obabel")

        self.assertEqual(_resolved(result), _resolved(out))
        mock_obabel.assert_called_once_with(
            inp.resolve(),
            out,
            extra_args=None,
            validate_receptor=False,
        )

    def test_pdbqt_to_sdf_non_obabel_not_supported(self) -> None:
        inp = self.tmpdir / "in.pdbqt"
        out = self.tmpdir / "out.sdf"
        inp.write_text("ATOM\n", encoding="utf-8")

        with self.assertRaises(NotImplementedError):
            pdbqt_to_sdf(inp, out, backend="meekox")  # type: ignore[arg-type]


@unittest.skipUnless(TEST_PDB.exists(), f"Missing test input: {TEST_PDB}")
@unittest.skipUnless(TEST_LIG.exists(), f"Missing test ligand: {TEST_LIG}")
class TestPublicConversionWrappersOpenBabelIntegration(unittest.TestCase):
    """
    Optional real-Open-Babel smoke tests for the public wrapper API.
    """

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory(
            prefix="prodock_test_public_obabel_"
        )
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    def test_public_sdf_to_pdb_and_back_to_pdbqt(self) -> None:
        pdb_path = self.tmpdir / "ligand.pdb"
        pdbqt_path = self.tmpdir / "ligand.pdbqt"

        out1 = sdf_to_pdb(TEST_LIG, pdb_path, backend="obabel")
        out2 = pdb_to_pdbqt(pdb_path, pdbqt_path, mode="ligand", backend="obabel")

        self.assertEqual(_resolved(out1), _resolved(pdb_path))
        self.assertEqual(_resolved(out2), _resolved(pdbqt_path))
        self.assertTrue(pdb_path.exists())
        self.assertTrue(pdbqt_path.exists())

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    def test_public_pdb_to_sdf_and_ensure_pdbqt(self) -> None:
        sdf_path = self.tmpdir / "protein.sdf"
        pdbqt_dir = self.tmpdir / "pdbqt_out"

        out1 = pdb_to_sdf(TEST_PDB, sdf_path, backend="obabel")
        out2 = ensure_pdbqt(TEST_PDB, pdbqt_dir, backend="obabel", mode="receptor")

        self.assertEqual(_resolved(out1), _resolved(sdf_path))
        self.assertTrue(sdf_path.exists())
        self.assertTrue(out2.exists())
        self.assertEqual(out2.suffix.lower(), ".pdbqt")


if __name__ == "__main__":
    unittest.main()
