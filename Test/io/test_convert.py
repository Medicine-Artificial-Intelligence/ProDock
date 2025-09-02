import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import prodock.io.convert as convert


def _write_file(dirpath: str, name: str, content: str) -> Path:
    p = Path(dirpath) / name
    p.write_text(content, encoding="utf-8")
    return p


class TestConvertUnittest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.tmpdir = self.td.name

    def tearDown(self):
        self.td.cleanup()

    def test_ensure_exists_missing_raises(self):
        p = Path(self.tmpdir) / "nope.pdb"
        with self.assertRaises(FileNotFoundError):
            convert._ensure_exists(p, "Input PDB")

    def test_require_exe_found_and_missing(self):
        # found
        with mock.patch.object(
            convert.shutil, "which", return_value="/usr/bin/somebin"
        ):
            path = convert._require_exe("somebin")
            self.assertTrue(path.endswith("somebin"))

        # not found -> RuntimeError
        with mock.patch.object(convert.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError):
                convert._require_exe("not_there_bin")

    def test_run_raises_on_nonzero(self):
        fake_proc = SimpleNamespace(returncode=2, stdout="out", stderr="err")
        with mock.patch.object(convert.subprocess, "run", return_value=fake_proc):
            with self.assertRaises(RuntimeError) as ctx:
                convert._run(["false"])
            msg = str(ctx.exception)
            self.assertIn("Command failed", msg)
            self.assertIn("STDOUT", msg)
            self.assertIn("STDERR", msg)

    def test_needs_meeko_sanitize_fast_invalid_fixed_column(self):
        # make line length >= 78 and place 'A' at columns 77-78 (1-index)
        # index 76:78 (0-based) -> put 'A' there
        ln = (
            "ATOM      1  C   LIG A   1      11.104  13.207  10.000  1.00  0.00".ljust(
                76
            )
            + "A\n"
        )
        p = _write_file(self.tmpdir, "bad1.pdbqt", ln)
        self.assertTrue(convert._needs_meeko_sanitize_fast(p))

    def test_needs_meeko_sanitize_fast_suspect_token_last(self):
        ln = "ATOM      1  C   LIG A   1      11.104  13.207  10.000  1.00  0.00    CG0\n"
        p = _write_file(self.tmpdir, "bad2.pdbqt", ln)
        self.assertTrue(convert._needs_meeko_sanitize_fast(p))

    def test_needs_meeko_sanitize_fast_alpha_digits_token(self):
        ln = "HETATM    1  C   LIG A   1      11.104  13.207  10.000  1.00  0.00    CL1\n"
        p = _write_file(self.tmpdir, "bad3.pdbqt", ln)
        self.assertTrue(convert._needs_meeko_sanitize_fast(p))

    def test_needs_meeko_sanitize_fast_prior_token_when_last_is_float(self):
        # last token numeric, prior token in suspect set
        ln = "ATOM      1  C   LIG A   1      11.104  13.207  10.000  CG0  0.123\n"
        p = _write_file(self.tmpdir, "bad4.pdbqt", ln)
        self.assertTrue(convert._needs_meeko_sanitize_fast(p))

    def test_needs_meeko_sanitize_fast_clean(self):
        # valid element "C" in fixed column
        ln = (
            "ATOM      1  C   LIG A   1      11.104  13.207  10.000  1.00  0.00".ljust(
                76
            )
            + "C\n"
        )
        p = _write_file(self.tmpdir, "good.pdbqt", ln)
        self.assertFalse(convert._needs_meeko_sanitize_fast(p))

    def test_sanitize_meeko_if_needed_calls_sanitizer(self):
        p = _write_file(self.tmpdir, "x.pdbqt", "ATOM      1\n")

        # force decision to sanitize
        with mock.patch.object(
            convert, "_needs_meeko_sanitize_fast", return_value=True
        ):
            called = {}

            class DummySan:
                @staticmethod
                def sanitize_file(
                    in_path, out_path=None, rebuild=True, aggressive=False, backup=False
                ):
                    called["ok"] = True
                    return None

            with mock.patch.object(convert, "PDBQTSanitizer", DummySan):
                # Should not raise
                convert._sanitize_meeko_if_needed(p)
                self.assertTrue(called.get("ok", False))

    def test_sanitize_meeko_if_needed_skips_when_not_needed(self):
        p = _write_file(self.tmpdir, "x2.pdbqt", "ATOM      1\n")
        with mock.patch.object(
            convert, "_needs_meeko_sanitize_fast", return_value=False
        ):

            class DummySan:
                @staticmethod
                def sanitize_file(*args, **kwargs):
                    raise AssertionError("Should not be called")

            with mock.patch.object(convert, "PDBQTSanitizer", DummySan):
                # Should simply return and not call sanitize_file
                convert._sanitize_meeko_if_needed(p)  # no exception

    def test_sanitize_meeko_if_needed_handles_exception(self):
        p = _write_file(self.tmpdir, "x3.pdbqt", "ATOM      1\n")
        with mock.patch.object(
            convert, "_needs_meeko_sanitize_fast", return_value=True
        ):

            class DummySan:
                @staticmethod
                def sanitize_file(*args, **kwargs):
                    raise RuntimeError("boom")

            with mock.patch.object(convert, "PDBQTSanitizer", DummySan):
                # should not raise, exception should be caught and logged
                convert._sanitize_meeko_if_needed(p)  # no exception

    def test_tmp_sdf_to_pdb_with_rdkit_success_and_fail(self):
        # create minimal sdf file
        sdf = _write_file(self.tmpdir, "m.sdf", "\n")

        # fake Chem object
        class DummyMol:
            pass

        def fake_SDMolSupplier(path, removeHs=False, sanitize=True):
            # when path contains "empty" yield None
            if "empty" in str(path):
                yield None
            else:
                yield DummyMol()

        def fake_MolToPDBFile(mol, out_path):
            with open(out_path, "w") as fh:
                fh.write("PDB")

        DummyChem = SimpleNamespace(
            SDMolSupplier=fake_SDMolSupplier,
            MolToPDBFile=fake_MolToPDBFile,
        )

        with mock.patch.object(convert, "Chem", DummyChem), mock.patch.object(
            convert, "_RDKIT_AVAILABLE", True
        ):
            out = Path(self.tmpdir) / "out.pdb"
            convert._tmp_sdf_to_pdb_with_rdkit(sdf, out)
            self.assertTrue(out.exists())

            # failure case: supplier yields only None
            empty_sdf = _write_file(self.tmpdir, "empty.sdf", "\n")
            with self.assertRaises(ValueError):
                convert._tmp_sdf_to_pdb_with_rdkit(
                    empty_sdf, Path(self.tmpdir) / "no.pdb"
                )

    def test_tmp_sdf_to_pdb_with_obabel_calls_run(self):
        in_sdf = _write_file(self.tmpdir, "a.sdf", "X")
        out_pdb = Path(self.tmpdir) / "a.pdb"

        # simulate obabel presence and intercept convert._run
        with mock.patch.object(convert.shutil, "which", return_value="/usr/bin/obabel"):
            called = {}

            def fake_run(args):
                called["args"] = args

            with mock.patch.object(convert, "_run", fake_run):
                convert._tmp_sdf_to_pdb_with_obabel(in_sdf, out_pdb)
                self.assertIn("obabel", called["args"][0] or called["args"][0])

    def test_converter_pdbqt_noop_and_missing_input(self):
        p = _write_file(self.tmpdir, "lig.pdbqt", "PDBQT")
        conv = convert.Converter()
        conv.set_input(p)
        conv.set_backend("meeko")
        conv.run()
        # output should be equal to input Path
        self.assertEqual(conv.output, p)

        conv2 = convert.Converter()
        conv2.set_backend("obabel")
        with self.assertRaises(RuntimeError):
            conv2.run()

    def test_ensure_pdbqt_input_missing(self):
        missing = Path(self.tmpdir) / "no.file"
        with self.assertRaises(FileNotFoundError):
            convert.ensure_pdbqt(missing, self.tmpdir, backend="obabel")

    def test_sdf_to_pdb_input_missing(self):
        missing = Path(self.tmpdir) / "no.sdf"
        with self.assertRaises(FileNotFoundError):
            convert.sdf_to_pdb(missing, Path(self.tmpdir) / "out.pdb", backend="obabel")


if __name__ == "__main__":
    unittest.main()
