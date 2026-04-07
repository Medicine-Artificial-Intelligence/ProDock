from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prodock.structure.conversion import convert_with_mekoo, convert_with_obabel
from prodock.structure.pdbqt_sanitizer import PDBQTSanitizer

TEST_PDB = Path("Data/testcase/4WKQ/receptor/4WKQ.pdb")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _has_torsion_tokens(text: str) -> bool:
    tokens = (
        "ROOT",
        "BRANCH",
        "ENDBRANCH",
        "ENDROOT",
        "TORSDOF",
        "active torsions",
    )
    low = text.lower()
    return any(tok.lower() in low for tok in tokens)


@unittest.skipUnless(TEST_PDB.exists(), f"Missing test input: {TEST_PDB}")
class TestPDBQTSanitizer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory(prefix="prodock_pdbqt_sanitizer_")
        self.tmp = Path(self.tmpdir_obj.name)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _write(self, name: str, content: str) -> Path:
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _make_atom_line(
        self,
        *,
        serial: int = 1,
        atom_name: str = "C",
        res_name: str = "LIG",
        res_seq: int = 1,
        x: float = 1.0,
        y: float = 2.0,
        z: float = 3.0,
        occ: float = 1.00,
        temp: float = 10.00,
        trailing: str = "",
        fixed_elem: str = "",
    ) -> str:
        """
        Build a minimally PDBQT-like ATOM line where:
        - atom name is placed in columns 13-16
        - optional element is placed in columns 77-78
        - optional trailing token remains visible to split()-based parsing
        """
        base = (
            f"{'ATOM':<6}{serial:>5} {atom_name:^4} {res_name:>3} "
            f"{res_seq:>4}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}"
        )

        if len(base) < 76:
            base = base + " " * (76 - len(base))

        line = base + f"{fixed_elem:>2}"
        if trailing:
            line += f" {trailing}"
        return line + "\n"

    # ------------------------------------------------------------------
    # Construction / basic behavior
    # ------------------------------------------------------------------
    def test_init_with_path_reads_file(self):
        p = self._write("init_read.pdbqt", "REMARK hello\n")
        s = PDBQTSanitizer(p)
        self.assertEqual(s.lines, ["REMARK hello"])
        self.assertFalse(s._sanitized)
        self.assertEqual(s.backend, "meeko")

    def test_read_missing_file_raises(self):
        s = PDBQTSanitizer()
        with self.assertRaises(FileNotFoundError):
            s.read(self.tmp / "missing.pdbqt")

    def test_write_before_sanitize_raises(self):
        p = self._write("write_before.pdbqt", "REMARK x\n")
        s = PDBQTSanitizer(p)
        with self.assertRaises(RuntimeError):
            s.write(self.tmp / "out.pdbqt")

    def test_validate_without_loaded_file_raises(self):
        s = PDBQTSanitizer()
        with self.assertRaises(RuntimeError):
            s.validate()

    def test_sanitize_without_loaded_file_raises(self):
        s = PDBQTSanitizer()
        with self.assertRaises(RuntimeError):
            s.sanitize()

    def test_sanitize_inplace_without_loaded_file_raises(self):
        s = PDBQTSanitizer()
        with self.assertRaises(RuntimeError):
            s.sanitize_inplace()

    def test_repr_and_help(self):
        s = PDBQTSanitizer()
        rep = repr(s)
        self.assertIsInstance(rep, str)
        self.assertIn("PDBQTSanitizer", rep)

        help_text = s.help()
        self.assertIsInstance(help_text, str)
        self.assertIn("validate", help_text)
        self.assertIn("sanitize", help_text)
        self.assertIn("write", help_text)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def test_unknown_top_level_tag_warns(self):
        p = self._write("unknown_tag.pdbqt", "FOOBAR some stuff\n")
        s = PDBQTSanitizer(p)
        warns = s.validate(strict=False)
        self.assertTrue(any("unknown top-level tag 'FOOBAR'" in w for w in warns))

    def test_tag_whitelist_does_not_warn(self):
        p = self._write("whitelist.pdbqt", "REMARK example\nROOT\nENDROOT\n")
        s = PDBQTSanitizer(p)
        warns = s.validate(strict=False)
        self.assertEqual(warns, [])

    def test_blank_lines_are_ignored_in_validate(self):
        p = self._write("blank_validate.pdbqt", "\n\nREMARK ok\n\n")
        s = PDBQTSanitizer(p)
        warns = s.validate(strict=False)
        self.assertEqual(warns, [])

    def test_trailing_token_noncanonical_warns_when_mappable(self):
        line = "ATOM   1  CG  LIG 1  12.345  13.456  14.567  1.00 20.00 CG0\n"
        p = self._write("trail_map.pdbqt", line)
        s = PDBQTSanitizer(p, backend="meeko")
        warns = s.validate(strict=False)
        self.assertTrue(
            any(
                "trailing PDBQT type 'CG0' is non-canonical; suggested='C'"
                in w
                for w in warns
            )
        )

    def test_trailing_token_unmappable_warns(self):
        line = "ATOM   2  XX  LIG 1  12.345  13.456  14.567  1.00 20.00 XXY\n"
        p = self._write("trail_bad.pdbqt", line)
        s = PDBQTSanitizer(p)
        warns = s.validate(strict=False)
        self.assertTrue(
            any(
                "trailing token 'XXY' is not a recognized PDBQT type"
                in w
                for w in warns
            )
        )

    def test_missing_trailing_type_warns_in_strict_mode_when_guessable(self):
        line = self._make_atom_line(
            serial=3,
            atom_name="CL1",
            fixed_elem="Cl",
            trailing="",
        )
        p = self._write("missing_trailing_guessable.pdbqt", line)
        s = PDBQTSanitizer(p)

        # Force the validator down the "missing trailing type" path by mocking
        # the split-based trailing-token extractor.
        with patch.object(s, "_extract_trailing_type", return_value=""):
            warns = s.validate(strict=True)

        self.assertTrue(
            any(
                "missing trailing PDBQT atom type; suggested='Cl'" in w
                for w in warns
            ),
            msg=f"Warnings were: {warns}",
        )

    def test_missing_trailing_type_warns_in_strict_mode_when_not_guessable(self):
        line = "ATOM   3  XXY LIG 1  1.000  2.000  3.000\n"
        p = self._write("missing_trailing_unguessable.pdbqt", line)
        s = PDBQTSanitizer(p)

        # Current parser would treat 'LIG' as trailing token on this short line,
        # so mock the trailing-token extractor to exercise the intended branch.
        with patch.object(s, "_extract_trailing_type", return_value=""):
            warns = s.validate(strict=True)

        self.assertTrue(
            any(
                "missing trailing PDBQT atom type and could not infer one" in w
                for w in warns
            ),
            msg=f"Warnings were: {warns}",
        )

    def test_could_not_parse_xyz_warns(self):
        line = "ATOM 1 CA LIG 1 X Y Z\n"
        p = self._write("bad_xyz_validate.pdbqt", line)
        s = PDBQTSanitizer(p)

        warns = s.validate(strict=False)

        self.assertTrue(
            any("could not parse x/y/z coordinates" in w for w in warns),
            msg=f"Warnings were: {warns}",
        )

    # ------------------------------------------------------------------
    # Sanitize
    # ------------------------------------------------------------------
    def test_sanitize_rebuild_false_still_outputs_fixed_width_line(self):
        line = "ATOM   1  CG  LIG 1  12.345  13.456  14.567  1.00 20.00 CG0\n"
        p = self._write("sanitize_meeko_minimal.pdbqt", line)
        s = PDBQTSanitizer(p, backend="meeko")

        s.sanitize(rebuild=False, aggressive=False)

        self.assertTrue(s._sanitized)
        self.assertEqual(len(s.sanitized_lines), 1)
        rebuilt = s.sanitized_lines[0]
        self.assertGreaterEqual(len(rebuilt), 80)
        self.assertEqual(rebuilt[78:80].strip(), "C")
        self.assertTrue(
            any("normalized ATOM/HETATM in fixed-width mode" in w for w in s.warnings)
        )

    def test_sanitize_rebuild_false_obabel_keeps_valid_type(self):
        line = "ATOM   1  CL  LIG 1  12.345  13.456  14.567  1.00 20.00 CL\n"
        p = self._write("sanitize_obabel_minimal.pdbqt", line)
        s = PDBQTSanitizer(p, backend="obabel")

        s.sanitize(rebuild=False, aggressive=False)

        self.assertTrue(s._sanitized)
        rebuilt = s.sanitized_lines[0]
        self.assertGreaterEqual(len(rebuilt), 80)
        self.assertEqual(rebuilt[78:80].strip(), "Cl")

    def test_short_atom_line_left_unchanged(self):
        line = "ATOM 1 C\n"
        p = self._write("short_atom.pdbqt", line)
        s = PDBQTSanitizer(p)

        s.sanitize(rebuild=True)

        self.assertEqual(s.sanitized_lines[0], "ATOM 1 C")
        self.assertTrue(
            any("short ATOM/HETATM left unchanged" in w for w in s.warnings)
        )

    def test_cannot_parse_coordinates_left_unchanged(self):
        line = "ATOM 1 CA LIG 1 X Y Z\n"
        p = self._write("bad_coords.pdbqt", line)
        s = PDBQTSanitizer(p)

        s.sanitize(rebuild=True)

        self.assertEqual(s.sanitized_lines[0], "ATOM 1 CA LIG 1 X Y Z")
        self.assertTrue(
            any("could not parse coordinates, left unchanged" in w for w in s.warnings)
        )

    def test_incomplete_coordinates_left_unchanged(self):
        line = "ATOM 1 CA LIG 1 1.0 2.0\n"
        p = self._write("incomplete_coords.pdbqt", line)
        s = PDBQTSanitizer(p)

        s.sanitize(rebuild=True)

        self.assertEqual(s.sanitized_lines[0], "ATOM 1 CA LIG 1 1.0 2.0")
        self.assertTrue(
            any("could not parse coordinates, left unchanged" in w for w in s.warnings)
        )

    def test_blank_lines_preserved_in_sanitize(self):
        text = "\nATOM 1 C LIG 1 X Y Z\n\n"
        p = self._write("blank_sanitize.pdbqt", text)
        s = PDBQTSanitizer(p)

        s.sanitize(rebuild=False)

        self.assertEqual(s.sanitized_lines[0], "")
        self.assertEqual(s.sanitized_lines[2], "")

    def test_sanitize_rebuild_sets_pdbqt_type_and_warns_rebuilt(self):
        line = "ATOM   2  CG  LIG 1  1.000  2.000  3.000  1.00 10.00 CG0\n"
        p = self._write("rebuild_basic.pdbqt", line)
        s = PDBQTSanitizer(p)

        s.sanitize(rebuild=True, aggressive=False)

        self.assertTrue(s._sanitized)
        rebuilt = s.sanitized_lines[0]
        self.assertGreaterEqual(len(rebuilt), 80)
        self.assertEqual(rebuilt[78:80].strip(), "C")
        self.assertTrue(any("rebuilt ATOM/HETATM" in w for w in s.warnings))

    def test_rebuild_uses_trailing_mapping_first_for_meeko(self):
        line = "ATOM   7  CA  LIG 1  1.000  2.000  3.000  1.00 10.00 OA\n"
        p = self._write("meeko_prefers_trailing.pdbqt", line)
        s = PDBQTSanitizer(p, backend="meeko")

        s.sanitize(rebuild=True, aggressive=False)

        rebuilt = s.sanitized_lines[0]
        self.assertEqual(rebuilt[78:80].strip(), "OA")

    def test_rebuild_obabel_also_prefers_valid_trailing_type(self):
        line = "ATOM   8  CL  LIG 1  1.000  2.000  3.000  1.00 10.00 OA\n"
        p = self._write("obabel_prefers_trailing.pdbqt", line)
        s = PDBQTSanitizer(p, backend="obabel")

        s.sanitize(rebuild=True, aggressive=False)

        rebuilt = s.sanitized_lines[0]
        self.assertEqual(rebuilt[78:80].strip(), "OA")

    def test_rebuild_aggressive_normalizes_multiletter_elements(self):
        line = "ATOM   9  CL1 LIG 1  1.000  2.000  3.000  1.00 10.00 CL1\n"
        p = self._write("aggressive_cl.pdbqt", line)
        s = PDBQTSanitizer(p, backend="meeko")

        s.sanitize(rebuild=True, aggressive=True)

        rebuilt = s.sanitized_lines[0]
        self.assertEqual(rebuilt[78:80].strip(), "Cl")

    def test_rebuild_without_aggressive_leaves_unknown_type_unchanged(self):
        line = "ATOM  10  XX  LIG 1  1.000  2.000  3.000  1.00 10.00 XXY\n"
        p = self._write("fallback_no_aggressive.pdbqt", line)
        s = PDBQTSanitizer(p)

        s.sanitize(rebuild=True, aggressive=False)

        self.assertEqual(s.sanitized_lines[0], line.rstrip("\n"))
        self.assertTrue(
            any(
                "could not infer PDBQT atom type, left unchanged" in w
                for w in s.warnings
            )
        )

    def test_rebuild_with_aggressive_falls_back_to_C(self):
        line = "ATOM  10  XX  LIG 1  1.000  2.000  3.000  1.00 10.00 XXY\n"
        p = self._write("fallback_aggressive_C.pdbqt", line)
        s = PDBQTSanitizer(p)

        s.sanitize(rebuild=True, aggressive=True)

        rebuilt = s.sanitized_lines[0]
        self.assertEqual(rebuilt[78:80].strip(), "C")

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def test_write_outputs_newline_terminated_file(self):
        line = "ATOM   1  CG  LIG 1  1.000  2.000  3.000  1.00 10.00 CG0\n"
        inp = self._write("write_test_in.pdbqt", line)
        out = self.tmp / "nested" / "write_test_out.pdbqt"

        s = PDBQTSanitizer(inp)
        s.sanitize(rebuild=False)
        written = s.write(out)

        self.assertEqual(written, out)
        self.assertTrue(out.exists())
        text = out.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))

    def test_sanitize_file_with_explicit_outpath_does_not_create_backup(self):
        line = "ATOM   3  C   LIG 1  1.0 2.0 3.0 1.00 10.00 CG0\n"
        inp = self._write("sanitize_file_in.pdbqt", line)
        out = self.tmp / "sanitize_file_out.pdbqt"

        result = PDBQTSanitizer.sanitize_file(
            inp,
            out_path=out,
            rebuild=False,
            aggressive=False,
            backup=True,
        )

        self.assertEqual(result, out)
        self.assertTrue(out.exists())
        self.assertFalse((inp.with_suffix(inp.suffix + ".bak")).exists())

    def test_sanitize_file_overwrite_creates_backup_when_requested(self):
        line = "ATOM   4  C   LIG 1  1.0 2.0 3.0 1.00 10.00 CG0\n"
        inp = self._write("overwrite_in.pdbqt", line)

        result = PDBQTSanitizer.sanitize_file(
            inp,
            out_path=None,
            rebuild=True,
            aggressive=False,
            backup=True,
        )

        self.assertEqual(result, inp)
        bak = inp.with_suffix(inp.suffix + ".bak")
        self.assertTrue(bak.exists())

    def test_sanitize_file_overwrite_without_backup_does_not_create_bak(self):
        line = "ATOM   5  C   LIG 1  1.0 2.0 3.0 1.00 10.00 CG0\n"
        inp = self._write("overwrite_no_bak_in.pdbqt", line)

        result = PDBQTSanitizer.sanitize_file(
            inp,
            out_path=None,
            rebuild=True,
            aggressive=False,
            backup=False,
        )

        self.assertEqual(result, inp)
        bak = inp.with_suffix(inp.suffix + ".bak")
        self.assertFalse(bak.exists())

    def test_sanitize_inplace_writes_backup_and_overwrites(self):
        line = "ATOM   5  OH  LIG 1  1.0 2.0 3.0 1.00 10.00 OH\n"
        inp = self._write("inplace.pdbqt", line)
        s = PDBQTSanitizer(inp)

        out = s.sanitize_inplace(rebuild=False, aggressive=True, backup=True)

        self.assertEqual(out, inp)
        self.assertTrue(inp.exists())
        self.assertTrue((inp.with_suffix(inp.suffix + ".bak")).exists())
        self.assertTrue(s._sanitized)
        self.assertGreaterEqual(len(s.sanitized_lines), 1)

    # ------------------------------------------------------------------
    # Internal helper coverage
    # ------------------------------------------------------------------
    def test_canonicalize_element(self):
        self.assertEqual(PDBQTSanitizer._canonicalize_element("cl"), "Cl")
        self.assertEqual(PDBQTSanitizer._canonicalize_element("br"), "Br")
        self.assertEqual(PDBQTSanitizer._canonicalize_element("c"), "C")
        self.assertEqual(PDBQTSanitizer._canonicalize_element(""), "")

    def test_strip_digits(self):
        self.assertEqual(PDBQTSanitizer._strip_digits("CG0"), "CG")
        self.assertEqual(PDBQTSanitizer._strip_digits("CL12"), "CL")
        self.assertEqual(PDBQTSanitizer._strip_digits("1234"), "")

    def test_is_valid_element_token(self):
        s = PDBQTSanitizer()
        self.assertTrue(s._is_valid_element_token("C"))
        self.assertTrue(s._is_valid_element_token("Cl"))
        self.assertFalse(s._is_valid_element_token("CL"))
        self.assertFalse(s._is_valid_element_token("ZZ"))
        self.assertFalse(s._is_valid_element_token(""))

    def test_is_valid_pdbqt_type(self):
        s = PDBQTSanitizer()
        self.assertTrue(s._is_valid_pdbqt_type("C"))
        self.assertTrue(s._is_valid_pdbqt_type("OA"))
        self.assertFalse(s._is_valid_pdbqt_type("CG0"))
        self.assertFalse(s._is_valid_pdbqt_type(""))

    def test_normalize_pdbqt_type(self):
        s = PDBQTSanitizer(backend="meeko")
        self.assertEqual(s._normalize_pdbqt_type("CG0"), "C")
        self.assertEqual(s._normalize_pdbqt_type("CL1"), "Cl")
        self.assertEqual(s._normalize_pdbqt_type("OA"), "OA")
        self.assertEqual(s._normalize_pdbqt_type("???"), "")

    def test_default_type_from_element(self):
        s = PDBQTSanitizer()
        self.assertEqual(s._default_type_from_element("Cl"), "Cl")
        self.assertEqual(s._default_type_from_element("C"), "C")
        self.assertEqual(s._default_type_from_element("ZZ"), "")

    def test_infer_element_from_atom_name(self):
        s = PDBQTSanitizer()
        self.assertEqual(s._infer_element_from_atom_name("CL1"), "Cl")
        self.assertEqual(s._infer_element_from_atom_name("C1"), "C")
        self.assertEqual(s._infer_element_from_atom_name(""), "")

    def test_extract_trailing_type(self):
        s = PDBQTSanitizer()
        line = "ATOM   1  CG  LIG 1  1.0 2.0 3.0 1.00 10.00 CG0"
        self.assertEqual(s._extract_trailing_type(line), "CG0")

    def test_extract_fixed_element(self):
        s = PDBQTSanitizer()
        line = ("ATOM   1  C   LIG 1  1.0 2.0 3.0 1.00 10.00").ljust(76) + "Cl"
        self.assertEqual(s._extract_fixed_element(line), "Cl")

    # ------------------------------------------------------------------
    # Integration with Open Babel-generated PDBQT
    # ------------------------------------------------------------------
    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    def test_validate_real_obabel_rigid_pdbqt_has_no_torsion_tokens(self):
        out = self.tmp / "obabel_rigid.pdbqt"

        convert_with_obabel(
            TEST_PDB,
            out,
            validate_receptor=True,
            flexibility=False,
        )

        self.assertTrue(out.exists())
        text = _read_text(out)
        self.assertFalse(
            _has_torsion_tokens(text),
            "Rigid receptor PDBQT should not contain torsion-tree markers",
        )

        s = PDBQTSanitizer(out, backend="obabel")
        warns = s.validate(strict=True)
        self.assertIsInstance(warns, list)

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    def test_validate_real_obabel_flexible_pdbqt(self):
        out = self.tmp / "obabel_flex.pdbqt"

        convert_with_obabel(
            TEST_PDB,
            out,
            validate_receptor=True,
            flexibility=True,
        )

        self.assertTrue(out.exists())
        text = _read_text(out)

        self.assertTrue(
            any(line.startswith(("ATOM", "HETATM")) for line in text.splitlines()),
            "Flexible receptor PDBQT should contain atom records",
        )

        s = PDBQTSanitizer(out, backend="obabel")
        warns = s.validate(strict=True)
        self.assertIsInstance(warns, list)

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    def test_rigid_vs_flexible_obabel_torsion_behavior(self):
        rigid = self.tmp / "rigid.pdbqt"
        flex = self.tmp / "flex.pdbqt"

        convert_with_obabel(
            TEST_PDB,
            rigid,
            validate_receptor=True,
            flexibility=False,
        )
        convert_with_obabel(
            TEST_PDB,
            flex,
            validate_receptor=True,
            flexibility=True,
        )

        rigid_text = _read_text(rigid)
        flex_text = _read_text(flex)

        self.assertFalse(
            _has_torsion_tokens(rigid_text),
            "Rigid receptor output must not contain torsion-tree markers",
        )
        self.assertIsInstance(_has_torsion_tokens(flex_text), bool)

    @unittest.skipUnless(
        shutil.which("obabel") or shutil.which("babel"),
        "Open Babel not available",
    )
    def test_sanitize_real_obabel_output_rewrite_to_new_file(self):
        out = self.tmp / "real_obabel.pdbqt"
        sanitized = self.tmp / "real_obabel_sanitized.pdbqt"

        convert_with_obabel(TEST_PDB, out, validate_receptor=True, flexibility=False)

        result = PDBQTSanitizer.sanitize_file(
            out,
            out_path=sanitized,
            backend="obabel",
            rebuild=True,
            aggressive=False,
            backup=True,
        )

        self.assertEqual(result, sanitized)
        self.assertTrue(sanitized.exists())
        self.assertFalse((out.with_suffix(out.suffix + ".bak")).exists())

        text = _read_text(sanitized)
        self.assertTrue(
            any(line.startswith(("ATOM", "HETATM")) for line in text.splitlines())
        )

    # ------------------------------------------------------------------
    # Integration with Meeko-generated PDBQT
    # ------------------------------------------------------------------
    def test_convert_with_mekoo_not_found_returns_info(self):
        out_basename = self.tmp / "meeko_out"
        write_pdbqt = self.tmp / "meeko_out.pdbqt"

        info = convert_with_mekoo(
            mekoo_cmd="definitely_not_a_real_meeko_command_12345",
            input_pdb=TEST_PDB,
            out_basename=out_basename,
            write_pdbqt=write_pdbqt,
        )

        self.assertIsInstance(info, dict)
        self.assertIn("stderr", info)
        self.assertIsNotNone(info["stderr"])
        self.assertIn("not found", str(info["stderr"]).lower())

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_mekoo_calls_sanitizer_for_written_pdbqt(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ):
        fake_exe = "/usr/bin/mk_prepare_receptor.py"
        mock_which.return_value = fake_exe

        write_pdbqt = self.tmp / "meeko_written.pdbqt"

        def _fake_run(*args, **kwargs):
            write_pdbqt.write_text(
                "ATOM      1  N   ALA A   1      11.104  13.207   9.455  1.00 20.00      A    N\n",
                encoding="utf-8",
            )

            class _Proc:
                returncode = 0
                stdout = "ok"
                stderr = ""

            return _Proc()

        mock_run.side_effect = _fake_run

        info = convert_with_mekoo(
            mekoo_cmd="mk_prepare_receptor.py",
            input_pdb=TEST_PDB,
            out_basename=self.tmp / "meeko_out",
            write_pdbqt=write_pdbqt,
        )

        self.assertEqual(info["rc"], 0)
        self.assertTrue(write_pdbqt.exists())
        mock_sanitize.assert_called()

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_convert_with_mekoo_collects_basename_products(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ):
        fake_exe = "/usr/bin/mk_prepare_receptor.py"
        mock_which.return_value = fake_exe

        out_basename = self.tmp / "bundle" / "receptor"
        out_basename.parent.mkdir(parents=True, exist_ok=True)

        def _fake_run(*args, **kwargs):
            out_basename.with_suffix(".pdbqt").write_text(
                "ATOM      1  C   UNL A   1       0.000   0.000   0.000  1.00  0.00      A    C\n",
                encoding="utf-8",
            )
            out_basename.with_suffix(".json").write_text(
                '{"status": "ok"}',
                encoding="utf-8",
            )

            class _Proc:
                returncode = 0
                stdout = "ok"
                stderr = ""

            return _Proc()

        mock_run.side_effect = _fake_run

        info = convert_with_mekoo(
            mekoo_cmd="mk_prepare_receptor.py",
            input_pdb=TEST_PDB,
            out_basename=out_basename,
            write_pdbqt=None,
        )

        self.assertEqual(info["rc"], 0)
        self.assertEqual(len(info["produced"]), 2)
        self.assertTrue(any(p.endswith(".pdbqt") for p in info["produced"]))
        self.assertTrue(any(p.endswith(".json") for p in info["produced"]))
        mock_sanitize.assert_called()

    @patch("prodock.structure.conversion.PDBQTSanitizer.sanitize_file")
    @patch("prodock.structure.conversion.subprocess.run")
    @patch("prodock.structure.conversion.shutil.which")
    def test_validate_mocked_meeko_generated_pdbqt_with_sanitizer_class(
        self,
        mock_which,
        mock_run,
        mock_sanitize,
    ):
        fake_exe = "/usr/bin/mk_prepare_receptor.py"
        mock_which.return_value = fake_exe

        out_pdbqt = self.tmp / "mocked_meeko_out.pdbqt"

        def _fake_run(*args, **kwargs):
            out_pdbqt.write_text(
                "ATOM   1  CG  LIG 1  12.345  13.456  14.567  1.00 20.00 CG0\n",
                encoding="utf-8",
            )

            class _Proc:
                returncode = 0
                stdout = "ok"
                stderr = ""

            return _Proc()

        mock_run.side_effect = _fake_run

        info = convert_with_mekoo(
            mekoo_cmd="mk_prepare_receptor.py",
            input_pdb=TEST_PDB,
            out_basename=self.tmp / "mocked_meeko_base",
            write_pdbqt=out_pdbqt,
        )

        self.assertEqual(info["rc"], 0)
        self.assertTrue(out_pdbqt.exists())

        s = PDBQTSanitizer(out_pdbqt, backend="meeko")
        warns = s.validate(strict=True)
        self.assertTrue(isinstance(warns, list))
        mock_sanitize.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)