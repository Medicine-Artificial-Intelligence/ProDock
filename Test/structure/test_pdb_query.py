# import tempfile
# import unittest
# from pathlib import Path

# from prodock.structure.pdbqt_sanitizer import PDBQTSanitizer


# class TestPDBQTSanitizer(unittest.TestCase):
#     def setUp(self) -> None:
#         self.tmpdir = tempfile.TemporaryDirectory()
#         self.tmp = Path(self.tmpdir.name)

#     def tearDown(self) -> None:
#         self.tmpdir.cleanup()

#     def _write(self, name: str, content: str) -> Path:
#         path = self.tmp / name
#         path.parent.mkdir(parents=True, exist_ok=True)
#         path.write_text(content, encoding="utf-8")
#         return path

#     def _make_atom_line(
#         self,
#         *,
#         serial: int = 1,
#         atom_name: str = "C",
#         res_name: str = "LIG",
#         res_seq: int = 1,
#         x: float = 1.0,
#         y: float = 2.0,
#         z: float = 3.0,
#         occ: float = 1.0,
#         temp: float = 10.0,
#         fixed_element: str = "",
#         trailing_token: str = "",
#         record: str = "ATOM",
#     ) -> str:
#         """
#         Build a fixed-width PDB/PDBQT-like ATOM/HETATM line.

#         The sanitizer reads:
#         - atom name from columns 13-16
#         - fixed element from columns 77-78

#         so tests that exercise validate() should prefer this helper instead of
#         purely whitespace-tokenized toy lines.
#         """
#         alt_loc = " "
#         chain_id = " "
#         i_code = " "

#         base = (
#             f"{record:<6}{serial:>5} {atom_name:^4}{alt_loc}{res_name:>3}"
#             f" {chain_id}{res_seq:>4}{i_code}   "
#             f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}"
#         )
#         if len(base) < 76:
#             base = base.ljust(76)
#         if fixed_element:
#             base = base[:76] + f"{fixed_element:>2}"
#         if trailing_token:
#             base = base + f" {trailing_token}"
#         return base + "\n"

#     # ------------------------------------------------------------------
#     # Construction / basic behavior
#     # ------------------------------------------------------------------
#     def test_init_with_path_reads_file(self):
#         p = self._write("init_read.pdbqt", "REMARK hello\n")
#         s = PDBQTSanitizer(p)
#         self.assertEqual(s.lines, ["REMARK hello"])
#         self.assertFalse(s._sanitized)
#         self.assertEqual(s.backend, "meeko")

#     def test_read_missing_file_raises(self):
#         s = PDBQTSanitizer()
#         with self.assertRaises(FileNotFoundError):
#             s.read(self.tmp / "missing.pdbqt")

#     def test_write_before_sanitize_raises(self):
#         p = self._write("write_before.pdbqt", "REMARK x\n")
#         s = PDBQTSanitizer(p)
#         with self.assertRaises(RuntimeError):
#             s.write(self.tmp / "out.pdbqt")

#     def test_validate_without_loaded_file_raises(self):
#         s = PDBQTSanitizer()
#         with self.assertRaises(RuntimeError):
#             s.validate()

#     def test_sanitize_without_loaded_file_raises(self):
#         s = PDBQTSanitizer()
#         with self.assertRaises(RuntimeError):
#             s.sanitize()

#     def test_sanitize_inplace_without_loaded_file_raises(self):
#         s = PDBQTSanitizer()
#         with self.assertRaises(RuntimeError):
#             s.sanitize_inplace()

#     def test_set_backend_returns_self(self):
#         s = PDBQTSanitizer()
#         returned = s.set_backend("obabel")
#         self.assertIs(returned, s)
#         self.assertEqual(s.backend, "obabel")

#     def test_repr_and_help(self):
#         s = PDBQTSanitizer()
#         rep = repr(s)
#         self.assertIsInstance(rep, str)
#         self.assertIn("PDBQTSanitizer", rep)

#         help_text = s.help()
#         self.assertIsInstance(help_text, str)
#         self.assertIn("validate", help_text)
#         self.assertIn("sanitize", help_text)
#         self.assertIn("write", help_text)

#     # ------------------------------------------------------------------
#     # Validation
#     # ------------------------------------------------------------------
#     def test_unknown_top_level_tag_warns(self):
#         p = self._write("unknown_tag.pdbqt", "FOOBAR some stuff\n")
#         s = PDBQTSanitizer(p)
#         warns = s.validate(strict=False)
#         self.assertTrue(any("unknown top-level tag 'FOOBAR'" in w for w in warns))

#     def test_tag_whitelist_does_not_warn(self):
#         p = self._write("whitelist.pdbqt", "REMARK example\nROOT\nENDROOT\n")
#         s = PDBQTSanitizer(p)
#         warns = s.validate(strict=False)
#         self.assertEqual(warns, [])

#     def test_blank_lines_are_ignored_in_validate(self):
#         p = self._write("blank_validate.pdbqt", "\n\nREMARK ok\n\n")
#         s = PDBQTSanitizer(p)
#         warns = s.validate(strict=False)
#         self.assertEqual(warns, [])

#     def test_trailing_token_noncanonical_warns_when_mappable(self):
#         line = self._make_atom_line(
#             serial=1,
#             atom_name="CG",
#             trailing_token="CG0",
#         )
#         p = self._write("trail_map.pdbqt", line)
#         s = PDBQTSanitizer(p, backend="meeko")
#         warns = s.validate(strict=False)
#         self.assertTrue(
#             any(
#                 "trailing token 'CG0' is non-canonical; suggested='C'" in w
#                 for w in warns
#             )
#         )

#     def test_trailing_token_unmappable_warns(self):
#         line = self._make_atom_line(
#             serial=2,
#             atom_name="XX",
#             trailing_token="XXY",
#         )
#         p = self._write("trail_bad.pdbqt", line)
#         s = PDBQTSanitizer(p)
#         warns = s.validate(strict=False)
#         self.assertTrue(
#             any("trailing token 'XXY' cannot be mapped" in w for w in warns)
#         )

#     def test_fixed_column_invalid_element_warns_and_includes_suggestion(self):
#         line = self._make_atom_line(
#             serial=10,
#             atom_name="C",
#             fixed_element="ZZ",
#             trailing_token="CG0",
#         )
#         p = self._write("fixed_invalid.pdbqt", line)
#         s = PDBQTSanitizer(p)

#         warns = s.validate(strict=True)

#         self.assertTrue(
#             any("fixed-column element token 'ZZ' is invalid" in w for w in warns)
#         )
#         self.assertTrue(any("suggested='C'" in w for w in warns))

#     def test_fixed_column_valid_but_trailing_differs_warns_in_strict_mode(self):
#         line = self._make_atom_line(
#             serial=11,
#             atom_name="C",
#             fixed_element="C",
#             trailing_token="OA",
#         )
#         p = self._write("strict_differs.pdbqt", line)
#         s = PDBQTSanitizer(p, backend="meeko")

#         warns = s.validate(strict=True)

#         self.assertTrue(
#             any(
#                 "trailing token 'OA' differs from fixed element 'C'; mapped='O'" in w
#                 for w in warns
#             )
#         )

#     def test_no_element_detected_warns_only_in_strict_mode(self):
#         line = self._make_atom_line(
#             serial=3,
#             atom_name="C1",
#             fixed_element="",
#             trailing_token="",
#         )
#         p = self._write("no_elem_strict.pdbqt", line)
#         s = PDBQTSanitizer(p)

#         warns_loose = s.validate(strict=False)
#         warns_strict = s.validate(strict=True)

#         self.assertFalse(any("no element detected" in w for w in warns_loose))
#         self.assertTrue(any("no element detected" in w for w in warns_strict))

#     def test_suspicious_atom_name_warns(self):
#         line = self._make_atom_line(
#             serial=6,
#             atom_name="C@1",
#             fixed_element="C",
#         )
#         p = self._write("suspicious_atom_name.pdbqt", line)
#         s = PDBQTSanitizer(p)
#         warns = s.validate(strict=False)
#         self.assertTrue(any("suspicious atom name 'C@1'" in w for w in warns))

#     # ------------------------------------------------------------------
#     # Sanitize without rebuild
#     # ------------------------------------------------------------------
#     def test_sanitize_without_rebuild_meeko_replaces_trailing_alias(self):
#         line = "ATOM   1  CG  LIG 1  12.345  13.456  14.567  1.00 20.00 CG0\n"
#         p = self._write("sanitize_meeko_minimal.pdbqt", line)
#         s = PDBQTSanitizer(p, backend="meeko")

#         s.sanitize(rebuild=False, aggressive=False)

#         self.assertTrue(s._sanitized)
#         self.assertEqual(len(s.sanitized_lines), 1)
#         self.assertTrue(s.sanitized_lines[0].endswith("C"))
#         self.assertTrue(
#             any("replaced trailing 'CG0' -> 'C'" in w for w in s.warnings)
#         )

#     def test_sanitize_without_rebuild_obabel_does_not_replace_trailing_alias(self):
#         line = "ATOM   1  CL  LIG 1  12.345  13.456  14.567  1.00 20.00 CL\n"
#         p = self._write("sanitize_obabel_minimal.pdbqt", line)
#         s = PDBQTSanitizer(p, backend="obabel")

#         s.sanitize(rebuild=False, aggressive=False)

#         self.assertTrue(s._sanitized)
#         self.assertEqual(s.sanitized_lines[0], line.rstrip("\n"))
#         self.assertFalse(any("replaced trailing" in w for w in s.warnings))

#     def test_short_atom_line_left_unchanged(self):
#         line = "ATOM 1 C\n"
#         p = self._write("short_atom.pdbqt", line)
#         s = PDBQTSanitizer(p)

#         s.sanitize(rebuild=True)

#         self.assertEqual(s.sanitized_lines[0], "ATOM 1 C")
#         self.assertTrue(
#             any("short ATOM/HETATM left unchanged" in w for w in s.warnings)
#         )

#     def test_cannot_parse_coordinates_left_unchanged(self):
#         line = "ATOM 1 CA LIG 1 X Y Z\n"
#         p = self._write("bad_coords.pdbqt", line)
#         s = PDBQTSanitizer(p)

#         s.sanitize(rebuild=True)

#         self.assertEqual(s.sanitized_lines[0], "ATOM 1 CA LIG 1 X Y Z")
#         self.assertTrue(any("cannot parse coordinates" in w for w in s.warnings))

#     def test_incomplete_coordinates_left_unchanged(self):
#         line = "ATOM 1 CA LIG 1 1.0 2.0\n"
#         p = self._write("incomplete_coords.pdbqt", line)
#         s = PDBQTSanitizer(p)

#         s.sanitize(rebuild=True)

#         self.assertEqual(s.sanitized_lines[0], "ATOM 1 CA LIG 1 1.0 2.0")
#         self.assertTrue(any("incomplete coordinates" in w for w in s.warnings))

#     def test_blank_lines_preserved_in_sanitize(self):
#         text = "\nATOM 1 C LIG 1 X Y Z\n\n"
#         p = self._write("blank_sanitize.pdbqt", text)
#         s = PDBQTSanitizer(p)

#         s.sanitize(rebuild=False)

#         self.assertEqual(s.sanitized_lines[0], "")
#         self.assertEqual(s.sanitized_lines[2], "")

#     # ------------------------------------------------------------------
#     # Sanitize with rebuild
#     # ------------------------------------------------------------------
#     def test_sanitize_rebuild_sets_fixed_element_and_warns_rebuilt(self):
#         line = "ATOM   2  CG  LIG 1  1.000  2.000  3.000  1.00 10.00 CG0\n"
#         p = self._write("rebuild_basic.pdbqt", line)
#         s = PDBQTSanitizer(p)

#         s.sanitize(rebuild=True, aggressive=False)

#         self.assertTrue(s._sanitized)
#         rebuilt = s.sanitized_lines[0]
#         self.assertGreaterEqual(len(rebuilt), 78)
#         self.assertEqual(rebuilt[76:78].strip(), "C")
#         self.assertTrue(any("rebuilt ATOM/HETATM" in w for w in s.warnings))

#     def test_rebuild_uses_trailing_mapping_first_for_meeko(self):
#         line = self._make_atom_line(
#             serial=7,
#             atom_name="CA",
#             trailing_token="OA",
#         )
#         p = self._write("meeko_prefers_trailing.pdbqt", line)
#         s = PDBQTSanitizer(p, backend="meeko")

#         s.sanitize(rebuild=True, aggressive=False)

#         rebuilt = s.sanitized_lines[0]
#         self.assertEqual(rebuilt[76:78].strip(), "O")

#     def test_rebuild_prefers_atom_name_over_trailing_for_obabel(self):
#         line = self._make_atom_line(
#             serial=8,
#             atom_name="CL",
#             trailing_token="OA",
#         )
#         p = self._write("obabel_prefers_atom_name.pdbqt", line)
#         s = PDBQTSanitizer(p, backend="obabel")

#         s.sanitize(rebuild=True, aggressive=False)

#         rebuilt = s.sanitized_lines[0]
#         self.assertEqual(rebuilt[76:78].strip(), "Cl")

#     def test_rebuild_aggressive_normalizes_multiletter_elements(self):
#         line = "ATOM   9  CL1 LIG 1  1.000  2.000  3.000  1.00 10.00 CL1\n"
#         p = self._write("aggressive_cl.pdbqt", line)
#         s = PDBQTSanitizer(p, backend="meeko")

#         s.sanitize(rebuild=True, aggressive=True)

#         rebuilt = s.sanitized_lines[0]
#         self.assertEqual(rebuilt[76:78].strip(), "Cl")

#     def test_rebuild_falls_back_to_default_element_C(self):
#         line = "ATOM  10  XX  LIG 1  1.000  2.000  3.000  1.00 10.00 XXY\n"
#         p = self._write("fallback_default_C.pdbqt", line)
#         s = PDBQTSanitizer(p)

#         s.sanitize(rebuild=True, aggressive=False)

#         rebuilt = s.sanitized_lines[0]
#         self.assertEqual(rebuilt[76:78].strip(), "C")

#     # ------------------------------------------------------------------
#     # File helpers
#     # ------------------------------------------------------------------
#     def test_write_outputs_newline_terminated_file(self):
#         line = "ATOM   1  CG  LIG 1  1.000  2.000  3.000  1.00 10.00 CG0\n"
#         inp = self._write("write_test_in.pdbqt", line)
#         out = self.tmp / "nested" / "write_test_out.pdbqt"

#         s = PDBQTSanitizer(inp)
#         s.sanitize(rebuild=False)
#         written = s.write(out)

#         self.assertEqual(written, out)
#         self.assertTrue(out.exists())
#         text = out.read_text(encoding="utf-8")
#         self.assertTrue(text.endswith("\n"))

#     def test_sanitize_file_with_explicit_outpath_does_not_create_backup(self):
#         line = "ATOM   3  C   LIG 1  1.0 2.0 3.0 1.00 10.00 CG0\n"
#         inp = self._write("sanitize_file_in.pdbqt", line)
#         out = self.tmp / "sanitize_file_out.pdbqt"

#         result = PDBQTSanitizer.sanitize_file(
#             inp,
#             out_path=out,
#             rebuild=False,
#             aggressive=False,
#             backup=True,
#         )

#         self.assertEqual(result, out)
#         self.assertTrue(out.exists())
#         self.assertFalse((inp.with_suffix(inp.suffix + ".bak")).exists())

#     def test_sanitize_file_overwrite_creates_backup_when_requested(self):
#         line = "ATOM   4  C   LIG 1  1.0 2.0 3.0 1.00 10.00 CG0\n"
#         inp = self._write("overwrite_in.pdbqt", line)

#         result = PDBQTSanitizer.sanitize_file(
#             inp,
#             out_path=None,
#             rebuild=True,
#             aggressive=False,
#             backup=True,
#         )

#         self.assertEqual(result, inp)
#         bak = inp.with_suffix(inp.suffix + ".bak")
#         self.assertTrue(bak.exists())

#     def test_sanitize_file_overwrite_without_backup_does_not_create_bak(self):
#         line = "ATOM   5  C   LIG 1  1.0 2.0 3.0 1.00 10.00 CG0\n"
#         inp = self._write("overwrite_no_bak_in.pdbqt", line)

#         result = PDBQTSanitizer.sanitize_file(
#             inp,
#             out_path=None,
#             rebuild=True,
#             aggressive=False,
#             backup=False,
#         )

#         self.assertEqual(result, inp)
#         bak = inp.with_suffix(inp.suffix + ".bak")
#         self.assertFalse(bak.exists())

#     def test_sanitize_inplace_writes_backup_and_overwrites(self):
#         line = "ATOM   5  OH  LIG 1  1.0 2.0 3.0 1.00 10.00 OH\n"
#         inp = self._write("inplace.pdbqt", line)
#         s = PDBQTSanitizer(inp)

#         out = s.sanitize_inplace(rebuild=False, aggressive=True, backup=True)

#         self.assertEqual(out, inp)
#         self.assertTrue(inp.exists())
#         self.assertTrue((inp.with_suffix(inp.suffix + ".bak")).exists())
#         self.assertTrue(s._sanitized)
#         self.assertGreaterEqual(len(s.sanitized_lines), 1)

#     # ------------------------------------------------------------------
#     # Internal helper coverage
#     # ------------------------------------------------------------------
#     def test_canonicalize_element(self):
#         self.assertEqual(PDBQTSanitizer._canonicalize_element("cl"), "Cl")
#         self.assertEqual(PDBQTSanitizer._canonicalize_element("br"), "Br")
#         self.assertEqual(PDBQTSanitizer._canonicalize_element("c"), "C")
#         self.assertEqual(PDBQTSanitizer._canonicalize_element(""), "")

#     def test_strip_digits(self):
#         self.assertEqual(PDBQTSanitizer._strip_digits("CG0"), "CG")
#         self.assertEqual(PDBQTSanitizer._strip_digits("CL12"), "CL")
#         self.assertEqual(PDBQTSanitizer._strip_digits("1234"), "")

#     def test_is_valid_element_token(self):
#         s = PDBQTSanitizer()
#         self.assertTrue(s._is_valid_element_token("C"))
#         self.assertTrue(s._is_valid_element_token("Cl"))
#         self.assertFalse(s._is_valid_element_token("CL"))
#         self.assertFalse(s._is_valid_element_token("ZZ"))
#         self.assertFalse(s._is_valid_element_token(""))

#     def test_map_alias_edge_cases(self):
#         s = PDBQTSanitizer()

#         self.assertEqual(s._map_alias("CG0", atomname="CG"), "C")
#         self.assertEqual(s._map_alias("CL1", atomname="CL"), "Cl")
#         self.assertEqual(s._map_alias("1234"), "")
#         self.assertEqual(s._map_alias("", atomname=""), "")

#     def test_map_alias_uses_atom_name_fallback(self):
#         s = PDBQTSanitizer()
#         self.assertEqual(s._map_alias("???", atomname="CL1"), "Cl")
#         self.assertEqual(s._map_alias("???", atomname="C1"), "C")

#     def test_extract_trailing_token(self):
#         s = PDBQTSanitizer()
#         line = "ATOM   1  CG  LIG 1  1.0 2.0 3.0 1.00 10.00 CG0"
#         self.assertEqual(s._extract_trailing_token(line), "CG0")

#     def test_fixed_element_reads_columns_77_78(self):
#         s = PDBQTSanitizer()
#         line = self._make_atom_line(
#             atom_name="C",
#             fixed_element="Cl",
#         ).rstrip("\n")
#         self.assertEqual(s._fixed_element(line), "Cl")


# if __name__ == "__main__":
#     unittest.main(verbosity=2)
