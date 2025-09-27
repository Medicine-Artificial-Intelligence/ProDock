import unittest
import tempfile
from pathlib import Path

from prodock.structure.pdbqt_sanitizer import PDBQTSanitizer


class TestPDBQTSanitizer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, name: str, content: str) -> Path:
        p = self.tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_unknown_top_level_tag_warns(self):
        txt = "FOOBAR some stuff\n"
        p = self._write("tag.pdbqt", txt)
        s = PDBQTSanitizer(p)
        warns = s.validate(strict=False)
        self.assertTrue(any("unknown top-level tag 'FOOBAR'" in w for w in warns))

    def test_trailing_token_alias_mapping_and_sanitize_minimal(self):
        # a simple ATOM-like tokenized line that uses trailing alias "CG0" -> C
        ln = "ATOM   1  CG  LIG 1  12.345  13.456  14.567  1.00 20.00 CG0\n"
        p = self._write("trail.pdbqt", ln)
        s = PDBQTSanitizer(p)
        warns = s.validate(strict=False)
        # mapping suggestion warning should be present
        self.assertTrue(
            any("trailing token 'CG0' is non-canonical" in w for w in warns)
        )

        # minimal sanitize (replacing trailing token -> mapped element)
        s.sanitize(rebuild=False, aggressive=False)
        self.assertTrue(s._sanitized)
        out = "\n".join(s.sanitized_lines) + "\n"
        # final token should be 'C' (mapped) instead of 'CG0'
        self.assertIn(" 1.00 20.00 C", out)

    def test_sanitize_rebuild_sets_fixed_element_and_warns_rebuilt(self):
        # similar line as above to force rebuild branch
        ln = "ATOM   2  CG  LIG 1  1.000  2.000  3.000  1.00 10.00 CG0\n"
        p = self._write("rebuild.pdbqt", ln)
        s = PDBQTSanitizer(p)
        s.validate(strict=False)
        s.sanitize(rebuild=True, aggressive=False)
        self.assertTrue(s._sanitized)
        self.assertGreaterEqual(len(s.sanitized_lines), 1)
        rebuilt_line = s.sanitized_lines[0]
        # rebuilt lines are padded and must have at least 78 chars (element at cols 77-78)
        self.assertGreaterEqual(len(rebuilt_line), 78)
        # element field (last two chars) should include 'C' (mapped from CG0)
        self.assertIn("C", rebuilt_line[76:78].strip())
        # a warning about rebuilt ATOM/HETATM should be recorded
        self.assertTrue(any("rebuilt ATOM/HETATM" in w for w in s.warnings))

    def test_fixed_column_invalid_element_warns_and_suggests(self):
        # craft a line with fixed-column element 'ZZ' (invalid) at cols 77-78
        base = "ATOM    10  C   LIG  1   1.00 2.00 3.00  1.00 20.00"
        # pad to column 76 and append invalid element 'ZZ'
        ln = base.ljust(76) + "ZZ" + "\n"
        p = self._write("fixed_invalid.pdbqt", ln)
        s = PDBQTSanitizer(p)
        warns = s.validate(strict=True)
        # warning should mention fixed-column element token is not valid
        self.assertTrue(
            any("fixed-column element token" in w and "not valid" in w for w in warns)
        )
        # and suggestion should be present in that message
        self.assertTrue(any("Suggested element" in w for w in warns))

    def test_short_atom_line_left_unchanged(self):
        # ATOM line with too few tokens
        ln = "ATOM 1 C\n"
        p = self._write("short.pdbqt", ln)
        s = PDBQTSanitizer(p)
        s.read(p)
        s.sanitize(rebuild=True)
        self.assertTrue(
            any("short ATOM/HETATM left unchanged" in w for w in s.warnings)
        )

    def test_cannot_parse_coordinates_left_unchanged(self):
        # tokens present but no numeric coords -> sanitize warns cannot parse coordinates
        ln = "ATOM 1 CA LIG 1 X Y Z\n"
        p = self._write("coords_bad.pdbqt", ln)
        s = PDBQTSanitizer(p)
        s.read(p)
        s.sanitize(rebuild=True)
        self.assertTrue(any("cannot parse coordinates" in w for w in s.warnings))
        # sanitized_lines should contain the original line unchanged
        self.assertIn("ATOM 1 CA LIG 1 X Y Z", s.sanitized_lines[0])

    def test_sanitize_file_creates_backup_and_writes_out(self):
        ln = "ATOM   3  C   LIG 1  1.0 2.0 3.0 1.00 10.00 CG0\n"
        p = self._write("file_in.pdbqt", ln)
        outp = self.tmp / "file_out.pdbqt"
        # call convenience method to sanitize and write to explicit out_path
        res = PDBQTSanitizer.sanitize_file(
            p, out_path=outp, rebuild=False, aggressive=False, backup=True
        )
        self.assertEqual(res, outp)
        self.assertTrue(outp.exists())
        # original file should be unchanged when out_path provided (no .bak created)
        self.assertFalse((p.with_suffix(p.suffix + ".bak")).exists())

    def test_sanitize_file_overwrite_creates_bak_when_no_outpath(self):
        ln = "ATOM   4  C   LIG 1  1.0 2.0 3.0 1.00 10.00 CG0\n"
        p = self._write("file_in2.pdbqt", ln)
        # call sanitize_file with out_path=None to overwrite input and create .bak
        res = PDBQTSanitizer.sanitize_file(
            p, out_path=None, rebuild=True, aggressive=False, backup=True
        )
        self.assertEqual(res, p)
        bak = p.with_suffix(p.suffix + ".bak")
        self.assertTrue(
            bak.exists(),
            "Backup file should be created when overwriting with backup=True",
        )
        # file content should have been changed (sanitized lines written)
        content = p.read_text(encoding="utf-8")
        self.assertIn("\n", content)

    def test_sanitize_inplace_writes_and_returns_path(self):
        ln = "ATOM   5  OH  LIG 1  1.0 2.0 3.0 1.00 10.00 OH\n"
        p = self._write("inplace.pdbqt", ln)
        s = PDBQTSanitizer(p)
        # sanitize_inplace should create a .bak and overwrite file
        out = s.sanitize_inplace(rebuild=False, aggressive=True, backup=True)
        self.assertEqual(out, p)
        self.assertTrue(p.exists())
        self.assertTrue((p.with_suffix(p.suffix + ".bak")).exists())
        # sanitized lines should be set
        self.assertTrue(s._sanitized)
        self.assertGreaterEqual(len(s.sanitized_lines), 1)

    def test_repr_and_help(self):
        s = PDBQTSanitizer()
        r = repr(s)
        self.assertIsInstance(r, str)
        h = s.help()
        self.assertIsInstance(h, str)
        self.assertIn("PDBQTSanitizer", h)

    def test_map_alias_edge_cases(self):
        s = PDBQTSanitizer()
        # direct known alias
        mapped = s._map_alias("CG0", atomname="CG")
        self.assertEqual(mapped, "C")
        # numeric-only token -> cannot map
        self.assertEqual(s._map_alias("1234"), "")
        # canonicalization of 'cl' -> 'Cl'
        self.assertEqual(s._canonicalize_element("cl"), "Cl")
        # strip digits
        self.assertEqual(s._strip_digits("CG0"), "CG")
        # suspicious atom name detection during validate
        ln = "ATOM   6  C@1 LIG 1  1.0 2.0 3.0 1.00 10.00 X\n"
        p = self._write("susp.pdbqt", ln)
        ss = PDBQTSanitizer(p)
        warns = ss.validate(strict=False)
        self.assertTrue(any("suspicious atom name" in w for w in warns))


if __name__ == "__main__":
    unittest.main(verbosity=2)
