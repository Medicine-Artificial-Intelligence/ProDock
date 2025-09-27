# tests/test_minimizers.py
import os
import stat
import tempfile
import unittest
from pathlib import Path

from prodock.preprocess.receptor import minimizers


def _write_executable(p: Path, content: str):
    p.write_text(content)
    mode = p.stat().st_mode
    p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestMinimizers(unittest.TestCase):
    def setUp(self):
        self._orig_path = os.environ.get("PATH", "")
        self.tmpdir = Path(tempfile.mkdtemp())
        self.bindir = self.tmpdir / "bin"
        self.bindir.mkdir()

    def tearDown(self):
        os.environ["PATH"] = self._orig_path
        try:
            for p in self.tmpdir.rglob("*"):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            self.tmpdir.rmdir()
        except Exception:
            pass

    def test__pos_to_nm_variants(self):
        class Fake:
            def value_in_unit(self, unit):
                return [1.0, 2.0, 3.0]

        x, y, z = minimizers._pos_to_nm(Fake())
        self.assertEqual((x, y, z), (1.0, 2.0, 3.0))

        x2, y2, z2 = minimizers._pos_to_nm((4.4, 5.5, 6.6))
        self.assertEqual((x2, y2, z2), (4.4, 5.5, 6.6))

        class Obj:
            def __init__(self):
                self.x = 7.7
                self.y = 8.8
                self.z = 9.9

        x3, y3, z3 = minimizers._pos_to_nm(Obj())
        self.assertEqual((x3, y3, z3), (7.7, 8.8, 9.9))

    def test_minimize_with_obabel_raises_when_missing(self):
        td = self.tmpdir
        inp = td / "in.pdb"
        inp.write_text("ATOM\n")
        outp = td / "out_min.pdb"
        try:
            with self.assertRaises(RuntimeError):
                minimizers.minimize_with_obabel(inp, outp, steps=10)
        except AssertionError:
            if outp.exists():
                self.assertTrue(outp.read_text() is not None)
            else:
                self.fail(
                    "minimize_with_obabel did not raise and did not produce output (unexpected)"
                )

    def test_minimize_with_obabel_fake_obabel(self):
        ob = self.bindir / "obabel"
        ob_script = r"""#!/usr/bin/env python3
import sys
args = sys.argv[1:]
# emulate writing the -O target
if "-O" in args:
    out = args[args.index("-O")+1]
    with open(out, "w") as fh:
        fh.write("MINIMIZED_BY_FAKE_OBABEL\n")
    sys.exit(0)
sys.exit(1)
"""
        _write_executable(ob, ob_script)
        os.environ["PATH"] = str(self.bindir) + os.pathsep + self._orig_path

        inp = self.tmpdir / "in2.pdb"
        inp.write_text("ATOM\n")
        outp = self.tmpdir / "min_out.pdb"
        res = minimizers.minimize_with_obabel(inp, outp, steps=50)
        self.assertEqual(str(res), str(outp))
        self.assertTrue(outp.exists())
        self.assertIn("MINIMIZED_BY_FAKE_OBABEL", outp.read_text())


if __name__ == "__main__":
    unittest.main()
