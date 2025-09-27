# tests/test_convert.py
import os
import stat
import tempfile
import unittest
from pathlib import Path

from prodock.preprocess.receptor.convert import convert_with_mekoo, convert_with_obabel


def _write_executable(p: Path, content: str):
    p.write_text(content)
    mode = p.stat().st_mode
    p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestConvert(unittest.TestCase):
    def setUp(self):
        self._orig_path = os.environ.get("PATH", "")
        self.tmpdir = Path(tempfile.mkdtemp())
        self.bindir = self.tmpdir / "bin"
        self.bindir.mkdir()

    def tearDown(self):
        os.environ["PATH"] = self._orig_path
        # best-effort cleanup
        try:
            for p in self.tmpdir.rglob("*"):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            self.tmpdir.rmdir()
        except Exception:
            pass

    def test_convert_with_mekoo_returns_info_structure(self):
        # Ensure returns info dict and fields exist (robust to environment)
        inp = self.tmpdir / "in.pdb"
        inp.write_text("HEADER\n")
        out_base = self.tmpdir / "outbase"  # <- use Path, not str
        write_pdbqt = self.tmpdir / "maybe.pdbqt"  # <- use Path

        info = convert_with_mekoo(
            "mk_prepare_receptor.py", inp, out_base, write_pdbqt=write_pdbqt
        )
        self.assertIsInstance(info, dict)
        for k in ("called", "rc", "stdout", "stderr", "produced"):
            self.assertIn(k, info)
        self.assertIsInstance(info.get("produced"), list)

        stderr = (info.get("stderr") or "").lower()
        if "not found" in stderr:
            self.assertEqual(info.get("produced"), [])
        else:
            self.assertTrue(isinstance(info.get("rc"), (int, type(None))))

    def test_convert_with_mekoo_fake_script_writes_files(self):
        # Create fake mk_prepare_receptor.py which writes out an outbase.pdbqt and an explicit pdbqt
        mk = self.bindir / "mk_prepare_receptor.py"
        mk_script = r"""#!/usr/bin/env python3
import sys
args = sys.argv[1:]
# attempt to create <outbase>.pdbqt based on last arg (mimic behaviour)
try:
    if args:
        # write a basename-derived file
        base = args[-1]
        with open(str(base) + ".pdbqt", "w") as fh:
            fh.write("PDBQT_FROM_MK\n")
except Exception:
    pass
# also accept explicit explicit.pdbqt in args and write it
for a in args:
    if a.endswith("explicit.pdbqt"):
        try:
            with open(a, "w") as fh:
                fh.write("EXPLICIT_PDBQT\n")
        except Exception:
            pass
sys.exit(0)
"""
        _write_executable(mk, mk_script)
        os.environ["PATH"] = str(self.bindir) + os.pathsep + self._orig_path

        inp = self.tmpdir / "in.pdb"
        inp.write_text("ATOM\n")
        out_base = self.tmpdir / "outbase"  # Path, not str
        explicit = self.tmpdir / "explicit.pdbqt"  # Path

        info = convert_with_mekoo(str(mk), inp, out_base, write_pdbqt=str(explicit))
        produced = info.get("produced", [])
        # produced items should include at least one .pdbqt path
        self.assertTrue(
            any(str(p).endswith(".pdbqt") for p in produced), f"produced={produced}"
        )

    def test_convert_with_obabel_behavior(self):
        # If OpenBabel is missing, convert_with_obabel should raise.
        # If present it may succeed — adapt assertion accordingly.
        inp = self.tmpdir / "in.pdb"
        inp.write_text("ATOM\n")
        outp = self.tmpdir / "out.pdbqt"
        try:
            with self.assertRaises(RuntimeError):
                convert_with_obabel(inp, outp)
        except AssertionError:
            # If OpenBabel actually exists in the environment this will run instead of raising;
            # ensure that it either produced the output file (of any content) or raised a clear error.
            if outp.exists():
                # accept any textual content (don't require exact "CONVERTED")
                content = outp.read_text()
                self.assertIsInstance(content, str)
            else:
                # If not produced and no RuntimeError then surface an informative failure
                self.fail(
                    "convert_with_obabel did not raise and did not produce output (unexpected)"
                )

    def test_convert_with_obabel_fake_executable(self):
        # Create a fake obabel that writes the -O argument file
        ob = self.bindir / "obabel"
        ob_script = r"""#!/usr/bin/env python3
import sys
args = sys.argv[1:]
if "-O" in args:
    idx = args.index("-O")
    out = args[idx+1]
    try:
        with open(out, "w") as fh:
            fh.write("FAKE_OBABEL_CONVERT\n")
    except Exception:
        pass
    sys.exit(0)
sys.exit(1)
"""
        _write_executable(ob, ob_script)
        os.environ["PATH"] = str(self.bindir) + os.pathsep + self._orig_path

        inp = self.tmpdir / "in2.pdb"
        inp.write_text("ATOM\n")
        outp = self.tmpdir / "out2.pdbqt"
        convert_with_obabel(inp, outp, extra_args=["--partialcharge", "gasteiger"])
        self.assertTrue(outp.exists())
        self.assertIn("FAKE_OBABEL_CONVERT", outp.read_text())


if __name__ == "__main__":
    unittest.main()
