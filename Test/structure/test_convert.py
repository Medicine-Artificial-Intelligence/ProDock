import unittest
import tempfile
import shutil
import os
from pathlib import Path
import stat

from prodock.structure.convert import convert_with_obabel, copy_fallback


class TestConvert(unittest.TestCase):
    def test_copy_fallback(self):
        td = Path(tempfile.mkdtemp())
        try:
            src = td / "in.pdb"
            dst = td / "out.sdf"
            src.write_text("MOL")
            ok = copy_fallback(src, dst)
            self.assertTrue(ok)
            self.assertTrue(dst.exists())
            self.assertEqual(dst.read_text(), "MOL")
        finally:
            shutil.rmtree(str(td))

    def test_convert_with_real_fake_obabel(self):
        """
        Create a small fake 'obabel' executable in a temp bin/ directory,
        put that directory at the front of PATH, and call convert_with_obabel()
        without mocking. The fake obabel will write the -O filename into cwd
        (which convert_with_obabel sets to src.parent) so the test can verify
        the produced file.
        """
        td = Path(tempfile.mkdtemp())
        old_path = os.environ.get("PATH", "")
        try:
            # prepare src and dst inside a subdir
            src_dir = td / "srcdir"
            src_dir.mkdir()
            src = src_dir / "lig_tmp.pdb"
            src.write_text("MOL")
            dst = src_dir / "LIG.sdf"

            # create a tiny fake obabel as a python script that writes the -O filename in cwd
            bin_dir = td / "bin"
            bin_dir.mkdir()
            obabel_path = bin_dir / "obabel"

            obabel_code = r"""#!/usr/bin/env python3
import sys
import os
# find "-O" and take next token as output filename
out = None
argv = sys.argv[1:]
for i, a in enumerate(argv):
    if a == "-O" and i+1 < len(argv):
        out = argv[i+1]
        break
# if not found, attempt last token as fallback
if out is None and len(argv)>0:
    out = argv[-1]
if out:
    try:
        with open(os.path.join(os.getcwd(), out), "w") as f:
            f.write("MOL_SDF")
    except Exception as e:
        print("ERROR writing output:", e, file=sys.stderr)
# return success
sys.exit(0)
"""
            obabel_path.write_text(obabel_code)
            # make executable
            st = obabel_path.stat().st_mode
            obabel_path.chmod(st | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            # put our fake bin at front of PATH
            os.environ["PATH"] = str(bin_dir) + os.pathsep + old_path

            # call the real function (no mocking)
            ok = convert_with_obabel(src, dst, extra_args=("-h",))

            # convert_with_obabel should report success (or detect fallback file)
            self.assertTrue(
                ok,
                "convert_with_obabel should return True when fake obabel wrote output",
            )

            # check that the expected file exists and content matches
            self.assertTrue(dst.exists(), f"Expected output file {dst} to exist")
            self.assertEqual(dst.read_text(), "MOL_SDF")
        finally:
            # restore PATH and cleanup
            os.environ["PATH"] = old_path
            shutil.rmtree(str(td))


if __name__ == "__main__":
    unittest.main()
