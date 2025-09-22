# test_pdb_query.py
import gzip
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import List

import prodock.io.pdb_query as pdb_query_mod
from prodock.io.pdb_query import PDBQuery

# -----------------------------
# Top-level option
# -----------------------------
usepymol = False


# -----------------------------
# Minimal Model used by FakeCmd
# -----------------------------
class _Model:
    def __init__(self, n=0):
        self.atom = [object() for _ in range(n)]


# -----------------------------
# Fake (hermetic) PyMOL cmd
# -----------------------------
class FakeCmd:
    """
    Minimal local stub of PyMOL's `cmd` used by PDBQuery.

    - fetch writes .pdb or .pdb.gz into requested path
    - load/save record calls and write tiny files
    - select/count_atoms/get_model/remove/delete implemented for assertions
    """

    def __init__(self, write_gz: bool = False, ligand_atoms: int = 12):
        self.write_gz = write_gz
        self._ligand_atoms = ligand_atoms
        self.selections = {}
        self.saved_paths: List[str] = []
        self.removed: List[str] = []
        self.loaded: List[str] = []
        self.deleted: List[str] = []

    def fetch(self, pdb_id, path="", type="pdb", async_=0):
        outdir = Path(path)
        outdir.mkdir(parents=True, exist_ok=True)
        if self.write_gz:
            outpath = outdir / f"{pdb_id}.pdb.gz"
            with gzip.open(outpath, "wb") as fh:
                fh.write(b"FAKE PDB CONTENT\nATOM ...\n")
        else:
            outpath = outdir / f"{pdb_id}.pdb"
            outpath.write_text("FAKE PDB CONTENT\nATOM ...\n", encoding="utf-8")

    def load(self, filename, *args, **kwargs):
        self.loaded.append(str(filename))

    def save(self, filename, selection="all"):
        p = Path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"SAVED {selection}\n", encoding="utf-8")
        self.saved_paths.append(str(p))

    def select(self, name, selection):
        self.selections[name] = selection

    def count_atoms(self, selection_name):
        if selection_name == "ligand":
            return self._ligand_atoms
        return 0

    def get_model(self, selection_name):
        if selection_name == "ligand":
            return _Model(n=self._ligand_atoms)
        return _Model(n=0)

    def remove(self, selection):
        self.removed.append(selection)

    def delete(self, name):
        self.deleted.append(name)


# -----------------------------
# Test Suite
# -----------------------------
class TestPDBQuery(unittest.TestCase):
    def setUp(self):
        # preserve original module cmd and PATH to restore later
        self._orig_cmd = getattr(pdb_query_mod, "cmd", None)
        self._orig_path = os.environ.get("PATH", "")
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.pdb_id = "5N2F"
        self.ligand_code = "8HW"
        self.chains = ["A", "B"]

        # Decide which cmd to use
        if usepymol:
            try:
                from pymol import cmd as real_cmd  # type: ignore

                pdb_query_mod.cmd = real_cmd
                self.using_real_pymol = True
            except Exception:
                # fallback to FakeCmd if real pymol import fails
                pdb_query_mod.cmd = FakeCmd()
                self.using_real_pymol = False
        else:
            pdb_query_mod.cmd = FakeCmd()
            self.using_real_pymol = False

    def tearDown(self):
        # restore original cmd and PATH
        pdb_query_mod.cmd = self._orig_cmd
        os.environ["PATH"] = self._orig_path
        self.tmp.cleanup()

    def _make_query(self) -> PDBQuery:
        out_base = str(self.tmpdir / "out")
        return PDBQuery(
            pdb_id=self.pdb_id,
            output_dir=out_base,
            chains=self.chains,
            ligand_code=self.ligand_code,
            protein_name=self.pdb_id,
        )

    def test_validate_creates_directories_and_paths(self):
        pq = self._make_query()
        pq.validate()

        # directories should exist
        self.assertTrue((self.tmpdir / "out" / "fetched_protein").exists())
        self.assertTrue((self.tmpdir / "out" / "filtered_protein").exists())
        self.assertTrue((self.tmpdir / "out" / "reference_ligand").exists())
        self.assertTrue((self.tmpdir / "out" / "cocrystal").exists())

        # derived paths set
        self.assertIn(self.pdb_id, pq.pdb_path or "")
        self.assertIn(self.pdb_id, pq.filtered_protein_path or "")
        self.assertIn(self.ligand_code, pq.reference_ligand_path or "")
        self.assertIn(self.pdb_id, pq.cocrystal_ligand_path or "")

    def test_fetch_prefers_and_decompresses_gz(self):
        # ensure module cmd is a FakeCmd that can write gz (skip for real pymol)
        if self.using_real_pymol:
            self.skipTest("Skipping gz-fetch test when running against real PyMOL")
        pdb_query_mod.cmd = FakeCmd(write_gz=True)
        pq = self._make_query()
        pq.validate().fetch()

        self.assertIsNotNone(pq.pdb_path)
        self.assertTrue(Path(pq.pdb_path).exists())
        self.assertTrue(str(pq.pdb_path).endswith(".pdb"))
        self.assertIn(str(Path(pq.pdb_path)), pdb_query_mod.cmd.loaded)

    def test_join_selection_static(self):
        s = PDBQuery._join_selection("chain", ["A", "B"])
        self.assertEqual(s, "chain A or chain B")

    def test_filter_chains_builds_selection(self):
        if self.using_real_pymol:
            self.skipTest(
                "Skipping filter_chains inspect test when running against real PyMOL"
            )
        pdb_query_mod.cmd = FakeCmd()
        pq = self._make_query()
        pq.validate().fetch().filter_chains()
        self.assertEqual(
            pdb_query_mod.cmd.selections.get("kept_chains"), "chain A or chain B"
        )

    def test_extract_ligand_without_obabel_falls_back_to_copy(self):
        # Ensure no obabel on PATH by setting PATH to empty (keeps behavior deterministic)
        os.environ["PATH"] = ""
        if self.using_real_pymol:
            self.skipTest(
                "Skipping ligand-extract fallback test when running against real PyMOL"
            )
        pdb_query_mod.cmd = FakeCmd(ligand_atoms=10)
        pq = self._make_query()
        pq.validate().fetch().extract_ligand()

        ref = Path(pq.reference_ligand_path)
        coco = Path(pq.cocrystal_ligand_path)
        self.assertTrue(
            ref.exists(), "reference ligand file should exist (fallback copy)"
        )
        self.assertTrue(coco.exists(), "cocrystal file should exist (fallback copy)")
        self.assertIn(f"resn {self.ligand_code}", pdb_query_mod.cmd.removed)

    def _write_fake_obabel_script(self, tmpdir: Path) -> str:
        """
        Create a small executable script named 'obabel' in tmpdir that writes the file
        provided after '-O' and exits 0. Returns absolute path to the script.
        """
        obabel_path = tmpdir / "obabel"
        script = (
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "argv = sys.argv[1:]\n"
            "if '-O' in argv:\n"
            "    idx = argv.index('-O')\n"
            "    if idx + 1 < len(argv):\n"
            "        out = argv[idx + 1]\n"
            "        with open(out, 'w', encoding='utf-8') as fh:\n"
            "            fh.write('OBABEL_WRITTEN\\n')\n"
            "sys.exit(0)\n"
        )
        obabel_path.write_text(script, encoding="utf-8")
        obabel_path.chmod(
            obabel_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        return str(obabel_path)

    def test_extract_ligand_with_obabel_conversion(self):
        # create a tmp bin dir with a fake obabel executable and prepend to PATH
        if self.using_real_pymol:
            self.skipTest(
                "Skipping obabel conversion test when running against real PyMOL"
            )
        bin_dir = self.tmpdir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        self._write_fake_obabel_script(bin_dir)
        os.environ["PATH"] = f"{str(bin_dir)}{os.pathsep}{self._orig_path}"

        pdb_query_mod.cmd = FakeCmd(ligand_atoms=8)
        pq = self._make_query()
        pq.validate().fetch().extract_ligand()

        ref = Path(pq.reference_ligand_path)
        coco = Path(pq.cocrystal_ligand_path)
        self.assertTrue(ref.exists(), "reference SDF should exist (obabel created)")
        self.assertTrue(coco.exists(), "cocrystal SDF should exist (obabel created)")
        self.assertIn("OBABEL_WRITTEN", ref.read_text())

    def test_clean_solvents_and_cofactors_selection(self):
        if self.using_real_pymol:
            self.skipTest("Skipping solvents/cofactors selection test with real PyMOL")
        pdb_query_mod.cmd = FakeCmd()
        pq = self._make_query()
        pq.cofactors = ["HEM", "NAD"]
        pq.validate().fetch().clean_solvents_and_cofactors()
        sel = pdb_query_mod.cmd.selections.get("removed_solvent", "")
        self.assertIn("solvents", sel)
        self.assertIn("not cofactors", sel)

    def test_save_filtered_protein_writes_and_clears(self):
        if self.using_real_pymol:
            self.skipTest(
                "Skipping save filtered protein inspect test when running against real PyMOL"
            )
        pdb_query_mod.cmd = FakeCmd()
        pq = self._make_query()
        pq.validate().fetch().save_filtered_protein()
        fpath = Path(pq.filtered_protein_path)
        self.assertTrue(fpath.exists(), "Filtered protein file should exist")
        self.assertIn("all", pdb_query_mod.cmd.deleted)

    def test_run_all_end_to_end_without_obabel(self):
        # end-to-end pipeline with fallback copies (no obabel)
        os.environ["PATH"] = ""
        if self.using_real_pymol:
            self.skipTest("Skipping run_all end-to-end with real PyMOL")
        pdb_query_mod.cmd = FakeCmd()
        pq = self._make_query()
        pq.run_all()
        self.assertTrue(Path(pq.filtered_protein_path).exists())
        self.assertTrue(Path(pq.cocrystal_ligand_path).exists())
        self.assertTrue(Path(pq.reference_ligand_path).exists())

    def test_list_reference_dir(self):
        if self.using_real_pymol:
            self.skipTest("Skipping list_reference_dir with real PyMOL")
        os.environ["PATH"] = ""
        pdb_query_mod.cmd = FakeCmd()
        pq = self._make_query()
        pq.validate().fetch().extract_ligand()
        files = pq.list_reference_dir()
        self.assertIn(f"{self.ligand_code}.sdf", files["reference"])
        self.assertIn(f"{self.pdb_id}.sdf", files["cocrystal"])

    def test_properties_return_strings(self):
        pdb_query_mod.cmd = FakeCmd()
        pq = self._make_query()
        pq.validate()
        self.assertIsInstance(pq.pdb_path, str)
        self.assertIsInstance(pq.filtered_protein_path, str)
        self.assertIsInstance(pq.reference_ligand_path, str)
        self.assertIsInstance(pq.cocrystal_ligand_path, str)

    def test_process_batch_minimal(self):
        # two small rows; no obabel on PATH -> fallback copy
        os.environ["PATH"] = ""
        if self.using_real_pymol:
            self.skipTest("Skipping batch test with real PyMOL")
        pdb_query_mod.cmd = FakeCmd()
        items = [
            {"pdb_id": "5N2F", "ligand_code": "8HW", "chains": ["A"]},
            {"pdb_id": "1ABC", "ligand_code": "LIG", "chains": []},
        ]
        outdir = str(self.tmpdir / "batch_out")
        results = PDBQuery.process_batch(items, output_dir=outdir)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r["success"], msg=f"Row failed: {r}")
            if r["filtered_protein"]:
                self.assertTrue(Path(r["filtered_protein"]).exists())
            if r["reference"]:
                self.assertTrue(Path(r["reference"]).exists())
            if r["cocrystal"]:
                self.assertTrue(Path(r["cocrystal"]).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
