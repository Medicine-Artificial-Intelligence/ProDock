import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path


class FakeCmd:
    def __init__(self):
        self.saved = []
        self.removed = []
        self.selected = []
        self.loaded = []
        self.deleted = []
        self._atoms = {"ligand": 10}
        self.raise_count_atoms = False
        self.raise_save_for_tmp_pdb = False
        self.raise_save_for_filtered = False

    def load(self, path):
        self.loaded.append(str(path))
        return None

    def select(self, name, sel):
        self.selected.append((name, sel))
        return None

    def count_atoms(self, sel):
        if self.raise_count_atoms:
            raise RuntimeError("count_atoms failed")
        return self._atoms.get(sel, 0)

    def save(self, path, sel):
        path = str(path)

        if self.raise_save_for_tmp_pdb and path.endswith("_tmp.pdb"):
            raise RuntimeError("tmp ligand save failed")

        if self.raise_save_for_filtered and sel == "all":
            raise RuntimeError("filtered protein save failed")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"SAVED:{sel}")
        self.saved.append((path, sel))

    def remove(self, sel):
        self.removed.append(sel)

    def delete(self, name):
        self.deleted.append(name)
        return None

    def get_model(self, sel):
        class M:
            atom = [1, 2, 3]

        return M()


# ----------------------------------------------------------------------
# Install fake pymol BEFORE importing target modules
# ----------------------------------------------------------------------
fake_pymol_mod = types.ModuleType("pymol")
fake_pymol_mod.cmd = FakeCmd()
sys.modules["pymol"] = fake_pymol_mod

pdb_engine_mod = importlib.import_module("prodock.structure.pdb_engine")
PDBEngine = pdb_engine_mod.PDBEngine

fetch_mod = importlib.import_module("prodock.structure.fetch")
convert_mod = importlib.import_module("prodock.structure.conversion")
selection_mod = importlib.import_module("prodock.structure.selection")


class TestPDBEngine(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.base_out = Path(self.td) / "out" / "5N2F"

        # Fresh fake cmd per test to avoid state leakage
        fake_pymol_mod.cmd = FakeCmd()
        self.fake_cmd = fake_pymol_mod.cmd
        pdb_engine_mod.cmd = self.fake_cmd

        self._orig_fetch = getattr(fetch_mod, "fetch_pdb_to_dir", None)
        self._orig_convert_obabel = getattr(convert_mod, "convert_with_obabel", None)
        self._orig_pdb_to_sdf = getattr(convert_mod, "pdb_to_sdf", None)

        def fake_fetch(pdb_id, fetch_dir):
            fn = os.path.join(str(fetch_dir), f"{pdb_id}.pdb")
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write("HEADER")
            return fn

        def fake_convert_obabel(src, dst, extra_args=None, **kwargs):
            dst = str(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write("SDF")
            return True

        def fake_pdb_to_sdf(src, dst, backend="rdkit", extra_args=None):
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("SDF", encoding="utf-8")
            return dst

        fetch_mod.fetch_pdb_to_dir = fake_fetch
        convert_mod.convert_with_obabel = fake_convert_obabel
        convert_mod.pdb_to_sdf = fake_pdb_to_sdf

    def tearDown(self):
        if self._orig_fetch is None:
            try:
                delattr(fetch_mod, "fetch_pdb_to_dir")
            except Exception:
                pass
        else:
            fetch_mod.fetch_pdb_to_dir = self._orig_fetch

        if self._orig_convert_obabel is None:
            try:
                delattr(convert_mod, "convert_with_obabel")
            except Exception:
                pass
        else:
            convert_mod.convert_with_obabel = self._orig_convert_obabel

        if self._orig_pdb_to_sdf is None:
            try:
                delattr(convert_mod, "pdb_to_sdf")
            except Exception:
                pass
        else:
            convert_mod.pdb_to_sdf = self._orig_pdb_to_sdf

        shutil.rmtree(self.td)

    def _engine(self, **kwargs):
        params = dict(
            pdb_id="5N2F",
            base_out=self.base_out,
            chains=["A"],
            ligand_code="HEM",
            cofactors=["HEM"],
            auto_create_dirs=True,
        )
        params.update(kwargs)
        return PDBEngine(**params)

    # ------------------------------------------------------------------
    # Initialization / validate
    # ------------------------------------------------------------------
    def test_init_sets_expected_attributes(self):
        eng = self._engine()

        self.assertEqual(eng.pdb_id, "5N2F")
        self.assertEqual(eng.base_out, self.base_out)
        self.assertEqual(eng.chains, ["A"])
        self.assertEqual(eng.ligand_code, "HEM")
        self.assertEqual(eng.cofactors, ["HEM"])
        self.assertTrue(eng.auto_create_dirs)

        self.assertEqual(eng.fetch_dir, self.base_out / "fetched_protein")
        self.assertEqual(eng.filtered_dir, self.base_out / "filtered_protein")
        self.assertEqual(eng.ref_dir, self.base_out / "reference_ligand")
        self.assertEqual(eng.cocrystal_dir, self.base_out / "cocrystal")

        self.assertIsNone(eng.pdb_path)
        self.assertIsNone(eng.ref_path)
        self.assertIsNone(eng.cocrystal_path)
        self.assertIsNone(eng.filtered_path)

    def test_validate_creates_dirs_and_sets_paths(self):
        eng = self._engine()

        returned = eng.validate()

        self.assertIs(returned, eng)
        self.assertTrue(eng.fetch_dir.exists())
        self.assertTrue(eng.filtered_dir.exists())
        self.assertTrue(eng.ref_dir.exists())
        self.assertTrue(eng.cocrystal_dir.exists())

        self.assertEqual(eng.pdb_path, eng.fetch_dir / "5N2F.pdb")
        self.assertEqual(eng.filtered_path, eng.filtered_dir / "5N2F.pdb")
        self.assertEqual(eng.ref_path, eng.ref_dir / "HEM.sdf")
        self.assertEqual(eng.cocrystal_path, eng.cocrystal_dir / "5N2F.sdf")

    def test_validate_without_ligand_code_sets_ref_path_none(self):
        eng = self._engine(ligand_code="")

        eng.validate()

        self.assertIsNone(eng.ref_path)
        self.assertEqual(eng.cocrystal_path, eng.cocrystal_dir / "5N2F.sdf")

    def test_ensure_dir_respects_auto_create_dirs_false(self):
        eng = self._engine(auto_create_dirs=False)

        eng.validate()

        self.assertFalse(eng.fetch_dir.exists())
        self.assertFalse(eng.filtered_dir.exists())
        self.assertFalse(eng.ref_dir.exists())
        self.assertFalse(eng.cocrystal_dir.exists())

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    def test_fetch_downloads_and_loads_structure(self):
        eng = self._engine().validate()

        returned = eng.fetch()

        self.assertIs(returned, eng)
        self.assertIsNotNone(eng.pdb_path)
        self.assertTrue(Path(eng.pdb_path).exists())
        self.assertIn(str(eng.pdb_path), self.fake_cmd.loaded)

    # ------------------------------------------------------------------
    # Chain filtering
    # ------------------------------------------------------------------
    def test_filter_chains_keeps_requested_chain(self):
        eng = self._engine().validate()

        returned = eng.filter_chains()

        self.assertIs(returned, eng)
        self.assertIn(
            ("kept_chains", selection_mod.chain_selection(["A"])),
            self.fake_cmd.selected,
        )
        self.assertIn(
            ("removed_complex", "all and not kept_chains"),
            self.fake_cmd.selected,
        )
        self.assertIn("removed_complex", self.fake_cmd.removed)

    def test_filter_chains_no_chains_is_noop(self):
        eng = self._engine(chains=[]).validate()

        returned = eng.filter_chains()

        self.assertIs(returned, eng)
        self.assertEqual(self.fake_cmd.selected, [])
        self.assertEqual(self.fake_cmd.removed, [])

    # ------------------------------------------------------------------
    # Internal ligand helpers
    # ------------------------------------------------------------------
    def test_ligand_selection_with_chain(self):
        eng = self._engine()
        self.assertEqual(eng._ligand_selection("A"), "resn HEM and chain A")

    def test_ligand_selection_without_chain(self):
        eng = self._engine(chains=[])
        self.assertEqual(eng._ligand_selection(None), "resn HEM")

    def test_count_selected_atoms_uses_count_atoms(self):
        eng = self._engine()
        count = eng._count_selected_atoms("ligand")
        self.assertEqual(count, 10)

    def test_count_selected_atoms_falls_back_to_get_model(self):
        eng = self._engine()
        self.fake_cmd.raise_count_atoms = True
        count = eng._count_selected_atoms("ligand")
        self.assertEqual(count, 3)

    def test_tmp_ligand_pdb_path(self):
        eng = self._engine().validate()
        self.assertEqual(eng._tmp_ligand_pdb_path(), eng.ref_dir / "HEM_tmp.pdb")

    def test_save_selected_ligand_to_tmp_success(self):
        eng = self._engine().validate()
        tmp_pdb = eng._tmp_ligand_pdb_path()

        ok = eng._save_selected_ligand_to_tmp(tmp_pdb)

        self.assertTrue(ok)
        self.assertTrue(tmp_pdb.exists())

    def test_save_selected_ligand_to_tmp_failure(self):
        eng = self._engine().validate()
        self.fake_cmd.raise_save_for_tmp_pdb = True
        tmp_pdb = eng._tmp_ligand_pdb_path()

        ok = eng._save_selected_ligand_to_tmp(tmp_pdb)

        self.assertFalse(ok)
        self.assertFalse(tmp_pdb.exists())

    def test_convert_reference_ligand_success(self):
        eng = self._engine().validate()
        tmp_pdb = eng._tmp_ligand_pdb_path()
        tmp_pdb.parent.mkdir(parents=True, exist_ok=True)
        tmp_pdb.write_text("TMP", encoding="utf-8")

        ok = eng._convert_reference_ligand(tmp_pdb)

        self.assertTrue(ok)
        self.assertTrue(eng.ref_path.exists())

    def test_convert_reference_ligand_failure(self):
        def fake_pdb_to_sdf_fail(src, dst, backend="rdkit", extra_args=None):
            raise ValueError("RDKit parse failed")

        convert_mod.pdb_to_sdf = fake_pdb_to_sdf_fail

        eng = self._engine().validate()
        tmp_pdb = eng._tmp_ligand_pdb_path()
        tmp_pdb.parent.mkdir(parents=True, exist_ok=True)
        tmp_pdb.write_text("TMP", encoding="utf-8")

        ok = eng._convert_reference_ligand(tmp_pdb)

        self.assertFalse(ok)
        self.assertFalse(eng.ref_path.exists())

    def test_convert_cocrystal_ligand_success(self):
        eng = self._engine().validate()
        eng.ref_path.write_text("REF", encoding="utf-8")

        ok = eng._convert_cocrystal_ligand()

        self.assertTrue(ok)
        self.assertTrue(eng.cocrystal_path.exists())

    def test_convert_cocrystal_ligand_failure(self):
        def fake_convert_fail(src, dst, extra_args=None, **kwargs):
            raise RuntimeError("Open Babel failed")

        convert_mod.convert_with_obabel = fake_convert_fail

        eng = self._engine().validate()
        eng.ref_path.write_text("REF", encoding="utf-8")

        ok = eng._convert_cocrystal_ligand()

        self.assertFalse(ok)
        self.assertFalse(eng.cocrystal_path.exists())

    def test_cleanup_partial_ligand_outputs(self):
        eng = self._engine().validate()
        eng.ref_path.write_text("REF", encoding="utf-8")
        eng.cocrystal_path.write_text("COC", encoding="utf-8")

        eng._cleanup_partial_ligand_outputs()

        self.assertFalse(eng.ref_path.exists())
        self.assertFalse(eng.cocrystal_path.exists())

    # ------------------------------------------------------------------
    # Ligand extraction
    # ------------------------------------------------------------------
    def test_extract_ligand_creates_ref_and_cocrystal(self):
        eng = self._engine().validate()

        returned = eng.extract_ligand()

        self.assertIs(returned, eng)
        self.assertIsNotNone(eng.ref_path)
        self.assertIsNotNone(eng.cocrystal_path)
        self.assertTrue(eng.ref_path.exists())
        self.assertTrue(eng.cocrystal_path.exists())

        self.assertIn(("ligand", "resn HEM and chain A"), self.fake_cmd.selected)
        self.assertIn("resn HEM", self.fake_cmd.removed)

    def test_extract_ligand_skips_when_no_ligand_code(self):
        eng = self._engine(ligand_code="").validate()

        returned = eng.extract_ligand()

        self.assertIs(returned, eng)
        self.assertEqual(self.fake_cmd.selected, [])
        self.assertEqual(self.fake_cmd.removed, [])

    def test_extract_ligand_raises_when_no_atoms_found(self):
        self.fake_cmd._atoms["ligand"] = 0
        eng = self._engine().validate()

        with self.assertRaises(RuntimeError) as ctx:
            eng.extract_ligand()

        self.assertIn("Failed to save reference ligand", str(ctx.exception))

    def test_extract_ligand_falls_back_to_get_model_when_count_atoms_fails(self):
        self.fake_cmd.raise_count_atoms = True
        eng = self._engine().validate()

        returned = eng.extract_ligand()

        self.assertIs(returned, eng)
        self.assertTrue(eng.ref_path.exists())
        self.assertTrue(eng.cocrystal_path.exists())

    def test_extract_ligand_raises_when_tmp_save_fails(self):
        self.fake_cmd.raise_save_for_tmp_pdb = True
        eng = self._engine().validate()

        with self.assertRaises(RuntimeError) as ctx:
            eng.extract_ligand()

        self.assertIn("Failed to save reference ligand", str(ctx.exception))

    def test_extract_ligand_raises_when_reference_conversion_fails(self):
        def fake_pdb_to_sdf_fail(src, dst, backend="rdkit", extra_args=None):
            raise ValueError("RDKit parse failed")

        convert_mod.pdb_to_sdf = fake_pdb_to_sdf_fail

        eng = self._engine().validate()

        with self.assertRaises(RuntimeError) as ctx:
            eng.extract_ligand()

        self.assertIn("Failed to save reference ligand", str(ctx.exception))

    def test_extract_ligand_removes_partial_outputs_when_cocrystal_conversion_fails(
        self,
    ):
        def fake_pdb_to_sdf_ok(src, dst, backend="rdkit", extra_args=None):
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("REF", encoding="utf-8")
            return dst

        def fake_convert_cocrystal_fail(src, dst, extra_args=None, **kwargs):
            raise RuntimeError("Open Babel failed")

        convert_mod.pdb_to_sdf = fake_pdb_to_sdf_ok
        convert_mod.convert_with_obabel = fake_convert_cocrystal_fail

        eng = self._engine().validate()

        with self.assertRaises(RuntimeError):
            eng.extract_ligand()

        self.assertFalse(eng.ref_path.exists())
        self.assertFalse(eng.cocrystal_path.exists())

    def test_extract_ligand_tries_multiple_chains_until_one_matches(self):
        selections_seen = []

        def custom_select(name, sel):
            selections_seen.append((name, sel))
            if name == "ligand" and sel == "resn HEM and chain B":
                self.fake_cmd._atoms["ligand"] = 10
            elif name == "ligand":
                self.fake_cmd._atoms["ligand"] = 0
            else:
                self.fake_cmd.selected.append((name, sel))
            return None

        self.fake_cmd.select = custom_select

        eng = self._engine(chains=["A", "B"]).validate()
        returned = eng.extract_ligand()

        self.assertIs(returned, eng)
        self.assertIn(("ligand", "resn HEM and chain A"), selections_seen)
        self.assertIn(("ligand", "resn HEM and chain B"), selections_seen)
        self.assertTrue(eng.ref_path.exists())

    # ------------------------------------------------------------------
    # Solvent / cofactor cleanup
    # ------------------------------------------------------------------
    def test_clean_solvents_and_cofactors_preserves_cofactors(self):
        eng = self._engine().validate()

        returned = eng.clean_solvents_and_cofactors()

        self.assertIs(returned, eng)

        selected_names = [x[0] for x in self.fake_cmd.selected]
        self.assertIn("solvents", selected_names)
        self.assertIn("cofactors", selected_names)
        self.assertIn(
            ("removed_solvent", "solvents and not cofactors"),
            self.fake_cmd.selected,
        )
        self.assertIn("removed_solvent", self.fake_cmd.removed)

    def test_clean_solvents_without_cofactors_removes_all_solvents(self):
        eng = self._engine(cofactors=[]).validate()

        returned = eng.clean_solvents_and_cofactors()

        self.assertIs(returned, eng)
        self.assertIn(("removed_solvent", "solvents"), self.fake_cmd.selected)
        self.assertIn("removed_solvent", self.fake_cmd.removed)

    # ------------------------------------------------------------------
    # Save filtered protein
    # ------------------------------------------------------------------
    def test_save_filtered_protein_writes_file_and_deletes_session(self):
        eng = self._engine().validate()

        returned = eng.save_filtered_protein()

        self.assertIs(returned, eng)
        self.assertIsNotNone(eng.filtered_path)
        self.assertTrue(eng.filtered_path.exists())
        self.assertIn((str(eng.filtered_path), "all"), self.fake_cmd.saved)
        self.assertIn("all", self.fake_cmd.deleted)

    def test_save_filtered_protein_still_deletes_all_if_save_fails(self):
        self.fake_cmd.raise_save_for_filtered = True
        eng = self._engine().validate()

        returned = eng.save_filtered_protein()

        self.assertIs(returned, eng)
        self.assertIn("all", self.fake_cmd.deleted)

    # ------------------------------------------------------------------
    # Full workflow
    # ------------------------------------------------------------------
    def test_run_all_executes_full_pipeline(self):
        eng = self._engine()

        returned = eng.run_all()

        self.assertIs(returned, eng)
        self.assertTrue(eng.fetch_dir.exists())
        self.assertTrue(eng.filtered_dir.exists())
        self.assertTrue(eng.ref_dir.exists())
        self.assertTrue(eng.cocrystal_dir.exists())

        self.assertTrue(eng.pdb_path.exists())
        self.assertTrue(eng.ref_path.exists())
        self.assertTrue(eng.cocrystal_path.exists())
        self.assertTrue(eng.filtered_path.exists())

        self.assertTrue(any("5N2F.pdb" in p for p in self.fake_cmd.loaded))
        self.assertIn("resn HEM", self.fake_cmd.removed)
        self.assertIn("removed_solvent", self.fake_cmd.removed)
        self.assertIn("all", self.fake_cmd.deleted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
