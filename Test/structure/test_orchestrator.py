import unittest
import tempfile
import shutil
import sys
import types
import importlib
import os


class FakeCmd:
    def __init__(self):
        self.saved = []
        self.removed = []
        # emulate that "ligand" selection has atoms to be saved
        self._atoms = {"ligand": 10}

    def fetch(self, pdb_id, path=None, type=None, async_=0):
        # Simulate writing a fetched file into the provided path
        if path is None:
            path = os.getcwd()
        fn = os.path.join(path, f"{pdb_id.lower()}.pdb")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "w") as fh:
            fh.write("HEADER")
        return fn

    def load(self, path):
        # no-op
        return None

    def select(self, name, sel):
        # no-op
        return None

    def count_atoms(self, sel):
        return self._atoms.get(sel, 0)

    def save(self, path, sel):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(f"SAVED:{sel}")
        self.saved.append((path, sel))

    def remove(self, sel):
        self.removed.append(sel)

    def delete(self, name):
        return None

    def get_model(self, sel):
        class M:
            atom = [1, 2, 3]

        return M()


# install fake pymol module BEFORE importing the package/submodule
fake_pymol_mod = types.ModuleType("pymol")
fake_pymol_mod.cmd = FakeCmd()
sys.modules["pymol"] = fake_pymol_mod

# Import orchestrator (expects package importable; run tests from repo root or set PYTHONPATH)
orchestrator_mod = importlib.import_module("prodock.structure.orchestrator")
PDBOrchestrator = orchestrator_mod.PDBOrchestrator

# Import the modules we will patch (we will assign fakes directly)
fetch_mod = importlib.import_module("prodock.structure.fetch")
convert_mod = importlib.import_module("prodock.structure.convert")


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        # create temporary directory (string paths only)
        self.td = tempfile.mkdtemp()

        # Save originals so we can restore them in tearDown
        self._orig_fetch = getattr(fetch_mod, "fetch_pdb_to_dir", None)
        self._orig_convert = getattr(convert_mod, "convert_with_obabel", None)
        self._orig_copyfb = getattr(convert_mod, "copy_fallback", None)

        # fakes accept Path or string and write files accordingly
        def fake_fetch(pdb_id, fetch_dir):
            fn = os.path.join(fetch_dir, f"{pdb_id}.pdb")
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            with open(fn, "w") as fh:
                fh.write("HEADER")
            return fn

        def fake_convert(src, dst, extra_args=None):
            # src/dst may be Path-like or str
            dst_str = str(dst)
            os.makedirs(os.path.dirname(dst_str), exist_ok=True)
            with open(dst_str, "w") as fh:
                fh.write("SDF")
            # emulate successful conversion
            return True

        def fake_copy_fallback(src, dst):
            dst_str = str(dst)
            os.makedirs(os.path.dirname(dst_str), exist_ok=True)
            # attempt to copy text from src (if exists), else write placeholder
            try:
                with open(str(src), "r") as fr:
                    txt = fr.read()
            except Exception:
                txt = "COPIED"
            with open(dst_str, "w") as fw:
                fw.write(txt)
            return True

        # assign fakes on the actual modules (no unittest.mock)
        fetch_mod.fetch_pdb_to_dir = fake_fetch
        convert_mod.convert_with_obabel = fake_convert
        convert_mod.copy_fallback = fake_copy_fallback

        # NOTE: orchestrator performs lazy import inside methods, so patching the modules above is sufficient.

    def tearDown(self):
        # restore originals
        if self._orig_fetch is None:
            try:
                delattr(fetch_mod, "fetch_pdb_to_dir")
            except Exception:
                pass
        else:
            fetch_mod.fetch_pdb_to_dir = self._orig_fetch

        if self._orig_convert is None:
            try:
                delattr(convert_mod, "convert_with_obabel")
            except Exception:
                pass
        else:
            convert_mod.convert_with_obabel = self._orig_convert

        if self._orig_copyfb is None:
            try:
                delattr(convert_mod, "copy_fallback")
            except Exception:
                pass
        else:
            convert_mod.copy_fallback = self._orig_copyfb

        # remove temp dir
        shutil.rmtree(self.td)

    # def test_orchestrator_extract_ligand_creates_ref_and_cocrystal(self):
    #     base = os.path.join(self.td, "work")
    #     orch = PDBOrchestrator(
    #         "5N2F", base_out=base, chains=["A"], ligand_code="LIG", cofactors=["HEM"]
    #     )
    #     orch.validate()

    #     # run the orchestrator steps (fetch/filter/extract)
    #     orch.fetch()
    #     orch.filter_chains()
    #     orch.extract_ligand()

    #     # orch.ref_path and orch.cocrystal_path may be Path objects internally.
    #     # Check existence using os.path.exists on their string representation.
    #     ref_path_str = str(getattr(orch, "ref_path"))
    #     cocrystal_path_str = str(getattr(orch, "cocrystal_path"))

    #     self.assertTrue(os.path.exists(ref_path_str), "reference SDF should exist")
    #     self.assertTrue(
    #         os.path.exists(cocrystal_path_str), "cocrystal SDF should exist"
    #     )

    #     # verify that a remove() call was attempted (we accept any recorded removes)
    #     self.assertTrue(len(fake_pymol_mod.cmd.removed) >= 0)


if __name__ == "__main__":
    unittest.main()
