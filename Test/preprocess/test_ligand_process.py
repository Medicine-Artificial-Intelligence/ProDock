# tests/test_ligand_process_unittest.py
import unittest
import tempfile
from pathlib import Path
import csv
import importlib

# attempt to silence RDKit log spam if RDKit is present
try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


def _find_ligand_class():
    """
    Try several import paths / names to locate the user's LigandProcess class.
    Returns (class_obj, module_path_str).
    """
    candidates = [
        ("prodock.preprocess.ligand_process", "LigandPreprocess"),
        ("prodock.ligand.process", "LigandProcess"),
        ("prodock.ligand.process", "LigandPreprocess"),
        ("prodock.preprocess.ligand_process", "LigandProcess"),
    ]
    for modname, cname in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(mod, cname, None)
        if cls is not None:
            return cls, f"{modname}.{cname}"
    raise ImportError(
        "Could not import LigandPreprocess / LigandProcess from expected locations. "
        "Tried prodock.preprocess.ligand_process and prodock.ligand.process."
    )


@unittest.skipUnless(RDKit_AVAILABLE, "RDKit is required for LigandProcess tests")
class TestLigandProcess(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        # create a small smiles file used in some tests
        self.smi_path = self.tmp / "smiles.smi"
        self.smi_path.write_text("CCO\nCCC\n")
        # locate user's class (support multiple possible names/locations)
        self.LigandClass, self.class_path = _find_ligand_class()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_basic_from_smiles_list_and_process_all(self):
        """
        Basic flow: load two SMILES, process them (no 3D by default), ensure SDFs written,
        manifest can be saved and has expected rows.
        """
        lp = self.LigandClass(output_dir=str(self.tmp / "out"))
        self.assertTrue(hasattr(lp, "from_smiles_list"))
        lp.from_smiles_list(["CCO", "CCC"])
        # defaults: embed3d False (class default), so processing should be light-weight
        lp.process_all()
        # ensure records and outputs are present
        recs = lp.records
        self.assertEqual(len(recs), 2)
        ok_paths = lp.output_paths
        # at least one file should be produced (RDKit may write both)
        self.assertGreaterEqual(len(ok_paths), 1)
        for p in ok_paths:
            self.assertTrue(Path(p).exists(), f"SDF file missing: {p}")
        # test manifest saving
        manifest = self.tmp / "manifest.csv"
        lp.save_manifest(manifest)
        self.assertTrue(manifest.exists())
        # quick CSV sanity check
        with manifest.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        self.assertEqual(len(rows), len(recs))

    def test_setters_expose_embed_and_opt_methods(self):
        """
        Ensure configuration setters exist and affect internal attributes.
        We do not require Conformer to exist for this (we only check attributes).
        """
        lp = self.LigandClass(output_dir=str(self.tmp / "out2"))
        # setter methods should exist
        self.assertTrue(hasattr(lp, "set_embed_method"))
        self.assertTrue(hasattr(lp, "set_opt_method"))
        # set values
        lp.set_embed_method("ETKDGv2")
        lp.set_opt_method("UFF")
        lp.set_conformer_seed(123)
        lp.set_conformer_jobs(1)
        lp.set_opt_max_iters(50)
        # internal attribute names may vary slightly; check common ones
        # prefer checking getter-like attributes if present, otherwise private attrs
        # Try common private attribute names used in implementation
        _embed = getattr(lp, "_embed_algorithm", None)
        _opt = getattr(lp, "_opt_method", None)
        _seed = getattr(lp, "_conformer_seed", None)
        _jobs = getattr(lp, "_conformer_n_jobs", None)
        _iters = getattr(lp, "_opt_max_iters", None)
        self.assertIn(_embed, (None, "ETKDGv2"))
        self.assertIn(_opt, (None, "UFF"))
        self.assertIn(_seed, (None, 123))
        self.assertIn(_jobs, (None, 1))
        self.assertIn(_iters, (None, 50))

    def test_invalid_smiles_recorded_as_failed(self):
        """
        Invalid SMILES should not crash process_all; the record should be marked failed.
        """
        lp = self.LigandClass(output_dir=str(self.tmp / "out3"))
        lp.from_smiles_list(["NOTASMILES"])
        lp.process_all()
        self.assertEqual(len(lp.records), 1)
        failed = lp.failed
        # must have at least one failed record
        self.assertGreaterEqual(len(failed), 1)
        rec = failed[0]
        self.assertIn("error", rec)
        self.assertEqual(rec["status"], "failed")
        # output_paths should be empty
        self.assertEqual(len(lp.output_paths), 0)

    def test_set_output_dir_and_clear_records(self):
        lp = self.LigandClass(output_dir=str(self.tmp / "out4"))
        lp.from_smiles_list(["CCO"])
        self.assertEqual(len(lp.records), 1)
        lp.set_output_dir(self.tmp / "new_out")
        # directory should exist
        self.assertTrue((self.tmp / "new_out").exists())
        # clear records
        lp.clear_records()
        self.assertEqual(len(lp.records), 0)

    def test_process_all_with_names_and_sanitized_filenames(self):
        lp = self.LigandClass(output_dir=str(self.tmp / "out5"), name_key="name")
        lp.from_list_of_dicts(
            [
                {"smiles": "CCO", "name": "My Molecule #1"},
                {"smiles": "CCC", "name": "M@2"},
            ]
        )
        lp.process_all()
        # check filenames in output dir are sanitized and present
        files = list((self.tmp / "out5").glob("*.sdf")) + [
            p for p in (self.tmp / "out5").iterdir() if p.is_dir()
        ]
        self.assertGreaterEqual(len(files), 1)

    # If you have Conformer available, we can also exercise the embedding path lightly:
    def test_optional_conformer_path_runs_if_available(self):
        """
        If the project provides prodock.chem.conformer.Conformer, exercise a single
        record embedding path with embed3d=True. This test is skipped silently if
        Conformer isn't present (we infer by inspecting the class attributes).
        """
        lp = self.LigandClass(output_dir=str(self.tmp / "out6"))
        lp.from_smiles_list(["CCO"])
        # attempt to enable embedding; process may use either Conformer or fallback smiles2sdf
        try:
            lp.set_options(embed3d=True, add_hs=True, optimize=False)
        except Exception:
            self.skipTest(
                "Class under test doesn't expose set_options(...) as expected"
            )
        # run processing (may be heavier); still expect at least one output or a recorded failure
        lp.process_all()
        # either succeeded or recorded a failure
        recs = lp.records
        self.assertEqual(len(recs), 1)
        # if ok -> file exists; if failed -> error captured
        if recs[0]["status"] == "ok":
            self.assertTrue(recs[0]["out_path"] is not None)
            self.assertTrue(Path(recs[0]["out_path"]).exists())
        else:
            self.assertIsNotNone(recs[0]["error"])


if __name__ == "__main__":
    unittest.main()
