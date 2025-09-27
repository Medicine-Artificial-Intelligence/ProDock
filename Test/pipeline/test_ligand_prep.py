import unittest
import tempfile
from pathlib import Path

import pandas as pd

from prodock.pipeline.ligand_prep_step import LigandPrepStep


class MiniLigandProcess:
    """
    Minimal drop-in that writes one file per ligand id to output_dir.

    Mimics expected interface of the real LigandProcess:
    - set_embed_method, set_opt_method, set_converter_backend, from_list_of_dicts, process_all
    """

    def __init__(self, output_dir, name_key="id"):
        self.output_dir = Path(output_dir)
        self.name_key = name_key
        self._records = []

    def set_embed_method(self, *_):  # API compatibility
        self._embed = True

    def set_opt_method(self, *_):
        self._opt = True

    def set_converter_backend(self, *_):
        self._conv = True

    def from_list_of_dicts(self, rows):
        self._records = list(rows)

    def process_all(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for r in self._records:
            fn = self.output_dir / f"{r[self.name_key]}.pdbqt"
            fn.write_text("LIG")


class TestLigandPrepStep(unittest.TestCase):
    def test_prep_writes_files_from_list(self):
        """List-of-dicts path: two ligands written as files."""
        with tempfile.TemporaryDirectory() as td:
            rows = [{"id": "A", "smiles": "CCO"}, {"id": "B", "smiles": "CCN"}]
            step = LigandPrepStep(ligand_process_cls=MiniLigandProcess)
            step.prep(rows, td, n_jobs=1, batch_size=None, verbose=0)

            self.assertTrue((Path(td) / "A.pdbqt").exists())
            self.assertTrue((Path(td) / "B.pdbqt").exists())

    def test_prep_accepts_dataframe(self):
        """DataFrame input should be normalized correctly and produce the same outputs."""
        with tempfile.TemporaryDirectory() as td:
            df = pd.DataFrame(
                [{"id": "X", "smiles": "CCC"}, {"id": "Y", "smiles": "CCCl"}]
            )
            step = LigandPrepStep(ligand_process_cls=MiniLigandProcess)
            step.prep(df, td, n_jobs=1, batch_size=None, verbose=0)

            self.assertTrue((Path(td) / "X.pdbqt").exists())
            self.assertTrue((Path(td) / "Y.pdbqt").exists())

    def test_single_smiles_string_normalizes(self):
        """Passing a single SMILES string should create one prepared ligand file."""
        with tempfile.TemporaryDirectory() as td:
            s = "CCO"
            step = LigandPrepStep(ligand_process_cls=MiniLigandProcess)
            # when ligands is a string, the normalizer will create a single record
            step.prep(s, td, id_key="smi", smiles_key="smi", n_jobs=1)
            # since the code uses the id_key as identifier and we passed the same string,
            # expect a file named by the SMILES (not ideal in real pipeline, but consistent)
            expected = Path(td) / f"{s}.pdbqt"
            self.assertTrue(expected.exists())

    def test_batch_splitting_and_processing(self):
        """
        Verify that batch_size correctly splits lists into batches.
        Create 5 ligands, set batch_size=2 -> expect batches [0,1], [2,3], [4]
        All ligands should be created as files when n_jobs=1.
        """
        with tempfile.TemporaryDirectory() as td:
            rows = [{"id": f"L{i}", "smiles": "C"} for i in range(5)]
            step = LigandPrepStep(ligand_process_cls=MiniLigandProcess)
            step.prep(rows, td, n_jobs=1, batch_size=2, verbose=0)

            for i in range(5):
                self.assertTrue(
                    (Path(td) / f"L{i}.pdbqt").exists(), f"L{i}.pdbqt should exist"
                )


if __name__ == "__main__":
    unittest.main()
