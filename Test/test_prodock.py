from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from prodock import ProDock


class TestProDock(unittest.TestCase):
    def setUp(self):
        # Temporary workspace (we keep the 'auto' project_dir inside this tmpdir)
        self.tmpdir = Path(tempfile.mkdtemp(prefix="test_prodock_integration_"))
        self.project_dir = self.tmpdir / "auto"
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # data files expected by your snippet (relative to repo root)
        self.receptor_path = Path("Data/testcase/pipeline/filtered_protein/5N2F.pdb")
        self.ligand_path = Path("Data/testcase/pipeline/reference_ligand/8HW.sdf")

        # ligand rows (from your snippet)
        self.rows = [
            {
                "smiles": "C1CC1C2=NC3=CC=CC=C3C(=C2OC4=CC=C(C=C4)C5=CC=CC=C5C6=NNN=N6)C(=O)O",
                "name": "BMS-183920",
            },
            {
                "smiles": "Cc1cccc2c1NC(=O)[C@@H](NC(=O)[C@H](CCC(F)(F)F)[C@H](CCC(F)(F)F)C(N)=O)N=C2c1cccc(F)c1",
                "name": "BMS-986115",
            },
            {"smiles": "O=c1[nH]c(C2CCCN2)nc2c1oc1ccc(Cl)cc12", "name": "BMS-863233"},
        ]

        # location to persist pipeline state
        self.state_path = self.project_dir / "pipeline_state.pkl"

    def tearDown(self):
        # remove temporary workspace
        try:
            shutil.rmtree(self.tmpdir)
        except Exception:
            pass

    def test_prodock_full_integration(self):
        # 1) Sanity checks for required files / binaries
        if not self.receptor_path.exists():
            self.fail(
                f"Required receptor file not found: {self.receptor_path.resolve()}"
            )
        if not self.ligand_path.exists():
            self.fail(f"Required ligand file not found: {self.ligand_path.resolve()}")

        # 2) Instantiate ProDock with the real files and a temporary project dir
        pdock = ProDock(
            target_path=str(self.receptor_path),
            crystal=False,
            ligand_path=str(self.ligand_path),
            project_dir=str(self.project_dir),
            cfg_box=None,
        )

        pdock.prepare_receptor(out_fmt="pdbqt")

        # 4) Save pipeline state to disk and verify file exists
        pdock.save(str(self.state_path))
        self.assertTrue(
            self.state_path.exists(),
            f"Pipeline state file should exist after save: {self.state_path}",
        )

        # 5) Load pipeline state back and verify we get a ProDock instance
        pdock2 = ProDock.load(str(self.state_path))
        self.assertIsInstance(pdock2, ProDock)

        # 6) Prep ligands (embedding/ligand processing) - real call
        pdock2.prep(ligands=self.rows, smiles_key="smiles", id_key="name", n_jobs=1)

        # 7) Run docking using smina (real call). Adjust cpu/n_jobs to your machine.
        results = pdock2.dock(
            backend="smina",
            cpu=4,
            n_jobs=1,
            batch_size=None,
            verbose=2,
            parallel_prefer="threads",
        )

        # 8) Basic post-run assertions: results must be non-empty and of expected shape/type.
        self.assertIsNotNone(results, "Docking returned None; expected results.")
        # Accept list/tuple or DataFrame or dict-like
        ok_type = (
            isinstance(results, (list, tuple))
            or hasattr(results, "shape")
            or isinstance(results, dict)
        )
        self.assertTrue(
            ok_type, f"Unexpected results type from dock(): {type(results)}"
        )

        # Print results so CI / console shows the primary output
        print("\n=== ProDock docking results (top-level) ===")
        print(results)


if __name__ == "__main__":
    unittest.main()
