import unittest
import tempfile
from pathlib import Path

from prodock.pipeline.dock_step import DockStep


class MiniSingleDock:
    """
    Minimal SingleDock-like class that writes out & log files when run() is called.
    The class mirrors the commonly used methods so the DockStep's best-effort calls work.
    """

    def __init__(self, engine: str):
        self.engine = engine
        self._out = None
        self._log = None

    def set_receptor(self, r: str):
        self._receptor = r

    def set_ligand(self, ligand: str):
        self._ligand = ligand

    def set_out(self, o: str):
        self._out = o

    def set_log(self, L: str):
        self._log = L

    def set_box(self, center, size):
        self._center = center
        self._size = size

    def set_exhaustiveness(self, e: int):
        self._ex = e

    def set_num_modes(self, m: int):
        self._m = m

    def set_cpu(self, c: int):
        self._cpu = c

    def set_seed(self, s: int):
        self._seed = s

    def run(self):
        outp = Path(self._out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("POSE")
        logp = Path(self._log)
        logp.parent.mkdir(parents=True, exist_ok=True)
        logp.write_text("LOG")


class TestDockStep(unittest.TestCase):
    def test_sequential_multiple_ligands_produces_files_and_results(self):
        """Two ligands should produce two mode files and two logs; results length must match."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            lig_dir = td / "ligs"
            lig_dir.mkdir()
            (lig_dir / "A.pdbqt").write_text("A")
            (lig_dir / "B.pdbqt").write_text("B")

            modes = td / "modes"
            logs = td / "logs"

            d = DockStep(dock_cls=MiniSingleDock)
            cfg = {
                "center_x": 0,
                "center_y": 0,
                "center_z": 0,
                "size_x": 10,
                "size_y": 10,
                "size_z": 10,
            }

            results = d.dock(
                receptor_path=str(td / "receptor.pdbqt"),
                ligand_dir=str(lig_dir),
                output_modes_dir=str(modes),
                logs_dir=str(logs),
                cfg_box=cfg,
                backend="smina",
                n_jobs=1,
                verbose=0,
            )

            # results must be list of dicts and include success True
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
            for r in results:
                self.assertIn("ligand", r)
                self.assertIn("out", r)
                self.assertIn("log", r)
                self.assertIn("success", r)
                self.assertTrue(r["success"])
            self.assertTrue((modes / "A.pdbqt").exists())
            self.assertTrue((logs / "B.txt").exists())

    def test_no_ligands_raises_file_not_found(self):
        """If ligand directory has no matching files, raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "ligs").mkdir()
            d = DockStep(dock_cls=MiniSingleDock)
            cfg = {
                "center_x": 0,
                "center_y": 0,
                "center_z": 0,
                "size_x": 5,
                "size_y": 5,
                "size_z": 5,
            }
            with self.assertRaises(FileNotFoundError):
                d.dock(
                    receptor_path=str(td / "receptor.pdbqt"),
                    ligand_dir=str(td / "ligs"),
                    output_modes_dir=str(td / "modes"),
                    logs_dir=str(td / "logs"),
                    cfg_box=cfg,
                    n_jobs=1,
                )

    def test_missing_cfg_box_raises_runtime_error(self):
        """cfg_box must include center_x/y/z and size_x/y/z; otherwise RuntimeError."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            lig_dir = td / "ligs"
            lig_dir.mkdir()
            (lig_dir / "A.pdbqt").write_text("A")
            d = DockStep(dock_cls=MiniSingleDock)

            # missing totally
            with self.assertRaises(RuntimeError):
                d.dock(
                    receptor_path=str(td / "receptor.pdbqt"),
                    ligand_dir=str(lig_dir),
                    output_modes_dir=str(td / "modes"),
                    logs_dir=str(td / "logs"),
                    cfg_box={},  # empty dict
                    n_jobs=1,
                )

            # incomplete keys
            with self.assertRaises(RuntimeError):
                d.dock(
                    receptor_path=str(td / "receptor.pdbqt"),
                    ligand_dir=str(lig_dir),
                    output_modes_dir=str(td / "modes"),
                    logs_dir=str(td / "logs"),
                    cfg_box={
                        "center_x": 0,
                        "center_y": 0,
                        "center_z": 0,
                    },  # no size keys
                    n_jobs=1,
                )

    def test_batch_mode_splits_and_processes_all(self):
        """Batch mode with batch_size should still produce outputs for all ligands."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            lig_dir = td / "ligs"
            lig_dir.mkdir()
            # Create 5 ligands
            for i in range(5):
                (lig_dir / f"L{i}.pdbqt").write_text("LIG")

            modes = td / "modes"
            logs = td / "logs"

            d = DockStep(dock_cls=MiniSingleDock)
            cfg = {
                "center_x": 0,
                "center_y": 0,
                "center_z": 0,
                "size_x": 10,
                "size_y": 10,
                "size_z": 10,
            }

            results = d.dock(
                receptor_path=str(td / "receptor.pdbqt"),
                ligand_dir=str(lig_dir),
                output_modes_dir=str(modes),
                logs_dir=str(logs),
                cfg_box=cfg,
                batch_size=2,  # will produce batches [0,1], [2,3], [4]
                n_jobs=1,
                verbose=0,
            )

            # Should have 5 results and corresponding files
            self.assertEqual(len(results), 5)
            for i in range(5):
                self.assertTrue((modes / f"L{i}.pdbqt").exists())
                self.assertTrue((logs / f"L{i}.txt").exists())


if __name__ == "__main__":
    unittest.main()
