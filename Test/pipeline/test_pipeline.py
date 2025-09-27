import unittest
import tempfile
from pathlib import Path

from prodock.pipeline.prodockpipeline import ProDock
from prodock.pipeline.receptor_step import ReceptorStep
from prodock.pipeline.ligand_prep_step import LigandPrepStep
from prodock.pipeline.dock_step import DockStep


class MiniReceptorPrep:
    def __init__(self, enable_logging=True):
        self.expected_output_path = None

    def prep(self, input_pdb, output_dir, **kwargs):
        out = Path(output_dir) / "prepared.pdbqt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("REC")
        self.expected_output_path = str(out)


class MiniLigandProcess:
    def __init__(self, output_dir, name_key="id"):
        self.output_dir = Path(output_dir)
        self.name_key = name_key
        self.rows = []

    def set_embed_method(self, *_):
        pass

    def set_opt_method(self, *_):
        pass

    def set_converter_backend(self, *_):
        pass

    def from_list_of_dicts(self, rows):
        self.rows = list(rows)

    def process_all(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for r in self.rows:
            (self.output_dir / f"{r[self.name_key]}.pdbqt").write_text("LIG")


class MiniSingleDock:
    def __init__(self, engine):
        self.engine = engine
        self._out = None
        self._log = None

    def set_receptor(self, r):
        pass

    def set_ligand(self, ligand):
        pass

    def set_out(self, o):
        self._out = o

    def set_log(self, L):
        self._log = L

    def set_box(self, center, size):
        pass

    def set_exhaustiveness(self, e):
        pass

    def set_num_modes(self, m):
        pass

    def set_cpu(self, c):
        pass

    def run(self):
        Path(self._out).parent.mkdir(parents=True, exist_ok=True)
        Path(self._out).write_text("POSE")
        Path(self._log).parent.mkdir(parents=True, exist_ok=True)
        Path(self._log).write_text("LOG")


class TestProDock(unittest.TestCase):
    def test_end_to_end_flow_and_state_roundtrip(self):
        """Full pipeline with injected steps should produce expected files and preserve state."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rec = td / "input.pdb"
            rec.write_text("ATOM")

            rs = ReceptorStep(receptor_prep_cls=MiniReceptorPrep)
            ls = LigandPrepStep(ligand_process_cls=MiniLigandProcess)
            ds = DockStep(dock_cls=MiniSingleDock)

            cfg_box = {
                "center_x": 0,
                "center_y": 0,
                "center_z": 0,
                "size_x": 20,
                "size_y": 20,
                "size_z": 20,
            }

            pipe = ProDock(
                target_path=rec,
                project_dir=td,
                cfg_box=cfg_box,
                receptor_step=rs,
                ligand_step=ls,
                dock_step=ds,
            )

            # prepare receptor should set pipe.target and write file
            prepared = pipe.prepare_receptor()
            self.assertTrue(prepared.exists())
            self.assertEqual(pipe.target.name, "prepared.pdbqt")

            # prepare ligands: create two ligands and expect files in pipeline.ligand_dir
            ligs = [{"id": "L1", "smiles": "CCO"}, {"id": "L2", "smiles": "CCN"}]
            pipe.prep(ligs, n_jobs=1)
            self.assertTrue((pipe.ligand_dir / "L1.pdbqt").exists())
            self.assertTrue((pipe.ligand_dir / "L2.pdbqt").exists())

            # dock produces two results; check files created in output/Modes and output/Log
            results = pipe.dock(n_jobs=1, verbose=0)
            self.assertEqual(len(results), 2)
            self.assertTrue((td / "output" / "Mode" / "L1.pdbqt").exists())
            self.assertTrue((td / "output" / "Log" / "L2.txt").exists())

            # summary contains keys we expect
            summary = pipe.summary()
            for k in ("target_path", "project_dir", "ligand_dir", "cfg_box_present"):
                self.assertIn(k, summary)

            # save + load roundtrip
            state_path = td / "state.pkl"
            pipe.save(state_path)
            self.assertTrue(state_path.exists())
            loaded = ProDock.load(state_path)
            self.assertEqual(loaded.state_dict()["project_dir"], str(td))

    def test_dock_without_prepare_receptor_raises(self):
        """If prepare_receptor was not called, docking should raise RuntimeError."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rec = td / "input.pdb"
            rec.write_text("ATOM")

            rs = ReceptorStep(receptor_prep_cls=MiniReceptorPrep)
            ls = LigandPrepStep(ligand_process_cls=MiniLigandProcess)
            ds = DockStep(dock_cls=MiniSingleDock)

            # Do not provide cfg_box and do not call prepare_receptor -> dock should raise
            pipe = ProDock(
                target_path=rec,
                project_dir=td,
                cfg_box=None,  # explicitly None to test error path
                receptor_step=rs,
                ligand_step=ls,
                dock_step=ds,
            )

            # create one ligand to prevent ligand-not-found error
            (pipe.ligand_dir).mkdir(parents=True, exist_ok=True)
            (pipe.ligand_dir / "X.pdbqt").write_text("LIG")

            with self.assertRaises(RuntimeError):
                pipe.dock(n_jobs=1)


if __name__ == "__main__":
    unittest.main()
