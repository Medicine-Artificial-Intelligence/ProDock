import unittest
import tempfile
from pathlib import Path

from prodock.pipeline.receptor_step import ReceptorStep


class MiniReceptorPrep:
    """Normal behavior: write a prepared pdbqt and set expected_output_path."""

    def __init__(self, enable_logging=True):
        self.enable_logging = enable_logging
        self.expected_output_path = None

    def prep(self, input_pdb, output_dir, **kwargs):
        inp = Path(input_pdb)
        out = Path(output_dir) / (inp.stem + ".pdbqt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("PDBQT")
        self.expected_output_path = str(out)


class BrokenReceptorPrep:
    """Simulates a prep implementation that fails to populate expected_output_path."""

    def __init__(self, enable_logging=True):
        self.enable_logging = enable_logging
        # deliberately do not set expected_output_path

    def prep(self, input_pdb, output_dir, **kwargs):
        # create no output (simulate failure)
        return


class TestReceptorStep(unittest.TestCase):
    def test_prepare_creates_output_and_returns_path(self):
        """Happy path: file written and returned path is correct."""
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "rec.pdb"
            outdir = Path(td) / "receptor"
            inp.write_text("ATOM")

            step = ReceptorStep(receptor_prep_cls=MiniReceptorPrep)
            out = step.prepare(inp, outdir)

            self.assertTrue(out.exists(), "Prepared receptor file should exist")
            self.assertEqual(
                out.suffix.lower(), ".pdbqt", "Prepared file should be .pdbqt"
            )

    def test_missing_input_raises_file_not_found(self):
        """If the input PDB doesn't exist, we should raise FileNotFoundError early."""
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "does_not_exist.pdb"
            outdir = Path(td) / "receptor"
            step = ReceptorStep(receptor_prep_cls=MiniReceptorPrep)
            with self.assertRaises(FileNotFoundError):
                step.prepare(missing, outdir)

    def test_broken_prep_raises_runtime_error(self):
        """If the underlying prep does not set expected_output_path, raise RuntimeError."""
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "rec.pdb"
            outdir = Path(td) / "receptor"
            inp.write_text("ATOM")
            step = ReceptorStep(receptor_prep_cls=BrokenReceptorPrep)
            with self.assertRaises(RuntimeError):
                step.prepare(inp, outdir)


if __name__ == "__main__":
    unittest.main()
