from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rdkit import Chem

from prodock.core import ProDockPipeline


class TestPipelineAutoboxDefaults(unittest.TestCase):
    def _write_reference_ligand(self, root: Path) -> Path:
        mol = Chem.MolFromSmiles("CCC")
        self.assertIsNotNone(mol)
        assert mol is not None

        conformer = Chem.Conformer(mol.GetNumAtoms())
        conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conformer.SetAtomPosition(1, (5.0, 2.0, 1.0))
        conformer.SetAtomPosition(2, (10.0, 4.0, 2.0))
        mol.RemoveAllConformers()
        mol.AddConformer(conformer)

        path = root / "reference.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
        return path

    def test_default_uses_isotropic_four_angstrom_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self._write_reference_ligand(root)
            pipeline = ProDockPipeline(root / "project", cpu=1, n_jobs=1, progress=False)

            center, size = pipeline._box_from_record({"pdb_id": "TEST", "reference_ligand": reference})

            self.assertEqual(pipeline.box_algorithm, "pad")
            self.assertEqual(pipeline.box_pad, 4.0)
            self.assertEqual(center, (5.0, 2.0, 1.0))
            self.assertEqual(size, (18.0, 18.0, 18.0))

    def test_explicit_box_scale_preserves_legacy_scale_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self._write_reference_ligand(root)
            pipeline = ProDockPipeline(
                root / "project",
                cpu=1,
                n_jobs=1,
                progress=False,
                box_scale=2.0,
            )

            _, size = pipeline._box_from_record({"pdb_id": "TEST", "reference_ligand": reference})

            self.assertEqual(pipeline.box_algorithm, "scale")
            self.assertEqual(size, (20.0, 20.0, 20.0))

    def test_record_can_request_anisotropic_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self._write_reference_ligand(root)
            pipeline = ProDockPipeline(root / "project", cpu=1, n_jobs=1, progress=False)

            _, size = pipeline._box_from_record(
                {
                    "pdb_id": "TEST",
                    "reference_ligand": reference,
                    "box_algorithm": "pad",
                    "box_pad": 4.0,
                    "box_isotropic": False,
                }
            )

            self.assertEqual(size, (18.0, 12.0, 10.0))

    def test_invalid_box_settings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "box_algorithm"):
                ProDockPipeline(
                    root / "bad-algorithm",
                    cpu=1,
                    n_jobs=1,
                    progress=False,
                    box_algorithm="unknown",
                )
            with self.assertRaisesRegex(ValueError, "box_pad"):
                ProDockPipeline(
                    root / "bad-padding",
                    cpu=1,
                    n_jobs=1,
                    progress=False,
                    box_pad=-1.0,
                )


if __name__ == "__main__":
    unittest.main()
