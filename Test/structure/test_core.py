import unittest
from unittest import mock
from pathlib import Path
import importlib

mod = importlib.import_module("prodock.structure.core")
PDBQuery = getattr(mod, "PDBQuery")


class TestPDBQueryCore(unittest.TestCase):
    def test_constructor_transforms_args_and_calls_orchestrator(self):
        """Constructor should convert chains/cofactors to lists and output_dir to Path."""
        mock_orch = mock.MagicMock(name="PDBOrchestrator() instance")
        with mock.patch.object(
            mod, "PDBOrchestrator", return_value=mock_orch
        ) as mock_ctor:
            pq = PDBQuery(
                pdb_id="5n2f",
                output_dir="/tmp/outdir",
                chains=("A", "B"),
                ligand_code="HEM",
                ligand_name="heme",
                cofactors=("HEM",),
                protein_name="myprot",
                auto_create_dirs=False,
            )

            # ensure constructor called once
            mock_ctor.assert_called_once()
            call_kwargs = mock_ctor.call_args.kwargs

            self.assertEqual(call_kwargs["pdb_id"], "5n2f")
            self.assertIsInstance(call_kwargs["base_out"], Path)
            self.assertEqual(call_kwargs["base_out"], Path("/tmp/outdir"))
            self.assertEqual(call_kwargs["chains"], ["A", "B"])
            self.assertEqual(call_kwargs["ligand_code"], "HEM")
            self.assertEqual(call_kwargs["cofactors"], ["HEM"])
            self.assertFalse(call_kwargs["auto_create_dirs"])
            # ensure the instance stored the orchestrator we returned
            self.assertIs(pq._orchestrator, mock_orch)

    def test_constructor_defaults_chains_cofactors_to_empty_lists(self):
        """Omitting chains/cofactors yields empty lists passed to orchestrator."""
        mock_orch = mock.MagicMock()
        with mock.patch.object(
            mod, "PDBOrchestrator", return_value=mock_orch
        ) as mock_ctor:
            pq = PDBQuery(
                pdb_id="1ABC",
                output_dir="out",
                chains=None,
                ligand_code="ABC",
                cofactors=None,
            )

            call_kwargs = mock_ctor.call_args.kwargs
            self.assertEqual(call_kwargs["chains"], [])
            self.assertEqual(call_kwargs["cofactors"], [])
            self.assertTrue(call_kwargs["auto_create_dirs"])
            self.assertIs(pq._orchestrator, mock_orch)

    def test_validate_forwards_and_returns_self(self):
        mock_orch = mock.MagicMock()
        with mock.patch.object(mod, "PDBOrchestrator", return_value=mock_orch):
            pq = PDBQuery("5N2F", "out")
            pq._orchestrator.validate.return_value = None
            returned = pq.validate()
            pq._orchestrator.validate.assert_called_once_with()
            self.assertIs(returned, pq)

    def test_run_all_forwards_and_returns_self(self):
        mock_orch = mock.MagicMock()
        with mock.patch.object(mod, "PDBOrchestrator", return_value=mock_orch):
            pq = PDBQuery("5N2F", "out")
            pq._orchestrator.run_all.return_value = None
            returned = pq.run_all()
            pq._orchestrator.run_all.assert_called_once_with()
            self.assertIs(returned, pq)

    def test_properties_return_str_paths_or_none(self):
        mock_orch = mock.MagicMock()
        mock_orch.pdb_path = Path("fetched_protein/5N2F.pdb")
        mock_orch.filtered_path = Path("filtered_protein/5N2F_filtered.pdb")
        mock_orch.ref_path = Path("reference_ligand/5N2F_lig.sdf")
        mock_orch.cocrystal_path = Path("cocrystal/5N2F_coxt.sdf")

        with mock.patch.object(mod, "PDBOrchestrator", return_value=mock_orch):
            pq = PDBQuery("5N2F", "out")
            self.assertEqual(pq.pdb_path, str(mock_orch.pdb_path))
            self.assertEqual(pq.filtered_protein_path, str(mock_orch.filtered_path))
            self.assertEqual(pq.reference_ligand_path, str(mock_orch.ref_path))
            self.assertEqual(pq.cocrystal_ligand_path, str(mock_orch.cocrystal_path))

        # Now check None propagation
        mock_orch2 = mock.MagicMock()
        mock_orch2.pdb_path = None
        mock_orch2.filtered_path = None
        mock_orch2.ref_path = None
        mock_orch2.cocrystal_path = None

        with mock.patch.object(mod, "PDBOrchestrator", return_value=mock_orch2):
            pq2 = PDBQuery("1XYZ", "out")
            self.assertIsNone(pq2.pdb_path)
            self.assertIsNone(pq2.filtered_protein_path)
            self.assertIsNone(pq2.reference_ligand_path)
            self.assertIsNone(pq2.cocrystal_ligand_path)

    def test_process_batch_forwards_to__process_batch_and_returns_value(self):
        sentinel = object()
        items = [{"pdb_id": "5N2F", "ligand_code": "HEM", "chains": ["A"]}]
        with mock.patch(
            "prodock.structure.batch.process_batch", return_value=sentinel
        ) as mock_proc:
            result = PDBQuery.process_batch(
                items=items, output_dir="out/batch", some_kw=True
            )
            mock_proc.assert_called_once_with(
                items=items, output_dir="out/batch", some_kw=True
            )
            self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
