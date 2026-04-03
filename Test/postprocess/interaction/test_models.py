from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from prodock.postprocess.interaction.models import (
    InteractionRunResult,
    PoseInteractionTableResult,
)


class FakeFingerprint:
    def __init__(self) -> None:
        self.saved_path = None

    def to_pickle(self, path) -> None:
        self.saved_path = Path(path)
        self.saved_path.write_text("fingerprint", encoding="utf-8")


class TestInteractionRunResult(unittest.TestCase):
    def test_pose_names_property(self) -> None:
        result = InteractionRunResult(
            receptor_path=None,
            molecule_names=["pose_1", "pose_2"],
        )
        self.assertEqual(result.pose_names, ["pose_1", "pose_2"])
        self.assertIsNot(result.pose_names, result.molecule_names)

    def test_ligand_molecules_property(self) -> None:
        mols = ["mol_a", "mol_b"]
        result = InteractionRunResult(
            receptor_path=None,
            molecule_names=["pose_1", "pose_2"],
            molecules=mols,
        )
        self.assertEqual(result.ligand_molecules, mols)
        self.assertIsNot(result.ligand_molecules, result.molecules)

    def test_molecule_table_with_smiles(self) -> None:
        result = InteractionRunResult(
            receptor_path=None,
            molecule_names=["pose_1", "pose_2"],
            molecules=["mol_a", "mol_b"],
        )
        with patch(
            "prodock.postprocess.interaction.models.mol_to_smiles",
            side_effect=lambda mol: f"SMILES:{mol}",
        ):
            df = result.molecule_table(include_smiles=True)

        self.assertEqual(list(df.columns), ["mol_index", "mol_name", "mol", "smiles"])
        self.assertEqual(df.loc[0, "mol_name"], "pose_1")
        self.assertEqual(df.loc[1, "smiles"], "SMILES:mol_b")

    def test_molecule_table_without_smiles(self) -> None:
        result = InteractionRunResult(
            receptor_path=None,
            molecule_names=["pose_1"],
            molecules=["mol_a"],
        )
        df = result.molecule_table(include_smiles=False)
        self.assertEqual(list(df.columns), ["mol_index", "mol_name", "mol"])

    def test_serializable_interaction_df_empty(self) -> None:
        result = InteractionRunResult(
            receptor_path=None,
            molecule_names=[],
            interaction_df=pd.DataFrame(),
        )
        out = result.serializable_interaction_df()
        self.assertTrue(out.empty)

    def test_serializable_interaction_df_replaces_mol(self) -> None:
        result = InteractionRunResult(
            receptor_path=None,
            molecule_names=["pose_1"],
            interaction_df=pd.DataFrame(
                [{"pose_id": "pose_1", "mol": "mol_a", "interaction": "Hydrophobic"}]
            ),
        )
        with patch(
            "prodock.postprocess.interaction.models.mol_to_smiles",
            return_value="CCO",
        ):
            out = result.serializable_interaction_df()

        self.assertIn("mol_smiles", out.columns)
        self.assertNotIn("mol", out.columns)
        self.assertEqual(out.loc[0, "mol_smiles"], "CCO")

    def test_save_tables_without_pickle(self) -> None:
        result = InteractionRunResult(
            receptor_path=None,
            molecule_names=["pose_1"],
            molecules=["mol_a"],
            fingerprint_df=pd.DataFrame({"a": [1]}),
            interaction_df=pd.DataFrame(
                [{"pose_id": "pose_1", "interaction": "Hydrophobic"}]
            ),
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "prodock.postprocess.interaction.models.mol_to_smiles",
                return_value="CCO",
            ),
        ):
            created = result.save_tables(tmpdir, prefix="run1")

            self.assertIn("fingerprint_csv", created)
            self.assertIn("interactions_csv", created)
            self.assertIn("molecules_csv", created)
            self.assertNotIn("fingerprint_pickle", created)
            self.assertTrue(created["fingerprint_csv"].exists())
            self.assertTrue(created["interactions_csv"].exists())
            self.assertTrue(created["molecules_csv"].exists())

    def test_save_tables_with_pickle(self) -> None:
        result = InteractionRunResult(
            receptor_path=None,
            molecule_names=["pose_1"],
            molecules=["mol_a"],
            fingerprint=FakeFingerprint(),
            fingerprint_df=pd.DataFrame({"a": [1]}),
            interaction_df=pd.DataFrame(
                [{"pose_id": "pose_1", "interaction": "Hydrophobic"}]
            ),
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "prodock.postprocess.interaction.models.mol_to_smiles",
                return_value="CCO",
            ),
        ):
            created = result.save_tables(tmpdir, prefix="run2")

            self.assertIn("fingerprint_pickle", created)
            self.assertTrue(created["fingerprint_csv"].exists())
            self.assertTrue(created["interactions_csv"].exists())
            self.assertTrue(created["molecules_csv"].exists())
            self.assertTrue(created["fingerprint_pickle"].exists())


class TestPoseInteractionTableResult(unittest.TestCase):
    def make_result(self) -> PoseInteractionTableResult:
        return PoseInteractionTableResult(
            merged_df=pd.DataFrame(
                [
                    {"pose_id": "pose_1", "mol": "mol_a", "score": -8.1},
                    {"pose_id": "pose_2", "mol": "mol_b", "score": -7.4},
                ]
            ),
            interaction_df=pd.DataFrame(
                [
                    {
                        "pose_id": "pose_1",
                        "interaction_events": [{"interaction": "Hydrophobic"}],
                        "interaction_events_json": json.dumps(
                            [{"interaction": "Hydrophobic"}],
                            sort_keys=True,
                        ),
                        "has_interactions": True,
                    },
                    {
                        "pose_id": "pose_2",
                        "interaction_events": None,
                        "interaction_events_json": None,
                        "has_interactions": False,
                    },
                ]
            ),
            summary_df=pd.DataFrame(
                [
                    {
                        "pose_id": "pose_1",
                        "interaction_compact": {"Hydrophobic": ["ASP160.A"]},
                        "interaction_compact_json": json.dumps(
                            {"Hydrophobic": ["ASP160.A"]},
                            sort_keys=True,
                        ),
                        "interaction_detail": {
                            "Hydrophobic": {"ASP160.A": [{"distance": 4.5}]}
                        },
                        "interaction_detail_json": json.dumps(
                            {"Hydrophobic": {"ASP160.A": [{"distance": 4.5}]}},
                            sort_keys=True,
                        ),
                        "has_interactions": True,
                    },
                    {
                        "pose_id": "pose_2",
                        "interaction_compact": None,
                        "interaction_compact_json": None,
                        "interaction_detail": None,
                        "interaction_detail_json": None,
                        "has_interactions": False,
                    },
                ]
            ),
            errors=[{"pose_id": "pose_x", "error": "failed"}],
            molecule_names=["pose_1", "pose_2"],
            bitvectors=["bv1", "bv2"],
            countvectors=["cv1", "cv2"],
        )

    def test_serializable_merged_df_empty(self) -> None:
        result = PoseInteractionTableResult()
        out = result.serializable_merged_df()
        self.assertTrue(out.empty)

    def test_serializable_merged_df_replaces_mol(self) -> None:
        result = self.make_result()
        with patch(
            "prodock.postprocess.interaction.models.mol_to_smiles",
            side_effect=lambda mol: f"SMILES:{mol}",
        ):
            out = result.serializable_merged_df()

        self.assertIn("mol_smiles", out.columns)
        self.assertNotIn("mol", out.columns)
        self.assertEqual(out.loc[0, "mol_smiles"], "SMILES:mol_a")

    def test_serializable_interaction_df_drops_payload_column(self) -> None:
        result = self.make_result()
        out = result.serializable_interaction_df()
        self.assertNotIn("interaction_events", out.columns)
        self.assertIn("interaction_events_json", out.columns)

    def test_serializable_summary_df_drops_nested_columns(self) -> None:
        result = self.make_result()
        out = result.serializable_summary_df()
        self.assertNotIn("interaction_compact", out.columns)
        self.assertNotIn("interaction_detail", out.columns)
        self.assertIn("interaction_compact_json", out.columns)
        self.assertIn("interaction_detail_json", out.columns)

    def test_summary_dict_compact(self) -> None:
        result = self.make_result()
        out = result.summary_dict(kind="compact")
        self.assertEqual(out["pose_1"]["Hydrophobic"], ["ASP160.A"])
        self.assertIsNone(out["pose_2"])

    def test_summary_dict_detail(self) -> None:
        result = self.make_result()
        out = result.summary_dict(kind="detail")
        self.assertIn("Hydrophobic", out["pose_1"])
        self.assertIsNone(out["pose_2"])

    def test_interaction_dict(self) -> None:
        result = self.make_result()
        out = result.interaction_dict()
        self.assertEqual(out["pose_1"], [{"interaction": "Hydrophobic"}])
        self.assertIsNone(out["pose_2"])

    def test_interaction_dict_drop_empty(self) -> None:
        result = self.make_result()
        out = result.interaction_dict(drop_empty=True)
        self.assertIn("pose_1", out)
        self.assertNotIn("pose_2", out)

    def test_interaction_dict_empty_df(self) -> None:
        result = PoseInteractionTableResult(interaction_df=pd.DataFrame())
        self.assertEqual(result.interaction_dict(), {})

    def test_similarity_matrix_bit(self) -> None:
        result = self.make_result()
        expected = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=["pose_1", "pose_2"],
            columns=["pose_1", "pose_2"],
        )
        with patch(
            "prodock.postprocess.interaction.similarity.tanimoto_similarity_matrix",
            return_value=expected,
        ) as mocked:
            out = result.similarity_matrix(kind="bit")

        mocked.assert_called_once_with(["bv1", "bv2"], ["pose_1", "pose_2"])
        pd.testing.assert_frame_equal(out, expected)

    def test_similarity_matrix_count(self) -> None:
        result = self.make_result()
        expected = pd.DataFrame(
            [[1.0, 0.2], [0.2, 1.0]],
            index=["pose_1", "pose_2"],
            columns=["pose_1", "pose_2"],
        )
        with patch(
            "prodock.postprocess.interaction.similarity.tanimoto_similarity_matrix",
            return_value=expected,
        ) as mocked:
            out = result.similarity_matrix(kind="count")

        mocked.assert_called_once_with(["cv1", "cv2"], ["pose_1", "pose_2"])
        pd.testing.assert_frame_equal(out, expected)

    def test_similarity_matrix_missing_bitvectors(self) -> None:
        result = PoseInteractionTableResult(molecule_names=["pose_1"])
        with self.assertRaises(ValueError):
            result.similarity_matrix(kind="bit")

    def test_similarity_matrix_missing_countvectors(self) -> None:
        result = PoseInteractionTableResult(molecule_names=["pose_1"])
        with self.assertRaises(ValueError):
            result.similarity_matrix(kind="count")

    def test_similarity_matrix_invalid_kind(self) -> None:
        result = self.make_result()
        with self.assertRaises(ValueError):
            result.similarity_matrix(kind="bad")

    def test_save_tables(self) -> None:
        result = self.make_result()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "prodock.postprocess.interaction.models.mol_to_smiles",
                side_effect=lambda mol: f"SMILES:{mol}",
            ),
        ):
            created = result.save_tables(tmpdir, prefix="pose_run")

            self.assertTrue(created["merged_csv"].exists())
            self.assertTrue(created["interactions_csv"].exists())
            self.assertTrue(created["summary_csv"].exists())
            self.assertTrue(created["errors_csv"].exists())
            self.assertTrue(created["compact_json"].exists())
            self.assertTrue(created["detail_json"].exists())

            compact = json.loads(created["compact_json"].read_text(encoding="utf-8"))
            detail = json.loads(created["detail_json"].read_text(encoding="utf-8"))

        self.assertIn("pose_1", compact)
        self.assertIn("pose_1", detail)
