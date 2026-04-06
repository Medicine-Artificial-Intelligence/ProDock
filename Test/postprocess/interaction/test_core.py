from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from prodock.postprocess.interaction.core import (
    InteractionProfiler,
    _iter_dataframe_chunks,
    extract_pose_table_interactions,
)


class FakeFingerprint:
    def __init__(self) -> None:
        self.ifp = {
            0: {
                ("pose_1", "ASP160.A"): {
                    "Hydrophobic": [
                        {
                            "indices": {"ligand": [2], "protein": [9]},
                            "parent_indices": {"ligand": [2], "protein": [2392]},
                            "distance": 4.49,
                        }
                    ]
                }
            }
        }

    def run_from_iterable(
        self,
        prolif_molecules,
        protein_molecule,
        residues=None,
        progress=False,
        n_jobs=1,
    ) -> None:
        return None

    def to_bitvectors(self):
        return ["bv1"]

    def to_countvectors(self):
        return ["cv1"]

    def to_dataframe(self, drop_empty=True):
        return pd.DataFrame({"fp_a": [1]})


class TestCoreHelpers(unittest.TestCase):
    def test_iter_dataframe_chunks_regular(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        chunks = list(_iter_dataframe_chunks(df, batch_size=2))
        self.assertEqual(len(chunks), 3)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])

    def test_iter_dataframe_chunks_nonpositive(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        chunks = list(_iter_dataframe_chunks(df, batch_size=0))
        self.assertEqual(len(chunks), 1)
        pd.testing.assert_frame_equal(chunks[0], df)

    def test_rename_fingerprint_index_match(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        out = InteractionProfiler._rename_fingerprint_index(df, ["pose_1", "pose_2"])
        self.assertEqual(list(out.index), ["pose_1", "pose_2"])
        self.assertEqual(out.index.name, "mol_name")

    def test_rename_fingerprint_index_no_match(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        out = InteractionProfiler._rename_fingerprint_index(df, ["pose_1", "pose_2"])
        self.assertEqual(list(out.index), [0, 1, 2])


class TestExtractPoseTableInteractionsFake(unittest.TestCase):
    def make_poses(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "receptor_id": "1M17",
                    "ligand_id": "lig1",
                    "engine": "qvina",
                    "pose_rank": 1,
                    "affinity": -7.5,
                    "mol": "mol_a",
                },
                {
                    "receptor_id": "4WKQ",
                    "ligand_id": "lig2",
                    "engine": "qvina",
                    "pose_rank": 1,
                    "affinity": -8.1,
                    "mol": "mol_b",
                },
            ]
        )

    def test_extract_pose_table_interactions_success(self) -> None:
        poses = self.make_poses()

        def fake_prepare_ligands(
            ligands,
            resname="LIG",
            resnumber=1,
            chain="",
            use_segid=False,
            sdf_sanitize=True,
        ):
            names = [name for name, _ in ligands]
            rdkit_mols = [mol for _, mol in ligands]
            prolif_mols = [f"plf_{name}" for name in names]
            return names, rdkit_mols, prolif_mols

        with (
            patch(
                "prodock.postprocess.interaction.core.load_receptor_molecule",
                return_value="fake_protein",
            ),
            patch(
                "prodock.postprocess.interaction.core.prepare_ligands",
                side_effect=fake_prepare_ligands,
            ),
            patch(
                "prodock.postprocess.interaction.core.InteractionProfiler._build_fingerprint",
                return_value=FakeFingerprint(),
            ),
        ):
            result = extract_pose_table_interactions(
                poses=poses,
                receptor_pdb_by_id={"1M17": "1m17.pdb", "4WKQ": "4wkq.pdb"},
                include_fingerprint_columns=True,
                include_interaction_events=True,
                include_bitvectors=True,
                include_countvectors=True,
                fail_fast=True,
            )

        self.assertEqual(len(result.merged_df), 2)
        self.assertEqual(len(result.interaction_df), 2)
        self.assertEqual(len(result.summary_df), 2)
        self.assertEqual(result.errors, [])

        self.assertIn("pose_id", result.merged_df.columns)

        self.assertIn("interaction_events_json", result.interaction_df.columns)
        self.assertIn("has_interactions", result.interaction_df.columns)

        self.assertIn("interaction_compact_json", result.summary_df.columns)
        self.assertIn("interaction_detail_json", result.summary_df.columns)
        self.assertIn("has_interactions", result.summary_df.columns)

        self.assertEqual(result.bitvectors, ["bv1", "bv1"])
        self.assertEqual(result.countvectors, ["cv1", "cv1"])

    def test_extract_pose_table_interactions_missing_required_columns(self) -> None:
        poses = pd.DataFrame([{"receptor_id": "1M17", "mol": "mol_a"}])

        with self.assertRaises(ValueError):
            extract_pose_table_interactions(
                poses=poses,
                receptor_pdb_by_id={"1M17": "1m17.pdb"},
            )

    def test_extract_pose_table_interactions_missing_receptor_mapping(self) -> None:
        poses = self.make_poses().iloc[[0]].copy()
        poses.loc[:, "receptor_id"] = "X"

        with self.assertRaises(KeyError):
            extract_pose_table_interactions(
                poses=poses,
                receptor_pdb_by_id={"1M17": "1m17.pdb"},
                fail_fast=False,
            )

    def test_extract_pose_table_interactions_fail_fast_on_processing_error(
        self,
    ) -> None:
        poses = self.make_poses().iloc[[0]].copy()

        with patch(
            "prodock.postprocess.interaction.core.load_receptor_molecule",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                extract_pose_table_interactions(
                    poses=poses,
                    receptor_pdb_by_id={"1M17": "1m17.pdb"},
                    fail_fast=True,
                )

    def test_extract_pose_table_interactions_collect_error_when_not_fail_fast(
        self,
    ) -> None:
        poses = self.make_poses().iloc[[0]].copy()

        with (
            patch(
                "prodock.postprocess.interaction.core.load_receptor_molecule",
                return_value="fake_protein",
            ),
            patch(
                "prodock.postprocess.interaction.core.prepare_ligands",
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = extract_pose_table_interactions(
                poses=poses,
                receptor_pdb_by_id={"1M17": "1m17.pdb"},
                fail_fast=False,
            )

        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["error_type"], "RuntimeError")
        self.assertEqual(len(result.interaction_df), 1)
        self.assertEqual(len(result.summary_df), 1)
        self.assertFalse(result.interaction_df.loc[0, "has_interactions"])
        self.assertFalse(result.summary_df.loc[0, "has_interactions"])
