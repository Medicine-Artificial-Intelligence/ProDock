from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from rdkit import Chem

from prodock.database.core import PoseDatabase


class TestPoseDatabaseCore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "poses.sqlite"
        self.db = PoseDatabase(self.db_path)

        self.mol1 = Chem.MolFromSmiles("CCO")
        self.mol2 = Chem.MolFromSmiles("CCN")
        self.pose1 = "1M17__erol__qvina__pose1"
        self.pose2 = "1M17__erol__qvina__pose2"

        self.df = pd.DataFrame(
            [
                {
                    "pose_id": self.pose1,
                    "receptor_id": "1M17",
                    "ligand_id": "erol",
                    "engine": "qvina",
                    "pose_rank": 1,
                    "affinity": -8.2,
                    "mol": self.mol1,
                    "pose_metadata": {"source": "test"},
                    "score_data": {"cnn_affinity": -7.9},
                },
                {
                    "pose_id": self.pose2,
                    "receptor_id": "1M17",
                    "ligand_id": "erol",
                    "engine": "qvina",
                    "pose_rank": 2,
                    "affinity": -7.8,
                    "mol": self.mol2,
                },
            ]
        )

    def tearDown(self) -> None:
        try:
            self.db.close()
        finally:
            self.tmpdir.cleanup()

    def test_insert_dataframe_and_query_pose(self) -> None:
        self.db.insert_dataframe(self.df)
        pose = self.db.get_pose(pose_id=self.pose1, include_mol=True)

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose.pose_id, self.pose1)
        self.assertEqual(pose.pose_rank, 1)
        self.assertAlmostEqual(pose.affinity, -8.2)
        self.assertIsNotNone(pose.mol)
        self.assertEqual(Chem.MolToSmiles(pose.mol), Chem.MolToSmiles(self.mol1))
        self.assertEqual(pose.pose_metadata["source"], "test")
        self.assertEqual(pose.score_data["cnn_affinity"], -7.9)

    def test_insert_dataframe_missing_columns_raises(self) -> None:
        bad = pd.DataFrame([{"receptor_id": "1M17"}])
        with self.assertRaises(ValueError):
            self.db.insert_dataframe(bad)

    def test_upsert_pose_updates_existing_logical_key(self) -> None:
        self.db.insert_dataframe(self.df)
        mol_new = Chem.MolFromSmiles("CCCC")
        pose_db_id = self.db.upsert_pose(
            receptor_id="1M17",
            ligand_id="erol",
            engine="qvina",
            pose_rank=1,
            affinity=-9.1,
            mol=mol_new,
            pose_id=self.pose1,
            score_data={"cnn_affinity": -8.8},
        )
        pose = self.db.get_pose(pose_db_id=pose_db_id, include_mol=True)

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.affinity, -9.1)
        self.assertEqual(pose.score_data["cnn_affinity"], -8.8)
        self.assertEqual(Chem.MolToSmiles(pose.mol), Chem.MolToSmiles(mol_new))

    def test_add_and_query_interaction(self) -> None:
        self.db.insert_dataframe(self.df)
        interaction_id = self.db.add_interaction(
            pose_id=self.pose1,
            interaction_type="Hydrophobic",
            residue_id="LEU23.A",
            ligand_residue="LIG1",
            ligand_atom_indices=[1, 2],
            protein_atom_indices=[11, 12],
            distance=4.2,
            metadata={"source": "manual"},
        )

        frame = self.db.query_interactions(
            interaction_id=interaction_id,
            as_dataframe=True,
        )
        self.assertEqual(frame.shape[0], 1)
        self.assertEqual(frame.iloc[0]["interaction_type"], "Hydrophobic")
        self.assertEqual(frame.iloc[0]["residue_name"], "LEU")
        self.assertEqual(frame.iloc[0]["residue_number"], 23)
        self.assertEqual(frame.iloc[0]["chain_id"], "A")
        self.assertEqual(frame.iloc[0]["ligand_atom_indices"], [1, 2])
        self.assertEqual(frame.iloc[0]["metadata"]["source"], "manual")

    def test_upsert_interaction_payload_summary_and_details(self) -> None:
        self.db.insert_dataframe(self.df)
        payload = {
            self.pose1: {
                "Hydrophobic": ["LEU23.A", "VAL31.A"],
                "HBDonor": {
                    "ASP160.A": [
                        {
                            "ligand_residue": "LIG1",
                            "distance": 2.8,
                            "indices": {"ligand": [2], "protein": [9]},
                            "parent_indices": {"ligand": [2], "protein": [2392]},
                            "metadata": {"score": 1.0},
                        }
                    ]
                },
            }
        }
        self.db.upsert_interaction_payload(payload, replace=True)

        summary = self.db.get_interaction_summary(pose_id=self.pose1)
        details = self.db.get_interaction_details(pose_id=self.pose1)

        self.assertIn(self.pose1, summary)
        self.assertEqual(summary[self.pose1]["Hydrophobic"], ["LEU23.A", "VAL31.A"])
        self.assertIn("HBDonor", details[self.pose1])
        event = details[self.pose1]["HBDonor"]["ASP160.A"][0]
        self.assertEqual(event["distance"], 2.8)
        self.assertEqual(event["indices"]["ligand"], [2])
        self.assertEqual(event["parent_indices"]["protein"], [2392])

    def test_query_poses_with_summary_interactions_dataframe(self) -> None:
        self.db.insert_dataframe(self.df)
        self.db.add_interaction(
            pose_id=self.pose1,
            interaction_type="Hydrophobic",
            residue_id="LEU23.A",
        )
        frame = self.db.query_poses(
            include_mol=False,
            include_interactions=True,
            interaction_mode="summary",
            as_dataframe=True,
        )

        self.assertEqual(frame.shape[0], 2)
        row = frame[frame["pose_id"] == self.pose1].iloc[0]
        self.assertEqual(row["interaction_summary"]["Hydrophobic"], ["LEU23.A"])

    def test_query_poses_with_detailed_interactions_records(self) -> None:
        self.db.insert_dataframe(self.df)
        self.db.add_interaction(
            pose_id=self.pose1,
            interaction_type="Hydrophobic",
            residue_id="LEU23.A",
            distance=4.5,
            ligand_atom_indices=[1],
            protein_atom_indices=[10],
        )

        pose = self.db.get_pose(
            pose_id=self.pose1,
            include_mol=False,
            include_interactions=True,
            interaction_mode="detailed",
        )
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertIn("Hydrophobic", pose.interaction_details)
        event = pose.interaction_details["Hydrophobic"]["LEU23.A"][0]
        self.assertEqual(event["distance"], 4.5)
        self.assertEqual(event["indices"]["protein"], [10])

    def test_query_scores_dataframe(self) -> None:
        self.db.insert_dataframe(self.df)
        frame = self.db.query_scores(as_dataframe=True, order_by=["pose_rank"])
        self.assertEqual(list(frame["pose_rank"]), [1, 2])
        self.assertEqual(frame.iloc[0]["pose_id"], self.pose1)

    def test_count_and_lists(self) -> None:
        self.db.insert_dataframe(self.df)
        self.assertEqual(self.db.count_poses(), 2)
        self.assertEqual(self.db.list_receptors(), ["1M17"])
        self.assertEqual(self.db.list_ligands(), ["erol"])
        self.assertEqual(self.db.list_engines(), ["qvina"])

    def test_interaction_fingerprint_binary_and_count(self) -> None:
        self.db.insert_dataframe(self.df)
        self.db.add_interaction(
            pose_id=self.pose1,
            interaction_type="Hydrophobic",
            residue_id="LEU23.A",
            occurrence_index=0,
        )
        self.db.add_interaction(
            pose_id=self.pose1,
            interaction_type="Hydrophobic",
            residue_id="LEU23.A",
            occurrence_index=1,
        )

        fp_count = self.db.interaction_fingerprint(mode="count", index_by="pose_id")
        fp_binary = self.db.interaction_fingerprint(mode="binary", index_by="pose_id")
        feature = "Hydrophobic::LEU23.A"

        self.assertEqual(int(fp_count.loc[self.pose1, feature]), 2)
        self.assertEqual(int(fp_binary.loc[self.pose1, feature]), 1)

    def test_interaction_fingerprint_invalid_mode_raises(self) -> None:
        self.db.insert_dataframe(self.df)
        with self.assertRaises(ValueError):
            self.db.interaction_fingerprint(mode="bad-mode")

    def test_delete_interactions_for_pose(self) -> None:
        self.db.insert_dataframe(self.df)
        self.db.add_interaction(
            pose_id=self.pose1,
            interaction_type="Hydrophobic",
            residue_id="LEU23.A",
        )
        deleted = self.db.delete_interactions_for_pose(pose_id=self.pose1)
        self.assertEqual(deleted, 1)
        self.assertEqual(self.db.query_interactions(as_dataframe=True).shape[0], 0)

    def test_delete_interactions_with_filter(self) -> None:
        self.db.insert_dataframe(self.df)
        self.db.add_interaction(
            pose_id=self.pose1,
            interaction_type="Hydrophobic",
            residue_id="LEU23.A",
        )
        self.db.add_interaction(
            pose_id=self.pose2,
            interaction_type="HBAcceptor",
            residue_id="ASP160.A",
        )
        deleted = self.db.delete_interactions(interaction_type="Hydrophobic")
        self.assertEqual(deleted, 1)

        frame = self.db.query_interactions(as_dataframe=True)
        self.assertEqual(frame.shape[0], 1)
        self.assertEqual(frame.iloc[0]["interaction_type"], "HBAcceptor")

    def test_delete_interactions_without_filter_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.db.delete_interactions()

    def test_delete_poses_with_filter(self) -> None:
        self.db.insert_dataframe(self.df)
        deleted = self.db.delete_poses(pose_rank=2)
        self.assertEqual(deleted, 1)
        self.assertEqual(self.db.count_poses(), 1)

    def test_delete_poses_without_filter_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.db.delete_poses()

    def test_flatten_payload_rejects_invalid_shape(self) -> None:
        with self.assertRaises(TypeError):
            self.db._flatten_one_pose_interaction_payload({"Hydrophobic": "LEU23.A"})

    def test_resolve_pose_db_id_errors(self) -> None:
        with self.assertRaises(ValueError):
            self.db._resolve_pose_db_id()

        with self.assertRaises(KeyError):
            self.db._resolve_pose_db_id(pose_id="missing")

    def test_from_dataframe_constructor(self) -> None:
        self.db.close()
        self.db = PoseDatabase.from_dataframe(self.db_path, self.df)
        self.assertEqual(self.db.count_poses(), 2)

    def test_insert_many_without_replace(self) -> None:
        self.db.insert_many(self.df.to_dict(orient="records"), replace=False)
        self.assertEqual(self.db.count_poses(), 2)

    def test_summarize(self) -> None:
        self.db.insert_dataframe(self.df)
        self.db.add_interaction(
            pose_id=self.pose1,
            interaction_type="Hydrophobic",
            residue_id="LEU23.A",
        )

        summary = self.db.summarize()
        self.assertEqual(summary.shape[0], 1)
        self.assertEqual(summary.iloc[0]["n_poses"], 2)
        self.assertAlmostEqual(summary.iloc[0]["best_affinity"], -8.2)

    def test_vacuum(self) -> None:
        self.db.insert_dataframe(self.df)
        self.db.vacuum()
        self.assertTrue(self.db_path.exists())


if __name__ == "__main__":
    unittest.main()
