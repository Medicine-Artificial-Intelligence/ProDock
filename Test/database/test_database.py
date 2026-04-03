# from __future__ import annotations

# import tempfile
# import unittest
# from pathlib import Path

# import pandas as pd
# from rdkit import Chem

# from prodock.database import PoseDatabase


# class TestPoseDatabase(unittest.TestCase):
#     def setUp(self) -> None:
#         self.tmpdir = tempfile.TemporaryDirectory()
#         self.db_path = Path(self.tmpdir.name) / "poses.sqlite"
#         self.db = PoseDatabase(self.db_path)

#     def tearDown(self) -> None:
#         self.db.close()
#         self.tmpdir.cleanup()

#     @staticmethod
#     def _make_mol(smiles: str):
#         mol = Chem.MolFromSmiles(smiles)
#         assert mol is not None
#         return mol

#     def test_upsert_and_get_pose_roundtrip(self) -> None:
#         mol = self._make_mol("CCO")
#         mol.SetProp("name", "ethanol")
#         pose_id = self.db.upsert_pose(
#             receptor_id="1M17",
#             ligand_id="erlotinib",
#             engine="qvina",
#             pose_rank=1,
#             affinity=-6.2,
#             mol=mol,
#             pose_metadata={"source": "unit"},
#             score_data={"cnn_score": 0.82},
#         )
#         self.assertGreater(pose_id, 0)

#         record = self.db.get_pose(
#             receptor_id="1M17",
#             ligand_id="erlotinib",
#             engine="qvina",
#             pose_rank=1,
#             include_mol=True,
#         )
#         self.assertIsNotNone(record)
#         assert record is not None
#         self.assertEqual(record.receptor_id, "1M17")
#         self.assertEqual(record.affinity, -6.2)
#         self.assertEqual(record.pose_metadata["source"], "unit")
#         self.assertEqual(record.score_data["cnn_score"], 0.82)
#         self.assertEqual(record.mol.GetProp("name"), "ethanol")

#     def test_insert_dataframe_and_filtered_queries(self) -> None:
#         df = pd.DataFrame(
#             [
#                 {
#                     "receptor_id": "1M17",
#                     "ligand_id": "erlotinib",
#                     "engine": "qvina",
#                     "pose_rank": 1,
#                     "affinity": -6.2,
#                     "mol": self._make_mol("CCO"),
#                 },
#                 {
#                     "receptor_id": "1M17",
#                     "ligand_id": "erlotinib",
#                     "engine": "qvina",
#                     "pose_rank": 2,
#                     "affinity": -5.9,
#                     "mol": self._make_mol("CCN"),
#                 },
#                 {
#                     "receptor_id": "1M17",
#                     "ligand_id": "gefitinib",
#                     "engine": "smina",
#                     "pose_rank": 1,
#                     "affinity": -7.3,
#                     "mol": self._make_mol("c1ccccc1"),
#                 },
#             ]
#         )
#         self.db.insert_dataframe(df)

#         top1 = self.db.query_poses(
#             receptor_id="1M17",
#             engine="qvina",
#             top_rank=1,
#             as_dataframe=True,
#             include_mol=False,
#         )
#         self.assertEqual(len(top1), 1)
#         self.assertEqual(float(top1.iloc[0]["affinity"]), -6.2)

#         filtered = self.db.query_scores(
#             receptor_id="1M17",
#             affinity_threshold=-6.0,
#             as_dataframe=True,
#         )
#         self.assertEqual(len(filtered), 2)
#         self.assertEqual(set(filtered["ligand_id"]), {"erlotinib", "gefitinib"})

#         self.assertEqual(self.db.count_poses(receptor_id="1M17"), 3)

#     def test_interaction_insert_query_and_delete(self) -> None:
#         pose_id = self.db.upsert_pose(
#             receptor_id="1M17",
#             ligand_id="erlotinib",
#             engine="vina",
#             pose_rank=1,
#             affinity=-6.1,
#             mol=self._make_mol("CCO"),
#         )
#         iid = self.db.add_interaction(
#             pose_id=pose_id,
#             interaction_type="Hydrophobic",
#             chain_id="A",
#             residue_name="LEU",
#             residue_number=718,
#             ligand_atom_indices=[1, 2],
#             protein_atom_indices=[10, 11],
#             distance=3.8,
#             metadata={"source": "prolif"},
#         )
#         self.assertGreater(iid, 0)

#         hits = self.db.query_interactions(
#             receptor_id="1M17",
#             interaction_type="Hydrophobic",
#         )
#         self.assertEqual(len(hits), 1)
#         self.assertEqual(hits[0].residue_name, "LEU")
#         self.assertEqual(hits[0].ligand_atom_indices, [1, 2])
#         self.assertEqual(hits[0].metadata["source"], "prolif")

#         deleted = self.db.delete_interactions(interaction_type="Hydrophobic")
#         self.assertEqual(deleted, 1)
#         self.assertEqual(len(self.db.query_interactions()), 0)

#     def test_summary_and_delete_pose(self) -> None:
#         self.db.upsert_pose(
#             receptor_id="1M17",
#             ligand_id="erlotinib",
#             engine="qvina",
#             pose_rank=1,
#             affinity=-6.2,
#             mol=self._make_mol("CCO"),
#         )
#         self.db.upsert_pose(
#             receptor_id="1M17",
#             ligand_id="erlotinib",
#             engine="qvina",
#             pose_rank=2,
#             affinity=-6.0,
#             mol=self._make_mol("CCN"),
#         )
#         summary = self.db.summarize()
#         self.assertEqual(len(summary), 1)
#         self.assertEqual(int(summary.iloc[0]["n_poses"]), 2)
#         self.assertEqual(float(summary.iloc[0]["best_affinity"]), -6.2)

#         deleted = self.db.delete_poses(receptor_id="1M17", top_rank=1)
#         self.assertEqual(deleted, 1)
#         self.assertEqual(self.db.count_poses(receptor_id="1M17"), 1)

#     def test_safe_delete_requires_filter(self) -> None:
#         with self.assertRaises(ValueError):
#             self.db.delete_poses()
#         with self.assertRaises(ValueError):
#             self.db.delete_interactions()


# if __name__ == "__main__":
#     unittest.main()
