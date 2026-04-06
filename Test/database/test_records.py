from __future__ import annotations

import unittest

from rdkit import Chem

from prodock.database.records import InteractionRecord, PoseRecord, ScoreRecord


class TestPoseRecord(unittest.TestCase):
    def test_pose_key_uses_pose_id_when_present(self) -> None:
        record = PoseRecord(
            pose_db_id=1,
            pose_id="custom_pose_id",
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="qvina",
            pose_rank=1,
            affinity=-6.2,
            mol=None,
            pose_metadata={},
            score_data={},
            score_metadata={},
        )
        self.assertEqual(record.pose_key, "custom_pose_id")

    def test_pose_key_falls_back_to_generated_key(self) -> None:
        record = PoseRecord(
            pose_db_id=1,
            pose_id=None,
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="qvina",
            pose_rank=1,
            affinity=-6.2,
            mol=None,
            pose_metadata={},
            score_data={},
            score_metadata={},
        )
        self.assertEqual(record.pose_key, "1M17__erlotinib__qvina__pose1")

    def test_pose_record_defaults(self) -> None:
        record = PoseRecord(
            pose_db_id=1,
            pose_id=None,
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="vina",
            pose_rank=2,
            affinity=None,
            mol=None,
            pose_metadata={},
            score_data={},
            score_metadata={},
        )
        self.assertEqual(record.interaction_summary, {})
        self.assertEqual(record.interaction_details, {})
        self.assertEqual(record.created_at, "")

    def test_pose_record_stores_rdkit_mol(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)

        record = PoseRecord(
            pose_db_id=2,
            pose_id=None,
            receptor_id="1M17",
            ligand_id="ethanol",
            engine="vina",
            pose_rank=1,
            affinity=-4.5,
            mol=mol,
            pose_metadata={"source": "test"},
            score_data={"affinity": -4.5},
            score_metadata={},
        )
        self.assertIsNotNone(record.mol)
        self.assertEqual(Chem.MolToSmiles(record.mol), "CCO")


class TestScoreRecord(unittest.TestCase):
    def test_pose_key_uses_pose_id_when_present(self) -> None:
        record = ScoreRecord(
            pose_db_id=1,
            pose_id="stored_pose_key",
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="vina",
            pose_rank=2,
            affinity=-7.1,
            score_data={"affinity": -7.1},
            metadata={},
        )
        self.assertEqual(record.pose_key, "stored_pose_key")

    def test_pose_key_falls_back_to_generated_key(self) -> None:
        record = ScoreRecord(
            pose_db_id=1,
            pose_id=None,
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="vina",
            pose_rank=2,
            affinity=-7.1,
            score_data={"affinity": -7.1},
            metadata={},
        )
        self.assertEqual(record.pose_key, "1M17__erlotinib__vina__pose2")

    def test_score_record_fields(self) -> None:
        record = ScoreRecord(
            pose_db_id=5,
            pose_id=None,
            receptor_id="4WKQ",
            ligand_id="ligA",
            engine="smina",
            pose_rank=3,
            affinity=-8.4,
            score_data={"affinity": -8.4, "cnn_pose": 0.91},
            metadata={"source": "log"},
        )
        self.assertEqual(record.pose_db_id, 5)
        self.assertEqual(record.score_data["cnn_pose"], 0.91)
        self.assertEqual(record.metadata["source"], "log")


class TestInteractionRecord(unittest.TestCase):
    def test_pose_key_uses_pose_id_when_present(self) -> None:
        record = InteractionRecord(
            interaction_id=1,
            pose_db_id=10,
            pose_id="stored_pose_id",
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="qvina",
            pose_rank=1,
            interaction_type="Hydrophobic",
            chain_id="A",
            residue_name="LEU",
            residue_number=149,
            residue_id="LEU149.A",
            ligand_residue="LIG1",
            occurrence_index=0,
            ligand_atom_indices=[2],
            protein_atom_indices=[9],
            ligand_parent_atom_indices=[2],
            protein_parent_atom_indices=[2392],
            distance=4.49,
            angle=None,
            metadata={},
            created_at="2026-04-02 10:00:00",
        )
        self.assertEqual(record.pose_key, "stored_pose_id")

    def test_pose_key_falls_back_to_generated_key(self) -> None:
        record = InteractionRecord(
            interaction_id=1,
            pose_db_id=10,
            pose_id=None,
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="qvina",
            pose_rank=1,
            interaction_type="Hydrophobic",
            chain_id="A",
            residue_name="LEU",
            residue_number=149,
            residue_id="LEU149.A",
            ligand_residue="LIG1",
            occurrence_index=0,
            ligand_atom_indices=[2],
            protein_atom_indices=[9],
            ligand_parent_atom_indices=[2],
            protein_parent_atom_indices=[2392],
            distance=4.49,
            angle=None,
            metadata={},
            created_at="2026-04-02 10:00:00",
        )
        self.assertEqual(record.pose_key, "1M17__erlotinib__qvina__pose1")

    def test_interaction_record_fields(self) -> None:
        record = InteractionRecord(
            interaction_id=2,
            pose_db_id=11,
            pose_id=None,
            receptor_id="4WKQ",
            ligand_id="ligB",
            engine="vina",
            pose_rank=2,
            interaction_type="HBDonor",
            chain_id="B",
            residue_name="ASP",
            residue_number=160,
            residue_id="ASP160.B",
            ligand_residue="LIG1",
            occurrence_index=1,
            ligand_atom_indices=[1, 3],
            protein_atom_indices=[10],
            ligand_parent_atom_indices=[1, 3],
            protein_parent_atom_indices=[2400],
            distance=2.8,
            angle=145.0,
            metadata={"strength": "strong"},
            created_at="2026-04-02 11:00:00",
        )
        self.assertEqual(record.interaction_type, "HBDonor")
        self.assertEqual(record.chain_id, "B")
        self.assertEqual(record.residue_id, "ASP160.B")
        self.assertEqual(record.ligand_atom_indices, [1, 3])
        self.assertEqual(record.protein_parent_atom_indices, [2400])
        self.assertEqual(record.metadata["strength"], "strong")

    def test_interaction_record_allows_optional_fields(self) -> None:
        record = InteractionRecord(
            interaction_id=3,
            pose_db_id=12,
            pose_id=None,
            receptor_id="X",
            ligand_id="Y",
            engine="vina",
            pose_rank=1,
            interaction_type="VdWContact",
            chain_id=None,
            residue_name=None,
            residue_number=None,
            residue_id=None,
            ligand_residue=None,
            occurrence_index=0,
            ligand_atom_indices=[],
            protein_atom_indices=[],
            ligand_parent_atom_indices=[],
            protein_parent_atom_indices=[],
            distance=None,
            angle=None,
            metadata={},
            created_at="",
        )
        self.assertIsNone(record.chain_id)
        self.assertIsNone(record.residue_name)
        self.assertIsNone(record.distance)
        self.assertEqual(record.ligand_atom_indices, [])


if __name__ == "__main__":
    unittest.main()
