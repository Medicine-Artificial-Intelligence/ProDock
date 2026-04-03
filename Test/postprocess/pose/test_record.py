from __future__ import annotations

import unittest
from pathlib import Path
from dataclasses import FrozenInstanceError

from prodock.postprocess.pose.record import PoseRecord


class TestPoseRecord(unittest.TestCase):
    def test_pose_record_fields(self) -> None:
        record = PoseRecord(
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="vina",
            pose_rank=1,
            affinity=-7.2,
            source_file=Path(
                "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
            ),
        )

        self.assertEqual(record.receptor_id, "1M17")
        self.assertEqual(record.ligand_id, "erlotinib")
        self.assertEqual(record.engine, "vina")
        self.assertEqual(record.pose_rank, 1)
        self.assertEqual(record.affinity, -7.2)
        self.assertEqual(
            record.source_file,
            Path("Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"),
        )

    def test_pose_record_allows_none_receptor_and_affinity(self) -> None:
        record = PoseRecord(
            receptor_id=None,
            ligand_id="erlotinib",
            engine="smina",
            pose_rank=2,
            affinity=None,
            source_file=Path(
                "Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"
            ),
        )

        self.assertIsNone(record.receptor_id)
        self.assertIsNone(record.affinity)
        self.assertEqual(record.engine, "smina")

    def test_pose_record_is_frozen(self) -> None:
        record = PoseRecord(
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="qvina",
            pose_rank=1,
            affinity=-7.1,
            source_file=Path(
                "Data/testcase/post/1M17/results/docked/qvina/erlotinib_docked.pdbqt"
            ),
        )

        with self.assertRaises(FrozenInstanceError):
            record.engine = "vina"

    def test_pose_record_equality(self) -> None:
        a = PoseRecord(
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="vina",
            pose_rank=1,
            affinity=-7.2,
            source_file=Path(
                "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
            ),
        )
        b = PoseRecord(
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="vina",
            pose_rank=1,
            affinity=-7.2,
            source_file=Path(
                "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
            ),
        )

        self.assertEqual(a, b)

    def test_pose_record_repr_contains_useful_fields(self) -> None:
        record = PoseRecord(
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="vina",
            pose_rank=1,
            affinity=-7.2,
            source_file=Path(
                "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
            ),
        )

        text = repr(record)
        self.assertIn("1M17", text)
        self.assertIn("erlotinib", text)
        self.assertIn("vina", text)


if __name__ == "__main__":
    unittest.main()
