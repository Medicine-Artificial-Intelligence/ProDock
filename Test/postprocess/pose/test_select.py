from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from prodock.postprocess.pose.record import PoseRecord
from prodock.postprocess.pose.select import (
    best_pose_per_group,
    pose_mols_to_dataframe,
    poses_to_dataframe,
)


class TestPosesToDataFrame(unittest.TestCase):
    def test_poses_to_dataframe_basic(self) -> None:
        records = [
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="vina",
                pose_rank=1,
                affinity=-7.2,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
                ),
            ),
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="smina",
                pose_rank=1,
                affinity=-7.4,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"
                ),
            ),
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="qvina",
                pose_rank=1,
                affinity=-7.1,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/qvina/erlotinib_docked.pdbqt"
                ),
            ),
        ]

        df = poses_to_dataframe(records)

        self.assertEqual(
            list(df.columns),
            ["receptor_id", "ligand_id", "engine", "pose_rank", "affinity"],
        )
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0]["receptor_id"], "1M17")
        self.assertEqual(df.iloc[0]["ligand_id"], "erlotinib")
        self.assertEqual(df.iloc[0]["engine"], "vina")
        self.assertEqual(df.iloc[0]["pose_rank"], 1)
        self.assertEqual(df.iloc[0]["affinity"], -7.2)

    def test_poses_to_dataframe_empty(self) -> None:
        df = poses_to_dataframe([])

        self.assertEqual(
            list(df.columns),
            ["receptor_id", "ligand_id", "engine", "pose_rank", "affinity"],
        )
        self.assertTrue(df.empty)


class TestPoseMolsToDataFrame(unittest.TestCase):
    def test_pose_mols_to_dataframe_basic(self) -> None:
        rows = [
            {
                "receptor_id": "1M17",
                "ligand_id": "erlotinib",
                "engine": "vina",
                "pose_rank": 1,
                "affinity": -7.2,
                "mol": "mol_vina",
            },
            {
                "receptor_id": "1M17",
                "ligand_id": "erlotinib",
                "engine": "smina",
                "pose_rank": 1,
                "affinity": -7.4,
                "mol": "mol_smina",
            },
            {
                "receptor_id": "1M17",
                "ligand_id": "erlotinib",
                "engine": "qvina",
                "pose_rank": 1,
                "affinity": -7.1,
                "mol": "mol_qvina",
            },
        ]

        df = pose_mols_to_dataframe(rows)

        self.assertEqual(
            list(df.columns),
            ["receptor_id", "ligand_id", "engine", "pose_rank", "affinity", "mol"],
        )
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[1]["engine"], "smina")
        self.assertEqual(df.iloc[1]["mol"], "mol_smina")

    def test_pose_mols_to_dataframe_empty(self) -> None:
        df = pose_mols_to_dataframe([])

        self.assertEqual(
            list(df.columns),
            ["receptor_id", "ligand_id", "engine", "pose_rank", "affinity", "mol"],
        )
        self.assertTrue(df.empty)


class TestBestPosePerGroup(unittest.TestCase):
    def test_best_pose_per_group_from_records(self) -> None:
        records = [
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="vina",
                pose_rank=2,
                affinity=-6.8,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
                ),
            ),
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="vina",
                pose_rank=1,
                affinity=-7.2,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
                ),
            ),
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="smina",
                pose_rank=2,
                affinity=-7.0,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"
                ),
            ),
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="smina",
                pose_rank=1,
                affinity=-7.4,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"
                ),
            ),
        ]

        best = best_pose_per_group(records)

        self.assertEqual(len(best), 2)

        vina_row = best[best["engine"] == "vina"].iloc[0]
        smina_row = best[best["engine"] == "smina"].iloc[0]

        self.assertEqual(vina_row["pose_rank"], 1)
        self.assertEqual(vina_row["affinity"], -7.2)

        self.assertEqual(smina_row["pose_rank"], 1)
        self.assertEqual(smina_row["affinity"], -7.4)

    def test_best_pose_per_group_from_dataframe(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "qvina",
                    "pose_rank": 2,
                    "affinity": -6.9,
                },
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "qvina",
                    "pose_rank": 1,
                    "affinity": -7.1,
                },
                {
                    "receptor_id": "1M17",
                    "ligand_id": "gefitinib",
                    "engine": "qvina",
                    "pose_rank": 1,
                    "affinity": -8.0,
                },
            ]
        )

        best = best_pose_per_group(df)

        self.assertEqual(len(best), 2)

        erlotinib_row = best[best["ligand_id"] == "erlotinib"].iloc[0]
        gefitinib_row = best[best["ligand_id"] == "gefitinib"].iloc[0]

        self.assertEqual(erlotinib_row["pose_rank"], 1)
        self.assertEqual(erlotinib_row["affinity"], -7.1)
        self.assertEqual(gefitinib_row["affinity"], -8.0)

    def test_best_pose_per_group_missing_affinity_goes_last(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "vina",
                    "pose_rank": 1,
                    "affinity": None,
                },
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "vina",
                    "pose_rank": 2,
                    "affinity": -6.5,
                },
            ]
        )

        best = best_pose_per_group(df)

        self.assertEqual(len(best), 1)
        self.assertEqual(best.iloc[0]["pose_rank"], 2)
        self.assertEqual(best.iloc[0]["affinity"], -6.5)

    def test_best_pose_per_group_tie_breaks_by_pose_rank(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "vina",
                    "pose_rank": 3,
                    "affinity": -7.2,
                },
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "vina",
                    "pose_rank": 1,
                    "affinity": -7.2,
                },
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "vina",
                    "pose_rank": 2,
                    "affinity": -7.2,
                },
            ]
        )

        best = best_pose_per_group(df)

        self.assertEqual(len(best), 1)
        self.assertEqual(best.iloc[0]["pose_rank"], 1)
        self.assertEqual(best.iloc[0]["affinity"], -7.2)

    def test_best_pose_per_group_custom_grouping(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "vina",
                    "pose_rank": 1,
                    "affinity": -7.2,
                },
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "smina",
                    "pose_rank": 1,
                    "affinity": -7.4,
                },
                {
                    "receptor_id": "1M17",
                    "ligand_id": "erlotinib",
                    "engine": "qvina",
                    "pose_rank": 1,
                    "affinity": -7.1,
                },
            ]
        )

        best = best_pose_per_group(df, by=("receptor_id", "ligand_id"))

        self.assertEqual(len(best), 1)
        self.assertEqual(best.iloc[0]["engine"], "smina")
        self.assertEqual(best.iloc[0]["affinity"], -7.4)

    def test_best_pose_per_group_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            columns=["receptor_id", "ligand_id", "engine", "pose_rank", "affinity"]
        )

        best = best_pose_per_group(df)

        self.assertTrue(best.empty)
        self.assertEqual(
            list(best.columns),
            ["receptor_id", "ligand_id", "engine", "pose_rank", "affinity"],
        )


if __name__ == "__main__":
    unittest.main()
