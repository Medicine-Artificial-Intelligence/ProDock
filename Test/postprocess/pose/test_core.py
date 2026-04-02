from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from prodock.postprocess.pose import core
from prodock.postprocess.pose.record import PoseRecord


class TestPoseCrawler(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_build_pose_records = core.build_pose_records
        self._orig_build_pose_mol_rows = core.build_pose_mol_rows
        self._orig_convert_pose_tree = core.convert_pose_tree

        self.sample_records = [
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

        self.sample_mol_rows = [
            {
                "receptor_id": "1M17",
                "ligand_id": "erlotinib",
                "engine": "vina",
                "pose_rank": 2,
                "affinity": -6.8,
                "mol": "mol_vina_2",
            },
            {
                "receptor_id": "1M17",
                "ligand_id": "erlotinib",
                "engine": "vina",
                "pose_rank": 1,
                "affinity": -7.2,
                "mol": "mol_vina_1",
            },
            {
                "receptor_id": "1M17",
                "ligand_id": "erlotinib",
                "engine": "smina",
                "pose_rank": 1,
                "affinity": -7.4,
                "mol": "mol_smina_1",
            },
            {
                "receptor_id": "1M17",
                "ligand_id": "erlotinib",
                "engine": "qvina",
                "pose_rank": 1,
                "affinity": -7.1,
                "mol": "mol_qvina_1",
            },
        ]

    def tearDown(self) -> None:
        core.build_pose_records = self._orig_build_pose_records
        core.build_pose_mol_rows = self._orig_build_pose_mol_rows
        core.convert_pose_tree = self._orig_convert_pose_tree

    def test_records(self) -> None:
        calls = []

        def fake_build_pose_records(roots, engine=None, recursive=True):
            calls.append((roots, engine, recursive))
            return list(self.sample_records)

        core.build_pose_records = fake_build_pose_records

        crawler = core.PoseCrawler(["Data/testcase/post"], engine=None, recursive=True)
        records = crawler.records()

        self.assertEqual(len(records), 4)
        self.assertEqual(records[0].receptor_id, "1M17")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], ["Data/testcase/post"])
        self.assertIsNone(calls[0][1])
        self.assertTrue(calls[0][2])

    def test_crawl(self) -> None:
        def fake_build_pose_records(roots, engine=None, recursive=True):
            return list(self.sample_records)

        core.build_pose_records = fake_build_pose_records

        crawler = core.PoseCrawler(["Data/testcase/post"])
        df = crawler.crawl()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(
            list(df.columns),
            ["receptor_id", "ligand_id", "engine", "pose_rank", "affinity"],
        )
        self.assertEqual(len(df), 4)

    def test_crawl_mols(self) -> None:
        calls = []

        def fake_build_pose_mol_rows(
            roots,
            engine=None,
            recursive=True,
            backend="obabel",
            sanitize=True,
            remove_hs=False,
            save_sdf=False,
            overwrite_sdf=False,
        ):
            calls.append(
                (
                    roots,
                    engine,
                    recursive,
                    backend,
                    sanitize,
                    remove_hs,
                    save_sdf,
                    overwrite_sdf,
                )
            )
            return list(self.sample_mol_rows)

        core.build_pose_mol_rows = fake_build_pose_mol_rows

        crawler = core.PoseCrawler(
            ["Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"],
            engine="smina",
        )
        df = crawler.crawl_mols(
            backend="obabel",
            sanitize=False,
            remove_hs=True,
            save_sdf=True,
            overwrite_sdf=True,
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(
            list(df.columns),
            ["receptor_id", "ligand_id", "engine", "pose_rank", "affinity", "mol"],
        )
        self.assertEqual(len(df), 4)
        self.assertEqual(
            calls[0][0],
            ["Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"],
        )
        self.assertEqual(calls[0][1], "smina")
        self.assertEqual(calls[0][3], "obabel")
        self.assertFalse(calls[0][4])
        self.assertTrue(calls[0][5])
        self.assertTrue(calls[0][6])
        self.assertTrue(calls[0][7])

    def test_best(self) -> None:
        def fake_build_pose_records(roots, engine=None, recursive=True):
            return list(self.sample_records)

        core.build_pose_records = fake_build_pose_records

        crawler = core.PoseCrawler(["Data/testcase/post"])
        best_df = crawler.best()

        self.assertEqual(len(best_df), 3)

        vina_row = best_df[best_df["engine"] == "vina"].iloc[0]
        smina_row = best_df[best_df["engine"] == "smina"].iloc[0]
        qvina_row = best_df[best_df["engine"] == "qvina"].iloc[0]

        self.assertEqual(vina_row["pose_rank"], 1)
        self.assertEqual(vina_row["affinity"], -7.2)
        self.assertEqual(smina_row["affinity"], -7.4)
        self.assertEqual(qvina_row["affinity"], -7.1)

    def test_best_mols(self) -> None:
        def fake_build_pose_mol_rows(
            roots,
            engine=None,
            recursive=True,
            backend="obabel",
            sanitize=True,
            remove_hs=False,
            save_sdf=False,
            overwrite_sdf=False,
        ):
            return list(self.sample_mol_rows)

        core.build_pose_mol_rows = fake_build_pose_mol_rows

        crawler = core.PoseCrawler(["Data/testcase/post"])
        best_df = crawler.best_mols()

        self.assertEqual(len(best_df), 3)

        vina_row = best_df[best_df["engine"] == "vina"].iloc[0]
        self.assertEqual(vina_row["pose_rank"], 1)
        self.assertEqual(vina_row["affinity"], -7.2)
        self.assertEqual(vina_row["mol"], "mol_vina_1")

    def test_best_mols_custom_grouping(self) -> None:
        def fake_build_pose_mol_rows(
            roots,
            engine=None,
            recursive=True,
            backend="obabel",
            sanitize=True,
            remove_hs=False,
            save_sdf=False,
            overwrite_sdf=False,
        ):
            return list(self.sample_mol_rows)

        core.build_pose_mol_rows = fake_build_pose_mol_rows

        crawler = core.PoseCrawler(["Data/testcase/post"])
        best_df = crawler.best_mols(by=("receptor_id", "ligand_id"))

        self.assertEqual(len(best_df), 1)
        self.assertEqual(best_df.iloc[0]["engine"], "smina")
        self.assertEqual(best_df.iloc[0]["affinity"], -7.4)

    def test_convert(self) -> None:
        calls = []
        expected = [
            Path("Data/testcase/post/converted_sdf/erlotinib_docked.sdf"),
            Path("Data/testcase/post/converted_sdf/erlotinib_docked_2.sdf"),
            Path("Data/testcase/post/converted_sdf/erlotinib_docked_3.sdf"),
        ]

        def fake_convert_pose_tree(
            roots,
            engine=None,
            recursive=True,
            backend="obabel",
            overwrite=False,
            out_dir=None,
        ):
            calls.append((roots, engine, recursive, backend, overwrite, out_dir))
            return list(expected)

        core.convert_pose_tree = fake_convert_pose_tree

        crawler = core.PoseCrawler(["Data/testcase/post"])
        outputs = crawler.convert(
            backend="obabel",
            overwrite=True,
            out_dir="Data/testcase/post/converted_sdf",
        )

        self.assertEqual(outputs, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], ["Data/testcase/post"])
        self.assertIsNone(calls[0][1])
        self.assertTrue(calls[0][2])
        self.assertEqual(calls[0][3], "obabel")
        self.assertTrue(calls[0][4])
        self.assertEqual(calls[0][5], "Data/testcase/post/converted_sdf")


class TestCoreWrappers(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_build_pose_records = core.build_pose_records
        self._orig_build_pose_mol_rows = core.build_pose_mol_rows

        self.sample_records = [
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

        self.sample_mol_rows = [
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

    def tearDown(self) -> None:
        core.build_pose_records = self._orig_build_pose_records
        core.build_pose_mol_rows = self._orig_build_pose_mol_rows

    def test_crawl_poses(self) -> None:
        def fake_build_pose_records(roots, engine=None, recursive=True):
            return list(self.sample_records)

        core.build_pose_records = fake_build_pose_records

        df = core.crawl_poses(["Data/testcase/post"])

        self.assertEqual(len(df), 3)
        self.assertEqual(
            list(df.columns),
            ["receptor_id", "ligand_id", "engine", "pose_rank", "affinity"],
        )

    def test_crawl_pose_mols(self) -> None:
        def fake_build_pose_mol_rows(
            roots,
            engine=None,
            recursive=True,
            backend="obabel",
            sanitize=True,
            remove_hs=False,
            save_sdf=False,
            overwrite_sdf=False,
        ):
            return list(self.sample_mol_rows)

        core.build_pose_mol_rows = fake_build_pose_mol_rows

        df = core.crawl_pose_mols(
            ["Data/testcase/post/1M17/results/docked/qvina/erlotinib_docked.pdbqt"],
            engine="qvina",
            save_sdf=True,
        )

        self.assertEqual(len(df), 3)
        self.assertEqual(
            list(df.columns),
            ["receptor_id", "ligand_id", "engine", "pose_rank", "affinity", "mol"],
        )

    def test_select_best_poses(self) -> None:
        def fake_build_pose_records(roots, engine=None, recursive=True):
            return list(self.sample_records)

        core.build_pose_records = fake_build_pose_records

        df = core.select_best_poses(["Data/testcase/post"])

        self.assertEqual(len(df), 3)
        self.assertIn("engine", df.columns)

    def test_select_best_pose_mols(self) -> None:
        def fake_build_pose_mol_rows(
            roots,
            engine=None,
            recursive=True,
            backend="obabel",
            sanitize=True,
            remove_hs=False,
            save_sdf=False,
            overwrite_sdf=False,
        ):
            return list(self.sample_mol_rows)

        core.build_pose_mol_rows = fake_build_pose_mol_rows

        df = core.select_best_pose_mols(["Data/testcase/post"])

        self.assertEqual(len(df), 3)
        self.assertIn("mol", df.columns)


if __name__ == "__main__":
    unittest.main()
