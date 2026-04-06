from __future__ import annotations

import os
import unittest
from pathlib import Path

from prodock.postprocess.pose.core import PoseCrawler
from prodock.postprocess.interaction.core import extract_pose_table_interactions


@unittest.skipUnless(
    os.environ.get("RUN_REAL_INTERACTION_TEST", "") == "1",
    "Real ProLIF integration test is skipped by default.",
)
class TestExtractPoseTableInteractionsReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pose_root = Path("./Data/testcase/post")
        cls.receptor_map = {
            "1M17": Path("Data/testcase/Multi/1M17/filtered_protein/1M17.pdb"),
            "4WKQ": Path("Data/testcase/Multi/4WKQ/filtered_protein/4WKQ.pdb"),
        }

    def setUp(self) -> None:
        if not self.pose_root.exists():
            self.skipTest(f"Missing testcase pose folder: {self.pose_root}")

        missing = [rid for rid, path in self.receptor_map.items() if not path.exists()]
        if missing:
            self.skipTest(
                "Missing receptor test files for: " + ", ".join(sorted(missing))
            )

    def test_real_pose_table_interactions_from_posecrawler(self) -> None:
        crawler = PoseCrawler([str(self.pose_root)])
        poses = crawler.crawl_mols(backend="obabel")

        self.assertFalse(poses.empty)
        self.assertIn("mol", poses.columns)
        self.assertIn("receptor_id", poses.columns)

        result = extract_pose_table_interactions(
            poses=poses,
            receptor_pdb_by_id=self.receptor_map,
            batch_size=1,
            progress=False,
            n_jobs=1,
            include_fingerprint_columns=True,
            include_interaction_events=True,
            include_bitvectors=False,
            include_countvectors=False,
            fail_fast=True,
            ultra_safe=True,
        )

        merged_df = result.merged_df
        interaction_df = result.interaction_df
        summary_df = result.summary_df

        self.assertFalse(merged_df.empty)
        self.assertEqual(len(merged_df), len(poses))
        self.assertIn("pose_id", merged_df.columns)

        self.assertFalse(interaction_df.empty)
        self.assertIn("pose_id", interaction_df.columns)
        self.assertIn("interaction_events_json", interaction_df.columns)
        self.assertIn("has_interactions", interaction_df.columns)

        self.assertFalse(summary_df.empty)
        self.assertIn("pose_id", summary_df.columns)
        self.assertIn("interaction_compact_json", summary_df.columns)
        self.assertIn("interaction_detail_json", summary_df.columns)
        self.assertIn("has_interactions", summary_df.columns)

        self.assertEqual(result.errors, [])
