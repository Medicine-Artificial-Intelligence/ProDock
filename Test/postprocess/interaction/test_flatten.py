from __future__ import annotations

import json
import unittest

import pandas as pd

from prodock.postprocess.interaction.flatten import (
    POSE_EVENT_COLUMNS,
    POSE_SUMMARY_COLUMNS,
    _iter_frame_interactions,
    _jsonable,
    _record_to_detail_entry,
    _safe_string,
    build_pose_interaction_table,
    build_pose_summary_table,
    flatten_ifp,
    summary_table_to_dict,
)


class FakeInteractionEntry:
    def __init__(self, ligand, protein, interaction, metadata=None):
        self.ligand = ligand
        self.protein = protein
        self.interaction = interaction
        self.metadata = metadata or {}


class FakeFrameWithInteractionsMethod:
    def __init__(self, entries):
        self._entries = list(entries)

    def interactions(self):
        return list(self._entries)


class TestHelpers(unittest.TestCase):
    def test_safe_string_none(self) -> None:
        self.assertEqual(_safe_string(None), "")

    def test_safe_string_regular_value(self) -> None:
        self.assertEqual(_safe_string(123), "123")

    def test_jsonable_scalar(self) -> None:
        self.assertEqual(_jsonable("abc"), "abc")
        self.assertEqual(_jsonable(3), 3)
        self.assertEqual(_jsonable(None), None)

    def test_jsonable_nested_containers(self) -> None:
        value = {
            "a": (1, 2),
            "b": [{"x": {3, 1}}],
        }
        out = _jsonable(value)
        self.assertEqual(out["a"], [1, 2])
        self.assertEqual(out["b"][0]["x"], [1, 3])

    def test_jsonable_fallback_to_string(self) -> None:
        class Dummy:
            def __str__(self) -> str:
                return "dummy"

        self.assertEqual(_jsonable(Dummy()), "dummy")

    def test_iter_frame_interactions_from_method(self) -> None:
        frame = FakeFrameWithInteractionsMethod(
            [
                FakeInteractionEntry(
                    ligand="LIG1.G",
                    protein="ASP160.A",
                    interaction="Hydrophobic",
                    metadata={"distance": 4.2},
                )
            ]
        )
        rows = list(_iter_frame_interactions(frame))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ligand_residue"], "LIG1.G")
        self.assertEqual(rows[0]["protein_residue"], "ASP160.A")
        self.assertEqual(rows[0]["interaction"], "Hydrophobic")
        self.assertEqual(rows[0]["metadata"]["distance"], 4.2)

    def test_iter_frame_interactions_from_mapping(self) -> None:
        frame = {
            ("LIG1.G", "LEU149.A"): {
                "Hydrophobic": [
                    {"distance": 4.4},
                    {"distance": 4.6},
                ]
            }
        }
        rows = list(_iter_frame_interactions(frame))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["ligand_residue"], "LIG1.G")
        self.assertEqual(rows[0]["protein_residue"], "LEU149.A")
        self.assertEqual(rows[0]["interaction"], "Hydrophobic")

    def test_iter_frame_interactions_mapping_with_bad_residue_pair(self) -> None:
        frame = {
            "bad_key": {
                "VdWContact": [{"distance": 3.1}],
            }
        }
        rows = list(_iter_frame_interactions(frame))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ligand_residue"], "")
        self.assertEqual(rows[0]["protein_residue"], "")

    def test_iter_frame_interactions_skips_non_mapping_interaction_map(self) -> None:
        frame = {("LIG1.G", "LEU149.A"): ["bad"]}
        rows = list(_iter_frame_interactions(frame))
        self.assertEqual(rows, [])

    def test_record_to_detail_entry_basic(self) -> None:
        record = {
            "protein_residue": "ASP160.A",
            "ligand_residue": "LIG1.G",
            "distance": 3.5,
            "angle": 120.0,
            "ligand_atom_indices": (1, 2),
            "protein_atom_indices": (9,),
            "ligand_parent_indices": (1, 2),
            "protein_parent_indices": (99,),
        }
        entry = _record_to_detail_entry(record)
        self.assertEqual(entry["protein_residue"], "ASP160.A")
        self.assertEqual(entry["indices"]["ligand"], [1, 2])
        self.assertEqual(entry["parent_indices"]["protein"], [99])
        self.assertNotIn("metadata", entry)

    def test_record_to_detail_entry_with_metadata(self) -> None:
        record = {
            "protein_residue": "ASP160.A",
            "ligand_residue": "LIG1.G",
            "metadata": {"extra": {2, 1}},
        }
        entry = _record_to_detail_entry(record)
        self.assertIn("metadata", entry)
        self.assertEqual(entry["metadata"]["extra"], [1, 2])


class TestFlattenIFP(unittest.TestCase):
    def test_flatten_ifp_empty(self) -> None:
        df = flatten_ifp({})
        self.assertTrue(df.empty)
        self.assertEqual(
            list(df.columns),
            [
                "frame",
                "mol_index",
                "mol_name",
                "mol",
                "ligand_residue",
                "protein_residue",
                "interaction",
                "occurrence_index",
                "ligand_atom_indices",
                "protein_atom_indices",
                "ligand_parent_indices",
                "protein_parent_indices",
                "distance",
                "angle",
                "metadata",
                "metadata_json",
            ],
        )

    def test_flatten_ifp_with_mapping_payload(self) -> None:
        ifp = {
            0: {
                ("LIG1.G", "ASP160.A"): {
                    "Hydrophobic": [
                        {
                            "indices": {"ligand": [2], "protein": [9]},
                            "parent_indices": {"ligand": [2], "protein": [2392]},
                            "distance": 4.49,
                            "angle": 110.0,
                        }
                    ]
                }
            }
        }
        mols = ["fake_mol_0"]
        df = flatten_ifp(ifp, mol_names=["pose_1"], mols=mols)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "frame"], 0)
        self.assertEqual(df.loc[0, "mol_name"], "pose_1")
        self.assertEqual(df.loc[0, "mol"], "fake_mol_0")
        self.assertEqual(df.loc[0, "interaction"], "Hydrophobic")
        self.assertEqual(df.loc[0, "ligand_atom_indices"], (2,))
        self.assertEqual(df.loc[0, "protein_parent_indices"], (2392,))
        self.assertEqual(df.loc[0, "distance"], 4.49)
        self.assertIsInstance(df.loc[0, "metadata_json"], str)

    def test_flatten_ifp_with_method_payload(self) -> None:
        ifp = {
            1: FakeFrameWithInteractionsMethod(
                [
                    FakeInteractionEntry(
                        ligand="LIG2.G",
                        protein="LEU149.A",
                        interaction="VdWContact",
                        metadata={
                            "indices": {"ligand": [3], "protein": [10]},
                            "parent_indices": {"ligand": [3], "protein": [2400]},
                            "distance": 2.7,
                        },
                    )
                ]
            )
        }
        df = flatten_ifp(ifp)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "mol_name"], "mol_0001")
        self.assertEqual(df.loc[0, "interaction"], "VdWContact")
        self.assertEqual(df.loc[0, "ligand_atom_indices"], (3,))
        self.assertEqual(df.loc[0, "protein_atom_indices"], (10,))

    def test_flatten_ifp_fallback_name_and_missing_mol(self) -> None:
        ifp = {2: {("LIG1.G", "PHE28.A"): {"PiStacking": [{}]}}}
        df = flatten_ifp(ifp, mol_names=["only_zero"])
        self.assertEqual(df.loc[0, "mol_name"], "mol_0002")
        self.assertIsNone(df.loc[0, "mol"])


class TestPoseTables(unittest.TestCase):
    def make_events_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pose_id": "pose_1",
                    "ligand_residue": "LIG1.G",
                    "protein_residue": "ASP160.A",
                    "interaction": "Hydrophobic",
                    "occurrence_index": 0,
                    "ligand_atom_indices": (2,),
                    "protein_atom_indices": (9,),
                    "ligand_parent_indices": (2,),
                    "protein_parent_indices": (2392,),
                    "distance": 4.49,
                    "angle": None,
                    "metadata": {"distance": 4.49},
                    "metadata_json": json.dumps({"distance": 4.49}, sort_keys=True),
                },
                {
                    "pose_id": "pose_1",
                    "ligand_residue": "LIG1.G",
                    "protein_residue": "LEU149.A",
                    "interaction": "Hydrophobic",
                    "occurrence_index": 1,
                    "ligand_atom_indices": (3,),
                    "protein_atom_indices": (10,),
                    "ligand_parent_indices": (3,),
                    "protein_parent_indices": (2400,),
                    "distance": 4.10,
                    "angle": None,
                    "metadata": {"distance": 4.10},
                    "metadata_json": json.dumps({"distance": 4.10}, sort_keys=True),
                },
                {
                    "pose_id": "pose_1",
                    "ligand_residue": "LIG1.G",
                    "protein_residue": "ASP160.A",
                    "interaction": "VdWContact",
                    "occurrence_index": 2,
                    "ligand_atom_indices": (4,),
                    "protein_atom_indices": (11,),
                    "ligand_parent_indices": (4,),
                    "protein_parent_indices": (2401,),
                    "distance": 3.20,
                    "angle": 95.0,
                    "metadata": {"distance": 3.20, "angle": 95.0},
                    "metadata_json": json.dumps(
                        {"angle": 95.0, "distance": 3.20},
                        sort_keys=True,
                    ),
                },
            ]
        )

    def test_build_pose_interaction_table(self) -> None:
        df = self.make_events_df()
        out = build_pose_interaction_table(df, pose_ids=["pose_1", "pose_2"])

        self.assertEqual(list(out.columns), POSE_EVENT_COLUMNS)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.loc[0, "pose_id"], "pose_1")
        self.assertTrue(out.loc[0, "has_interactions"])
        self.assertEqual(len(out.loc[0, "interaction_events"]), 3)
        self.assertIsNone(out.loc[1, "interaction_events"])
        self.assertFalse(out.loc[1, "has_interactions"])

    def test_build_pose_interaction_table_missing_pose_id_column(self) -> None:
        df = pd.DataFrame([{"interaction": "Hydrophobic"}])
        out = build_pose_interaction_table(df, pose_ids=["pose_1"])
        self.assertIsNone(out.loc[0, "interaction_events"])
        self.assertFalse(out.loc[0, "has_interactions"])

    def test_build_pose_summary_table(self) -> None:
        df = self.make_events_df()
        out = build_pose_summary_table(df, pose_ids=["pose_1", "pose_2"])

        self.assertEqual(list(out.columns), POSE_SUMMARY_COLUMNS)
        self.assertEqual(len(out), 2)

        compact = out.loc[0, "interaction_compact"]
        detail = out.loc[0, "interaction_detail"]

        self.assertIn("Hydrophobic", compact)
        self.assertEqual(compact["Hydrophobic"], ["ASP160.A", "LEU149.A"])
        self.assertIn("Hydrophobic", detail)
        self.assertIn("ASP160.A", detail["Hydrophobic"])
        self.assertEqual(len(detail["Hydrophobic"]["ASP160.A"]), 1)

        self.assertIsNone(out.loc[1, "interaction_compact"])
        self.assertIsNone(out.loc[1, "interaction_detail"])
        self.assertFalse(out.loc[1, "has_interactions"])

    def test_build_pose_summary_table_missing_required_columns(self) -> None:
        df = pd.DataFrame([{"pose_id": "pose_1"}])
        out = build_pose_summary_table(df, pose_ids=["pose_1"])
        self.assertIsNone(out.loc[0, "interaction_compact"])
        self.assertIsNone(out.loc[0, "interaction_detail"])
        self.assertFalse(out.loc[0, "has_interactions"])


class TestSummaryTableToDict(unittest.TestCase):
    def make_summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pose_id": "pose_1",
                    "interaction_compact": {"Hydrophobic": ["ASP160.A", "LEU149.A"]},
                    "interaction_compact_json": None,
                    "interaction_detail": {
                        "Hydrophobic": {
                            "ASP160.A": [{"distance": 4.5}],
                        }
                    },
                    "interaction_detail_json": None,
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
        )

    def test_summary_table_to_dict_empty(self) -> None:
        out = summary_table_to_dict(pd.DataFrame())
        self.assertEqual(out, {})

    def test_summary_table_to_dict_compact(self) -> None:
        out = summary_table_to_dict(self.make_summary_df(), kind="compact")
        self.assertIn("pose_1", out)
        self.assertEqual(out["pose_1"]["Hydrophobic"], ["ASP160.A", "LEU149.A"])
        self.assertIsNone(out["pose_2"])

    def test_summary_table_to_dict_compact_as_sets(self) -> None:
        out = summary_table_to_dict(
            self.make_summary_df(),
            kind="compact",
            as_sets=True,
        )
        self.assertEqual(out["pose_1"]["Hydrophobic"], {"ASP160.A", "LEU149.A"})

    def test_summary_table_to_dict_detail(self) -> None:
        out = summary_table_to_dict(self.make_summary_df(), kind="detail")
        self.assertIn("Hydrophobic", out["pose_1"])
        self.assertIsNone(out["pose_2"])

    def test_summary_table_to_dict_drop_empty(self) -> None:
        out = summary_table_to_dict(
            self.make_summary_df(),
            kind="compact",
            drop_empty=True,
        )
        self.assertIn("pose_1", out)
        self.assertNotIn("pose_2", out)

    def test_summary_table_to_dict_invalid_kind(self) -> None:
        with self.assertRaises(ValueError):
            summary_table_to_dict(self.make_summary_df(), kind="bad")
