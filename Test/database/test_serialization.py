from __future__ import annotations

import json
import unittest

from rdkit import Chem

from prodock.database.serialization import (
    as_many,
    compose_residue_id,
    deserialize_mol,
    json_dumps,
    json_dumps_list,
    json_loads_dict,
    json_loads_int_list,
    json_loads_list,
    make_pose_key,
    parse_residue_id,
    serialize_mol,
)


class TestJsonHelpers(unittest.TestCase):
    def test_json_dumps_with_mapping(self) -> None:
        result = json_dumps({"engine": "vina", "rank": 1})
        self.assertEqual(json.loads(result), {"engine": "vina", "rank": 1})

    def test_json_dumps_with_none(self) -> None:
        self.assertEqual(json_dumps(None), "{}")

    def test_json_dumps_list_with_sequence(self) -> None:
        result = json_dumps_list([1, "a", 3.5])
        self.assertEqual(json.loads(result), [1, "a", 3.5])

    def test_json_dumps_list_with_none(self) -> None:
        self.assertEqual(json_dumps_list(None), "[]")

    def test_json_loads_dict_valid(self) -> None:
        self.assertEqual(json_loads_dict('{"a": 1, "b": 2}'), {"a": 1, "b": 2})

    def test_json_loads_dict_none(self) -> None:
        self.assertEqual(json_loads_dict(None), {})

    def test_json_loads_dict_empty(self) -> None:
        self.assertEqual(json_loads_dict(""), {})

    def test_json_loads_dict_wrong_json_type(self) -> None:
        self.assertEqual(json_loads_dict("[1, 2, 3]"), {})

    def test_json_loads_list_valid(self) -> None:
        self.assertEqual(json_loads_list('[1, "x", 3]'), [1, "x", 3])

    def test_json_loads_list_none(self) -> None:
        self.assertEqual(json_loads_list(None), [])

    def test_json_loads_list_empty(self) -> None:
        self.assertEqual(json_loads_list(""), [])

    def test_json_loads_list_wrong_json_type(self) -> None:
        self.assertEqual(json_loads_list('{"a": 1}'), [])

    def test_json_loads_int_list_valid(self) -> None:
        self.assertEqual(json_loads_int_list('[1, "2", 3.0, "-4"]'), [1, 2, 3, -4])

    def test_json_loads_int_list_skips_invalid(self) -> None:
        self.assertEqual(json_loads_int_list('[1, "x", null, 2]'), [1, 2])

    def test_json_loads_int_list_none(self) -> None:
        self.assertEqual(json_loads_int_list(None), [])


class TestScalarSequenceHelpers(unittest.TestCase):
    def test_as_many_none(self) -> None:
        self.assertIsNone(as_many(None))

    def test_as_many_string(self) -> None:
        self.assertEqual(as_many("vina"), ["vina"])

    def test_as_many_sequence(self) -> None:
        self.assertEqual(as_many(["vina", "smina"]), ["vina", "smina"])

    def test_as_many_mixed_sequence(self) -> None:
        self.assertEqual(as_many([1, "2", 3.5]), ["1", "2", "3.5"])


class TestPoseKeyHelpers(unittest.TestCase):
    def test_make_pose_key(self) -> None:
        result = make_pose_key("1M17", "erlotinib", "qvina", 1)
        self.assertEqual(result, "1M17__erlotinib__qvina__pose1")

    def test_make_pose_key_casts_rank_to_int(self) -> None:
        result = make_pose_key("1M17", "erlotinib", "vina", 2.0)
        self.assertEqual(result, "1M17__erlotinib__vina__pose2")


class TestResidueHelpers(unittest.TestCase):
    def test_parse_residue_id_full(self) -> None:
        self.assertEqual(parse_residue_id("LEU149.A"), ("LEU", 149, "A"))

    def test_parse_residue_id_without_chain(self) -> None:
        self.assertEqual(parse_residue_id("GLY24"), ("GLY", 24, None))

    def test_parse_residue_id_name_only(self) -> None:
        self.assertEqual(parse_residue_id("ATP"), ("ATP", None, None))

    def test_parse_residue_id_negative_number(self) -> None:
        self.assertEqual(parse_residue_id("ASP-10.B"), ("ASP", -10, "B"))

    def test_parse_residue_id_none(self) -> None:
        self.assertEqual(parse_residue_id(None), (None, None, None))

    def test_parse_residue_id_empty(self) -> None:
        self.assertEqual(parse_residue_id(""), (None, None, None))

    def test_parse_residue_id_whitespace(self) -> None:
        self.assertEqual(parse_residue_id("   "), (None, None, None))

    def test_parse_residue_id_invalid(self) -> None:
        self.assertEqual(parse_residue_id("LEU149.A.extra"), (None, None, None))

    def test_compose_residue_id_full(self) -> None:
        self.assertEqual(compose_residue_id("LEU", 149, "A"), "LEU149.A")

    def test_compose_residue_id_without_chain(self) -> None:
        self.assertEqual(compose_residue_id("GLY", 24, None), "GLY24")

    def test_compose_residue_id_name_only(self) -> None:
        self.assertEqual(compose_residue_id("ATP", None, None), "ATP")

    def test_compose_residue_id_missing_name(self) -> None:
        self.assertIsNone(compose_residue_id(None, 24, "A"))

    def test_parse_compose_roundtrip(self) -> None:
        residue_id = compose_residue_id("TYR", 100, "B")
        self.assertEqual(parse_residue_id(residue_id), ("TYR", 100, "B"))


class TestMolSerialization(unittest.TestCase):
    def test_serialize_deserialize_mol_compressed(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)

        blob = serialize_mol(mol, compress=True, include_props=True)
        restored = deserialize_mol(blob, compressed=True)

        self.assertIsNotNone(restored)
        self.assertEqual(Chem.MolToSmiles(restored), Chem.MolToSmiles(mol))

    def test_serialize_deserialize_mol_uncompressed(self) -> None:
        mol = Chem.MolFromSmiles("c1ccccc1")
        self.assertIsNotNone(mol)

        blob = serialize_mol(mol, compress=False, include_props=True)
        restored = deserialize_mol(blob, compressed=False)

        self.assertIsNotNone(restored)
        self.assertEqual(Chem.MolToSmiles(restored), Chem.MolToSmiles(mol))

    def test_serialize_mol_with_props(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)
        mol.SetProp("_Name", "ethanol")

        blob = serialize_mol(mol, compress=True, include_props=True)
        restored = deserialize_mol(blob, compressed=True)

        self.assertTrue(restored.HasProp("_Name"))
        self.assertEqual(restored.GetProp("_Name"), "ethanol")

    def test_serialize_mol_without_props(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)
        mol.SetProp("_Name", "ethanol")

        blob = serialize_mol(mol, compress=True, include_props=False)
        restored = deserialize_mol(blob, compressed=True)

        self.assertFalse(restored.HasProp("_Name"))

    def test_serialize_mol_none_raises(self) -> None:
        with self.assertRaises(ValueError):
            serialize_mol(None)  # type: ignore[arg-type]

    def test_deserialize_mol_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            deserialize_mol(b"")

    def test_deserialize_mol_invalid_payload_raises(self) -> None:
        with self.assertRaises(Exception):
            deserialize_mol(b"not-a-valid-rdkit-payload", compressed=False)


if __name__ == "__main__":
    unittest.main()
