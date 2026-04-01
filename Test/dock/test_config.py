import json
import tempfile
import unittest
from pathlib import Path

from prodock.dock.config import (
    BatchConfig,
    Box,
    DockRow,
    LigandSpec,
    ReceptorSpec,
    SingleConfig,
    SoftwareSpec,
    _load_mapping,
    _tuplize3,
)


class TestConfigModels(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_tuplize3_none(self) -> None:
        self.assertIsNone(_tuplize3(None))

    def test_tuplize3_valid_list(self) -> None:
        self.assertEqual(_tuplize3([1, 2, 3]), (1.0, 2.0, 3.0))

    def test_tuplize3_invalid_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            _tuplize3([1, 2])

    def test_box_from_mapping_dict(self) -> None:
        box = Box.from_mapping(
            {
                "center": [1, 2, 3],
                "size": [10, 11, 12],
            }
        )
        self.assertEqual(box.center, (1.0, 2.0, 3.0))
        self.assertEqual(box.size, (10.0, 11.0, 12.0))

    def test_box_from_mapping_pair(self) -> None:
        box = Box.from_mapping(([1, 2, 3], [4, 5, 6]))
        self.assertEqual(box.center, (1.0, 2.0, 3.0))
        self.assertEqual(box.size, (4.0, 5.0, 6.0))

    def test_box_from_mapping_existing_box(self) -> None:
        original = Box(center=(1.0, 2.0, 3.0), size=(4.0, 5.0, 6.0))
        new_box = Box.from_mapping(original)
        self.assertIs(new_box, original)

    def test_box_from_mapping_missing_fields_raises(self) -> None:
        with self.assertRaises(ValueError):
            Box.from_mapping({"center": [1, 2, 3]})

    def test_single_config_from_dict_with_box(self) -> None:
        cfg = SingleConfig.from_dict(
            {
                "engine": "vina",
                "receptor": "protein.pdbqt",
                "ligand": "ligand.pdbqt",
                "box": {"center": [1, 2, 3], "size": [20, 20, 20]},
            }
        )
        self.assertEqual(cfg.engine, "vina")
        self.assertIsInstance(cfg.box, Box)
        self.assertEqual(cfg.box.center, (1.0, 2.0, 3.0))

    def test_ligand_spec_from_dict_infers_id_and_path_alias(self) -> None:
        lig = LigandSpec.from_dict({"path": "inputs/erlotinib.pdbqt"})
        self.assertEqual(lig.id, "erlotinib")
        self.assertEqual(lig.ligand, "inputs/erlotinib.pdbqt")

    def test_software_spec_from_dict_builds_ligands(self) -> None:
        sw = SoftwareSpec.from_dict(
            {
                "name": "vina",
                "ligands": [
                    {"path": "lig1.pdbqt"},
                    {"id": "lig2", "ligand": "lig2.pdbqt"},
                ],
            }
        )
        self.assertEqual(sw.name, "vina")
        self.assertEqual(len(sw.ligands), 2)
        self.assertIsInstance(sw.ligands[0], LigandSpec)
        self.assertEqual(sw.ligands[0].id, "lig1")

    def test_receptor_spec_from_dict_infers_id_and_engine_alias(self) -> None:
        receptor = ReceptorSpec.from_dict(
            {
                "path": "receptors/4WKQ.pdbqt",
                "box": {"center": [1, 2, 3], "size": [20, 20, 20]},
                "engines": [
                    {
                        "name": "vina",
                        "ligands": [{"path": "lig1.pdbqt"}],
                    }
                ],
            }
        )
        self.assertEqual(receptor.id, "4WKQ")
        self.assertEqual(receptor.receptor, "receptors/4WKQ.pdbqt")
        self.assertIsInstance(receptor.box, Box)
        self.assertEqual(len(receptor.softwares), 1)
        self.assertIsInstance(receptor.softwares[0], SoftwareSpec)

    def test_dockrow_from_dict_with_center_size(self) -> None:
        row = DockRow.from_dict(
            {
                "id": "job1",
                "receptor": "protein.pdbqt",
                "ligand": "ligand.pdbqt",
                "center": [1, 2, 3],
                "size": [10, 10, 10],
            }
        )
        self.assertEqual(row.center, (1.0, 2.0, 3.0))
        self.assertEqual(row.size, (10.0, 10.0, 10.0))

    def test_dockrow_resolved_box_prefers_box(self) -> None:
        box = Box(center=(1.0, 2.0, 3.0), size=(4.0, 5.0, 6.0))
        row = DockRow(
            id="job1",
            receptor="protein.pdbqt",
            ligand="ligand.pdbqt",
            box=box,
            center=(9.0, 9.0, 9.0),
            size=(10.0, 10.0, 10.0),
        )
        self.assertIs(row.resolved_box(), box)

    def test_dockrow_resolved_box_from_center_size(self) -> None:
        row = DockRow(
            id="job1",
            receptor="protein.pdbqt",
            ligand="ligand.pdbqt",
            center=(1.0, 2.0, 3.0),
            size=(4.0, 5.0, 6.0),
        )
        box = row.resolved_box()
        self.assertIsInstance(box, Box)
        assert box is not None
        self.assertEqual(box.center, (1.0, 2.0, 3.0))
        self.assertEqual(box.size, (4.0, 5.0, 6.0))

    def test_batch_config_from_dict_rows_and_receptors(self) -> None:
        batch = BatchConfig.from_dict(
            {
                "engine": "vina",
                "rows": [
                    {
                        "id": "job1",
                        "receptor": "protein.pdbqt",
                        "ligand": "lig1.pdbqt",
                    }
                ],
                "receptors": [
                    {
                        "id": "4WKQ",
                        "receptor": "4WKQ.pdbqt",
                        "softwares": [
                            {
                                "name": "vina",
                                "ligands": [{"id": "lig1", "ligand": "lig1.pdbqt"}],
                            }
                        ],
                    }
                ],
            }
        )
        self.assertEqual(batch.engine, "vina")
        self.assertEqual(len(batch.rows), 1)
        self.assertEqual(len(batch.receptors), 1)
        self.assertIsInstance(batch.rows[0], DockRow)
        self.assertIsInstance(batch.receptors[0], ReceptorSpec)

    def test_batch_config_from_dict_accepts_ligands_alias_for_rows(self) -> None:
        batch = BatchConfig.from_dict(
            {
                "ligands": [
                    {
                        "id": "job1",
                        "receptor": "protein.pdbqt",
                        "ligand": "lig1.pdbqt",
                    }
                ]
            }
        )
        self.assertEqual(len(batch.rows), 1)
        self.assertEqual(batch.rows[0].id, "job1")

    def test_to_dict_roundtrip_single_config(self) -> None:
        cfg = SingleConfig(
            engine="vina",
            receptor="protein.pdbqt",
            ligand="ligand.pdbqt",
            box=Box(center=(1.0, 2.0, 3.0), size=(20.0, 20.0, 20.0)),
        )
        payload = cfg.to_dict()
        rebuilt = SingleConfig.from_dict(payload)
        self.assertEqual(rebuilt.engine, cfg.engine)
        self.assertEqual(rebuilt.receptor, cfg.receptor)
        self.assertEqual(rebuilt.ligand, cfg.ligand)
        self.assertEqual(rebuilt.box, cfg.box)

    def test_load_mapping_json(self) -> None:
        path = self.tmp / "config.json"
        path.write_text(json.dumps({"engine": "vina", "n_jobs": 2}))
        data = _load_mapping(path)
        self.assertEqual(data["engine"], "vina")
        self.assertEqual(data["n_jobs"], 2)

    def test_load_mapping_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _load_mapping(self.tmp / "missing.json")

    def test_load_mapping_non_mapping_root_raises(self) -> None:
        path = self.tmp / "bad.json"
        path.write_text(json.dumps([1, 2, 3]))
        with self.assertRaises(TypeError):
            _load_mapping(path)

    def test_batch_config_from_file_json(self) -> None:
        path = self.tmp / "batch.json"
        path.write_text(
            json.dumps(
                {
                    "engine": "vina",
                    "rows": [
                        {
                            "id": "job1",
                            "receptor": "protein.pdbqt",
                            "ligand": "lig1.pdbqt",
                        }
                    ],
                }
            )
        )
        batch = BatchConfig.from_file(path)
        self.assertEqual(batch.engine, "vina")
        self.assertEqual(len(batch.rows), 1)

    def test_single_config_from_file_json(self) -> None:
        path = self.tmp / "single.json"
        path.write_text(
            json.dumps(
                {
                    "engine": "vina",
                    "receptor": "protein.pdbqt",
                    "ligand": "lig1.pdbqt",
                }
            )
        )
        cfg = SingleConfig.from_file(path)
        self.assertEqual(cfg.engine, "vina")
        self.assertEqual(cfg.receptor, "protein.pdbqt")
        self.assertEqual(cfg.ligand, "lig1.pdbqt")


if __name__ == "__main__":
    unittest.main()
