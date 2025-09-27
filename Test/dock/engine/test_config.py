import json
import tempfile
import shutil
import unittest
from pathlib import Path

from prodock.dock.engine.config import Box, SingleConfig, BatchConfig, LigandTask


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_box_from_mapping_dict_and_list(self):
        b1 = Box.from_mapping({"center": [1, 2, 3], "size": [4, 5, 6]})
        self.assertEqual(b1.center, (1.0, 2.0, 3.0))
        self.assertEqual(b1.size, (4.0, 5.0, 6.0))

        b2 = Box.from_mapping(([7, 8, 9], [10, 11, 12]))
        self.assertEqual(b2.center, (7.0, 8.0, 9.0))
        self.assertEqual(b2.size, (10.0, 11.0, 12.0))

    def test_singleconfig_roundtrip_json(self):
        cfg = SingleConfig(
            engine="smina",
            receptor="rec.pdbqt",
            ligand="lig.pdbqt",
            box=Box(center=(1, 2, 3), size=(10, 11, 12)),
            exhaustiveness=8,
            n_poses=9,
            cpu=4,
            seed=42,
            out="out/pose.pdbqt",
            log="out/run.log",
            engine_options={"foo": "bar"},
            validate_receptor=False,
        )
        p = self.td / "single.json"
        p.write_text(json.dumps(cfg.to_dict()))
        loaded = SingleConfig.from_file(p)
        self.assertEqual(loaded.engine, "smina")
        self.assertEqual(loaded.box.center, (1.0, 2.0, 3.0))
        self.assertEqual(loaded.n_poses, 9)
        self.assertEqual(loaded.engine_options["foo"], "bar")

    def test_batchconfig_rows_from_dict(self):
        data = {
            "engine": "smina",
            "rows": [
                {
                    "id": "L1",
                    "receptor": "rec.pdbqt",
                    "ligand": "l1.pdbqt",
                    "box": {"center": [1, 2, 3], "size": [10, 10, 10]},
                },
                {"id": "L2", "receptor": "rec.pdbqt", "ligand": "l2.pdbqt"},
            ],
        }
        cfg = BatchConfig.from_dict(data)
        self.assertEqual(cfg.engine, "smina")
        self.assertIsInstance(cfg.rows[0], LigandTask)
        self.assertEqual(cfg.rows[0].box.center, (1.0, 2.0, 3.0))
        self.assertEqual(cfg.rows[1].ligand, "l2.pdbqt")


if __name__ == "__main__":
    unittest.main()
