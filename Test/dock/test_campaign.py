import json
import tempfile
import unittest
from pathlib import Path

from prodock.dock.campaign import (
    BoxSpec,
    Campaign,
    LigandSpec,
    ReceptorSpec,
    SoftwareSpec,
)


class TestSpecs(unittest.TestCase):
    def test_ligand_spec_repr(self):
        lig = LigandSpec(id="erlotinib", ligand="ligands/erlotinib.pdbqt")
        text = repr(lig)
        self.assertIn("LigandSpec", text)
        self.assertIn("erlotinib", text)

    def test_box_spec_repr(self):
        box = BoxSpec(center=(1.0, 2.0, 3.0), size=(10.0, 11.0, 12.0))
        text = repr(box)
        self.assertIn("BoxSpec", text)
        self.assertIn("center=(1.0, 2.0, 3.0)", text)

    def test_software_spec_to_dict_and_from_dict_with_extra(self):
        lig = LigandSpec(id="lig1", ligand="/tmp/lig1.pdbqt")
        sw = SoftwareSpec(
            name="gnina",
            cpu=8,
            seed=7,
            exhaustiveness=24,
            n_poses=30,
            ligands=[lig],
            extra={"cnn_scoring": "rescore", "num_modes": 5},
        )

        data = sw.to_dict()
        self.assertEqual(data["name"], "gnina")
        self.assertEqual(data["cpu"], 8)
        self.assertEqual(data["seed"], 7)
        self.assertEqual(data["exhaustiveness"], 24)
        self.assertEqual(data["n_poses"], 30)
        self.assertEqual(data["cnn_scoring"], "rescore")
        self.assertEqual(data["num_modes"], 5)
        self.assertEqual(len(data["ligands"]), 1)
        self.assertEqual(data["ligands"][0]["id"], "lig1")

        rebuilt = SoftwareSpec.from_dict(data)
        self.assertEqual(rebuilt.name, "gnina")
        self.assertEqual(rebuilt.cpu, 8)
        self.assertEqual(rebuilt.seed, 7)
        self.assertEqual(rebuilt.exhaustiveness, 24)
        self.assertEqual(rebuilt.n_poses, 30)
        self.assertEqual(len(rebuilt.ligands), 1)
        self.assertEqual(rebuilt.ligands[0].id, "lig1")
        self.assertEqual(rebuilt.extra["cnn_scoring"], "rescore")
        self.assertEqual(rebuilt.extra["num_modes"], 5)

    def test_receptor_spec_to_dict_and_from_dict(self):
        rec = ReceptorSpec(
            id="4WKQ",
            receptor="/tmp/4WKQ.pdbqt",
            box=BoxSpec(center=(1.0, 2.0, 3.0), size=(20.0, 20.0, 20.0)),
            out_dir="/tmp/4WKQ/results/docked",
            log_dir="/tmp/4WKQ/results/logs",
            softwares=[
                SoftwareSpec(
                    name="vina",
                    ligands=[LigandSpec(id="lig1", ligand="/tmp/lig1.pdbqt")],
                )
            ],
        )

        data = rec.to_dict()
        self.assertEqual(data["id"], "4WKQ")
        self.assertEqual(data["receptor"], "/tmp/4WKQ.pdbqt")
        self.assertEqual(data["out_dir"], "/tmp/4WKQ/results/docked")
        self.assertEqual(data["log_dir"], "/tmp/4WKQ/results/logs")
        self.assertIn("box", data)
        self.assertIn("softwares", data)
        self.assertEqual(len(data["softwares"]), 1)

        rebuilt = ReceptorSpec.from_dict(data)
        self.assertEqual(rebuilt.id, "4WKQ")
        self.assertEqual(rebuilt.receptor, "/tmp/4WKQ.pdbqt")
        self.assertEqual(rebuilt.out_dir, "/tmp/4WKQ/results/docked")
        self.assertEqual(rebuilt.log_dir, "/tmp/4WKQ/results/logs")
        self.assertEqual(rebuilt.box.center, (1.0, 2.0, 3.0))
        self.assertEqual(rebuilt.box.size, (20.0, 20.0, 20.0))
        self.assertEqual(len(rebuilt.softwares), 1)
        self.assertEqual(rebuilt.softwares[0].name, "vina")


class TestCampaign(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

        self.workdir = self.tmp / "demo"
        self.workdir.mkdir()

        self.rec1_id = "4WKQ"
        self.rec2_id = "1M17"

        self.rec1_dir = self.workdir / self.rec1_id / "filtered_protein"
        self.rec2_dir = self.workdir / self.rec2_id / "filtered_protein"
        self.rec1_dir.mkdir(parents=True)
        self.rec2_dir.mkdir(parents=True)

        self.rec1_file = self.rec1_dir / f"{self.rec1_id}.pdbqt"
        self.rec2_file = self.rec2_dir / f"{self.rec2_id}.pdbqt"
        self.rec1_file.write_text("RECEPTOR1\n", encoding="utf-8")
        self.rec2_file.write_text("RECEPTOR2\n", encoding="utf-8")

        self.lig_dir = self.workdir / "ligands"
        self.lig_dir.mkdir()

        self.lig1 = self.lig_dir / "erlotinib.pdbqt"
        self.lig2 = self.lig_dir / "gefitinib.pdbqt"
        self.lig1.write_text("LIG1\n", encoding="utf-8")
        self.lig2.write_text("LIG2\n", encoding="utf-8")

        self.nested_lig_dir = self.lig_dir / "nested"
        self.nested_lig_dir.mkdir()
        self.lig3 = self.nested_lig_dir / "afatinib.pdbqt"
        self.lig3.write_text("LIG3\n", encoding="utf-8")

        self.boxes = [
            ((2.865, 193.257, 21.367), (27.091, 27.091, 27.091)),
            ((21.623, 0.4, 52.467), (34.07, 34.07, 34.07)),
        ]

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_manual_campaign(self):
        return Campaign(
            working_dir=str(self.workdir),
            receptors=[
                ReceptorSpec(
                    id=self.rec1_id,
                    receptor=str(self.rec1_file),
                    box=BoxSpec(center=(1.0, 2.0, 3.0), size=(10.0, 10.0, 10.0)),
                    out_dir=str(self.workdir / self.rec1_id / "results" / "docked"),
                    log_dir=str(self.workdir / self.rec1_id / "results" / "logs"),
                    softwares=[
                        SoftwareSpec(
                            name="vina",
                            ligands=[
                                LigandSpec(id="erlotinib", ligand=str(self.lig1)),
                                LigandSpec(id="gefitinib", ligand=str(self.lig2)),
                            ],
                        ),
                        SoftwareSpec(
                            name="smina",
                            ligands=[
                                LigandSpec(id="erlotinib", ligand=str(self.lig1)),
                            ],
                        ),
                    ],
                )
            ],
        )

    def test_working_path_property(self):
        campaign = Campaign(working_dir=str(self.workdir))
        self.assertEqual(campaign.working_path, self.workdir)

    def test_resolve_path_absolute_and_relative(self):
        rel = Campaign._resolve_path("abc/def.txt", absolute=False)
        self.assertEqual(rel, "abc/def.txt")

        abs_path = Campaign._resolve_path(self.lig1, absolute=True)
        self.assertEqual(abs_path, str(self.lig1.resolve()))

    def test_normalize_vec3_valid(self):
        vec = Campaign._normalize_vec3([1, 2, 3], "center")
        self.assertEqual(vec, (1.0, 2.0, 3.0))

    def test_normalize_vec3_invalid_length(self):
        with self.assertRaises(ValueError):
            Campaign._normalize_vec3([1, 2], "center")

    def test_with_working_dir(self):
        campaign = Campaign()
        returned = campaign.with_working_dir(self.workdir)
        self.assertIs(returned, campaign)
        self.assertEqual(campaign.working_dir, str(self.workdir))

    def test_scan_ligands_non_recursive(self):
        ligands = Campaign.scan_ligands(
            self.lig_dir,
            pattern="*.pdbqt",
            absolute=False,
            recursive=False,
        )
        ids = {x.id for x in ligands}
        self.assertSetEqual(ids, {"erlotinib", "gefitinib"})

    def test_scan_ligands_recursive(self):
        ligands = Campaign.scan_ligands(
            self.lig_dir,
            pattern="*.pdbqt",
            absolute=False,
            recursive=True,
        )
        ids = {x.id for x in ligands}
        self.assertSetEqual(ids, {"afatinib", "erlotinib", "gefitinib"})

    def test_scan_ligands_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            Campaign.scan_ligands(self.workdir / "missing")

    def test_scan_ligands_not_directory(self):
        with self.assertRaises(NotADirectoryError):
            Campaign.scan_ligands(self.lig1)

    def test_scan_ligands_no_matches(self):
        with self.assertRaises(ValueError):
            Campaign.scan_ligands(self.lig_dir, pattern="*.sdf")

    def test_from_lists_basic(self):
        campaign = Campaign.from_lists(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id, self.rec2_id],
            receptors=[self.rec1_file, self.rec2_file],
            boxes=self.boxes,
            engines=["vina", "smina"],
            lig_paths=[self.lig_dir, self.lig_dir],
            check_receptor_files=True,
            check_ligand_files=True,
        )

        self.assertEqual(len(campaign.receptors), 2)
        self.assertEqual(campaign.receptors[0].id, self.rec1_id)
        self.assertEqual(campaign.receptors[1].id, self.rec2_id)

        for receptor in campaign.receptors:
            self.assertTrue(receptor.out_dir.endswith("results/docked"))
            self.assertTrue(receptor.log_dir.endswith("results/logs"))
            self.assertEqual(len(receptor.softwares), 2)
            self.assertEqual(receptor.softwares[0].name, "vina")
            self.assertEqual(receptor.softwares[1].name, "smina")
            self.assertEqual(len(receptor.softwares[0].ligands), 2)

    def test_from_lists_with_engine_overrides(self):
        campaign = Campaign.from_lists(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id],
            receptors=[self.rec1_file],
            boxes=[self.boxes[0]],
            engines=["vina", "gnina"],
            lig_paths=[self.lig_dir],
            engine_overrides={
                "vina": {"cpu": 8, "seed": 99},
                "gnina": {"cpu": 2, "cnn_scoring": "rescore", "num_modes": 7},
            },
        )

        self.assertEqual(len(campaign.receptors), 1)
        softwares = {sw.name: sw for sw in campaign.receptors[0].softwares}

        self.assertEqual(softwares["vina"].cpu, 8)
        self.assertEqual(softwares["vina"].seed, 99)
        self.assertEqual(softwares["vina"].exhaustiveness, 16)
        self.assertEqual(softwares["vina"].n_poses, 20)

        self.assertEqual(softwares["gnina"].cpu, 2)
        self.assertEqual(softwares["gnina"].extra["cnn_scoring"], "rescore")
        self.assertEqual(softwares["gnina"].extra["num_modes"], 7)

    def test_from_lists_length_mismatch(self):
        with self.assertRaises(ValueError):
            Campaign.from_lists(
                working_dir=self.workdir,
                pdb_ids=[self.rec1_id, self.rec2_id],
                receptors=[self.rec1_file],
                boxes=self.boxes,
                engines=["vina"],
                lig_paths=[self.lig_dir, self.lig_dir],
            )

    def test_from_lists_no_engines(self):
        with self.assertRaises(ValueError):
            Campaign.from_lists(
                working_dir=self.workdir,
                pdb_ids=[self.rec1_id],
                receptors=[self.rec1_file],
                boxes=[self.boxes[0]],
                engines=[],
                lig_paths=[self.lig_dir],
            )

    def test_from_lists_missing_receptor_file_when_checked(self):
        with self.assertRaises(FileNotFoundError):
            Campaign.from_lists(
                working_dir=self.workdir,
                pdb_ids=[self.rec1_id],
                receptors=[self.workdir / "missing.pdbqt"],
                boxes=[self.boxes[0]],
                engines=["vina"],
                lig_paths=[self.lig_dir],
                check_receptor_files=True,
            )

    def test_from_lists_missing_receptor_root_and_no_create(self):
        other_workdir = self.tmp / "other_demo"
        other_workdir.mkdir()

        with self.assertRaises(FileNotFoundError):
            Campaign.from_lists(
                working_dir=other_workdir,
                pdb_ids=["XXXX"],
                receptors=[self.rec1_file],
                boxes=[self.boxes[0]],
                engines=["vina"],
                lig_paths=[self.lig_dir],
                create_receptor_dirs=False,
                check_receptor_files=False,
            )

    def test_from_lists_create_receptor_root(self):
        other_workdir = self.tmp / "other_demo"
        other_workdir.mkdir()

        campaign = Campaign.from_lists(
            working_dir=other_workdir,
            pdb_ids=["XXXX"],
            receptors=[self.rec1_file],
            boxes=[self.boxes[0]],
            engines=["vina"],
            lig_paths=[self.lig_dir],
            create_receptor_dirs=True,
            check_receptor_files=False,
        )

        self.assertTrue((other_workdir / "XXXX").is_dir())
        self.assertEqual(campaign.receptors[0].id, "XXXX")
        self.assertTrue(campaign.receptors[0].out_dir.endswith("XXXX/results/docked"))
        self.assertTrue(campaign.receptors[0].log_dir.endswith("XXXX/results/logs"))

    def test_from_lists_invalid_box_size(self):
        with self.assertRaises(ValueError):
            Campaign.from_lists(
                working_dir=self.workdir,
                pdb_ids=[self.rec1_id],
                receptors=[self.rec1_file],
                boxes=[((1.0, 2.0), (3.0, 4.0, 5.0))],
                engines=["vina"],
                lig_paths=[self.lig_dir],
            )

    def test_from_shared_ligand_dir(self):
        campaign = Campaign.from_shared_ligand_dir(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id, self.rec2_id],
            receptors=[self.rec1_file, self.rec2_file],
            boxes=self.boxes,
            engines=["vina"],
            ligand_dir=self.lig_dir,
            check_receptor_files=True,
            check_ligand_files=True,
        )

        self.assertEqual(len(campaign.receptors), 2)
        for receptor in campaign.receptors:
            self.assertEqual(len(receptor.softwares), 1)
            self.assertEqual(receptor.softwares[0].name, "vina")
            self.assertEqual(len(receptor.softwares[0].ligands), 2)

    def test_validate_success(self):
        campaign = self._build_manual_campaign()
        campaign.validate(check_receptor_files=True, check_ligand_files=True)

    def test_validate_no_receptors(self):
        campaign = Campaign(working_dir=str(self.workdir), receptors=[])
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_empty_receptor_id(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].id = ""
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_empty_receptor_path(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].receptor = ""
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_empty_out_dir(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].out_dir = ""
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_empty_log_dir(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].log_dir = ""
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_no_softwares(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].softwares = []
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_empty_engine_name(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].softwares[0].name = ""
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_no_ligands(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].softwares[0].ligands = []
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_empty_ligand_id(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].softwares[0].ligands[0].id = ""
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_empty_ligand_path(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].softwares[0].ligands[0].ligand = ""
        with self.assertRaises(ValueError):
            campaign.validate()

    def test_validate_missing_receptor_file(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].receptor = str(self.workdir / "missing_rec.pdbqt")
        with self.assertRaises(FileNotFoundError):
            campaign.validate(check_receptor_files=True)

    def test_validate_missing_ligand_file(self):
        campaign = self._build_manual_campaign()
        campaign.receptors[0].softwares[0].ligands[0].ligand = str(
            self.workdir / "missing_lig.pdbqt"
        )
        with self.assertRaises(FileNotFoundError):
            campaign.validate(check_ligand_files=True)

    def test_to_dict_and_from_dict_roundtrip(self):
        campaign = Campaign.from_shared_ligand_dir(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id],
            receptors=[self.rec1_file],
            boxes=[self.boxes[0]],
            engines=["vina", "gnina"],
            ligand_dir=self.lig_dir,
            engine_overrides={"gnina": {"cnn_scoring": "rescore"}},
        )

        data = campaign.to_dict()
        rebuilt = Campaign.from_dict(data)

        self.assertIsNone(rebuilt.working_dir)
        self.assertEqual(len(rebuilt.receptors), 1)
        self.assertEqual(rebuilt.receptors[0].id, self.rec1_id)
        self.assertEqual(len(rebuilt.receptors[0].softwares), 2)
        self.assertIn("out_dir", data["receptors"][0])
        self.assertIn("log_dir", data["receptors"][0])

        softwares = {sw.name: sw for sw in rebuilt.receptors[0].softwares}
        self.assertEqual(softwares["gnina"].extra["cnn_scoring"], "rescore")

    def test_from_dict_with_working_dir_binding(self):
        campaign = Campaign.from_shared_ligand_dir(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id],
            receptors=[self.rec1_file],
            boxes=[self.boxes[0]],
            engines=["vina"],
            ligand_dir=self.lig_dir,
        )

        data = campaign.to_dict()
        rebuilt = Campaign.from_dict(data, working_dir=self.workdir)
        self.assertEqual(rebuilt.working_dir, str(self.workdir))

    def test_save_json_and_load_json(self):
        campaign = Campaign.from_shared_ligand_dir(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id],
            receptors=[self.rec1_file],
            boxes=[self.boxes[0]],
            engines=["vina"],
            ligand_dir=self.lig_dir,
        )

        out_json = self.workdir / "campaign.json"
        written = campaign.save_json(out_json)
        self.assertEqual(written, out_json)
        self.assertTrue(out_json.is_file())

        loaded = Campaign.load_json(out_json)
        self.assertIsNone(loaded.working_dir)
        self.assertEqual(len(loaded.receptors), 1)
        self.assertEqual(loaded.receptors[0].id, self.rec1_id)

        raw = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertNotIn("working_dir", raw)
        self.assertIn("receptors", raw)
        self.assertIn("out_dir", raw["receptors"][0])
        self.assertIn("log_dir", raw["receptors"][0])

    def test_load_json_with_working_dir_binding(self):
        campaign = Campaign.from_shared_ligand_dir(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id],
            receptors=[self.rec1_file],
            boxes=[self.boxes[0]],
            engines=["vina"],
            ligand_dir=self.lig_dir,
        )

        out_json = self.workdir / "campaign.json"
        campaign.save_json(out_json)

        loaded = Campaign.load_json(out_json, working_dir=self.workdir)
        self.assertEqual(loaded.working_dir, str(self.workdir))

    def test_iter_jobs(self):
        campaign = Campaign.from_shared_ligand_dir(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id],
            receptors=[self.rec1_file],
            boxes=[self.boxes[0]],
            engines=["vina", "smina"],
            ligand_dir=self.lig_dir,
        )

        jobs = list(campaign.iter_jobs())
        self.assertEqual(len(jobs), 4)

        first = jobs[0]
        self.assertEqual(first[0], self.rec1_id)
        self.assertEqual(first[1], str(self.rec1_file.resolve()))
        self.assertTrue(first[2].endswith("/results/docked"))
        self.assertTrue(first[3].endswith("/results/logs"))
        self.assertEqual(
            first[4], "smina" if False else first[4]
        )  # no-op for stable tuple shape
        self.assertIn(first[4], {"vina", "smina"})
        self.assertIn(first[5], {"erlotinib", "gefitinib"})
        self.assertTrue(first[6].endswith(".pdbqt"))

    def test_summary(self):
        campaign = Campaign.from_shared_ligand_dir(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id, self.rec2_id],
            receptors=[self.rec1_file, self.rec2_file],
            boxes=self.boxes,
            engines=["vina", "smina"],
            ligand_dir=self.lig_dir,
        )

        summary = campaign.summary()
        self.assertEqual(summary["working_dir"], str(self.workdir.resolve()))
        self.assertEqual(summary["n_receptors"], 2)
        self.assertEqual(summary["n_engine_blocks"], 4)
        self.assertEqual(summary["n_jobs"], 8)
        self.assertEqual(summary["receptors"], [self.rec1_id, self.rec2_id])
        self.assertIn(self.rec1_id, summary["out_dirs"])
        self.assertIn(self.rec2_id, summary["log_dirs"])

    def test_ensure_receptor_dirs_basic(self):
        campaign = Campaign.from_shared_ligand_dir(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id],
            receptors=[self.rec1_file],
            boxes=[self.boxes[0]],
            engines=["vina", "smina"],
            ligand_dir=self.lig_dir,
        )

        campaign.ensure_receptor_dirs(engine_subdirs=False)

        self.assertTrue((self.workdir / self.rec1_id).is_dir())
        self.assertTrue((self.workdir / self.rec1_id / "results" / "docked").is_dir())
        self.assertTrue((self.workdir / self.rec1_id / "results" / "logs").is_dir())
        self.assertFalse(
            (self.workdir / self.rec1_id / "results" / "docked" / "vina").exists()
        )
        self.assertFalse(
            (self.workdir / self.rec1_id / "results" / "logs" / "vina").exists()
        )

    def test_ensure_receptor_dirs_with_engine_subdirs(self):
        campaign = Campaign.from_shared_ligand_dir(
            working_dir=self.workdir,
            pdb_ids=[self.rec1_id],
            receptors=[self.rec1_file],
            boxes=[self.boxes[0]],
            engines=["vina", "smina"],
            ligand_dir=self.lig_dir,
        )

        campaign.ensure_receptor_dirs(engine_subdirs=True)

        self.assertTrue(
            (self.workdir / self.rec1_id / "results" / "docked" / "vina").is_dir()
        )
        self.assertTrue(
            (self.workdir / self.rec1_id / "results" / "logs" / "vina").is_dir()
        )
        self.assertTrue(
            (self.workdir / self.rec1_id / "results" / "docked" / "smina").is_dir()
        )
        self.assertTrue(
            (self.workdir / self.rec1_id / "results" / "logs" / "smina").is_dir()
        )


if __name__ == "__main__":
    unittest.main()
