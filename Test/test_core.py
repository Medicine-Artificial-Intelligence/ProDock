from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from prodock import prodock

HAS_SMINA = True
RUN_NETWORK_TESTS = True
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MULTI = REPO_ROOT / "Data" / "testcase" / "Multi"


class TestProDockPipeline(unittest.TestCase):
    """Integration tests for the public ``prodock(...)`` entry point."""

    maxDiff = None

    def _load_campaign(self, campaign_json: Path) -> dict:
        """Load a campaign JSON file."""
        with campaign_json.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _copy_prepared_fixture_project(self, dst_project: Path) -> None:
        """
        Copy the prepared-receptor fixture project into a temporary location.

        Expected source layout::

            Data/testcase/Multi/
                4WKQ/
                1M17/
                ligands/
        """
        if not SOURCE_MULTI.exists():
            self.skipTest(f"Fixture project not found: {SOURCE_MULTI}")

        required_paths = [
            SOURCE_MULTI / "4WKQ" / "filtered_protein" / "4WKQ.pdbqt",
            SOURCE_MULTI / "1M17" / "filtered_protein" / "1M17.pdbqt",
            SOURCE_MULTI / "ligands",
        ]
        missing = [str(p) for p in required_paths if not p.exists()]
        if missing:
            self.skipTest(
                "Prepared fixture data is incomplete. Missing:\n- "
                + "\n- ".join(missing)
            )

        shutil.copytree(SOURCE_MULTI / "4WKQ", dst_project / "4WKQ")
        shutil.copytree(SOURCE_MULTI / "1M17", dst_project / "1M17")
        shutil.copytree(SOURCE_MULTI / "ligands", dst_project / "ligands")

    @unittest.skipUnless(
        HAS_SMINA, "smina binary is required for ProDock integration tests."
    )
    def test_prodock_with_prepared_receptors_and_ligand_dir(self) -> None:
        """
        Run the pipeline using prepared receptors and an existing ligand directory.

        This mirrors the prepared-receptor workflow, but writes everything into a
        temporary project directory so the source fixture is not modified.
        """
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "Multi"
            project.mkdir(parents=True, exist_ok=True)
            self._copy_prepared_fixture_project(project)

            result = prodock(
                project,
                prepared_receptors=[
                    {
                        "receptor_id": "4WKQ",
                        "receptor_pdbqt": str(
                            project / "4WKQ" / "filtered_protein" / "4WKQ.pdbqt"
                        ),
                        "center": (2.865, 193.257, 21.367),
                        "size": (27.091, 27.091, 27.091),
                    },
                    {
                        "receptor_id": "1M17",
                        "receptor_pdbqt": str(
                            project / "1M17" / "filtered_protein" / "1M17.pdbqt"
                        ),
                        "center": (21.623, 0.4, 52.467),
                        "size": (34.07, 34.07, 34.07),
                    },
                ],
                ligand_dir=str(project / "ligands"),
                engines=["smina"],
            )

            self.assertEqual(result.project_dir, project.resolve())
            self.assertEqual(result.ligand_dir, (project / "ligands").resolve())
            self.assertTrue(result.campaign_json.exists())
            self.assertEqual(result.campaign_json.name, "campaign.json")

            self.assertEqual(len(result.receptors), 2)
            self.assertEqual(
                {spec.receptor_id for spec in result.receptors},
                {"4WKQ", "1M17"},
            )
            self.assertTrue(
                all(spec.receptor_pdbqt.exists() for spec in result.receptors)
            )

            payload = self._load_campaign(result.campaign_json)
            self.assertIn("receptors", payload)
            self.assertEqual(len(payload["receptors"]), 2)

            receptor_ids = {entry["id"] for entry in payload["receptors"]}
            self.assertEqual(receptor_ids, {"4WKQ", "1M17"})

            for entry in payload["receptors"]:
                self.assertIn("softwares", entry)
                self.assertEqual([sw["name"] for sw in entry["softwares"]], ["smina"])
                self.assertGreater(len(entry["softwares"][0]["ligands"]), 0)

    @unittest.skipUnless(
        HAS_SMINA, "smina binary is required for ProDock integration tests."
    )
    @unittest.skipUnless(
        RUN_NETWORK_TESTS,
        "Set PRODOCK_RUN_NETWORK_TESTS=1 to enable the raw full-pipeline test.",
    )
    def test_prodock_full_pipeline_from_raw_receptor_and_smiles(self) -> None:
        """
        Run the full pipeline from raw receptor input and SMILES ligands.

        This test performs real work:

        - downloads/processes the receptor through ``PDBQuery.process_batch``
        - prepares receptor/ligands
        - builds a campaign
        - runs docking

        It is therefore intentionally guarded behind
        ``PRODOCK_RUN_NETWORK_TESTS=1``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "raw_case"
            project.mkdir(parents=True, exist_ok=True)

            receptors = [
                {
                    "pdb_id": "4WKQ",
                    "receptor_name": "EGFR_4WKQ",
                    "ligand_code": "IRE",
                    "chains": ["A"],
                    "cofactors": [],
                },
            ]
            ligands = [
                {
                    "id": "erlotinib",
                    "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C",
                },
                {
                    "id": "gefitinib",
                    "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F",
                },
            ]

            result = prodock(
                project,
                receptors=receptors,
                ligands=ligands,
                engines=["smina"],
            )

            self.assertEqual(result.project_dir, project.resolve())
            self.assertTrue(result.campaign_json.exists())
            self.assertEqual(result.ligand_dir, (project / "ligands").resolve())

            expected_receptor_pdbqt = (
                project / "4WKQ" / "filtered_protein" / "4WKQ.pdbqt"
            ).resolve()
            self.assertTrue(expected_receptor_pdbqt.exists())

            self.assertEqual(len(result.receptors), 1)
            self.assertEqual(result.receptors[0].receptor_id, "4WKQ")
            self.assertEqual(
                result.receptors[0].receptor_pdbqt, expected_receptor_pdbqt
            )

            ligand_files = sorted(p.name for p in result.ligand_dir.glob("*.pdbqt"))
            self.assertEqual(ligand_files, ["erlotinib.pdbqt", "gefitinib.pdbqt"])

            payload = self._load_campaign(result.campaign_json)
            self.assertIn("receptors", payload)
            self.assertEqual(len(payload["receptors"]), 1)

            receptor_entry = payload["receptors"][0]
            self.assertEqual(receptor_entry["id"], "4WKQ")
            self.assertEqual(
                [sw["name"] for sw in receptor_entry["softwares"]],
                ["smina"],
            )

            ligand_ids = sorted(
                ligand["id"] for ligand in receptor_entry["softwares"][0]["ligands"]
            )
            self.assertEqual(ligand_ids, ["erlotinib", "gefitinib"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
