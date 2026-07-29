from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from rdkit import Chem

from prodock.database import PoseDatabase
from prodock.database.pose_db import _SCHEMA_VERSION


def _pose_df():
    return pd.DataFrame(
        [
            {
                "receptor_id": "R1",
                "ligand_id": "L1",
                "engine": "vina",
                "pose_rank": 1,
                "affinity": -7.0,
                "mol": Chem.MolFromSmiles("CCO"),
            }
        ]
    )


class TestSchemaV2Fresh(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "fresh.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fresh_db_is_at_current_version(self) -> None:
        db = PoseDatabase(self.path, create=True)
        try:
            version = db.connection.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, _SCHEMA_VERSION)
        finally:
            db.close()

    def test_create_run_and_stamp_pose(self) -> None:
        db = PoseDatabase(self.path, create=True)
        try:
            run_id = db.create_run(
                name="camp",
                config={"engines": ["vina"]},
                prodock_version="0.5.0",
            )
            self.assertEqual(db.active_run_id, run_id)

            db.insert_dataframe(_pose_df(), replace=True)

            pose_run = db.connection.execute("SELECT run_id FROM poses").fetchone()[0]
            self.assertEqual(pose_run, run_id)

            run = db.connection.execute(
                "SELECT name, config_json, prodock_version FROM runs"
            ).fetchone()
            self.assertEqual(run["name"], "camp")
            self.assertIn("vina", run["config_json"])
            self.assertEqual(run["prodock_version"], "0.5.0")
        finally:
            db.close()

    def test_ligand_smiles_derived_from_mol(self) -> None:
        db = PoseDatabase(self.path, create=True)
        try:
            db.insert_dataframe(_pose_df(), replace=True)
            smiles = db.connection.execute(
                "SELECT smiles FROM ligands WHERE ligand_id = 'L1'"
            ).fetchone()[0]
            self.assertEqual(smiles, "CCO")
        finally:
            db.close()

    def test_ligand_metadata_overrides_derived_smiles(self) -> None:
        db = PoseDatabase(self.path, create=True)
        try:
            df = _pose_df()
            df["ligand_metadata"] = [{"smiles": "OCC", "inchikey": "ABC-XYZ"}]
            db.insert_dataframe(df, replace=True)
            row = db.connection.execute(
                "SELECT smiles, inchikey FROM ligands WHERE ligand_id = 'L1'"
            ).fetchone()
            self.assertEqual(row["smiles"], "OCC")
            self.assertEqual(row["inchikey"], "ABC-XYZ")
        finally:
            db.close()

    def test_flat_view_reports_joined_columns(self) -> None:
        db = PoseDatabase(self.path, create=True)
        try:
            db.create_run(name="c", config={}, prodock_version="0.5.0")
            db.insert_dataframe(_pose_df(), replace=True)
            row = db.connection.execute(
                "SELECT receptor_id, ligand_id, smiles, affinity, run_id "
                "FROM v_poses_flat"
            ).fetchone()
            self.assertEqual(row["ligand_id"], "L1")
            self.assertEqual(row["smiles"], "CCO")
            self.assertEqual(row["affinity"], -7.0)
            self.assertIsNotNone(row["run_id"])
        finally:
            db.close()

    def test_runs_config_json_must_be_valid_json(self) -> None:
        db = PoseDatabase(self.path, create=True)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                db.connection.execute(
                    "INSERT INTO runs (config_json) VALUES (?)",
                    ("not json",),
                )
        finally:
            db.close()


class TestSchemaV2Migration(unittest.TestCase):
    _V1_SQL = """
    CREATE TABLE receptors (receptor_id TEXT PRIMARY KEY,
        metadata_json TEXT NOT NULL DEFAULT '{}');
    CREATE TABLE ligands (ligand_id TEXT PRIMARY KEY,
        metadata_json TEXT NOT NULL DEFAULT '{}');
    CREATE TABLE engines (engine TEXT PRIMARY KEY,
        metadata_json TEXT NOT NULL DEFAULT '{}');
    CREATE TABLE poses (pose_db_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pose_id TEXT UNIQUE, receptor_id TEXT NOT NULL, ligand_id TEXT NOT NULL,
        engine TEXT NOT NULL, pose_rank INTEGER NOT NULL, mol_blob BLOB NOT NULL,
        mol_is_compressed INTEGER NOT NULL DEFAULT 1,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (receptor_id, ligand_id, engine, pose_rank));
    CREATE TABLE pose_scores (pose_db_id INTEGER PRIMARY KEY,
        pose_rank INTEGER NOT NULL, affinity REAL,
        score_json TEXT NOT NULL DEFAULT '{}',
        metadata_json TEXT NOT NULL DEFAULT '{}');
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "old.db")
        conn = sqlite3.connect(self.path)
        conn.executescript(self._V1_SQL)
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_v1_database_is_migrated_in_place(self) -> None:
        db = PoseDatabase(self.path, create=False)
        try:
            version = db.connection.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, _SCHEMA_VERSION)

            ligand_cols = {
                r["name"] for r in db.connection.execute("PRAGMA table_info(ligands)")
            }
            self.assertIn("smiles", ligand_cols)
            self.assertIn("inchikey", ligand_cols)

            pose_cols = {
                r["name"] for r in db.connection.execute("PRAGMA table_info(poses)")
            }
            self.assertIn("run_id", pose_cols)

            self.assertTrue(db._table_exists("runs"))

            # View is usable and inserts still work post-migration.
            db.insert_dataframe(_pose_df(), replace=True)
            self.assertEqual(
                db.connection.execute("SELECT COUNT(*) FROM v_poses_flat").fetchone()[
                    0
                ],
                1,
            )
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
