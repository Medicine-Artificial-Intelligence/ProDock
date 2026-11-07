"""Regression checks for campaign provenance and connection lifecycle."""

import hashlib
import json
import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest
from rdkit import Chem

from prodock.core import ProDockPipeline
from prodock.database import PoseDatabase


def test_saved_campaign_snapshot_survives_external_edit(tmp_path):
    pipeline = ProDockPipeline(tmp_path, progress=False, log_colored=False)
    campaign = {"receptors": [{"id": "R", "box": {"center": [1, 2, 3]}}]}
    payload = json.dumps(campaign).encode()
    path = tmp_path / "custom.json"
    path.write_bytes(payload)
    pipeline._saved_campaign_path = path
    df = pd.DataFrame(
        [
            dict(
                receptor_id="R",
                ligand_id="L",
                engine="vina",
                pose_rank=1,
                affinity=-7.0,
                mol=Chem.MolFromSmiles("CCO"),
            )
        ]
    )
    db_path = pipeline.save_database(df=df)
    path.write_text("{}")
    with sqlite3.connect(db_path) as conn:
        config = json.loads(conn.execute("SELECT config_json FROM runs").fetchone()[0])
        assert config["campaign"] == campaign
        assert config["campaign_sha256"] == hashlib.sha256(payload).hexdigest()
        assert conn.execute("SELECT run_id FROM poses").fetchone()[0] is not None


def test_database_connection_closes_after_insert_failure(tmp_path):
    pipeline = ProDockPipeline(tmp_path, progress=False, log_colored=False)
    db = PoseDatabase(tmp_path / "failure.db")
    with (
        patch("prodock.core.PoseDatabase", return_value=db),
        patch.object(
            db, "insert_dataframe", side_effect=RuntimeError("insertion failed")
        ),
    ):
        with pytest.raises(RuntimeError, match="insertion failed"):
            pipeline.save_database(df=pd.DataFrame())
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        db.connection.execute("SELECT 1")
