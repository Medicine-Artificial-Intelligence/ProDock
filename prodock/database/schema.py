from __future__ import annotations

"""SQLite schema definition for ProDock pose, score, and interaction storage."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS receptors (
    receptor_id   TEXT PRIMARY KEY,
    pdb_id        TEXT,
    receptor_name TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ligands (
    ligand_id     TEXT PRIMARY KEY,
    smiles        TEXT,
    inchikey      TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS engines (
    engine        TEXT PRIMARY KEY,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    config_json     TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(config_json)),
    prodock_version TEXT,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS poses (
    pose_db_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pose_id             TEXT UNIQUE,
    receptor_id         TEXT NOT NULL,
    ligand_id           TEXT NOT NULL,
    engine              TEXT NOT NULL,
    pose_rank           INTEGER NOT NULL CHECK (pose_rank >= 1),
    run_id              INTEGER,
    mol_blob            BLOB NOT NULL,
    mol_is_compressed   INTEGER NOT NULL DEFAULT 1 CHECK (mol_is_compressed IN (0, 1)),
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (receptor_id) REFERENCES receptors(receptor_id) ON DELETE CASCADE,
    FOREIGN KEY (ligand_id) REFERENCES ligands(ligand_id) ON DELETE CASCADE,
    FOREIGN KEY (engine) REFERENCES engines(engine) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE SET NULL,
    UNIQUE (receptor_id, ligand_id, engine, pose_rank)
);

CREATE TABLE IF NOT EXISTS pose_scores (
    pose_db_id     INTEGER PRIMARY KEY,
    pose_rank      INTEGER NOT NULL CHECK (pose_rank >= 1),
    affinity       REAL,
    score_json     TEXT NOT NULL DEFAULT '{}',
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (pose_db_id) REFERENCES poses(pose_db_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    pose_db_id                        INTEGER NOT NULL,
    interaction_type                  TEXT NOT NULL,
    chain_id                          TEXT,
    residue_name                      TEXT,
    residue_number                    INTEGER,
    residue_id                        TEXT,
    ligand_residue                    TEXT,
    occurrence_index                  INTEGER NOT NULL DEFAULT 0 CHECK (occurrence_index >= 0),
    ligand_atom_indices_json          TEXT NOT NULL DEFAULT '[]',
    protein_atom_indices_json         TEXT NOT NULL DEFAULT '[]',
    ligand_parent_atom_indices_json   TEXT NOT NULL DEFAULT '[]',
    protein_parent_atom_indices_json  TEXT NOT NULL DEFAULT '[]',
    distance                          REAL,
    angle                             REAL,
    metadata_json                     TEXT NOT NULL DEFAULT '{}',
    created_at                        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pose_db_id) REFERENCES poses(pose_db_id) ON DELETE CASCADE,
    UNIQUE (pose_db_id, interaction_type, residue_id, occurrence_index)
);

CREATE TRIGGER IF NOT EXISTS trg_pose_scores_insert_rank_check
BEFORE INSERT ON pose_scores
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN (SELECT pose_rank FROM poses WHERE pose_db_id = NEW.pose_db_id) IS NULL THEN
            RAISE(ABORT, 'pose_db_id not found in poses')
        WHEN NEW.pose_rank != (SELECT pose_rank FROM poses WHERE pose_db_id = NEW.pose_db_id) THEN
            RAISE(ABORT, 'pose_scores.pose_rank must match poses.pose_rank')
    END;
END;

CREATE TRIGGER IF NOT EXISTS trg_pose_scores_update_rank_check
BEFORE UPDATE OF pose_rank ON pose_scores
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN NEW.pose_rank != (SELECT pose_rank FROM poses WHERE pose_db_id = NEW.pose_db_id) THEN
            RAISE(ABORT, 'pose_scores.pose_rank must match poses.pose_rank')
    END;
END;

CREATE INDEX IF NOT EXISTS idx_poses_pose_id
    ON poses (pose_id);

CREATE INDEX IF NOT EXISTS idx_poses_receptor
    ON poses (receptor_id);

CREATE INDEX IF NOT EXISTS idx_poses_ligand
    ON poses (ligand_id);

CREATE INDEX IF NOT EXISTS idx_poses_engine
    ON poses (engine);

CREATE INDEX IF NOT EXISTS idx_poses_rank
    ON poses (pose_rank);

CREATE INDEX IF NOT EXISTS idx_poses_filter_main
    ON poses (receptor_id, ligand_id, engine, pose_rank);

CREATE INDEX IF NOT EXISTS idx_scores_affinity
    ON pose_scores (affinity);

CREATE INDEX IF NOT EXISTS idx_scores_pose_rank
    ON pose_scores (pose_rank);

CREATE INDEX IF NOT EXISTS idx_interactions_pose
    ON interactions (pose_db_id);

CREATE INDEX IF NOT EXISTS idx_interactions_type
    ON interactions (interaction_type);

CREATE INDEX IF NOT EXISTS idx_interactions_residue_id
    ON interactions (residue_id);

CREATE INDEX IF NOT EXISTS idx_interactions_residue
    ON interactions (chain_id, residue_name, residue_number);

CREATE INDEX IF NOT EXISTS idx_interactions_pose_type_residue
    ON interactions (pose_db_id, interaction_type, residue_id);
"""

# Objects that reference the typed columns added in schema version 2. These are
# applied only after :meth:`PoseDatabase._migrate` has ensured those columns
# exist, so they are safe on databases upgraded in place.
VIEW_SQL = """
CREATE INDEX IF NOT EXISTS idx_poses_run
    ON poses (run_id);

CREATE INDEX IF NOT EXISTS idx_ligands_inchikey
    ON ligands (inchikey);

CREATE VIEW IF NOT EXISTS v_poses_flat AS
SELECT
    p.pose_db_id,
    p.pose_id,
    p.receptor_id,
    p.ligand_id,
    p.engine,
    p.pose_rank,
    p.run_id,
    p.created_at,
    r.pdb_id,
    r.receptor_name,
    l.smiles,
    l.inchikey,
    s.affinity,
    s.score_json
FROM poses AS p
LEFT JOIN receptors AS r ON p.receptor_id = r.receptor_id
LEFT JOIN ligands AS l ON p.ligand_id = l.ligand_id
LEFT JOIN pose_scores AS s ON p.pose_db_id = s.pose_db_id;
"""
