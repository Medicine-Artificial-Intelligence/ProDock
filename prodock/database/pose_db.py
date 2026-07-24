from __future__ import annotations

"""
Core SQLite database wrapper for ProDock pose, score, and interaction storage.

This module defines :class:`PoseDatabase`, a high-level wrapper around a small
normalized SQLite schema for docking campaigns. It stores:

- receptor, ligand, and engine dimension tables
- docked poses and serialized RDKit molecules
- pose-level score payloads
- residue-level interaction rows

Two identifiers can be used to address a pose:

- ``pose_db_id``: internal SQLite integer primary key
- ``pose_id``: optional external stable identifier, for example
  ``"1M17__erlotinib__qvina__pose1"``

When an external ``pose_id`` is not provided, the logical unique pose key is
still the tuple ``(receptor_id, ligand_id, engine, pose_rank)``.

Examples
--------
.. code-block:: python

    import pandas as pd
    from rdkit import Chem
    from prodock.database.core import PoseDatabase

    mol = Chem.MolFromSmiles("CCO")
    df = pd.DataFrame(
        [
            {
                "pose_id": "1M17__erol__qvina__pose1",
                "receptor_id": "1M17",
                "ligand_id": "erol",
                "engine": "qvina",
                "pose_rank": 1,
                "affinity": -8.2,
                "mol": mol,
            }
        ]
    )

    with PoseDatabase("poses.sqlite") as db:
        db.insert_dataframe(df)
        db.add_interaction(
            pose_id="1M17__erol__qvina__pose1",
            interaction_type="Hydrophobic",
            residue_id="LEU23.A",
        )

        poses = db.query_poses(include_interactions=True)
        summary = db.get_interaction_summary()
"""

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd
from rdkit.Chem import rdchem

from .query import (
    build_interaction_where_clause,
    build_pose_where_clause,
    resolve_order_by,
)
from .records import InteractionRecord, PoseRecord, ScoreRecord
from .schema import SCHEMA_SQL
from .serialization import (
    compose_residue_id,
    deserialize_mol,
    json_dumps,
    json_dumps_list,
    json_loads_dict,
    json_loads_int_list,
    make_pose_key,
    parse_residue_id,
    serialize_mol,
)

PathLike = Union[str, Path]


class PoseDatabase:
    """
    SQLite database wrapper for docking pose, score, and interaction storage.

    The wrapper exposes convenience APIs for three common workflows:

    1. Insert docking poses from row mappings or a pandas DataFrame
    2. Store interactions either row-by-row or from pose-keyed dictionaries
    3. Query poses, scores, and interactions with flexible filters

    Tables
    ------
    - ``receptors``: receptor dimension table
    - ``ligands``: ligand dimension table
    - ``engines``: docking engine dimension table
    - ``poses``: pose identity, optional external ``pose_id``, and molecules
    - ``pose_scores``: affinity and score payloads
    - ``interactions``: one row per interaction event or summary interaction

    If a DataFrame does not provide an external ``pose_id`` column, the logical
    unique key remains ``(receptor_id, ligand_id, engine, pose_rank)``. If an
    external ``pose_id`` is present, it is stored and can later be used to
    import interactions from pose-keyed dictionaries.

    :param db_path:
        SQLite database file path.
    :type db_path: PathLike
    :param compress_mol:
        Whether serialized RDKit molecules should be compressed with ``zlib``.
    :type compress_mol: bool
    :param create:
        Whether to create the schema on initialization.
    :type create: bool
    :param timeout:
        SQLite connection timeout in seconds.
    :type timeout: float

    Example
    -------
    .. code-block:: python

        from prodock.database import PoseDatabase

        db = PoseDatabase("poses.sqlite")
        db.insert_dataframe(df)
        db.upsert_interaction_payload(interactions_by_pose)

        fp = db.interaction_fingerprint(mode="binary")
    """

    def __init__(
        self,
        db_path: PathLike,
        *,
        compress_mol: bool = True,
        create: bool = True,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize a database connection.

        :param db_path:
            Path to the SQLite database file.
        :type db_path: PathLike
        :param compress_mol:
            Whether serialized RDKit molecules should be compressed.
        :type compress_mol: bool
        :param create:
            Whether to create the database schema immediately.
        :type create: bool
        :param timeout:
            SQLite connection timeout in seconds.
        :type timeout: float
        """
        self.db_path = Path(db_path)
        self.compress_mol = bool(compress_mol)
        self._conn = sqlite3.connect(str(self.db_path), timeout=timeout)
        self._conn.row_factory = sqlite3.Row
        self._configure_connection()
        if create:
            self.create_schema()

    def _configure_connection(self) -> None:
        """
        Configure recommended SQLite pragmas for this connection.

        :returns:
            None
        :rtype: None
        """
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")

    @property
    def connection(self) -> sqlite3.Connection:
        """
        Return the underlying SQLite connection.

        :returns:
            Active SQLite connection.
        :rtype: sqlite3.Connection
        """
        return self._conn

    def close(self) -> None:
        """
        Close the active SQLite connection.

        :returns:
            None
        :rtype: None
        """
        self._conn.close()

    def __enter__(self) -> "PoseDatabase":
        """
        Enter context-manager scope.

        :returns:
            Current database instance.
        :rtype: PoseDatabase
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Exit context-manager scope and close the connection.

        :param exc_type:
            Exception type raised inside the context, if any.
        :type exc_type: Any
        :param exc:
            Exception instance raised inside the context, if any.
        :type exc: Any
        :param tb:
            Traceback object, if any.
        :type tb: Any

        :returns:
            None
        :rtype: None
        """
        self.close()

    def create_schema(self) -> None:
        """
        Create the database schema if it does not yet exist.

        :returns:
            None
        :rtype: None

        Example
        -------
        .. code-block:: python

            db = PoseDatabase("poses.sqlite", create=False)
            db.create_schema()
        """
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    @staticmethod
    def _norm_value(value: Any) -> Any:
        """
        Normalize pandas-style missing values to ``None``.

        :param value:
            Input scalar or object value.
        :type value: Any

        :returns:
            Normalized value.
        :rtype: Any
        """
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return value

    @staticmethod
    def _clean_pose_id(value: Any) -> Optional[str]:
        """
        Normalize an optional external pose identifier.

        :param value:
            Raw pose identifier.
        :type value: Any

        :returns:
            Stripped pose id or ``None`` when empty.
        :rtype: Optional[str]
        """
        value = PoseDatabase._norm_value(value)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _pose_key_from_row(row: Mapping[str, Any]) -> str:
        """
        Return stored external ``pose_id`` or a generated logical key.

        :param row:
            Mapping-like row object containing pose columns.
        :type row: Mapping[str, Any]

        :returns:
            External pose id when present, otherwise a generated logical key.
        :rtype: str
        """
        try:
            pose_id = row["pose_id"]
        except Exception:
            pose_id = None
        if pose_id:
            return str(pose_id)
        return make_pose_key(
            str(row["receptor_id"]),
            str(row["ligand_id"]),
            str(row["engine"]),
            int(row["pose_rank"]),
        )

    def _ensure_reference_row(
        self,
        table: str,
        id_column: str,
        value: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Insert a dimension-table row if it does not already exist.

        :param table:
            Dimension table name such as ``"receptors"``.
        :type table: str
        :param id_column:
            Identifier column name in the dimension table.
        :type id_column: str
        :param value:
            Identifier value to insert.
        :type value: str
        :param metadata:
            Optional metadata payload to serialize into ``metadata_json``.
        :type metadata: Optional[Mapping[str, Any]]

        :returns:
            None
        :rtype: None
        """
        sql = f"""
            INSERT INTO {table} ({id_column}, metadata_json)
            VALUES (?, ?)
            ON CONFLICT({id_column}) DO NOTHING
        """
        self._conn.execute(sql, (value, json_dumps(metadata)))

    def _insert_or_update_pose_only(
        self,
        *,
        receptor_id: str,
        ligand_id: str,
        engine: str,
        pose_rank: int,
        mol: rdchem.Mol,
        pose_id: Optional[str] = None,
        pose_metadata: Optional[Mapping[str, Any]] = None,
        receptor_metadata: Optional[Mapping[str, Any]] = None,
        ligand_metadata: Optional[Mapping[str, Any]] = None,
        engine_metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """
        Insert or update a pose row and return its internal id.

        :param receptor_id:
            Receptor identifier.
        :type receptor_id: str
        :param ligand_id:
            Ligand identifier.
        :type ligand_id: str
        :param engine:
            Docking engine name.
        :type engine: str
        :param pose_rank:
            One-based pose rank.
        :type pose_rank: int
        :param mol:
            RDKit molecule to serialize and store.
        :type mol: rdchem.Mol
        :param pose_id:
            Optional external stable pose identifier.
        :type pose_id: Optional[str]
        :param pose_metadata:
            Optional pose-level metadata payload.
        :type pose_metadata: Optional[Mapping[str, Any]]
        :param receptor_metadata:
            Optional receptor metadata.
        :type receptor_metadata: Optional[Mapping[str, Any]]
        :param ligand_metadata:
            Optional ligand metadata.
        :type ligand_metadata: Optional[Mapping[str, Any]]
        :param engine_metadata:
            Optional engine metadata.
        :type engine_metadata: Optional[Mapping[str, Any]]

        :returns:
            Internal ``pose_db_id``.
        :rtype: int

        :raises ValueError:
            If ``pose_rank < 1``.
        """
        if pose_rank < 1:
            raise ValueError("pose_rank must be >= 1")

        pose_id = self._clean_pose_id(pose_id)
        self._ensure_reference_row(
            "receptors",
            "receptor_id",
            receptor_id,
            receptor_metadata,
        )
        self._ensure_reference_row(
            "ligands",
            "ligand_id",
            ligand_id,
            ligand_metadata,
        )
        self._ensure_reference_row(
            "engines",
            "engine",
            engine,
            engine_metadata,
        )

        mol_blob = serialize_mol(mol, compress=self.compress_mol, include_props=True)

        if pose_id is not None:
            existing = self._conn.execute(
                "SELECT pose_db_id FROM poses WHERE pose_id = ?",
                (pose_id,),
            ).fetchone()
            if existing is not None:
                self._conn.execute(
                    """
                    UPDATE poses
                    SET receptor_id = ?,
                        ligand_id = ?,
                        engine = ?,
                        pose_rank = ?,
                        mol_blob = ?,
                        mol_is_compressed = ?,
                        metadata_json = ?,
                        pose_id = ?
                    WHERE pose_db_id = ?
                    """,
                    (
                        receptor_id,
                        ligand_id,
                        engine,
                        int(pose_rank),
                        sqlite3.Binary(mol_blob),
                        int(self.compress_mol),
                        json_dumps(pose_metadata),
                        pose_id,
                        int(existing["pose_db_id"]),
                    ),
                )
                return int(existing["pose_db_id"])

        self._conn.execute(
            """
            INSERT INTO poses (
                pose_id,
                receptor_id,
                ligand_id,
                engine,
                pose_rank,
                mol_blob,
                mol_is_compressed,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(receptor_id, ligand_id, engine, pose_rank)
            DO UPDATE SET
                pose_id = COALESCE(excluded.pose_id, poses.pose_id),
                mol_blob = excluded.mol_blob,
                mol_is_compressed = excluded.mol_is_compressed,
                metadata_json = excluded.metadata_json
            """,
            (
                pose_id,
                receptor_id,
                ligand_id,
                engine,
                int(pose_rank),
                sqlite3.Binary(mol_blob),
                int(self.compress_mol),
                json_dumps(pose_metadata),
            ),
        )
        row = self._conn.execute(
            """
            SELECT pose_db_id
            FROM poses
            WHERE receptor_id = ? AND ligand_id = ? AND engine = ? AND pose_rank = ?
            """,
            (receptor_id, ligand_id, engine, int(pose_rank)),
        ).fetchone()
        return int(row["pose_db_id"])

    def upsert_pose(
        self,
        *,
        receptor_id: str,
        ligand_id: str,
        engine: str,
        pose_rank: int,
        affinity: Optional[float],
        mol: rdchem.Mol,
        pose_id: Optional[str] = None,
        pose_metadata: Optional[Mapping[str, Any]] = None,
        score_data: Optional[Mapping[str, Any]] = None,
        score_metadata: Optional[Mapping[str, Any]] = None,
        receptor_metadata: Optional[Mapping[str, Any]] = None,
        ligand_metadata: Optional[Mapping[str, Any]] = None,
        engine_metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """
        Insert or update one docking pose and its score row.

        :param receptor_id:
            Receptor identifier.
        :type receptor_id: str
        :param ligand_id:
            Ligand identifier.
        :type ligand_id: str
        :param engine:
            Docking engine name.
        :type engine: str
        :param pose_rank:
            Pose rank within the receptor-ligand-engine group.
        :type pose_rank: int
        :param affinity:
            Primary affinity score.
        :type affinity: Optional[float]
        :param mol:
            RDKit molecule to store.
        :type mol: rdchem.Mol
        :param pose_id:
            Optional external stable pose identifier.
        :type pose_id: Optional[str]
        :param pose_metadata:
            Optional pose metadata payload.
        :type pose_metadata: Optional[Mapping[str, Any]]
        :param score_data:
            Optional structured score payload.
        :type score_data: Optional[Mapping[str, Any]]
        :param score_metadata:
            Optional score metadata payload.
        :type score_metadata: Optional[Mapping[str, Any]]
        :param receptor_metadata:
            Optional receptor metadata.
        :type receptor_metadata: Optional[Mapping[str, Any]]
        :param ligand_metadata:
            Optional ligand metadata.
        :type ligand_metadata: Optional[Mapping[str, Any]]
        :param engine_metadata:
            Optional engine metadata.
        :type engine_metadata: Optional[Mapping[str, Any]]

        :returns:
            Internal ``pose_db_id``.
        :rtype: int

        Example
        -------
        .. code-block:: python

            from rdkit import Chem

            mol = Chem.MolFromSmiles("CCO")
            pose_db_id = db.upsert_pose(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="qvina",
                pose_rank=1,
                affinity=-8.1,
                mol=mol,
                pose_id="1M17__erlotinib__qvina__pose1",
            )
        """
        with self._conn:
            pose_db_id = self._insert_or_update_pose_only(
                receptor_id=receptor_id,
                ligand_id=ligand_id,
                engine=engine,
                pose_rank=pose_rank,
                mol=mol,
                pose_id=pose_id,
                pose_metadata=pose_metadata,
                receptor_metadata=receptor_metadata,
                ligand_metadata=ligand_metadata,
                engine_metadata=engine_metadata,
            )
            self._conn.execute(
                """
                INSERT INTO pose_scores (
                    pose_db_id,
                    pose_rank,
                    affinity,
                    score_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pose_db_id)
                DO UPDATE SET
                    pose_rank = excluded.pose_rank,
                    affinity = excluded.affinity,
                    score_json = excluded.score_json,
                    metadata_json = excluded.metadata_json
                """,
                (
                    pose_db_id,
                    int(pose_rank),
                    affinity,
                    json_dumps(score_data),
                    json_dumps(score_metadata),
                ),
            )
        return pose_db_id

    def insert_many(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        replace: bool = True,
    ) -> None:
        """
        Insert many pose rows inside one transaction.

        Each row should contain at least ``receptor_id``, ``ligand_id``,
        ``engine``, ``pose_rank``, ``affinity``, and ``mol``. An optional
        external string ``pose_id`` is supported.

        :param rows:
            Iterable of row-like mappings.
        :type rows: Iterable[Mapping[str, Any]]
        :param replace:
            Whether to upsert existing logical keys.
        :type replace: bool

        :returns:
            None
        :rtype: None

        Example
        -------
        .. code-block:: python

            rows = [
                {
                    "pose_id": "1M17__erol__qvina__pose1",
                    "receptor_id": "1M17",
                    "ligand_id": "erol",
                    "engine": "qvina",
                    "pose_rank": 1,
                    "affinity": -8.2,
                    "mol": mol,
                }
            ]
            db.insert_many(rows, replace=True)
        """
        with self._conn:
            for row in rows:
                receptor_id = str(row["receptor_id"])
                ligand_id = str(row["ligand_id"])
                engine = str(row["engine"])
                pose_rank = int(row["pose_rank"])
                affinity = self._norm_value(row.get("affinity"))
                mol = row["mol"]
                pose_id = self._clean_pose_id(row.get("pose_id"))

                pose_metadata = self._norm_value(
                    row.get("pose_metadata") or row.get("metadata")
                )
                score_data = self._norm_value(row.get("score_data"))
                score_metadata = self._norm_value(row.get("score_metadata"))
                receptor_metadata = self._norm_value(row.get("receptor_metadata"))
                ligand_metadata = self._norm_value(row.get("ligand_metadata"))
                engine_metadata = self._norm_value(row.get("engine_metadata"))

                if replace:
                    self.upsert_pose(
                        receptor_id=receptor_id,
                        ligand_id=ligand_id,
                        engine=engine,
                        pose_rank=pose_rank,
                        affinity=affinity,
                        mol=mol,
                        pose_id=pose_id,
                        pose_metadata=pose_metadata,
                        score_data=score_data,
                        score_metadata=score_metadata,
                        receptor_metadata=receptor_metadata,
                        ligand_metadata=ligand_metadata,
                        engine_metadata=engine_metadata,
                    )
                    continue

                if pose_rank < 1:
                    raise ValueError("pose_rank must be >= 1")

                self._ensure_reference_row(
                    "receptors",
                    "receptor_id",
                    receptor_id,
                    receptor_metadata,
                )
                self._ensure_reference_row(
                    "ligands",
                    "ligand_id",
                    ligand_id,
                    ligand_metadata,
                )
                self._ensure_reference_row(
                    "engines",
                    "engine",
                    engine,
                    engine_metadata,
                )

                mol_blob = serialize_mol(
                    mol,
                    compress=self.compress_mol,
                    include_props=True,
                )

                cur = self._conn.execute(
                    """
                    INSERT INTO poses (
                        pose_id,
                        receptor_id,
                        ligand_id,
                        engine,
                        pose_rank,
                        mol_blob,
                        mol_is_compressed,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pose_id,
                        receptor_id,
                        ligand_id,
                        engine,
                        pose_rank,
                        sqlite3.Binary(mol_blob),
                        int(self.compress_mol),
                        json_dumps(pose_metadata),
                    ),
                )
                pose_db_id = int(cur.lastrowid)

                self._conn.execute(
                    """
                    INSERT INTO pose_scores (
                        pose_db_id,
                        pose_rank,
                        affinity,
                        score_json,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        pose_db_id,
                        pose_rank,
                        affinity,
                        json_dumps(score_data),
                        json_dumps(score_metadata),
                    ),
                )

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        *,
        replace: bool = True,
        interactions_by_pose: Optional[Mapping[str, Mapping[str, Any]]] = None,
        replace_interactions: bool = True,
    ) -> None:
        """
        Insert a pandas DataFrame of docking poses.

        Required columns are ``receptor_id``, ``ligand_id``, ``engine``,
        ``pose_rank``, ``affinity``, and ``mol``. An optional ``pose_id``
        column is stored when present.

        If ``interactions_by_pose`` is supplied, it must be keyed by the stored
        external ``pose_id`` values.

        :param df:
            Input DataFrame.
        :type df: pd.DataFrame
        :param replace:
            Whether existing pose rows should be updated.
        :type replace: bool
        :param interactions_by_pose:
            Optional interaction payload keyed by external ``pose_id``.
        :type interactions_by_pose: Optional[Mapping[str, Mapping[str, Any]]]
        :param replace_interactions:
            Whether existing interactions for affected poses should first be
            deleted.
        :type replace_interactions: bool

        :returns:
            None
        :rtype: None

        :raises ValueError:
            If required DataFrame columns are missing.

        Example
        -------
        .. code-block:: python

            db.insert_dataframe(df, replace=True)
        """
        required = {
            "receptor_id",
            "ligand_id",
            "engine",
            "pose_rank",
            "affinity",
            "mol",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing required DataFrame columns: {missing}")

        rows: list[dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            rows.append(
                {
                    "pose_id": self._clean_pose_id(row.get("pose_id")),
                    "receptor_id": str(row["receptor_id"]),
                    "ligand_id": str(row["ligand_id"]),
                    "engine": str(row["engine"]),
                    "pose_rank": int(row["pose_rank"]),
                    "affinity": self._norm_value(row.get("affinity")),
                    "mol": row["mol"],
                    "pose_metadata": self._norm_value(
                        row.get("pose_metadata") or row.get("metadata")
                    ),
                    "score_data": self._norm_value(row.get("score_data")),
                    "score_metadata": self._norm_value(row.get("score_metadata")),
                    "receptor_metadata": self._norm_value(row.get("receptor_metadata")),
                    "ligand_metadata": self._norm_value(row.get("ligand_metadata")),
                    "engine_metadata": self._norm_value(row.get("engine_metadata")),
                }
            )
        self.insert_many(rows, replace=replace)

        if interactions_by_pose is not None:
            self.upsert_interaction_payload(
                interactions_by_pose,
                replace=replace_interactions,
            )

    @classmethod
    def from_dataframe(
        cls,
        db_path: PathLike,
        df: pd.DataFrame,
        *,
        compress_mol: bool = True,
        replace: bool = True,
        interactions_by_pose: Optional[Mapping[str, Mapping[str, Any]]] = None,
        replace_interactions: bool = True,
    ) -> "PoseDatabase":
        """
        Build a new database file from a DataFrame.

        :param db_path:
            Output SQLite file path.
        :type db_path: PathLike
        :param df:
            Input DataFrame containing docking poses.
        :type df: pd.DataFrame
        :param compress_mol:
            Whether stored molecule blobs should be compressed.
        :type compress_mol: bool
        :param replace:
            Whether duplicate logical keys should be updated.
        :type replace: bool
        :param interactions_by_pose:
            Optional interaction payloads keyed by external pose id.
        :type interactions_by_pose: Optional[Mapping[str, Mapping[str, Any]]]
        :param replace_interactions:
            Whether to replace existing interactions when interaction payloads
            are supplied.
        :type replace_interactions: bool

        :returns:
            Initialized database instance.
        :rtype: PoseDatabase
        """
        db = cls(db_path, compress_mol=compress_mol, create=True)
        db.insert_dataframe(
            df,
            replace=replace,
            interactions_by_pose=interactions_by_pose,
            replace_interactions=replace_interactions,
        )
        return db

    def _resolve_pose_db_id(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[str] = None,
        receptor_id: Optional[str] = None,
        ligand_id: Optional[str] = None,
        engine: Optional[str] = None,
        pose_rank: Optional[int] = None,
    ) -> int:
        """
        Resolve an internal pose id from several supported selectors.

        :param pose_db_id:
            Internal SQLite pose identifier.
        :type pose_db_id: Optional[int]
        :param pose_id:
            External stable pose identifier.
        :type pose_id: Optional[str]
        :param receptor_id:
            Receptor identifier for logical-key lookup.
        :type receptor_id: Optional[str]
        :param ligand_id:
            Ligand identifier for logical-key lookup.
        :type ligand_id: Optional[str]
        :param engine:
            Engine name for logical-key lookup.
        :type engine: Optional[str]
        :param pose_rank:
            Pose rank for logical-key lookup.
        :type pose_rank: Optional[int]

        :returns:
            Resolved internal ``pose_db_id``.
        :rtype: int

        :raises KeyError:
            If the requested pose cannot be found.
        :raises ValueError:
            If not enough information is supplied.
        """
        if pose_db_id is not None:
            row = self._conn.execute(
                "SELECT pose_db_id FROM poses WHERE pose_db_id = ?",
                (int(pose_db_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown pose_db_id: {pose_db_id}")
            return int(row["pose_db_id"])

        pose_id = self._clean_pose_id(pose_id)
        if pose_id is not None:
            row = self._conn.execute(
                "SELECT pose_db_id FROM poses WHERE pose_id = ?",
                (pose_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown pose_id: {pose_id}")
            return int(row["pose_db_id"])

        if None in (receptor_id, ligand_id, engine, pose_rank):
            raise ValueError(
                "Provide either pose_db_id, pose_id, or the full logical key: "
                "receptor_id, ligand_id, engine, and pose_rank."
            )

        row = self._conn.execute(
            """
            SELECT pose_db_id
            FROM poses
            WHERE receptor_id = ? AND ligand_id = ? AND engine = ? AND pose_rank = ?
            """,
            (str(receptor_id), str(ligand_id), str(engine), int(pose_rank)),
        ).fetchone()
        if row is None:
            raise KeyError(
                "Unknown pose logical key: "
                f"({receptor_id}, {ligand_id}, {engine}, {pose_rank})"
            )
        return int(row["pose_db_id"])

    def query_poses(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[Union[str, Sequence[str]]] = None,
        receptor_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_id: Optional[Union[str, Sequence[str]]] = None,
        engine: Optional[Union[str, Sequence[str]]] = None,
        pose_rank: Optional[int] = None,
        top_rank: Optional[int] = None,
        affinity_threshold: Optional[float] = None,
        affinity_min: Optional[float] = None,
        interaction_type: Optional[Union[str, Sequence[str]]] = None,
        residue_id: Optional[Union[str, Sequence[str]]] = None,
        chain_id: Optional[Union[str, Sequence[str]]] = None,
        residue_name: Optional[Union[str, Sequence[str]]] = None,
        residue_number: Optional[int] = None,
        include_mol: bool = True,
        include_interactions: bool = False,
        interaction_mode: str = "summary",
        as_dataframe: bool = False,
        order_by: Optional[Union[str, Sequence[str]]] = None,
        limit: Optional[int] = None,
    ) -> Union[list[PoseRecord], pd.DataFrame]:
        """
        Query poses using flexible logical and interaction-aware filters.

        Interaction filters can be used to return only poses that contain
        particular interactions, for example ``interaction_type="Hydrophobic"``
        and ``residue_id="LEU23.A"``.

        If ``include_interactions`` is enabled, pose rows are enriched with
        either summary or detailed interaction payloads.

        :param pose_db_id:
            Optional internal pose id filter.
        :type pose_db_id: Optional[int]
        :param pose_id:
            Optional external pose id or sequence of ids.
        :type pose_id: Optional[Union[str, Sequence[str]]]
        :param receptor_id:
            Optional receptor id or sequence of receptor ids.
        :type receptor_id: Optional[Union[str, Sequence[str]]]
        :param ligand_id:
            Optional ligand id or sequence of ligand ids.
        :type ligand_id: Optional[Union[str, Sequence[str]]]
        :param engine:
            Optional engine name or sequence of engine names.
        :type engine: Optional[Union[str, Sequence[str]]]
        :param pose_rank:
            Optional exact pose rank.
        :type pose_rank: Optional[int]
        :param top_rank:
            Optional maximum pose rank to keep.
        :type top_rank: Optional[int]
        :param affinity_threshold:
            Optional maximum affinity threshold.
        :type affinity_threshold: Optional[float]
        :param affinity_min:
            Optional minimum affinity threshold.
        :type affinity_min: Optional[float]
        :param interaction_type:
            Optional interaction type filter.
        :type interaction_type: Optional[Union[str, Sequence[str]]]
        :param residue_id:
            Optional residue id filter such as ``"LEU23.A"``.
        :type residue_id: Optional[Union[str, Sequence[str]]]
        :param chain_id:
            Optional chain filter.
        :type chain_id: Optional[Union[str, Sequence[str]]]
        :param residue_name:
            Optional residue-name filter.
        :type residue_name: Optional[Union[str, Sequence[str]]]
        :param residue_number:
            Optional residue-number filter.
        :type residue_number: Optional[int]
        :param include_mol:
            Whether deserialized RDKit molecules should be included.
        :type include_mol: bool
        :param include_interactions:
            Whether interaction payloads should be attached.
        :type include_interactions: bool
        :param interaction_mode:
            Interaction payload style, either ``"summary"`` or ``"detailed"``.
        :type interaction_mode: str
        :param as_dataframe:
            Whether to return a pandas DataFrame instead of dataclass records.
        :type as_dataframe: bool
        :param order_by:
            Optional ordering clause definition passed to
            :func:`resolve_order_by`.
        :type order_by: Optional[Union[str, Sequence[str]]]
        :param limit:
            Optional maximum number of returned rows.
        :type limit: Optional[int]

        :returns:
            List of :class:`PoseRecord` objects or a pandas DataFrame.
        :rtype: Union[list[PoseRecord], pd.DataFrame]
        """
        where_sql, params = build_pose_where_clause(
            pose_db_id=pose_db_id,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            top_rank=top_rank,
            affinity_threshold=affinity_threshold,
            affinity_min=affinity_min,
            interaction_type=interaction_type,
            residue_id=residue_id,
            chain_id=chain_id,
            residue_name=residue_name,
            residue_number=residue_number,
        )
        order_sql = resolve_order_by(order_by)
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(int(limit))

        rows = self._conn.execute(
            """
            SELECT
                p.pose_db_id,
                p.pose_id,
                p.receptor_id,
                p.ligand_id,
                p.engine,
                p.pose_rank,
                s.affinity,
                s.score_json,
                s.metadata_json AS score_metadata_json,
                p.mol_blob,
                p.mol_is_compressed,
                p.metadata_json AS pose_metadata_json,
                p.created_at
            FROM poses AS p
            LEFT JOIN pose_scores AS s
                ON s.pose_db_id = p.pose_db_id
            """ + where_sql + order_sql + limit_sql,
            params,
        ).fetchall()

        interaction_map: dict[int, Any] = {}
        if include_interactions and rows:
            pose_db_ids = [int(row["pose_db_id"]) for row in rows]
            if interaction_mode == "summary":
                interaction_map = self.get_interaction_summary(
                    pose_db_id=pose_db_ids,
                    return_by="pose_db_id",
                )
            elif interaction_mode == "detailed":
                interaction_map = self.get_interaction_details(
                    pose_db_id=pose_db_ids,
                    return_by="pose_db_id",
                )
            else:
                raise ValueError("interaction_mode must be 'summary' or 'detailed'")

        if as_dataframe:
            payload: list[dict[str, Any]] = []
            for row in rows:
                item = {
                    "pose_db_id": row["pose_db_id"],
                    "pose_id": row["pose_id"],
                    "pose_key": self._pose_key_from_row(row),
                    "receptor_id": row["receptor_id"],
                    "ligand_id": row["ligand_id"],
                    "engine": row["engine"],
                    "pose_rank": row["pose_rank"],
                    "affinity": row["affinity"],
                    "pose_metadata": json_loads_dict(row["pose_metadata_json"]),
                    "score_data": json_loads_dict(row["score_json"]),
                    "score_metadata": json_loads_dict(row["score_metadata_json"]),
                    "created_at": row["created_at"],
                }
                if include_mol:
                    item["mol"] = deserialize_mol(
                        row["mol_blob"],
                        compressed=bool(row["mol_is_compressed"]),
                    )
                if include_interactions:
                    key = (
                        "interaction_summary"
                        if interaction_mode == "summary"
                        else "interaction_details"
                    )
                    item[key] = interaction_map.get(int(row["pose_db_id"]), {})
                payload.append(item)
            return pd.DataFrame(payload)

        records: list[PoseRecord] = []
        for row in rows:
            mol: Optional[rdchem.Mol] = None
            if include_mol:
                mol = deserialize_mol(
                    row["mol_blob"],
                    compressed=bool(row["mol_is_compressed"]),
                )

            summary: dict[str, list[str]] = {}
            details: dict[str, Any] = {}
            if include_interactions:
                if interaction_mode == "summary":
                    summary = interaction_map.get(int(row["pose_db_id"]), {})
                else:
                    details = interaction_map.get(int(row["pose_db_id"]), {})

            records.append(
                PoseRecord(
                    pose_db_id=row["pose_db_id"],
                    pose_id=row["pose_id"],
                    receptor_id=row["receptor_id"],
                    ligand_id=row["ligand_id"],
                    engine=row["engine"],
                    pose_rank=row["pose_rank"],
                    affinity=row["affinity"],
                    mol=mol,
                    pose_metadata=json_loads_dict(row["pose_metadata_json"]),
                    score_data=json_loads_dict(row["score_json"]),
                    score_metadata=json_loads_dict(row["score_metadata_json"]),
                    interaction_summary=summary,
                    interaction_details=details,
                    created_at=row["created_at"],
                )
            )
        return records

    def get_pose(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[str] = None,
        receptor_id: Optional[str] = None,
        ligand_id: Optional[str] = None,
        engine: Optional[str] = None,
        pose_rank: Optional[int] = None,
        include_mol: bool = True,
        include_interactions: bool = False,
        interaction_mode: str = "summary",
    ) -> Optional[PoseRecord]:
        """
        Fetch one exact pose by internal id, external id, or logical key.

        :param pose_db_id:
            Internal pose id.
        :type pose_db_id: Optional[int]
        :param pose_id:
            External stable pose id.
        :type pose_id: Optional[str]
        :param receptor_id:
            Receptor identifier.
        :type receptor_id: Optional[str]
        :param ligand_id:
            Ligand identifier.
        :type ligand_id: Optional[str]
        :param engine:
            Engine name.
        :type engine: Optional[str]
        :param pose_rank:
            Pose rank within the receptor-ligand-engine group.
        :type pose_rank: Optional[int]
        :param include_mol:
            Whether to include the RDKit molecule.
        :type include_mol: bool
        :param include_interactions:
            Whether to attach interactions.
        :type include_interactions: bool
        :param interaction_mode:
            ``"summary"`` or ``"detailed"``.
        :type interaction_mode: str

        :returns:
            Matching pose or ``None`` if no match exists.
        :rtype: Optional[PoseRecord]
        """
        rows = self.query_poses(
            pose_db_id=pose_db_id,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            include_mol=include_mol,
            include_interactions=include_interactions,
            interaction_mode=interaction_mode,
            as_dataframe=False,
            limit=1,
        )
        return rows[0] if rows else None

    def query_scores(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[Union[str, Sequence[str]]] = None,
        receptor_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_id: Optional[Union[str, Sequence[str]]] = None,
        engine: Optional[Union[str, Sequence[str]]] = None,
        pose_rank: Optional[int] = None,
        top_rank: Optional[int] = None,
        affinity_threshold: Optional[float] = None,
        affinity_min: Optional[float] = None,
        as_dataframe: bool = False,
        order_by: Optional[Union[str, Sequence[str]]] = None,
        limit: Optional[int] = None,
    ) -> Union[list[ScoreRecord], pd.DataFrame]:
        """
        Query the dedicated ``pose_scores`` table joined to pose identity.

        :param pose_db_id:
            Optional internal pose id filter.
        :type pose_db_id: Optional[int]
        :param pose_id:
            Optional external pose id or sequence of ids.
        :type pose_id: Optional[Union[str, Sequence[str]]]
        :param receptor_id:
            Optional receptor id filter.
        :type receptor_id: Optional[Union[str, Sequence[str]]]
        :param ligand_id:
            Optional ligand id filter.
        :type ligand_id: Optional[Union[str, Sequence[str]]]
        :param engine:
            Optional engine filter.
        :type engine: Optional[Union[str, Sequence[str]]]
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param top_rank:
            Optional maximum pose rank.
        :type top_rank: Optional[int]
        :param affinity_threshold:
            Optional maximum affinity threshold.
        :type affinity_threshold: Optional[float]
        :param affinity_min:
            Optional minimum affinity threshold.
        :type affinity_min: Optional[float]
        :param as_dataframe:
            Whether to return a DataFrame.
        :type as_dataframe: bool
        :param order_by:
            Optional ordering clause definition.
        :type order_by: Optional[Union[str, Sequence[str]]]
        :param limit:
            Optional maximum number of rows.
        :type limit: Optional[int]

        :returns:
            List of :class:`ScoreRecord` or a DataFrame.
        :rtype: Union[list[ScoreRecord], pd.DataFrame]
        """
        where_sql, params = build_pose_where_clause(
            pose_db_id=pose_db_id,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            top_rank=top_rank,
            affinity_threshold=affinity_threshold,
            affinity_min=affinity_min,
        )
        order_sql = resolve_order_by(order_by)
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(int(limit))

        rows = self._conn.execute(
            """
            SELECT
                p.pose_db_id,
                p.pose_id,
                p.receptor_id,
                p.ligand_id,
                p.engine,
                p.pose_rank,
                s.affinity,
                s.score_json,
                s.metadata_json
            FROM poses AS p
            LEFT JOIN pose_scores AS s
                ON s.pose_db_id = p.pose_db_id
            """ + where_sql + order_sql + limit_sql,
            params,
        ).fetchall()

        if as_dataframe:
            return pd.DataFrame(
                [
                    {
                        "pose_db_id": row["pose_db_id"],
                        "pose_id": row["pose_id"],
                        "pose_key": self._pose_key_from_row(row),
                        "receptor_id": row["receptor_id"],
                        "ligand_id": row["ligand_id"],
                        "engine": row["engine"],
                        "pose_rank": row["pose_rank"],
                        "affinity": row["affinity"],
                        "score_data": json_loads_dict(row["score_json"]),
                        "metadata": json_loads_dict(row["metadata_json"]),
                    }
                    for row in rows
                ]
            )

        return [
            ScoreRecord(
                pose_db_id=row["pose_db_id"],
                pose_id=row["pose_id"],
                receptor_id=row["receptor_id"],
                ligand_id=row["ligand_id"],
                engine=row["engine"],
                pose_rank=row["pose_rank"],
                affinity=row["affinity"],
                score_data=json_loads_dict(row["score_json"]),
                metadata=json_loads_dict(row["metadata_json"]),
            )
            for row in rows
        ]

    def count_poses(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[Union[str, Sequence[str]]] = None,
        receptor_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_id: Optional[Union[str, Sequence[str]]] = None,
        engine: Optional[Union[str, Sequence[str]]] = None,
        pose_rank: Optional[int] = None,
        top_rank: Optional[int] = None,
        affinity_threshold: Optional[float] = None,
        affinity_min: Optional[float] = None,
        interaction_type: Optional[Union[str, Sequence[str]]] = None,
        residue_id: Optional[Union[str, Sequence[str]]] = None,
        chain_id: Optional[Union[str, Sequence[str]]] = None,
        residue_name: Optional[Union[str, Sequence[str]]] = None,
        residue_number: Optional[int] = None,
    ) -> int:
        """
        Count poses matching the supplied filters.

        :returns:
            Number of matching pose rows.
        :rtype: int
        """
        where_sql, params = build_pose_where_clause(
            pose_db_id=pose_db_id,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            top_rank=top_rank,
            affinity_threshold=affinity_threshold,
            affinity_min=affinity_min,
            interaction_type=interaction_type,
            residue_id=residue_id,
            chain_id=chain_id,
            residue_name=residue_name,
            residue_number=residue_number,
        )
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM poses AS p
            LEFT JOIN pose_scores AS s
                ON s.pose_db_id = p.pose_db_id
            """ + where_sql,
            params,
        ).fetchone()
        return int(row["n"])

    @staticmethod
    def _interaction_upsert_suffix() -> str:
        """
        Return the reusable ``ON CONFLICT`` clause for interaction upserts.

        :returns:
            SQL suffix for upserting interaction rows.
        :rtype: str
        """
        return """
            ON CONFLICT(pose_db_id, interaction_type, residue_id, occurrence_index)
            DO UPDATE SET
                chain_id = excluded.chain_id,
                residue_name = excluded.residue_name,
                residue_number = excluded.residue_number,
                ligand_residue = excluded.ligand_residue,
                ligand_atom_indices_json = excluded.ligand_atom_indices_json,
                protein_atom_indices_json = excluded.protein_atom_indices_json,
                ligand_parent_atom_indices_json = excluded.ligand_parent_atom_indices_json,
                protein_parent_atom_indices_json = excluded.protein_parent_atom_indices_json,
                distance = excluded.distance,
                angle = excluded.angle,
                metadata_json = excluded.metadata_json
        """

    def add_interaction(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[str] = None,
        receptor_id: Optional[str] = None,
        ligand_id: Optional[str] = None,
        engine: Optional[str] = None,
        pose_rank: Optional[int] = None,
        interaction_type: str,
        chain_id: Optional[str] = None,
        residue_name: Optional[str] = None,
        residue_number: Optional[int] = None,
        residue_id: Optional[str] = None,
        ligand_residue: Optional[str] = None,
        occurrence_index: int = 0,
        ligand_atom_indices: Optional[Sequence[int]] = None,
        protein_atom_indices: Optional[Sequence[int]] = None,
        ligand_parent_atom_indices: Optional[Sequence[int]] = None,
        protein_parent_atom_indices: Optional[Sequence[int]] = None,
        distance: Optional[float] = None,
        angle: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        replace: bool = False,
    ) -> int:
        """
        Insert one interaction linked to a stored pose.

        A single row can represent either a summarized interaction or one
        detailed interaction event.

        :param pose_db_id:
            Internal pose id.
        :type pose_db_id: Optional[int]
        :param pose_id:
            External stable pose id.
        :type pose_id: Optional[str]
        :param receptor_id:
            Receptor id for logical-key lookup.
        :type receptor_id: Optional[str]
        :param ligand_id:
            Ligand id for logical-key lookup.
        :type ligand_id: Optional[str]
        :param engine:
            Engine name for logical-key lookup.
        :type engine: Optional[str]
        :param pose_rank:
            Pose rank for logical-key lookup.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Interaction family, for example ``"Hydrophobic"``.
        :type interaction_type: str
        :param chain_id:
            Protein chain identifier.
        :type chain_id: Optional[str]
        :param residue_name:
            Residue name, for example ``"LEU"``.
        :type residue_name: Optional[str]
        :param residue_number:
            Residue sequence number.
        :type residue_number: Optional[int]
        :param residue_id:
            Combined residue id such as ``"LEU23.A"``.
        :type residue_id: Optional[str]
        :param ligand_residue:
            Ligand residue identifier if available.
        :type ligand_residue: Optional[str]
        :param occurrence_index:
            Zero-based event index within one pose / residue / interaction type.
        :type occurrence_index: int
        :param ligand_atom_indices:
            Ligand atom indices for the specific event.
        :type ligand_atom_indices: Optional[Sequence[int]]
        :param protein_atom_indices:
            Protein atom indices for the specific event.
        :type protein_atom_indices: Optional[Sequence[int]]
        :param ligand_parent_atom_indices:
            Parent ligand atom indices when available.
        :type ligand_parent_atom_indices: Optional[Sequence[int]]
        :param protein_parent_atom_indices:
            Parent protein atom indices when available.
        :type protein_parent_atom_indices: Optional[Sequence[int]]
        :param distance:
            Optional interaction distance.
        :type distance: Optional[float]
        :param angle:
            Optional interaction angle.
        :type angle: Optional[float]
        :param metadata:
            Optional arbitrary metadata payload.
        :type metadata: Optional[Mapping[str, Any]]
        :param replace:
            Whether an existing unique interaction row should be updated.
        :type replace: bool

        :returns:
            New interaction identifier.
        :rtype: int
        """
        resolved_pose_db_id = self._resolve_pose_db_id(
            pose_db_id=pose_db_id,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
        )

        residue_id = self._norm_value(residue_id)
        if residue_id is not None:
            parsed_name, parsed_number, parsed_chain = parse_residue_id(str(residue_id))
            residue_name = residue_name or parsed_name
            residue_number = (
                residue_number if residue_number is not None else parsed_number
            )
            chain_id = chain_id or parsed_chain
        else:
            residue_id = compose_residue_id(residue_name, residue_number, chain_id)

        sql = """
            INSERT INTO interactions (
                pose_db_id,
                interaction_type,
                chain_id,
                residue_name,
                residue_number,
                residue_id,
                ligand_residue,
                occurrence_index,
                ligand_atom_indices_json,
                protein_atom_indices_json,
                ligand_parent_atom_indices_json,
                protein_parent_atom_indices_json,
                distance,
                angle,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        if replace:
            sql += self._interaction_upsert_suffix()

        with self._conn:
            cur = self._conn.execute(
                sql,
                (
                    resolved_pose_db_id,
                    str(interaction_type),
                    chain_id,
                    residue_name,
                    residue_number,
                    residue_id,
                    ligand_residue,
                    int(occurrence_index),
                    json_dumps_list(ligand_atom_indices),
                    json_dumps_list(protein_atom_indices),
                    json_dumps_list(ligand_parent_atom_indices),
                    json_dumps_list(protein_parent_atom_indices),
                    distance,
                    angle,
                    json_dumps(metadata),
                ),
            )
        return int(cur.lastrowid)

    def delete_interactions_for_pose(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[str] = None,
        receptor_id: Optional[str] = None,
        ligand_id: Optional[str] = None,
        engine: Optional[str] = None,
        pose_rank: Optional[int] = None,
    ) -> int:
        """
        Delete all interactions linked to a single pose.

        :returns:
            Number of deleted interaction rows.
        :rtype: int
        """
        resolved_pose_db_id = self._resolve_pose_db_id(
            pose_db_id=pose_db_id,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
        )
        cur = self._conn.execute(
            "DELETE FROM interactions WHERE pose_db_id = ?",
            (resolved_pose_db_id,),
        )
        self._conn.commit()
        return int(cur.rowcount)

    @staticmethod
    def _flatten_one_pose_interaction_payload(
        payload: Optional[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Flatten one summary or detailed interaction payload.

        Supported payload shapes are:

        - summary:
          ``{"Hydrophobic": ["LEU23.A", "VAL31.A"]}``
        - detailed:
          ``{"Hydrophobic": {"LEU23.A": [{...}, {...}]}}``

        ``None`` and empty mappings are treated as "no interactions" and return
        an empty list.

        :param payload:
            Pose-specific interaction payload.
        :type payload: Optional[Mapping[str, Any]]

        :returns:
            Flat list of row dictionaries ready for insertion.
        :rtype: list[dict[str, Any]]

        :raises TypeError:
            If the payload shape is not recognized.
        """
        if payload is None:
            return []

        if not isinstance(payload, Mapping):
            raise TypeError(
                "Interaction payload must be a mapping or None. "
                f"Got {type(payload).__name__}."
            )

        rows: list[dict[str, Any]] = []
        for interaction_type, value in payload.items():
            if value is None:
                continue

            # Detailed format:
            # {"Hydrophobic": {"LEU23.A": [{...}, {...}]}}
            if isinstance(value, Mapping):
                for residue_id, events in value.items():
                    if events is None:
                        continue

                    if isinstance(events, Mapping):
                        event_list = [events]
                    elif isinstance(events, Sequence) and not isinstance(
                        events,
                        (str, bytes),
                    ):
                        event_list = list(events)
                    else:
                        raise TypeError(
                            "Detailed interaction payload must map residue ids "
                            "to a dict or list of dicts."
                        )

                    for occurrence_index, event in enumerate(event_list):
                        event_dict = dict(event or {})
                        protein_residue = (
                            event_dict.get("protein_residue") or residue_id
                        )
                        ligand_residue = event_dict.get("ligand_residue")
                        distance = event_dict.get("distance")
                        angle = event_dict.get("angle")
                        indices = event_dict.get("indices") or {}
                        parent_indices = event_dict.get("parent_indices") or {}
                        metadata = dict(event_dict.get("metadata") or {})

                        extra_fields = {
                            key: val
                            for key, val in event_dict.items()
                            if key
                            not in {
                                "protein_residue",
                                "ligand_residue",
                                "distance",
                                "angle",
                                "indices",
                                "parent_indices",
                                "metadata",
                            }
                        }
                        if extra_fields:
                            metadata["extra_fields"] = extra_fields
                        metadata.setdefault("source_format", "detailed")

                        rows.append(
                            {
                                "interaction_type": str(interaction_type),
                                "residue_id": str(protein_residue),
                                "ligand_residue": ligand_residue,
                                "occurrence_index": int(occurrence_index),
                                "ligand_atom_indices": indices.get("ligand") or [],
                                "protein_atom_indices": indices.get("protein") or [],
                                "ligand_parent_atom_indices": (
                                    parent_indices.get("ligand") or []
                                ),
                                "protein_parent_atom_indices": (
                                    parent_indices.get("protein") or []
                                ),
                                "distance": distance,
                                "angle": angle,
                                "metadata": metadata,
                            }
                        )

            # Summary format:
            # {"Hydrophobic": ["LEU23.A", "VAL31.A"]}
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for residue in value:
                    if residue is None:
                        continue
                    rows.append(
                        {
                            "interaction_type": str(interaction_type),
                            "residue_id": str(residue),
                            "ligand_residue": None,
                            "occurrence_index": 0,
                            "ligand_atom_indices": [],
                            "protein_atom_indices": [],
                            "ligand_parent_atom_indices": [],
                            "protein_parent_atom_indices": [],
                            "distance": None,
                            "angle": None,
                            "metadata": {"source_format": "summary"},
                        }
                    )
            else:
                raise TypeError(
                    "Interaction payload values must be either a list of residue "
                    "ids, a nested residue->events mapping, or None."
                )

        return rows

    def upsert_interaction_payload(
        self,
        interactions_by_pose: Mapping[str, Optional[Mapping[str, Any]]],
        *,
        replace: bool = True,
    ) -> None:
        """
        Insert interaction payloads keyed by external ``pose_id``.

        Supported payload formats per pose are:

        - summary:
          ``{"Hydrophobic": ["LEU23.A", "VAL31.A"]}``
        - detailed:
          ``{"Hydrophobic": {"LEU23.A": [{...}, {...}]}}``

        ``None`` or empty payloads are treated as "no interactions". If
        ``replace=True``, existing interactions for that pose are deleted and no
        new rows are inserted.

        :param interactions_by_pose:
            Mapping from external ``pose_id`` to interaction payload.
        :type interactions_by_pose: Mapping[str, Optional[Mapping[str, Any]]]
        :param replace:
            Whether existing interactions for each affected pose should first be
            deleted.
        :type replace: bool

        :returns:
            None
        :rtype: None
        """
        insert_sql = """
            INSERT INTO interactions (
                pose_db_id,
                interaction_type,
                chain_id,
                residue_name,
                residue_number,
                residue_id,
                ligand_residue,
                occurrence_index,
                ligand_atom_indices_json,
                protein_atom_indices_json,
                ligand_parent_atom_indices_json,
                protein_parent_atom_indices_json,
                distance,
                angle,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """ + self._interaction_upsert_suffix()

        with self._conn:
            for pose_id, payload in interactions_by_pose.items():
                resolved_pose_db_id = self._resolve_pose_db_id(pose_id=str(pose_id))

                if replace:
                    self._conn.execute(
                        "DELETE FROM interactions WHERE pose_db_id = ?",
                        (resolved_pose_db_id,),
                    )

                # No interactions for this pose -> done
                if payload is None:
                    continue

                if not isinstance(payload, Mapping):
                    raise TypeError(
                        "Each interaction payload must be a mapping or None. "
                        f"Pose {pose_id!r} has type {type(payload).__name__}."
                    )

                rows = self._flatten_one_pose_interaction_payload(payload)
                if not rows:
                    continue

                for row in rows:
                    residue_name, residue_number, chain_id = parse_residue_id(
                        row.get("residue_id")
                    )
                    self._conn.execute(
                        insert_sql,
                        (
                            resolved_pose_db_id,
                            row["interaction_type"],
                            chain_id,
                            residue_name,
                            residue_number,
                            row.get("residue_id"),
                            row.get("ligand_residue"),
                            int(row.get("occurrence_index", 0)),
                            json_dumps_list(row.get("ligand_atom_indices")),
                            json_dumps_list(row.get("protein_atom_indices")),
                            json_dumps_list(row.get("ligand_parent_atom_indices")),
                            json_dumps_list(row.get("protein_parent_atom_indices")),
                            row.get("distance"),
                            row.get("angle"),
                            json_dumps(row.get("metadata")),
                        ),
                    )

    def insert_interactions(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        replace: bool = False,
    ) -> None:
        """
        Insert many interaction rows inside one transaction.

        Each row must provide either ``pose_db_id``, external ``pose_id``, or
        the full logical pose key.

        :param rows:
            Iterable of interaction row mappings.
        :type rows: Iterable[Mapping[str, Any]]
        :param replace:
            Whether conflicting interaction rows should be updated.
        :type replace: bool

        :returns:
            None
        :rtype: None
        """
        with self._conn:
            for row in rows:
                self.add_interaction(replace=replace, **dict(row))

    def query_interactions(
        self,
        *,
        interaction_id: Optional[int] = None,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[Union[str, Sequence[str]]] = None,
        receptor_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_id: Optional[Union[str, Sequence[str]]] = None,
        engine: Optional[Union[str, Sequence[str]]] = None,
        pose_rank: Optional[int] = None,
        interaction_type: Optional[Union[str, Sequence[str]]] = None,
        chain_id: Optional[Union[str, Sequence[str]]] = None,
        residue_name: Optional[Union[str, Sequence[str]]] = None,
        residue_number: Optional[int] = None,
        residue_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_residue: Optional[Union[str, Sequence[str]]] = None,
        as_dataframe: bool = False,
        order_by: Optional[Union[str, Sequence[str]]] = None,
        limit: Optional[int] = None,
    ) -> Union[list[InteractionRecord], pd.DataFrame]:
        """
        Query stored interactions using pose-level and residue-level filters.

        :param interaction_id:
            Optional interaction primary-key filter.
        :type interaction_id: Optional[int]
        :param pose_db_id:
            Optional internal pose id filter.
        :type pose_db_id: Optional[int]
        :param pose_id:
            Optional external pose id or sequence of ids.
        :type pose_id: Optional[Union[str, Sequence[str]]]
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: Optional[Union[str, Sequence[str]]]
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: Optional[Union[str, Sequence[str]]]
        :param engine:
            Optional engine filter.
        :type engine: Optional[Union[str, Sequence[str]]]
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Optional interaction type filter.
        :type interaction_type: Optional[Union[str, Sequence[str]]]
        :param chain_id:
            Optional chain filter.
        :type chain_id: Optional[Union[str, Sequence[str]]]
        :param residue_name:
            Optional residue-name filter.
        :type residue_name: Optional[Union[str, Sequence[str]]]
        :param residue_number:
            Optional residue-number filter.
        :type residue_number: Optional[int]
        :param residue_id:
            Optional combined residue-id filter.
        :type residue_id: Optional[Union[str, Sequence[str]]]
        :param ligand_residue:
            Optional ligand residue filter.
        :type ligand_residue: Optional[Union[str, Sequence[str]]]
        :param as_dataframe:
            Whether to return a DataFrame.
        :type as_dataframe: bool
        :param order_by:
            Optional ordering clause definition.
        :type order_by: Optional[Union[str, Sequence[str]]]
        :param limit:
            Optional maximum number of rows.
        :type limit: Optional[int]

        :returns:
            List of :class:`InteractionRecord` or a DataFrame.
        :rtype: Union[list[InteractionRecord], pd.DataFrame]
        """
        where_sql, params = build_interaction_where_clause(
            interaction_id=interaction_id,
            pose_db_id=pose_db_id,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            interaction_type=interaction_type,
            chain_id=chain_id,
            residue_name=residue_name,
            residue_number=residue_number,
            residue_id=residue_id,
            ligand_residue=ligand_residue,
        )
        order_sql = resolve_order_by(
            order_by
            or ["pose_db_id", "interaction_type", "residue_id", "occurrence_index"]
        )
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(int(limit))

        rows = self._conn.execute(
            """
            SELECT
                i.interaction_id,
                i.pose_db_id,
                p.pose_id,
                p.receptor_id,
                p.ligand_id,
                p.engine,
                p.pose_rank,
                i.interaction_type,
                i.chain_id,
                i.residue_name,
                i.residue_number,
                i.residue_id,
                i.ligand_residue,
                i.occurrence_index,
                i.ligand_atom_indices_json,
                i.protein_atom_indices_json,
                i.ligand_parent_atom_indices_json,
                i.protein_parent_atom_indices_json,
                i.distance,
                i.angle,
                i.metadata_json,
                i.created_at
            FROM interactions AS i
            INNER JOIN poses AS p
                ON p.pose_db_id = i.pose_db_id
            """ + where_sql + order_sql + limit_sql,
            params,
        ).fetchall()

        if as_dataframe:
            return pd.DataFrame(
                [
                    {
                        "interaction_id": row["interaction_id"],
                        "pose_db_id": row["pose_db_id"],
                        "pose_id": row["pose_id"],
                        "pose_key": self._pose_key_from_row(row),
                        "receptor_id": row["receptor_id"],
                        "ligand_id": row["ligand_id"],
                        "engine": row["engine"],
                        "pose_rank": row["pose_rank"],
                        "interaction_type": row["interaction_type"],
                        "chain_id": row["chain_id"],
                        "residue_name": row["residue_name"],
                        "residue_number": row["residue_number"],
                        "residue_id": row["residue_id"],
                        "ligand_residue": row["ligand_residue"],
                        "occurrence_index": row["occurrence_index"],
                        "ligand_atom_indices": json_loads_int_list(
                            row["ligand_atom_indices_json"]
                        ),
                        "protein_atom_indices": json_loads_int_list(
                            row["protein_atom_indices_json"]
                        ),
                        "ligand_parent_atom_indices": json_loads_int_list(
                            row["ligand_parent_atom_indices_json"]
                        ),
                        "protein_parent_atom_indices": json_loads_int_list(
                            row["protein_parent_atom_indices_json"]
                        ),
                        "distance": row["distance"],
                        "angle": row["angle"],
                        "metadata": json_loads_dict(row["metadata_json"]),
                        "created_at": row["created_at"],
                    }
                    for row in rows
                ]
            )

        return [
            InteractionRecord(
                interaction_id=row["interaction_id"],
                pose_db_id=row["pose_db_id"],
                pose_id=row["pose_id"],
                receptor_id=row["receptor_id"],
                ligand_id=row["ligand_id"],
                engine=row["engine"],
                pose_rank=row["pose_rank"],
                interaction_type=row["interaction_type"],
                chain_id=row["chain_id"],
                residue_name=row["residue_name"],
                residue_number=row["residue_number"],
                residue_id=row["residue_id"],
                ligand_residue=row["ligand_residue"],
                occurrence_index=row["occurrence_index"],
                ligand_atom_indices=json_loads_int_list(
                    row["ligand_atom_indices_json"]
                ),
                protein_atom_indices=json_loads_int_list(
                    row["protein_atom_indices_json"]
                ),
                ligand_parent_atom_indices=json_loads_int_list(
                    row["ligand_parent_atom_indices_json"]
                ),
                protein_parent_atom_indices=json_loads_int_list(
                    row["protein_parent_atom_indices_json"]
                ),
                distance=row["distance"],
                angle=row["angle"],
                metadata=json_loads_dict(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _group_key_from_record(
        row: Mapping[str, Any],
        *,
        return_by: str,
    ) -> Union[int, str]:
        """
        Resolve the grouping key used in nested interaction outputs.

        :param row:
            Row-like mapping with pose columns.
        :type row: Mapping[str, Any]
        :param return_by:
            One of ``"pose_db_id"``, ``"pose_id"``, or ``"pose_key"``.
        :type return_by: str

        :returns:
            Group key used in nested output mappings.
        :rtype: Union[int, str]

        :raises ValueError:
            If ``return_by`` is invalid or ``pose_id`` is required but missing.
        """
        if return_by == "pose_db_id":
            return int(row["pose_db_id"])
        if return_by == "pose_id":
            pose_id = row.get("pose_id")
            if pose_id is None:
                raise ValueError(
                    "return_by='pose_id' requires all selected poses to have "
                    "a stored pose_id"
                )
            return str(pose_id)
        if return_by == "pose_key":
            pose_id = row.get("pose_id")
            if pose_id:
                return str(pose_id)
            return make_pose_key(
                str(row["receptor_id"]),
                str(row["ligand_id"]),
                str(row["engine"]),
                int(row["pose_rank"]),
            )
        raise ValueError("return_by must be 'pose_db_id', 'pose_id', or 'pose_key'")

    def get_interaction_summary(
        self,
        *,
        pose_db_id: Optional[Union[int, Sequence[int]]] = None,
        pose_id: Optional[Union[str, Sequence[str]]] = None,
        receptor_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_id: Optional[Union[str, Sequence[str]]] = None,
        engine: Optional[Union[str, Sequence[str]]] = None,
        pose_rank: Optional[int] = None,
        interaction_type: Optional[Union[str, Sequence[str]]] = None,
        residue_id: Optional[Union[str, Sequence[str]]] = None,
        return_by: str = "pose_key",
    ) -> dict[Union[int, str], dict[str, list[str]]]:
        """
        Return summarized interactions grouped by pose.

        The output payload is compatible with the compact interaction format:
        ``{pose_key: {interaction_type: [residue_id, ...]}}``.

        :param pose_db_id:
            Optional pose id or sequence of internal pose ids.
        :type pose_db_id: Optional[Union[int, Sequence[int]]]
        :param pose_id:
            Optional external pose id filter.
        :type pose_id: Optional[Union[str, Sequence[str]]]
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: Optional[Union[str, Sequence[str]]]
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: Optional[Union[str, Sequence[str]]]
        :param engine:
            Optional engine filter.
        :type engine: Optional[Union[str, Sequence[str]]]
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Optional interaction-type filter.
        :type interaction_type: Optional[Union[str, Sequence[str]]]
        :param residue_id:
            Optional residue-id filter.
        :type residue_id: Optional[Union[str, Sequence[str]]]
        :param return_by:
            One of ``"pose_db_id"``, ``"pose_id"``, or ``"pose_key"``.
        :type return_by: str

        :returns:
            Nested summary mapping grouped by pose.
        :rtype: dict[Union[int, str], dict[str, list[str]]]
        """
        if isinstance(pose_db_id, int):
            pose_db_id_filter = pose_db_id
        else:
            pose_db_id_filter = None

        frame = self.query_interactions(
            pose_db_id=pose_db_id_filter,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            interaction_type=interaction_type,
            residue_id=residue_id,
            as_dataframe=True,
        )
        if (
            not frame.empty
            and pose_db_id is not None
            and not isinstance(pose_db_id, int)
        ):
            ids = {int(x) for x in pose_db_id}
            frame = frame[frame["pose_db_id"].isin(ids)].copy()

        out: dict[Union[int, str], dict[str, list[str]]] = {}
        for row in frame.to_dict(orient="records"):
            key = self._group_key_from_record(row, return_by=return_by)
            bucket = out.setdefault(key, {})
            residues = bucket.setdefault(str(row["interaction_type"]), [])
            residue_text = row.get("residue_id")
            if residue_text is None:
                continue
            residue_text = str(residue_text)
            if residue_text not in residues:
                residues.append(residue_text)

        for bucket in out.values():
            for residues in bucket.values():
                residues.sort()
        return out

    def get_interaction_details(
        self,
        *,
        pose_db_id: Optional[Union[int, Sequence[int]]] = None,
        pose_id: Optional[Union[str, Sequence[str]]] = None,
        receptor_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_id: Optional[Union[str, Sequence[str]]] = None,
        engine: Optional[Union[str, Sequence[str]]] = None,
        pose_rank: Optional[int] = None,
        interaction_type: Optional[Union[str, Sequence[str]]] = None,
        residue_id: Optional[Union[str, Sequence[str]]] = None,
        return_by: str = "pose_key",
    ) -> dict[Union[int, str], dict[str, dict[str, list[dict[str, Any]]]]]:
        """
        Return detailed interactions grouped by pose.

        The output mirrors the nested detailed format:
        ``{pose_key: {interaction_type: {residue_id: [event, ...]}}}``.

        :param pose_db_id:
            Optional pose id or sequence of internal pose ids.
        :type pose_db_id: Optional[Union[int, Sequence[int]]]
        :param pose_id:
            Optional external pose id filter.
        :type pose_id: Optional[Union[str, Sequence[str]]]
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: Optional[Union[str, Sequence[str]]]
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: Optional[Union[str, Sequence[str]]]
        :param engine:
            Optional engine filter.
        :type engine: Optional[Union[str, Sequence[str]]]
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Optional interaction-type filter.
        :type interaction_type: Optional[Union[str, Sequence[str]]]
        :param residue_id:
            Optional residue-id filter.
        :type residue_id: Optional[Union[str, Sequence[str]]]
        :param return_by:
            One of ``"pose_db_id"``, ``"pose_id"``, or ``"pose_key"``.
        :type return_by: str

        :returns:
            Nested detailed mapping grouped by pose.
        :rtype: dict[Union[int, str], dict[str, dict[str, list[dict[str, Any]]]]]
        """
        if isinstance(pose_db_id, int):
            pose_db_id_filter = pose_db_id
        else:
            pose_db_id_filter = None

        frame = self.query_interactions(
            pose_db_id=pose_db_id_filter,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            interaction_type=interaction_type,
            residue_id=residue_id,
            as_dataframe=True,
            order_by=[
                "pose_db_id",
                "interaction_type",
                "residue_id",
                "occurrence_index",
            ],
        )
        if (
            not frame.empty
            and pose_db_id is not None
            and not isinstance(pose_db_id, int)
        ):
            ids = {int(x) for x in pose_db_id}
            frame = frame[frame["pose_db_id"].isin(ids)].copy()

        out: dict[Union[int, str], dict[str, dict[str, list[dict[str, Any]]]]] = {}
        for row in frame.to_dict(orient="records"):
            key = self._group_key_from_record(row, return_by=return_by)
            type_bucket = out.setdefault(key, {}).setdefault(
                str(row["interaction_type"]),
                {},
            )
            residue_text = row.get("residue_id") or "UNKNOWN"
            events = type_bucket.setdefault(str(residue_text), [])

            distance = row.get("distance")
            angle = row.get("angle")
            if self._norm_value(distance) is None:
                distance = None
            if self._norm_value(angle) is None:
                angle = None

            events.append(
                {
                    "protein_residue": row.get("residue_id"),
                    "ligand_residue": row.get("ligand_residue"),
                    "distance": distance,
                    "angle": angle,
                    "indices": {
                        "ligand": row.get("ligand_atom_indices") or [],
                        "protein": row.get("protein_atom_indices") or [],
                    },
                    "parent_indices": {
                        "ligand": row.get("ligand_parent_atom_indices") or [],
                        "protein": row.get("protein_parent_atom_indices") or [],
                    },
                    "metadata": row.get("metadata") or {},
                }
            )
        return out

    def interaction_fingerprint(
        self,
        *,
        pose_db_id: Optional[Union[int, Sequence[int]]] = None,
        pose_id: Optional[Union[str, Sequence[str]]] = None,
        receptor_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_id: Optional[Union[str, Sequence[str]]] = None,
        engine: Optional[Union[str, Sequence[str]]] = None,
        pose_rank: Optional[int] = None,
        interaction_type: Optional[Union[str, Sequence[str]]] = None,
        residue_id: Optional[Union[str, Sequence[str]]] = None,
        mode: str = "binary",
        feature_sep: str = "::",
        index_by: str = "pose_key",
    ) -> pd.DataFrame:
        """
        Build a pose-by-feature interaction fingerprint matrix.

        Features are named as ``<interaction_type><feature_sep><residue_id>``.

        :param pose_db_id:
            Optional pose id or sequence of internal pose ids.
        :type pose_db_id: Optional[Union[int, Sequence[int]]]
        :param pose_id:
            Optional external pose id filter.
        :type pose_id: Optional[Union[str, Sequence[str]]]
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: Optional[Union[str, Sequence[str]]]
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: Optional[Union[str, Sequence[str]]]
        :param engine:
            Optional engine filter.
        :type engine: Optional[Union[str, Sequence[str]]]
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Optional interaction-type filter.
        :type interaction_type: Optional[Union[str, Sequence[str]]]
        :param residue_id:
            Optional residue-id filter.
        :type residue_id: Optional[Union[str, Sequence[str]]]
        :param mode:
            Either ``"binary"`` or ``"count"``.
        :type mode: str
        :param feature_sep:
            Separator used when building feature names.
        :type feature_sep: str
        :param index_by:
            One of ``"pose_db_id"``, ``"pose_id"``, or ``"pose_key"``.
        :type index_by: str

        :returns:
            Fingerprint matrix as a pandas DataFrame.
        :rtype: pd.DataFrame

        :raises ValueError:
            If ``mode`` or ``index_by`` is invalid.
        """
        if mode not in {"binary", "count"}:
            raise ValueError("mode must be 'binary' or 'count'")
        if index_by not in {"pose_db_id", "pose_id", "pose_key"}:
            raise ValueError("index_by must be 'pose_db_id', 'pose_id', or 'pose_key'")

        if isinstance(pose_db_id, int):
            pose_db_id_filter = pose_db_id
        else:
            pose_db_id_filter = None

        frame = self.query_interactions(
            pose_db_id=pose_db_id_filter,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            interaction_type=interaction_type,
            residue_id=residue_id,
            as_dataframe=True,
        )
        if (
            not frame.empty
            and pose_db_id is not None
            and not isinstance(pose_db_id, int)
        ):
            ids = {int(x) for x in pose_db_id}
            frame = frame[frame["pose_db_id"].isin(ids)].copy()

        if frame.empty:
            return pd.DataFrame()

        if index_by == "pose_db_id":
            frame["_pose_index"] = frame["pose_db_id"]
        elif index_by == "pose_id":
            if frame["pose_id"].isna().any():
                raise ValueError(
                    "index_by='pose_id' requires all selected poses to have "
                    "a stored pose_id"
                )
            frame["_pose_index"] = frame["pose_id"]
        else:
            frame["_pose_index"] = frame["pose_key"]

        frame["_feature"] = (
            frame["interaction_type"].astype(str)
            + feature_sep
            + frame["residue_id"].astype(str)
        )
        counts = (
            frame.groupby(["_pose_index", "_feature"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        if mode == "binary":
            counts = (counts > 0).astype(int)
        return counts

    def summarize(self) -> pd.DataFrame:
        """
        Return one summary row per receptor-ligand-engine group.

        :returns:
            Summary DataFrame with pose counts and best affinity values.
        :rtype: pd.DataFrame
        """
        rows = self._conn.execute("""
            SELECT
                p.receptor_id,
                p.ligand_id,
                p.engine,
                COUNT(DISTINCT p.pose_db_id) AS n_poses,
                MIN(s.affinity) AS best_affinity,
                MAX(p.pose_rank) AS max_pose_rank,
                COUNT(i.interaction_id) AS n_interactions
            FROM poses AS p
            LEFT JOIN pose_scores AS s
                ON s.pose_db_id = p.pose_db_id
            LEFT JOIN interactions AS i
                ON i.pose_db_id = p.pose_db_id
            GROUP BY p.receptor_id, p.ligand_id, p.engine
            ORDER BY p.receptor_id, p.ligand_id, p.engine
            """).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def list_receptors(self) -> list[str]:
        """
        List all receptor identifiers present in the database.

        :returns:
            Sorted receptor identifiers.
        :rtype: list[str]
        """
        rows = self._conn.execute(
            "SELECT receptor_id FROM receptors ORDER BY receptor_id"
        ).fetchall()
        return [str(row["receptor_id"]) for row in rows]

    def list_ligands(self) -> list[str]:
        """
        List all ligand identifiers present in the database.

        :returns:
            Sorted ligand identifiers.
        :rtype: list[str]
        """
        rows = self._conn.execute(
            "SELECT ligand_id FROM ligands ORDER BY ligand_id"
        ).fetchall()
        return [str(row["ligand_id"]) for row in rows]

    def list_engines(self) -> list[str]:
        """
        List all docking engine names present in the database.

        :returns:
            Sorted engine names.
        :rtype: list[str]
        """
        rows = self._conn.execute(
            "SELECT engine FROM engines ORDER BY engine"
        ).fetchall()
        return [str(row["engine"]) for row in rows]

    def delete_poses(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[Union[str, Sequence[str]]] = None,
        receptor_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_id: Optional[Union[str, Sequence[str]]] = None,
        engine: Optional[Union[str, Sequence[str]]] = None,
        pose_rank: Optional[int] = None,
        top_rank: Optional[int] = None,
        affinity_threshold: Optional[float] = None,
        affinity_min: Optional[float] = None,
    ) -> int:
        """
        Delete poses matching the supplied filters.

        At least one filter must be provided to prevent accidental deletion of
        the entire table.

        :param pose_db_id:
            Optional internal pose id filter.
        :type pose_db_id: Optional[int]
        :param pose_id:
            Optional external pose id or sequence of ids.
        :type pose_id: Optional[Union[str, Sequence[str]]]
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: Optional[Union[str, Sequence[str]]]
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: Optional[Union[str, Sequence[str]]]
        :param engine:
            Optional engine filter.
        :type engine: Optional[Union[str, Sequence[str]]]
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param top_rank:
            Optional maximum pose rank.
        :type top_rank: Optional[int]
        :param affinity_threshold:
            Optional maximum affinity threshold.
        :type affinity_threshold: Optional[float]
        :param affinity_min:
            Optional minimum affinity threshold.
        :type affinity_min: Optional[float]

        :returns:
            Number of deleted pose rows.
        :rtype: int

        :raises ValueError:
            If no filters are provided.
        """
        where_sql, params = build_pose_where_clause(
            pose_db_id=pose_db_id,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            top_rank=top_rank,
            affinity_threshold=affinity_threshold,
            affinity_min=affinity_min,
        )
        if not where_sql:
            raise ValueError("Refusing to delete all poses without any filter.")

        sql = (
            "DELETE FROM poses WHERE pose_db_id IN ("
            "SELECT p.pose_db_id FROM poses AS p "
            "LEFT JOIN pose_scores AS s ON s.pose_db_id = p.pose_db_id"
            + where_sql
            + ")"
        )
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return int(cur.rowcount)

    def delete_interactions(
        self,
        *,
        interaction_id: Optional[int] = None,
        pose_db_id: Optional[int] = None,
        pose_id: Optional[Union[str, Sequence[str]]] = None,
        receptor_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_id: Optional[Union[str, Sequence[str]]] = None,
        engine: Optional[Union[str, Sequence[str]]] = None,
        pose_rank: Optional[int] = None,
        interaction_type: Optional[Union[str, Sequence[str]]] = None,
        chain_id: Optional[Union[str, Sequence[str]]] = None,
        residue_name: Optional[Union[str, Sequence[str]]] = None,
        residue_number: Optional[int] = None,
        residue_id: Optional[Union[str, Sequence[str]]] = None,
        ligand_residue: Optional[Union[str, Sequence[str]]] = None,
    ) -> int:
        """
        Delete interactions matching the supplied filters.

        :param interaction_id:
            Optional interaction primary-key filter.
        :type interaction_id: Optional[int]
        :param pose_db_id:
            Optional pose id filter.
        :type pose_db_id: Optional[int]
        :param pose_id:
            Optional external pose id filter.
        :type pose_id: Optional[Union[str, Sequence[str]]]
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: Optional[Union[str, Sequence[str]]]
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: Optional[Union[str, Sequence[str]]]
        :param engine:
            Optional engine filter.
        :type engine: Optional[Union[str, Sequence[str]]]
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Optional interaction-type filter.
        :type interaction_type: Optional[Union[str, Sequence[str]]]
        :param chain_id:
            Optional chain filter.
        :type chain_id: Optional[Union[str, Sequence[str]]]
        :param residue_name:
            Optional residue-name filter.
        :type residue_name: Optional[Union[str, Sequence[str]]]
        :param residue_number:
            Optional residue-number filter.
        :type residue_number: Optional[int]
        :param residue_id:
            Optional residue-id filter.
        :type residue_id: Optional[Union[str, Sequence[str]]]
        :param ligand_residue:
            Optional ligand-residue filter.
        :type ligand_residue: Optional[Union[str, Sequence[str]]]

        :returns:
            Number of deleted interaction rows.
        :rtype: int

        :raises ValueError:
            If no filters are provided.
        """
        where_sql, params = build_interaction_where_clause(
            interaction_id=interaction_id,
            pose_db_id=pose_db_id,
            pose_id=pose_id,
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine,
            pose_rank=pose_rank,
            interaction_type=interaction_type,
            chain_id=chain_id,
            residue_name=residue_name,
            residue_number=residue_number,
            residue_id=residue_id,
            ligand_residue=ligand_residue,
        )
        if not where_sql:
            raise ValueError("Refusing to delete all interactions without any filter.")

        sql = (
            "DELETE FROM interactions WHERE interaction_id IN ("
            "SELECT i.interaction_id FROM interactions AS i "
            "INNER JOIN poses AS p ON p.pose_db_id = i.pose_db_id" + where_sql + ")"
        )
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return int(cur.rowcount)

    def vacuum(self) -> None:
        """
        Run SQLite ``VACUUM`` to compact the database file.

        :returns:
            None
        :rtype: None
        """
        self._conn.execute("VACUUM")
