from __future__ import annotations

"""
Standalone read/query API for the ProDock SQLite database.

This module provides :class:`PoseQuery`, a read-focused companion to
:class:`PoseDatabase`. Unlike :class:`PoseDatabase`, which is oriented toward
schema creation and data insertion/updating, :class:`PoseQuery` is designed to
open an existing database and expose convenient, analysis-friendly query
methods.

The class supports:

- querying stored poses
- querying pose scores
- querying residue-level interactions
- building compact interaction summaries
- rebuilding detailed interaction payloads
- generating interaction fingerprint matrices
- listing receptors, ligands, and engines
- computing campaign-level summary tables

The query API can be used either directly from a database path or by reusing an
existing SQLite connection.

Example
-------
Open an existing ProDock database and run several common queries:

.. code-block:: python

    from prodock.database import PoseQuery

    q = PoseQuery("prodock.db")

    # Query poses as a pandas DataFrame
    poses = q.poses(
        receptor_id="1M17",
        engine="qvina",
        as_dataframe=True,
    )
    print(poses[["pose_id", "pose_rank", "affinity"]].head())

    # Query one exact pose
    top_pose = q.pose(
        receptor_id="1M17",
        ligand_id="erlotinib",
        engine="qvina",
        pose_rank=1,
    )
    print(top_pose)

    # Query interactions
    hydrophobic = q.interactions(
        receptor_id="1M17",
        interaction_type="Hydrophobic",
        as_dataframe=True,
    )
    print(hydrophobic.head())

    # Query poses and attach compact interaction summaries
    pose_table = q.poses(
        receptor_id="1M17",
        include_interactions=True,
        interaction_mode="summary",
        as_dataframe=True,
    )
    print(pose_table[["pose_id", "interaction_summary"]].head())

    # Build a binary interaction fingerprint matrix
    fp = q.fingerprint(
        receptor_id="1M17",
        mode="binary",
        index_by="pose_key",
    )
    print(fp.head())

    # Show a campaign-level summary
    print(q.summary())

The class can also reuse an existing connection:

.. code-block:: python

    from prodock.database import PoseDatabase, PoseQuery

    db = PoseDatabase("prodock.db", create=False)
    q = PoseQuery(connection=db.connection)

    print(q.receptors())

By default, opening from ``db_path`` uses SQLite read-only mode.
"""

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd
from rdkit.Chem import rdchem

from .query import (
    build_interaction_where_clause,
    build_pose_where_clause,
    resolve_order_by,
)
from .records import InteractionRecord, PoseRecord, ScoreRecord
from .serialization import (
    deserialize_mol,
    json_loads_dict,
    json_loads_int_list,
    make_pose_key,
)

PathLike = Union[str, Path]
FilterStr = Optional[Union[str, Sequence[str]]]
FilterInt = Optional[Union[int, Sequence[int]]]


class PoseQuery:
    """
    Standalone query client for an existing ProDock SQLite database.

    The class opens an existing database file or attaches to an existing SQLite
    connection, then provides read/query helpers for stored poses, score rows,
    interactions, interaction summaries, and fingerprint matrices.

    :param db_path:
        Path to an existing ProDock SQLite database. Required when
        ``connection`` is not supplied.
    :type db_path: Optional[PathLike]
    :param connection:
        Existing SQLite connection to reuse.
    :type connection: Optional[sqlite3.Connection]
    :param timeout:
        SQLite connection timeout in seconds.
    :type timeout: float
    :param read_only:
        Whether to open ``db_path`` in SQLite read-only mode.
    :type read_only: bool

    :raises ValueError:
        If neither ``db_path`` nor ``connection`` is provided.
    :raises FileNotFoundError:
        If ``db_path`` does not exist.

    Example
    -------
    .. code-block:: python

        from prodock.database import PoseQuery

        q = PoseQuery("prodock.db")
        df = q.poses(as_dataframe=True)
        print(df.head())
    """

    def __init__(
        self,
        db_path: Optional[PathLike] = None,
        *,
        connection: Optional[sqlite3.Connection] = None,
        timeout: float = 30.0,
        read_only: bool = True,
    ) -> None:
        if connection is None and db_path is None:
            raise ValueError("Provide either db_path or connection.")

        self._owns_connection = connection is None
        self.db_path = Path(db_path) if db_path is not None else None

        if connection is not None:
            self._conn = connection
            self._conn.row_factory = sqlite3.Row
            return

        assert self.db_path is not None
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file does not exist: {self.db_path}")

        if read_only:
            uri = f"file:{self.db_path}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, timeout=timeout)
        else:
            self._conn = sqlite3.connect(str(self.db_path), timeout=timeout)

        self._conn.row_factory = sqlite3.Row

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
        Close the connection owned by this query object.

        Connections passed in through ``connection=...`` are not closed here.

        :returns:
            None
        :rtype: None
        """
        if self._owns_connection:
            self._conn.close()

    def __enter__(self) -> "PoseQuery":
        """
        Enter context-manager scope.

        :returns:
            Current query instance.
        :rtype: PoseQuery
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Exit context-manager scope and close the owned connection.

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
        value = PoseQuery._norm_value(value)
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

    def poses(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: FilterStr = None,
        receptor_id: FilterStr = None,
        ligand_id: FilterStr = None,
        engine: FilterStr = None,
        pose_rank: Optional[int] = None,
        top_rank: Optional[int] = None,
        affinity_threshold: Optional[float] = None,
        affinity_min: Optional[float] = None,
        interaction_type: FilterStr = None,
        residue_id: FilterStr = None,
        chain_id: FilterStr = None,
        residue_name: FilterStr = None,
        residue_number: Optional[int] = None,
        include_mol: bool = True,
        include_interactions: bool = False,
        interaction_mode: str = "summary",
        as_dataframe: bool = False,
        order_by: Optional[Union[str, Sequence[str]]] = None,
        limit: Optional[int] = None,
    ) -> Union[list[PoseRecord], pd.DataFrame]:
        """
        Query poses using logical and interaction-aware filters.

        Interaction filters can be used to return only poses that contain
        particular interaction patterns, for example
        ``interaction_type="Hydrophobic"`` and ``residue_id="LEU23.A"``.

        When ``include_interactions`` is enabled, each returned pose is enriched
        with either a compact summary payload or a detailed nested interaction
        payload.

        :param pose_db_id:
            Optional internal pose id filter.
        :type pose_db_id: Optional[int]
        :param pose_id:
            Optional external pose id or sequence of ids.
        :type pose_id: FilterStr
        :param receptor_id:
            Optional receptor id or sequence of receptor ids.
        :type receptor_id: FilterStr
        :param ligand_id:
            Optional ligand id or sequence of ligand ids.
        :type ligand_id: FilterStr
        :param engine:
            Optional engine name or sequence of engine names.
        :type engine: FilterStr
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
        :type interaction_type: FilterStr
        :param residue_id:
            Optional residue id filter such as ``"LEU23.A"``.
        :type residue_id: FilterStr
        :param chain_id:
            Optional chain filter.
        :type chain_id: FilterStr
        :param residue_name:
            Optional residue-name filter.
        :type residue_name: FilterStr
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

        Example
        -------
        .. code-block:: python

            q = PoseQuery("prodock.db")

            df = q.poses(
                receptor_id="1M17",
                top_rank=3,
                include_interactions=True,
                interaction_mode="summary",
                as_dataframe=True,
            )
            print(df[["pose_id", "affinity", "interaction_summary"]].head())
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
                interaction_map = self.interaction_summary(
                    pose_db_id=pose_db_ids,
                    return_by="pose_db_id",
                )
            elif interaction_mode == "detailed":
                interaction_map = self.interaction_details(
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

    def pose(
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

        Example
        -------
        .. code-block:: python

            q = PoseQuery("prodock.db")

            pose = q.pose(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="qvina",
                pose_rank=1,
            )
            print(pose)
        """
        rows = self.poses(
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

    def scores(
        self,
        *,
        pose_db_id: Optional[int] = None,
        pose_id: FilterStr = None,
        receptor_id: FilterStr = None,
        ligand_id: FilterStr = None,
        engine: FilterStr = None,
        pose_rank: Optional[int] = None,
        top_rank: Optional[int] = None,
        affinity_threshold: Optional[float] = None,
        affinity_min: Optional[float] = None,
        as_dataframe: bool = False,
        order_by: Optional[Union[str, Sequence[str]]] = None,
        limit: Optional[int] = None,
    ) -> Union[list[ScoreRecord], pd.DataFrame]:
        """
        Query score rows joined to pose identity.

        :param pose_db_id:
            Optional internal pose id filter.
        :type pose_db_id: Optional[int]
        :param pose_id:
            Optional external pose id or sequence of ids.
        :type pose_id: FilterStr
        :param receptor_id:
            Optional receptor id filter.
        :type receptor_id: FilterStr
        :param ligand_id:
            Optional ligand id filter.
        :type ligand_id: FilterStr
        :param engine:
            Optional engine filter.
        :type engine: FilterStr
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
            List of :class:`ScoreRecord` or a pandas DataFrame.
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
        pose_id: FilterStr = None,
        receptor_id: FilterStr = None,
        ligand_id: FilterStr = None,
        engine: FilterStr = None,
        pose_rank: Optional[int] = None,
        top_rank: Optional[int] = None,
        affinity_threshold: Optional[float] = None,
        affinity_min: Optional[float] = None,
        interaction_type: FilterStr = None,
        residue_id: FilterStr = None,
        chain_id: FilterStr = None,
        residue_name: FilterStr = None,
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

    def interactions(
        self,
        *,
        interaction_id: Optional[int] = None,
        pose_db_id: Optional[int] = None,
        pose_id: FilterStr = None,
        receptor_id: FilterStr = None,
        ligand_id: FilterStr = None,
        engine: FilterStr = None,
        pose_rank: Optional[int] = None,
        interaction_type: FilterStr = None,
        chain_id: FilterStr = None,
        residue_name: FilterStr = None,
        residue_number: Optional[int] = None,
        residue_id: FilterStr = None,
        ligand_residue: FilterStr = None,
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
        :type pose_id: FilterStr
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: FilterStr
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: FilterStr
        :param engine:
            Optional engine filter.
        :type engine: FilterStr
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Optional interaction type filter.
        :type interaction_type: FilterStr
        :param chain_id:
            Optional chain filter.
        :type chain_id: FilterStr
        :param residue_name:
            Optional residue-name filter.
        :type residue_name: FilterStr
        :param residue_number:
            Optional residue-number filter.
        :type residue_number: Optional[int]
        :param residue_id:
            Optional combined residue-id filter.
        :type residue_id: FilterStr
        :param ligand_residue:
            Optional ligand residue filter.
        :type ligand_residue: FilterStr
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
            List of :class:`InteractionRecord` or a pandas DataFrame.
        :rtype: Union[list[InteractionRecord], pd.DataFrame]

        Example
        -------
        .. code-block:: python

            q = PoseQuery("prodock.db")

            df = q.interactions(
                receptor_id="1M17",
                interaction_type="Hydrophobic",
                as_dataframe=True,
            )
            print(df[["pose_id", "interaction_type", "residue_id"]].head())
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

    def interaction_summary(
        self,
        *,
        pose_db_id: FilterInt = None,
        pose_id: FilterStr = None,
        receptor_id: FilterStr = None,
        ligand_id: FilterStr = None,
        engine: FilterStr = None,
        pose_rank: Optional[int] = None,
        interaction_type: FilterStr = None,
        residue_id: FilterStr = None,
        return_by: str = "pose_key",
    ) -> dict[Union[int, str], dict[str, list[str]]]:
        """
        Return summarized interactions grouped by pose.

        The returned payload uses the compact format
        ``{pose_key: {interaction_type: [residue_id, ...]}}``.

        :param pose_db_id:
            Optional pose id or sequence of internal pose ids.
        :type pose_db_id: FilterInt
        :param pose_id:
            Optional external pose id filter.
        :type pose_id: FilterStr
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: FilterStr
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: FilterStr
        :param engine:
            Optional engine filter.
        :type engine: FilterStr
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Optional interaction-type filter.
        :type interaction_type: FilterStr
        :param residue_id:
            Optional residue-id filter.
        :type residue_id: FilterStr
        :param return_by:
            One of ``"pose_db_id"``, ``"pose_id"``, or ``"pose_key"``.
        :type return_by: str

        :returns:
            Nested summary mapping grouped by pose.
        :rtype: dict[Union[int, str], dict[str, list[str]]]

        Example
        -------
        .. code-block:: python

            q = PoseQuery("prodock.db")

            summary = q.interaction_summary(
                receptor_id="1M17",
                return_by="pose_id",
            )
            print(summary)
        """
        if isinstance(pose_db_id, int):
            pose_db_id_filter = pose_db_id
        else:
            pose_db_id_filter = None

        frame = self.interactions(
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

    def interaction_details(
        self,
        *,
        pose_db_id: FilterInt = None,
        pose_id: FilterStr = None,
        receptor_id: FilterStr = None,
        ligand_id: FilterStr = None,
        engine: FilterStr = None,
        pose_rank: Optional[int] = None,
        interaction_type: FilterStr = None,
        residue_id: FilterStr = None,
        return_by: str = "pose_key",
    ) -> dict[Union[int, str], dict[str, dict[str, list[dict[str, Any]]]]]:
        """
        Return detailed interactions grouped by pose.

        The returned payload mirrors the nested detailed format
        ``{pose_key: {interaction_type: {residue_id: [event, ...]}}}``.

        :param pose_db_id:
            Optional pose id or sequence of internal pose ids.
        :type pose_db_id: FilterInt
        :param pose_id:
            Optional external pose id filter.
        :type pose_id: FilterStr
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: FilterStr
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: FilterStr
        :param engine:
            Optional engine filter.
        :type engine: FilterStr
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Optional interaction-type filter.
        :type interaction_type: FilterStr
        :param residue_id:
            Optional residue-id filter.
        :type residue_id: FilterStr
        :param return_by:
            One of ``"pose_db_id"``, ``"pose_id"``, or ``"pose_key"``.
        :type return_by: str

        :returns:
            Nested detailed mapping grouped by pose.
        :rtype: dict[Union[int, str], dict[str, dict[str, list[dict[str, Any]]]]]

        Example
        -------
        .. code-block:: python

            q = PoseQuery("prodock.db")

            details = q.interaction_details(
                receptor_id="1M17",
                return_by="pose_key",
            )
            print(details)
        """
        if isinstance(pose_db_id, int):
            pose_db_id_filter = pose_db_id
        else:
            pose_db_id_filter = None

        frame = self.interactions(
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

    def fingerprint(
        self,
        *,
        pose_db_id: FilterInt = None,
        pose_id: FilterStr = None,
        receptor_id: FilterStr = None,
        ligand_id: FilterStr = None,
        engine: FilterStr = None,
        pose_rank: Optional[int] = None,
        interaction_type: FilterStr = None,
        residue_id: FilterStr = None,
        mode: str = "binary",
        feature_sep: str = "::",
        index_by: str = "pose_key",
    ) -> pd.DataFrame:
        """
        Build a pose-by-feature interaction fingerprint matrix.

        Features are named as ``<interaction_type><feature_sep><residue_id>``.

        :param pose_db_id:
            Optional pose id or sequence of internal pose ids.
        :type pose_db_id: FilterInt
        :param pose_id:
            Optional external pose id filter.
        :type pose_id: FilterStr
        :param receptor_id:
            Optional receptor filter.
        :type receptor_id: FilterStr
        :param ligand_id:
            Optional ligand filter.
        :type ligand_id: FilterStr
        :param engine:
            Optional engine filter.
        :type engine: FilterStr
        :param pose_rank:
            Optional exact pose-rank filter.
        :type pose_rank: Optional[int]
        :param interaction_type:
            Optional interaction-type filter.
        :type interaction_type: FilterStr
        :param residue_id:
            Optional residue-id filter.
        :type residue_id: FilterStr
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

        Example
        -------
        .. code-block:: python

            q = PoseQuery("prodock.db")

            fp = q.fingerprint(
                receptor_id="1M17",
                mode="binary",
                index_by="pose_key",
            )
            print(fp.head())
        """
        if mode not in {"binary", "count"}:
            raise ValueError("mode must be 'binary' or 'count'")
        if index_by not in {"pose_db_id", "pose_id", "pose_key"}:
            raise ValueError("index_by must be 'pose_db_id', 'pose_id', or 'pose_key'")

        if isinstance(pose_db_id, int):
            pose_db_id_filter = pose_db_id
        else:
            pose_db_id_filter = None

        frame = self.interactions(
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

    def summary(self) -> pd.DataFrame:
        """
        Return one summary row per receptor-ligand-engine group.

        The summary includes the number of stored poses, best affinity value,
        maximum stored pose rank, and the number of linked interaction rows.

        :returns:
            Summary DataFrame with pose counts and best affinity values.
        :rtype: pd.DataFrame

        Example
        -------
        .. code-block:: python

            q = PoseQuery("prodock.db")
            print(q.summary())
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

    def receptors(self) -> list[str]:
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

    def ligands(self) -> list[str]:
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

    def engines(self) -> list[str]:
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

    def query_poses(self, **kwargs: Any) -> Union[list[PoseRecord], pd.DataFrame]:
        """Alias for :meth:`poses`."""
        return self.poses(**kwargs)

    def get_pose(self, **kwargs: Any) -> Optional[PoseRecord]:
        """Alias for :meth:`pose`."""
        return self.pose(**kwargs)

    def query_scores(self, **kwargs: Any) -> Union[list[ScoreRecord], pd.DataFrame]:
        """Alias for :meth:`scores`."""
        return self.scores(**kwargs)

    def query_interactions(
        self,
        **kwargs: Any,
    ) -> Union[list[InteractionRecord], pd.DataFrame]:
        """Alias for :meth:`interactions`."""
        return self.interactions(**kwargs)

    def get_interaction_summary(
        self,
        **kwargs: Any,
    ) -> dict[Union[int, str], dict[str, list[str]]]:
        """Alias for :meth:`interaction_summary`."""
        return self.interaction_summary(**kwargs)

    def get_interaction_details(
        self,
        **kwargs: Any,
    ) -> dict[Union[int, str], dict[str, dict[str, list[dict[str, Any]]]]]:
        """Alias for :meth:`interaction_details`."""
        return self.interaction_details(**kwargs)

    def interaction_fingerprint(self, **kwargs: Any) -> pd.DataFrame:
        """Alias for :meth:`fingerprint`."""
        return self.fingerprint(**kwargs)

    def summarize(self) -> pd.DataFrame:
        """Alias for :meth:`summary`."""
        return self.summary()

    def list_receptors(self) -> list[str]:
        """Alias for :meth:`receptors`."""
        return self.receptors()

    def list_ligands(self) -> list[str]:
        """Alias for :meth:`ligands`."""
        return self.ligands()

    def list_engines(self) -> list[str]:
        """Alias for :meth:`engines`."""
        return self.engines()
