from __future__ import annotations

"""SQL query helpers for ProDock database filters and ordering."""

from typing import Any, Optional, Sequence, Union

from .serialization import as_many

SORTABLE_COLUMNS = {
    "pose_db_id": "p.pose_db_id",
    "pose_id": "p.pose_id",
    "receptor_id": "p.receptor_id",
    "ligand_id": "p.ligand_id",
    "engine": "p.engine",
    "pose_rank": "p.pose_rank",
    "affinity": "s.affinity",
    "created_at": "p.created_at",
    "interaction_id": "i.interaction_id",
    "interaction_type": "i.interaction_type",
    "residue_id": "i.residue_id",
    "residue_number": "i.residue_number",
    "distance": "i.distance",
    "angle": "i.angle",
    "occurrence_index": "i.occurrence_index",
}

FilterType = Optional[Union[str, Sequence[str]]]


def _apply_many_filter(
    clauses: list[str],
    params: list[Any],
    *,
    column: str,
    value: FilterType,
) -> None:
    """Append a scalar-or-list SQL filter clause in place."""
    many = as_many(value)
    if many is None:
        return
    if len(many) == 1:
        clauses.append(f"{column} = ?")
        params.append(many[0])
    else:
        placeholders = ", ".join("?" for _ in many)
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(many)


def build_pose_where_clause(
    *,
    pose_db_id: Optional[int] = None,
    pose_id: FilterType = None,
    receptor_id: FilterType = None,
    ligand_id: FilterType = None,
    engine: FilterType = None,
    pose_rank: Optional[int] = None,
    top_rank: Optional[int] = None,
    affinity_threshold: Optional[float] = None,
    affinity_min: Optional[float] = None,
    interaction_type: FilterType = None,
    residue_id: FilterType = None,
    chain_id: FilterType = None,
    residue_name: FilterType = None,
    residue_number: Optional[int] = None,
) -> tuple[str, list[Any]]:
    """
    Build a ``WHERE`` clause for pose queries.

    Interaction filters are translated to an ``EXISTS`` subquery so poses can be
    queried directly by interaction content.

    :returns:
        SQL fragment and bound parameters.
    :rtype: tuple[str, list[Any]]
    """
    clauses: list[str] = []
    params: list[Any] = []

    if pose_db_id is not None:
        clauses.append("p.pose_db_id = ?")
        params.append(int(pose_db_id))

    for column, value in (
        ("p.pose_id", pose_id),
        ("p.receptor_id", receptor_id),
        ("p.ligand_id", ligand_id),
        ("p.engine", engine),
    ):
        _apply_many_filter(clauses, params, column=column, value=value)

    if pose_rank is not None:
        clauses.append("p.pose_rank = ?")
        params.append(int(pose_rank))
    if top_rank is not None:
        clauses.append("p.pose_rank <= ?")
        params.append(int(top_rank))
    if affinity_threshold is not None:
        clauses.append("s.affinity <= ?")
        params.append(float(affinity_threshold))
    if affinity_min is not None:
        clauses.append("s.affinity >= ?")
        params.append(float(affinity_min))

    interaction_clauses: list[str] = ["i.pose_db_id = p.pose_db_id"]
    interaction_params: list[Any] = []
    for column, value in (
        ("i.interaction_type", interaction_type),
        ("i.residue_id", residue_id),
        ("i.chain_id", chain_id),
        ("i.residue_name", residue_name),
    ):
        _apply_many_filter(
            interaction_clauses,
            interaction_params,
            column=column,
            value=value,
        )
    if residue_number is not None:
        interaction_clauses.append("i.residue_number = ?")
        interaction_params.append(int(residue_number))
    if len(interaction_clauses) > 1:
        clauses.append(
            "EXISTS (SELECT 1 FROM interactions AS i WHERE "
            + " AND ".join(interaction_clauses)
            + ")"
        )
        params.extend(interaction_params)

    if not clauses:
        return "", params
    return " WHERE " + " AND ".join(clauses), params


def build_interaction_where_clause(
    *,
    interaction_id: Optional[int] = None,
    pose_db_id: Optional[int] = None,
    pose_id: FilterType = None,
    receptor_id: FilterType = None,
    ligand_id: FilterType = None,
    engine: FilterType = None,
    pose_rank: Optional[int] = None,
    interaction_type: FilterType = None,
    chain_id: FilterType = None,
    residue_name: FilterType = None,
    residue_number: Optional[int] = None,
    residue_id: FilterType = None,
    ligand_residue: FilterType = None,
) -> tuple[str, list[Any]]:
    """
    Build a ``WHERE`` clause for interaction queries.

    :returns:
        SQL fragment and bound parameters.
    :rtype: tuple[str, list[Any]]
    """
    clauses: list[str] = []
    params: list[Any] = []

    if interaction_id is not None:
        clauses.append("i.interaction_id = ?")
        params.append(int(interaction_id))
    if pose_db_id is not None:
        clauses.append("i.pose_db_id = ?")
        params.append(int(pose_db_id))

    for column, value in (
        ("p.pose_id", pose_id),
        ("p.receptor_id", receptor_id),
        ("p.ligand_id", ligand_id),
        ("p.engine", engine),
        ("i.interaction_type", interaction_type),
        ("i.chain_id", chain_id),
        ("i.residue_name", residue_name),
        ("i.residue_id", residue_id),
        ("i.ligand_residue", ligand_residue),
    ):
        _apply_many_filter(clauses, params, column=column, value=value)

    if pose_rank is not None:
        clauses.append("p.pose_rank = ?")
        params.append(int(pose_rank))
    if residue_number is not None:
        clauses.append("i.residue_number = ?")
        params.append(int(residue_number))

    if not clauses:
        return "", params
    return " WHERE " + " AND ".join(clauses), params


def resolve_order_by(order_by: Optional[Union[str, Sequence[str]]]) -> str:
    """
    Resolve a public order key to a SQL ``ORDER BY`` clause.

    :param order_by:
        Column name or list of column names. Prefix with ``-`` for descending.
    :type order_by: Optional[Union[str, Sequence[str]]]

    :returns:
        SQL ``ORDER BY`` clause.
    :rtype: str

    :raises ValueError:
        If an unsupported key is supplied.
    """
    if order_by is None:
        return " ORDER BY p.receptor_id, p.ligand_id, p.engine, p.pose_rank"

    keys = [order_by] if isinstance(order_by, str) else list(order_by)
    parts: list[str] = []
    for key in keys:
        descending = key.startswith("-")
        clean_key = key[1:] if descending else key
        sql_col = SORTABLE_COLUMNS.get(clean_key)
        if sql_col is None:
            allowed = ", ".join(sorted(SORTABLE_COLUMNS))
            raise ValueError(
                f"Unsupported order_by key: {clean_key!r}. Allowed keys: {allowed}"
            )
        parts.append(f"{sql_col} {'DESC' if descending else 'ASC'}")
    return " ORDER BY " + ", ".join(parts)
