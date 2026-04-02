from __future__ import annotations

"""Flattening and pose-level summarization helpers for ProLIF metadata."""

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd

POSE_EVENT_COLUMNS = [
    "pose_id",
    "interaction_events",
    "interaction_events_json",
    "has_interactions",
]

POSE_SUMMARY_COLUMNS = [
    "pose_id",
    "interaction_compact",
    "interaction_compact_json",
    "interaction_detail",
    "interaction_detail_json",
    "has_interactions",
]


def _safe_string(value: Any) -> str:
    """
    Convert a value to a stable string representation.

    :param value:
        Value to stringify.
    :type value: Any

    :returns:
        String representation.
    :rtype: str
    """
    return str(value) if value is not None else ""


def _jsonable(value: Any) -> Any:
    """
    Convert metadata values to JSON-serializable objects.

    :param value:
        Arbitrary metadata value.
    :type value: Any

    :returns:
        JSON-serializable representation.
    :rtype: Any
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return str(value)


def _iter_frame_interactions(frame_ifp: Any) -> Iterable[Dict[str, Any]]:
    """
    Iterate over interactions from a single ProLIF frame in a version-tolerant way.

    :param frame_ifp:
        Per-frame entry from ``fingerprint.ifp``.
    :type frame_ifp: Any

    :returns:
        Iterator of flat interaction records.
    :rtype: Iterable[Dict[str, Any]]
    """
    interactions_method = getattr(frame_ifp, "interactions", None)
    if callable(interactions_method):
        for entry in interactions_method():
            metadata = dict(getattr(entry, "metadata", {}) or {})
            yield {
                "ligand_residue": _safe_string(getattr(entry, "ligand", "")),
                "protein_residue": _safe_string(getattr(entry, "protein", "")),
                "interaction": _safe_string(getattr(entry, "interaction", "")),
                "metadata": metadata,
            }
        return

    if isinstance(frame_ifp, Mapping):
        for residue_pair, interaction_map in frame_ifp.items():
            try:
                ligand_residue, protein_residue = residue_pair
            except Exception:
                ligand_residue, protein_residue = "", ""
            if not isinstance(interaction_map, Mapping):
                continue
            for interaction_name, metadata_items in interaction_map.items():
                for metadata in metadata_items:
                    yield {
                        "ligand_residue": _safe_string(ligand_residue),
                        "protein_residue": _safe_string(protein_residue),
                        "interaction": _safe_string(interaction_name),
                        "metadata": dict(metadata or {}),
                    }


def flatten_ifp(
    ifp: Mapping[int, Any],
    mol_names: Sequence[str] | None = None,
    mols: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """
    Flatten ``Fingerprint.ifp`` into a long-form event table.

    :param ifp:
        Mapping from frame index to per-frame ProLIF interaction metadata.
    :type ifp: Mapping[int, Any]
    :param mol_names:
        Optional molecule names aligned with frame indices.
    :type mol_names: Optional[Sequence[str]]
    :param mols:
        Optional RDKit molecules aligned with frame indices.
    :type mols: Optional[Sequence[Any]]

    :returns:
        Long-form interaction table.
    :rtype: pandas.DataFrame
    """
    rows: List[MutableMapping[str, Any]] = []
    columns = [
        "frame",
        "mol_index",
        "mol_name",
        "mol",
        "ligand_residue",
        "protein_residue",
        "interaction",
        "occurrence_index",
        "ligand_atom_indices",
        "protein_atom_indices",
        "ligand_parent_indices",
        "protein_parent_indices",
        "distance",
        "angle",
        "metadata",
        "metadata_json",
    ]

    for frame, frame_ifp in ifp.items():
        mol_index = int(frame)
        mol_name = (
            mol_names[mol_index]
            if mol_names is not None and 0 <= mol_index < len(mol_names)
            else f"mol_{mol_index:04d}"
        )
        mol = (
            mols[mol_index] if mols is not None and 0 <= mol_index < len(mols) else None
        )

        for occurrence_index, record in enumerate(_iter_frame_interactions(frame_ifp)):
            metadata = dict(record.get("metadata", {}) or {})
            indices = dict(metadata.get("indices", {}) or {})
            parent_indices = dict(metadata.get("parent_indices", {}) or {})

            row: MutableMapping[str, Any] = {
                "frame": mol_index,
                "mol_index": mol_index,
                "mol_name": mol_name,
                "mol": mol,
                "ligand_residue": record.get("ligand_residue", ""),
                "protein_residue": record.get("protein_residue", ""),
                "interaction": record.get("interaction", ""),
                "occurrence_index": occurrence_index,
                "ligand_atom_indices": tuple(indices.get("ligand", ()) or ()),
                "protein_atom_indices": tuple(indices.get("protein", ()) or ()),
                "ligand_parent_indices": tuple(parent_indices.get("ligand", ()) or ()),
                "protein_parent_indices": tuple(
                    parent_indices.get("protein", ()) or ()
                ),
                "distance": metadata.get("distance"),
                "angle": metadata.get("angle"),
                "metadata": _jsonable(metadata),
                "metadata_json": json.dumps(_jsonable(metadata), sort_keys=True),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_pose_interaction_table(
    interaction_events_df: pd.DataFrame,
    pose_ids: Sequence[str],
) -> pd.DataFrame:
    """
    Collapse raw interaction events into one row per pose.

    :param interaction_events_df:
        Long-form interaction event table containing a ``pose_id`` column.
    :type interaction_events_df: pandas.DataFrame
    :param pose_ids:
        Pose ids to preserve and order in the output table.
    :type pose_ids: Sequence[str]

    :returns:
        Pose-level table of raw event payloads.
    :rtype: pandas.DataFrame
    """
    payload_by_pose: Dict[str, Any] = {str(pose_id): None for pose_id in pose_ids}
    if not interaction_events_df.empty and "pose_id" in interaction_events_df.columns:
        cleaned = interaction_events_df.copy()
        keep_columns = [
            "ligand_residue",
            "protein_residue",
            "interaction",
            "occurrence_index",
            "ligand_atom_indices",
            "protein_atom_indices",
            "ligand_parent_indices",
            "protein_parent_indices",
            "distance",
            "angle",
            "metadata",
            "metadata_json",
        ]
        keep_columns = [column for column in keep_columns if column in cleaned.columns]
        for pose_id, pose_df in cleaned.groupby("pose_id", sort=False):
            payload_by_pose[str(pose_id)] = (
                pose_df[keep_columns].to_dict(orient="records") or None
            )

    rows: List[Dict[str, Any]] = []
    for pose_id in pose_ids:
        payload = payload_by_pose.get(str(pose_id))
        rows.append(
            {
                "pose_id": str(pose_id),
                "interaction_events": payload,
                "interaction_events_json": (
                    json.dumps(_jsonable(payload), sort_keys=True)
                    if payload is not None
                    else None
                ),
                "has_interactions": bool(payload),
            }
        )
    return pd.DataFrame(rows, columns=POSE_EVENT_COLUMNS)


def _record_to_detail_entry(record: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert one flat event record to a compact detail entry.

    :param record:
        Raw event dictionary.
    :type record: Mapping[str, Any]

    :returns:
        Compact detail entry.
    :rtype: Dict[str, Any]
    """
    entry: Dict[str, Any] = {
        "protein_residue": _safe_string(record.get("protein_residue")),
        "ligand_residue": _safe_string(record.get("ligand_residue")),
        "distance": record.get("distance"),
        "angle": record.get("angle"),
        "indices": {
            "ligand": list(record.get("ligand_atom_indices") or []),
            "protein": list(record.get("protein_atom_indices") or []),
        },
        "parent_indices": {
            "ligand": list(record.get("ligand_parent_indices") or []),
            "protein": list(record.get("protein_parent_indices") or []),
        },
    }
    metadata = record.get("metadata")
    if metadata not in (None, {}):
        entry["metadata"] = _jsonable(metadata)
    return entry


def build_pose_summary_table(
    interaction_events_df: pd.DataFrame,
    pose_ids: Sequence[str],
) -> pd.DataFrame:
    """
    Build one summary row per pose.

    The returned table keeps two payload styles per pose:

    - ``interaction_compact``: ``interaction_type -> sorted list[protein_residue]``
    - ``interaction_detail``: ``interaction_type -> protein_residue -> list[detail entries]``

    :param interaction_events_df:
        Long-form event table containing ``pose_id``, ``interaction``, and
        ``protein_residue`` columns.
    :type interaction_events_df: pandas.DataFrame
    :param pose_ids:
        Pose ids to preserve and order in the output table.
    :type pose_ids: Sequence[str]

    :returns:
        Pose-level summary table.
    :rtype: pandas.DataFrame
    """
    compact_by_pose: Dict[str, Any] = {str(pose_id): None for pose_id in pose_ids}
    detail_by_pose: Dict[str, Any] = {str(pose_id): None for pose_id in pose_ids}

    required = {"pose_id", "interaction", "protein_residue"}
    if not interaction_events_df.empty and required.issubset(
        interaction_events_df.columns
    ):
        for pose_id, pose_df in interaction_events_df.groupby("pose_id", sort=False):
            compact_map: Dict[str, List[str]] = {}
            detail_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for interaction_type, interaction_block in pose_df.groupby(
                "interaction", sort=False
            ):
                residues = sorted(
                    {
                        _safe_string(value)
                        for value in interaction_block["protein_residue"].tolist()
                        if _safe_string(value)
                    }
                )
                compact_map[str(interaction_type)] = residues

                residue_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for record in interaction_block.to_dict(orient="records"):
                    residue = _safe_string(record.get("protein_residue"))
                    residue_map[residue].append(_record_to_detail_entry(record))
                detail_map[str(interaction_type)] = {
                    str(residue): entries for residue, entries in residue_map.items()
                }

            compact_by_pose[str(pose_id)] = compact_map or None
            detail_by_pose[str(pose_id)] = detail_map or None

    rows: List[Dict[str, Any]] = []
    for pose_id in pose_ids:
        compact_payload = compact_by_pose.get(str(pose_id))
        detail_payload = detail_by_pose.get(str(pose_id))
        rows.append(
            {
                "pose_id": str(pose_id),
                "interaction_compact": compact_payload,
                "interaction_compact_json": (
                    json.dumps(_jsonable(compact_payload), sort_keys=True)
                    if compact_payload is not None
                    else None
                ),
                "interaction_detail": detail_payload,
                "interaction_detail_json": (
                    json.dumps(_jsonable(detail_payload), sort_keys=True)
                    if detail_payload is not None
                    else None
                ),
                "has_interactions": bool(compact_payload),
            }
        )
    return pd.DataFrame(rows, columns=POSE_SUMMARY_COLUMNS)


def summary_table_to_dict(
    summary_df: pd.DataFrame,
    *,
    kind: str = "compact",
    as_sets: bool = False,
    drop_empty: bool = False,
) -> Dict[str, Any]:
    """
    Convert a pose-level summary table into a dictionary.

    :param summary_df:
        Pose-level summary table from :func:`build_pose_summary_table`.
    :type summary_df: pandas.DataFrame
    :param kind:
        Either ``'compact'`` or ``'detail'``.
    :type kind: str
    :param as_sets:
        Whether compact residue collections should be returned as sets.
    :type as_sets: bool
    :param drop_empty:
        Whether poses without interactions should be omitted.
    :type drop_empty: bool

    :returns:
        Dictionary keyed by pose id.
    :rtype: Dict[str, Any]
    """
    if summary_df.empty:
        return {}

    normalized = kind.strip().lower()
    if normalized not in {"compact", "detail"}:
        raise ValueError("kind must be either 'compact' or 'detail'.")
    payload_column = (
        "interaction_compact" if normalized == "compact" else "interaction_detail"
    )

    result: Dict[str, Any] = {}
    for row in summary_df.itertuples(index=False):
        pose_id = str(getattr(row, "pose_id"))
        payload = getattr(row, payload_column, None)
        if payload is None:
            if not drop_empty:
                result[pose_id] = None
            continue
        if normalized == "compact" and isinstance(payload, Mapping) and as_sets:
            result[pose_id] = {str(k): set(v or []) for k, v in payload.items()}
        else:
            result[pose_id] = payload
    return result
