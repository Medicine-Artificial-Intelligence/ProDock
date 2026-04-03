from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd

from .record import PoseRecord


def poses_to_dataframe(records: Iterable[PoseRecord]) -> pd.DataFrame:
    """
    Convert pose records into a standardized public :class:`pandas.DataFrame`.

    The returned schema is intentionally minimal and omits internal bookkeeping
    fields such as source file paths. This makes the output suitable for
    downstream filtering, ranking, aggregation, and export.

    The output columns are always returned in the fixed order:

    - ``receptor_id``
    - ``ligand_id``
    - ``engine``
    - ``pose_rank``
    - ``affinity``

    :param records:
        Iterable of pose records to convert.
    :type records: Iterable[prodock.postprocess.pose.model.PoseRecord]

    :returns:
        DataFrame with the standardized public pose schema.
    :rtype: pandas.DataFrame

    Example
    -------
    .. code-block:: python

        from pathlib import Path
        from prodock.postprocess.pose.model import PoseRecord
        from prodock.postprocess.pose.select import poses_to_dataframe

        records = [
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="vina",
                pose_rank=1,
                affinity=-7.2,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
                ),
            ),
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="smina",
                pose_rank=1,
                affinity=-7.4,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"
                ),
            ),
        ]

        df = poses_to_dataframe(records)
        print(df)
    """
    rows = [
        {
            "receptor_id": rec.receptor_id,
            "ligand_id": rec.ligand_id,
            "engine": rec.engine,
            "pose_rank": rec.pose_rank,
            "affinity": rec.affinity,
        }
        for rec in records
    ]
    return pd.DataFrame(
        rows,
        columns=["receptor_id", "ligand_id", "engine", "pose_rank", "affinity"],
    )


def pose_mols_to_dataframe(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert pose-plus-molecule row dictionaries into a standardized DataFrame.

    This helper is intended for tabular outputs produced by molecule-loading
    utilities, where each row contains both pose metadata and an RDKit molecule
    object stored under the ``mol`` key.

    The output columns are always returned in the fixed order:

    - ``receptor_id``
    - ``ligand_id``
    - ``engine``
    - ``pose_rank``
    - ``affinity``
    - ``mol``

    :param rows:
        Sequence of row dictionaries containing pose metadata and molecule
        objects.
    :type rows: Sequence[dict[str, Any]]

    :returns:
        DataFrame with the standardized pose-plus-molecule schema.
    :rtype: pandas.DataFrame

    Example
    -------
    .. code-block:: python

        from prodock.postprocess.pose.select import pose_mols_to_dataframe

        rows = [
            {
                "receptor_id": "1M17",
                "ligand_id": "erlotinib",
                "engine": "qvina",
                "pose_rank": 1,
                "affinity": -7.1,
                "mol": "mock_mol_qvina",
            },
            {
                "receptor_id": "1M17",
                "ligand_id": "erlotinib",
                "engine": "vina",
                "pose_rank": 1,
                "affinity": -7.2,
                "mol": "mock_mol_vina",
            },
        ]

        df = pose_mols_to_dataframe(rows)
        print(df)
    """
    return pd.DataFrame(
        list(rows),
        columns=["receptor_id", "ligand_id", "engine", "pose_rank", "affinity", "mol"],
    )


def best_pose_per_group(
    records_or_df,
    *,
    by: Sequence[str] = ("receptor_id", "ligand_id", "engine"),
) -> pd.DataFrame:
    """
    Select the best-scoring pose row within each group.

    Lower affinity is treated as better. Missing affinity values are placed
    after numeric values, so they are only selected when a group has no
    non-missing affinity. Ties are resolved stably by ``pose_rank`` after
    sorting, so lower-ranked poses are preferred when affinities are equal.

    The input may be either:

    - an iterable of :class:`PoseRecord` objects, or
    - a :class:`pandas.DataFrame` already following the pose schema

    By default, grouping is performed by:

    - ``receptor_id``
    - ``ligand_id``
    - ``engine``

    :param records_or_df:
        Either an iterable of :class:`PoseRecord` objects or a pose DataFrame.
    :type records_or_df: Iterable[prodock.postprocess.pose.model.PoseRecord] | pandas.DataFrame
    :param by:
        Grouping columns used to define independent pose-selection groups.
    :type by: Sequence[str]

    :returns:
        DataFrame containing one best row per group.
    :rtype: pandas.DataFrame

    Example
    -------
    .. code-block:: python

        from pathlib import Path
        from prodock.postprocess.pose.model import PoseRecord
        from prodock.postprocess.pose.select import best_pose_per_group

        records = [
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="vina",
                pose_rank=1,
                affinity=-7.2,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
                ),
            ),
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="vina",
                pose_rank=2,
                affinity=-6.8,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
                ),
            ),
            PoseRecord(
                receptor_id="1M17",
                ligand_id="erlotinib",
                engine="smina",
                pose_rank=1,
                affinity=-7.4,
                source_file=Path(
                    "Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"
                ),
            ),
        ]

        best_df = best_pose_per_group(records)
        print(best_df)
    """
    if isinstance(records_or_df, pd.DataFrame):
        df = records_or_df.copy()
    else:
        df = poses_to_dataframe(records_or_df)

    if df.empty:
        return df

    df = df.sort_values(
        by=[*by, "affinity", "pose_rank"],
        ascending=[True] * len(by) + [True, True],
        kind="stable",
        na_position="last",
    )
    return df.groupby(list(by), as_index=False, dropna=False, sort=False).first()
