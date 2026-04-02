from __future__ import annotations

"""
Dataclasses used by :mod:`prodock.postprocess.interaction`.

This module defines lightweight result containers for two common interaction
analysis workflows:

- :class:`InteractionRunResult` stores outputs from a single fingerprint run,
  including raw molecules, ProLIF molecules, fingerprint tables, and
  event-level interaction tables.
- :class:`PoseInteractionTableResult` stores pose-centric tables suitable for
  downstream database insertion, export, summarization, and similarity-based
  comparison.

These containers provide convenience helpers for:

- building molecule-level summary tables,
- converting interaction tables into CSV-friendly forms,
- exporting pose summaries as dictionaries,
- saving generated tables to disk,
- computing similarity matrices from stored fingerprints.

Example
-------
.. code-block:: python

    result = InteractionRunResult(
        receptor_path="Data/receptor/egfr.pdb",
        molecule_names=["pose_1", "pose_2"],
        molecules=[mol1, mol2],
        fingerprint_df=fingerprint_df,
        interaction_df=interaction_df,
    )

    molecule_df = result.molecule_table()
    serializable_df = result.serializable_interaction_df()

    pose_result = PoseInteractionTableResult(
        merged_df=merged_df,
        interaction_df=pose_events_df,
        summary_df=pose_summary_df,
        molecule_names=["pose_1", "pose_2"],
    )

    compact_map = pose_result.summary_dict(kind="compact")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .flatten import summary_table_to_dict
from .io import mol_to_smiles


@dataclass
class InteractionRunResult:
    """
    Container holding the outputs of a single interaction fingerprint run.

    This result object groups together the receptor path, ligand identities,
    original molecule objects, ProLIF-converted molecules, fingerprint outputs,
    event-level interaction tables, and optional vector representations.

    :param receptor_path:
        Path to the receptor structure used for the interaction run.
    :type receptor_path: Optional[pathlib.Path]
    :param molecule_names:
        Ordered ligand or pose names aligned with ``molecules`` and
        ``prolif_molecules``.
    :type molecule_names: List[str]
    :param molecules:
        Original ligand molecule objects, typically RDKit molecules.
    :type molecules: List[Any]
    :param prolif_molecules:
        ProLIF molecule objects aligned with ``molecule_names``.
    :type prolif_molecules: List[Any]
    :param fingerprint:
        Raw fingerprint object returned by ProLIF, when available.
    :type fingerprint: Any
    :param fingerprint_df:
        Tabular fingerprint representation.
    :type fingerprint_df: pandas.DataFrame
    :param interaction_df:
        Long-form interaction event table.
    :type interaction_df: pandas.DataFrame
    :param bitvectors:
        Optional bitvector fingerprints aligned with ``molecule_names``.
    :type bitvectors: Optional[List[Any]]
    :param countvectors:
        Optional countvector fingerprints aligned with ``molecule_names``.
    :type countvectors: Optional[List[Any]]
    :param protein_molecule:
        Protein molecule object used internally for interaction analysis.
    :type protein_molecule: Any
    :param settings:
        Optional run settings or provenance metadata.
    :type settings: Dict[str, Any]

    Example
    -------
    .. code-block:: python

        result = InteractionRunResult(
            receptor_path=Path("Data/receptor/egfr.pdb"),
            molecule_names=["pose_1", "pose_2"],
            molecules=[mol1, mol2],
            fingerprint_df=fingerprint_df,
            interaction_df=interaction_df,
            settings={"count": True},
        )
    """

    receptor_path: Optional[Path]
    molecule_names: List[str]
    molecules: List[Any] = field(default_factory=list, repr=False)
    prolif_molecules: List[Any] = field(default_factory=list, repr=False)
    fingerprint: Any = None
    fingerprint_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    interaction_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bitvectors: Optional[List[Any]] = None
    countvectors: Optional[List[Any]] = None
    protein_molecule: Any = field(default=None, repr=False)
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def pose_names(self) -> List[str]:
        """
        Return a backward-compatible alias for ``molecule_names``.

        :returns:
            Copy of the stored molecule names.
        :rtype: List[str]

        Example
        -------
        .. code-block:: python

            names = result.pose_names
        """
        return list(self.molecule_names)

    @property
    def ligand_molecules(self) -> List[Any]:
        """
        Return a backward-compatible alias for ``molecules``.

        :returns:
            Copy of the stored molecule objects.
        :rtype: List[Any]

        Example
        -------
        .. code-block:: python

            ligands = result.ligand_molecules
        """
        return list(self.molecules)

    def molecule_table(self, include_smiles: bool = True) -> pd.DataFrame:
        """
        Build a per-molecule table with direct object references.

        The returned table contains one row per molecule and includes the raw
        molecule object in the ``mol`` column. This is useful for in-memory
        inspection but may not be suitable for CSV export unless the ``mol``
        column is removed.

        :param include_smiles:
            Whether to include a SMILES representation computed from each
            molecule.
        :type include_smiles: bool

        :returns:
            Per-molecule table with columns such as ``mol_index``, ``mol_name``,
            ``mol``, and optionally ``smiles``.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            molecule_df = result.molecule_table(include_smiles=True)
        """
        rows = []
        for index, (name, mol) in enumerate(zip(self.molecule_names, self.molecules)):
            row = {"mol_index": index, "mol_name": name, "mol": mol}
            if include_smiles:
                row["smiles"] = mol_to_smiles(mol)
            rows.append(row)
        return pd.DataFrame(rows)

    def serializable_interaction_df(self) -> pd.DataFrame:
        """
        Return a CSV-friendly interaction event table.

        If a ``mol`` column is present, it is replaced by a ``mol_smiles``
        column derived from :func:`mol_to_smiles`.

        :returns:
            Serializable interaction table suitable for CSV export.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            csv_df = result.serializable_interaction_df()
            csv_df.to_csv("interaction_events.csv", index=False)
        """
        if self.interaction_df.empty:
            return self.interaction_df.copy()
        serializable = self.interaction_df.copy()
        if "mol" in serializable.columns:
            serializable["mol_smiles"] = serializable["mol"].map(mol_to_smiles)
            serializable = serializable.drop(columns=["mol"])
        return serializable

    def save_tables(
        self,
        out_dir: str | Path,
        prefix: str = "interaction",
    ) -> Dict[str, Path]:
        """
        Save single-run tables and optional fingerprint pickle to disk.

        The following files are generated:

        - ``{prefix}_fingerprint.csv``
        - ``{prefix}_events.csv``
        - ``{prefix}_molecules.csv``

        In addition, if ``fingerprint`` exposes a ``to_pickle`` method, the
        method also writes:

        - ``{prefix}_fingerprint.pkl``

        :param out_dir:
            Output directory in which all files should be created.
        :type out_dir: str | pathlib.Path
        :param prefix:
            Filename prefix for exported tables.
        :type prefix: str

        :returns:
            Mapping from output artifact name to written file path.
        :rtype: Dict[str, pathlib.Path]

        Example
        -------
        .. code-block:: python

            created = result.save_tables("Results/interaction_run", prefix="egfr")
            print(created["interactions_csv"])
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        fingerprint_csv = out_path / f"{prefix}_fingerprint.csv"
        interactions_csv = out_path / f"{prefix}_events.csv"
        molecules_csv = out_path / f"{prefix}_molecules.csv"

        self.fingerprint_df.to_csv(fingerprint_csv)
        self.serializable_interaction_df().to_csv(interactions_csv, index=False)
        self.molecule_table(include_smiles=True).drop(
            columns=["mol"], errors="ignore"
        ).to_csv(molecules_csv, index=False)

        created = {
            "fingerprint_csv": fingerprint_csv,
            "interactions_csv": interactions_csv,
            "molecules_csv": molecules_csv,
        }
        if self.fingerprint is not None and hasattr(self.fingerprint, "to_pickle"):
            fp_pickle = out_path / f"{prefix}_fingerprint.pkl"
            self.fingerprint.to_pickle(fp_pickle)
            created["fingerprint_pickle"] = fp_pickle
        return created


@dataclass
class PoseInteractionTableResult:
    """
    Container holding pose-centric interaction outputs.

    All main tables are aligned one-row-per-pose:

    - ``merged_df`` contains the original pose table plus pose identifiers and
      optional extra columns.
    - ``interaction_df`` contains one row per pose with raw grouped interaction
      events.
    - ``summary_df`` contains one row per pose with compact and detailed
      interaction summaries.

    :param merged_df:
        Pose-level merged table, often derived from the original pose metadata
        plus interaction-related columns.
    :type merged_df: pandas.DataFrame
    :param interaction_df:
        Pose-level raw-event table, typically produced by
        ``build_pose_interaction_table``.
    :type interaction_df: pandas.DataFrame
    :param summary_df:
        Pose-level summary table, typically produced by
        ``build_pose_summary_table``.
    :type summary_df: pandas.DataFrame
    :param receptor_pdb_by_id:
        Mapping from receptor identifier to receptor PDB path.
    :type receptor_pdb_by_id: Dict[str, pathlib.Path]
    :param batch_size:
        Batch size used during processing, when relevant.
    :type batch_size: int
    :param settings:
        Optional processing settings or provenance metadata.
    :type settings: Dict[str, Any]
    :param errors:
        List of structured error payloads captured during processing.
    :type errors: List[Dict[str, Any]]
    :param bitvectors:
        Optional bitvector fingerprints aligned with ``molecule_names``.
    :type bitvectors: Optional[List[Any]]
    :param countvectors:
        Optional countvector fingerprints aligned with ``molecule_names``.
    :type countvectors: Optional[List[Any]]
    :param molecule_names:
        Ordered molecule or pose names aligned with similarity vectors.
    :type molecule_names: List[str]

    Example
    -------
    .. code-block:: python

        pose_result = PoseInteractionTableResult(
            merged_df=merged_df,
            interaction_df=pose_events_df,
            summary_df=pose_summary_df,
            molecule_names=["pose_1", "pose_2"],
            errors=[],
        )
    """

    merged_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    interaction_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    receptor_pdb_by_id: Dict[str, Path] = field(default_factory=dict)
    batch_size: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    bitvectors: Optional[List[Any]] = None
    countvectors: Optional[List[Any]] = None
    molecule_names: List[str] = field(default_factory=list)

    def serializable_merged_df(self) -> pd.DataFrame:
        """
        Return a CSV-friendly merged pose table.

        If a ``mol`` column is present, it is replaced by a ``mol_smiles``
        column derived from :func:`mol_to_smiles`.

        :returns:
            Serializable merged pose table.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            merged_csv_df = pose_result.serializable_merged_df()
        """
        if self.merged_df.empty:
            return self.merged_df.copy()
        serializable = self.merged_df.copy()
        if "mol" in serializable.columns:
            serializable["mol_smiles"] = serializable["mol"].map(mol_to_smiles)
            serializable = serializable.drop(columns=["mol"])
        return serializable

    def serializable_interaction_df(self) -> pd.DataFrame:
        """
        Return a CSV-friendly pose-level raw-event table.

        The in-memory ``interaction_events`` payload column is removed because
        the JSON string column is typically the export-safe representation.

        :returns:
            Serializable pose-level raw-event table.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            events_csv_df = pose_result.serializable_interaction_df()
        """
        if self.interaction_df.empty:
            return self.interaction_df.copy()
        serializable = self.interaction_df.copy()
        if "interaction_events" in serializable.columns:
            serializable = serializable.drop(columns=["interaction_events"])
        return serializable

    def serializable_summary_df(self) -> pd.DataFrame:
        """
        Return a CSV-friendly pose-level summary table.

        The in-memory nested payload columns ``interaction_compact`` and
        ``interaction_detail`` are removed because their JSON string
        counterparts are more suitable for flat-file export.

        :returns:
            Serializable pose-level summary table.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            summary_csv_df = pose_result.serializable_summary_df()
        """
        if self.summary_df.empty:
            return self.summary_df.copy()
        serializable = self.summary_df.copy()
        for column in ["interaction_compact", "interaction_detail"]:
            if column in serializable.columns:
                serializable = serializable.drop(columns=[column])
        return serializable

    def summary_dict(
        self,
        *,
        kind: str = "compact",
        as_sets: bool = False,
        drop_empty: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert ``summary_df`` into a ``pose_id -> payload`` dictionary.

        This is a thin wrapper around :func:`summary_table_to_dict`.

        :param kind:
            Summary payload type to export. Must be either ``"compact"`` or
            ``"detail"``.
        :type kind: str
        :param as_sets:
            Whether compact residue lists should be converted to sets.
        :type as_sets: bool
        :param drop_empty:
            Whether poses without interactions should be omitted.
        :type drop_empty: bool

        :returns:
            Dictionary keyed by pose id.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            compact_map = pose_result.summary_dict(kind="compact")
            detail_map = pose_result.summary_dict(kind="detail")
        """
        return summary_table_to_dict(
            self.summary_df,
            kind=kind,
            as_sets=as_sets,
            drop_empty=drop_empty,
        )

    def interaction_dict(self, *, drop_empty: bool = False) -> Dict[str, Any]:
        """
        Convert ``interaction_df`` into a ``pose_id -> event_list`` dictionary.

        :param drop_empty:
            Whether poses with no grouped event payload should be omitted.
        :type drop_empty: bool

        :returns:
            Dictionary keyed by pose id with grouped event payloads.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            event_map = pose_result.interaction_dict(drop_empty=True)
        """
        if self.interaction_df.empty:
            return {}
        result: Dict[str, Any] = {}
        for row in self.interaction_df.itertuples(index=False):
            pose_id = str(getattr(row, "pose_id"))
            payload = getattr(row, "interaction_events", None)
            if payload is None and drop_empty:
                continue
            result[pose_id] = payload
        return result

    def similarity_matrix(self, kind: str = "bit") -> pd.DataFrame:
        """
        Compute a Tanimoto similarity matrix from stored vectors.

        This method dispatches to
        :func:`prodock.postprocess.interaction.similarity.tanimoto_similarity_matrix`
        using either stored bitvectors or countvectors.

        :param kind:
            Vector type to use. Must be either ``"bit"`` or ``"count"``.
        :type kind: str

        :returns:
            Square similarity matrix indexed by ``molecule_names``.
        :rtype: pandas.DataFrame

        :raises ValueError:
            If ``kind`` is invalid or if the required vectors are unavailable.

        Example
        -------
        .. code-block:: python

            sim_df = pose_result.similarity_matrix(kind="bit")
        """
        from .similarity import tanimoto_similarity_matrix

        normalized = kind.strip().lower()
        if normalized == "bit":
            if not self.bitvectors:
                raise ValueError("No bitvectors are available in this result.")
            return tanimoto_similarity_matrix(self.bitvectors, self.molecule_names)
        if normalized == "count":
            if not self.countvectors:
                raise ValueError("No countvectors are available in this result.")
            return tanimoto_similarity_matrix(self.countvectors, self.molecule_names)
        raise ValueError("kind must be either 'bit' or 'count'.")

    def save_tables(
        self,
        out_dir: str | Path,
        prefix: str = "pose_interaction",
    ) -> Dict[str, Path]:
        """
        Save pose-level tables and summary mappings to disk.

        The following files are generated:

        - ``{prefix}_merged.csv``
        - ``{prefix}_events.csv``
        - ``{prefix}_summary.csv``
        - ``{prefix}_errors.csv``
        - ``{prefix}_compact_map.json``
        - ``{prefix}_detail_map.json``

        :param out_dir:
            Output directory in which all files should be created.
        :type out_dir: str | pathlib.Path
        :param prefix:
            Filename prefix for exported tables and JSON files.
        :type prefix: str

        :returns:
            Mapping from artifact name to written file path.
        :rtype: Dict[str, pathlib.Path]

        Example
        -------
        .. code-block:: python

            created = pose_result.save_tables("Results/pose_tables", prefix="egfr")
            print(created["summary_csv"])
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        merged_csv = out_path / f"{prefix}_merged.csv"
        interactions_csv = out_path / f"{prefix}_events.csv"
        summary_csv = out_path / f"{prefix}_summary.csv"
        errors_csv = out_path / f"{prefix}_errors.csv"
        compact_json = out_path / f"{prefix}_compact_map.json"
        detail_json = out_path / f"{prefix}_detail_map.json"

        self.serializable_merged_df().to_csv(merged_csv, index=False)
        self.serializable_interaction_df().to_csv(interactions_csv, index=False)
        self.serializable_summary_df().to_csv(summary_csv, index=False)
        pd.DataFrame(self.errors).to_csv(errors_csv, index=False)
        compact_json.write_text(
            json.dumps(self.summary_dict(kind="compact"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        detail_json.write_text(
            json.dumps(self.summary_dict(kind="detail"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return {
            "merged_csv": merged_csv,
            "interactions_csv": interactions_csv,
            "summary_csv": summary_csv,
            "errors_csv": errors_csv,
            "compact_json": compact_json,
            "detail_json": detail_json,
        }
