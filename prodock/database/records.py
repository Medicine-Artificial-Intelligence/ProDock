from __future__ import annotations

"""Typed record containers for the ProDock SQLite database layer."""

from dataclasses import dataclass, field
from typing import Any, Optional

from rdkit.Chem import rdchem

from .serialization import make_pose_key


@dataclass(frozen=True)
class PoseRecord:
    """
    Immutable in-memory representation of a docking pose row.

    This record combines core pose identity fields with optional molecule
    content, score payloads, and aggregated interaction summaries. It is
    designed to act as a typed container for rows reconstructed from the
    ProDock SQLite database layer.

    :param pose_db_id:
        Internal SQLite integer primary key for the pose row.
    :type pose_db_id: int
    :param pose_id:
        Optional external stable pose identifier, for example
        ``1M17__erlotinib__qvina__pose1``. When absent, a logical pose key can
        still be generated from the receptor, ligand, engine, and rank fields.
    :type pose_id: Optional[str]
    :param receptor_id:
        Receptor identifier associated with the pose.
    :type receptor_id: str
    :param ligand_id:
        Ligand identifier associated with the pose.
    :type ligand_id: str
    :param engine:
        Docking engine name, for example ``vina``, ``smina``, or ``qvina``.
    :type engine: str
    :param pose_rank:
        One-based pose rank within the receptor-ligand-engine group.
    :type pose_rank: int
    :param affinity:
        Primary affinity value associated with the pose, if available.
    :type affinity: Optional[float]
    :param mol:
        Deserialized RDKit molecule for the pose. This may be ``None`` when
        molecule blobs are not loaded from the database.
    :type mol: Optional[rdchem.Mol]
    :param pose_metadata:
        Free-form pose-level metadata stored with the ``poses`` table row.
    :type pose_metadata: dict[str, Any]
    :param score_data:
        Structured score payload stored in the related ``pose_scores`` row.
    :type score_data: dict[str, Any]
    :param score_metadata:
        Additional metadata associated with the score payload.
    :type score_metadata: dict[str, Any]
    :param interaction_summary:
        Optional grouped interaction summary in the form
        ``{interaction_type: [residue_id, ...]}``.
    :type interaction_summary: dict[str, list[str]]
    :param interaction_details:
        Optional grouped detailed interaction payload, typically shaped like
        ``{interaction_type: {residue_id: [event, ...]}}`` or a similar
        nested structure.
    :type interaction_details: dict[str, Any]
    :param created_at:
        SQLite insertion timestamp for the pose row.
    :type created_at: str

    Example:
        >>> record = PoseRecord(
        ...     pose_db_id=1,
        ...     pose_id=None,
        ...     receptor_id="1M17",
        ...     ligand_id="erlotinib",
        ...     engine="qvina",
        ...     pose_rank=1,
        ...     affinity=-6.2,
        ...     mol=None,
        ...     pose_metadata={},
        ...     score_data={"affinity": -6.2},
        ...     score_metadata={},
        ... )
        >>> record.pose_key
        '1M17__erlotinib__qvina__pose1'
    """

    pose_db_id: int
    pose_id: Optional[str]
    receptor_id: str
    ligand_id: str
    engine: str
    pose_rank: int
    affinity: Optional[float]
    mol: Optional[rdchem.Mol]
    pose_metadata: dict[str, Any]
    score_data: dict[str, Any]
    score_metadata: dict[str, Any]
    interaction_summary: dict[str, list[str]] = field(default_factory=dict)
    interaction_details: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    @property
    def pose_key(self) -> str:
        """
        Return the best available human-readable pose key.

        This property prefers the stored external ``pose_id`` when present.
        Otherwise, it deterministically reconstructs a logical key from the
        receptor identifier, ligand identifier, engine, and pose rank.

        :returns:
            Stable pose key suitable for display, export, or downstream
            matching.
        :rtype: str

        Example:
            >>> record.pose_key
            '1M17__erlotinib__qvina__pose1'
        """
        return self.pose_id or make_pose_key(
            self.receptor_id,
            self.ligand_id,
            self.engine,
            self.pose_rank,
        )


@dataclass(frozen=True)
class ScoreRecord:
    """
    Immutable in-memory representation of a pose score row.

    This record stores resolved pose identity fields together with a structured
    score payload and optional metadata. It is typically used for score-centric
    queries where loading the full pose molecule or interaction details is not
    necessary.

    :param pose_db_id:
        Internal SQLite pose primary key.
    :type pose_db_id: int
    :param pose_id:
        Optional external stable pose identifier.
    :type pose_id: Optional[str]
    :param receptor_id:
        Receptor identifier resolved through the associated pose row.
    :type receptor_id: str
    :param ligand_id:
        Ligand identifier resolved through the associated pose row.
    :type ligand_id: str
    :param engine:
        Docking engine name resolved through the associated pose row.
    :type engine: str
    :param pose_rank:
        One-based pose rank mirrored from the pose row.
    :type pose_rank: int
    :param affinity:
        Primary affinity value, if available.
    :type affinity: Optional[float]
    :param score_data:
        Structured score payload, for example containing raw engine-specific
        score terms.
    :type score_data: dict[str, Any]
    :param metadata:
        Additional metadata associated with the score record.
    :type metadata: dict[str, Any]

    Example:
        >>> record = ScoreRecord(
        ...     pose_db_id=1,
        ...     pose_id=None,
        ...     receptor_id="1M17",
        ...     ligand_id="erlotinib",
        ...     engine="vina",
        ...     pose_rank=2,
        ...     affinity=-7.1,
        ...     score_data={"affinity": -7.1, "cnn_pose": 0.82},
        ...     metadata={},
        ... )
        >>> record.pose_key
        '1M17__erlotinib__vina__pose2'
    """

    pose_db_id: int
    pose_id: Optional[str]
    receptor_id: str
    ligand_id: str
    engine: str
    pose_rank: int
    affinity: Optional[float]
    score_data: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def pose_key(self) -> str:
        """
        Return the best available human-readable pose key.

        This property prefers the stored external ``pose_id`` when available.
        Otherwise, it reconstructs a deterministic logical key from the pose
        identity fields.

        :returns:
            Stable pose key suitable for display and record matching.
        :rtype: str
        """
        return self.pose_id or make_pose_key(
            self.receptor_id,
            self.ligand_id,
            self.engine,
            self.pose_rank,
        )


@dataclass(frozen=True)
class InteractionRecord:
    """
    Immutable in-memory representation of a pose interaction row.

    Each instance represents one detailed interaction event associated with a
    specific docking pose. The record includes resolved pose identity fields,
    residue-level annotations, atom index mappings, geometric descriptors, and
    arbitrary extra metadata.

    :param interaction_id:
        Internal SQLite integer primary key for the interaction row.
    :type interaction_id: int
    :param pose_db_id:
        Foreign-key link to the associated pose row.
    :type pose_db_id: int
    :param pose_id:
        Optional external stable pose identifier.
    :type pose_id: Optional[str]
    :param receptor_id:
        Receptor identifier resolved through the associated pose row.
    :type receptor_id: str
    :param ligand_id:
        Ligand identifier resolved through the associated pose row.
    :type ligand_id: str
    :param engine:
        Docking engine name resolved through the associated pose row.
    :type engine: str
    :param pose_rank:
        One-based pose rank resolved through the associated pose row.
    :type pose_rank: int
    :param interaction_type:
        Interaction label such as ``Hydrophobic``, ``VdWContact``, or
        ``HBDonor``.
    :type interaction_type: str
    :param chain_id:
        Optional protein chain identifier.
    :type chain_id: Optional[str]
    :param residue_name:
        Optional residue name, for example ``LEU``.
    :type residue_name: Optional[str]
    :param residue_number:
        Optional residue number, for example ``149``.
    :type residue_number: Optional[int]
    :param residue_id:
        Optional compact residue identifier such as ``LEU149.A``.
    :type residue_id: Optional[str]
    :param ligand_residue:
        Optional ligand residue label such as ``LIG1``.
    :type ligand_residue: Optional[str]
    :param occurrence_index:
        Zero-based occurrence index for repeated interactions of the same type
        at the same residue.
    :type occurrence_index: int
    :param ligand_atom_indices:
        Ligand atom indices participating directly in the interaction.
    :type ligand_atom_indices: list[int]
    :param protein_atom_indices:
        Protein atom indices participating directly in the interaction.
    :type protein_atom_indices: list[int]
    :param ligand_parent_atom_indices:
        Parent ligand atom indices when available from the upstream interaction
        extractor.
    :type ligand_parent_atom_indices: list[int]
    :param protein_parent_atom_indices:
        Parent protein atom indices when available from the upstream
        interaction extractor.
    :type protein_parent_atom_indices: list[int]
    :param distance:
        Optional interaction distance value.
    :type distance: Optional[float]
    :param angle:
        Optional interaction angle value.
    :type angle: Optional[float]
    :param metadata:
        Additional free-form metadata associated with the interaction event.
    :type metadata: dict[str, Any]
    :param created_at:
        SQLite insertion timestamp for the interaction row.
    :type created_at: str

    Example:
        >>> record = InteractionRecord(
        ...     interaction_id=1,
        ...     pose_db_id=10,
        ...     pose_id=None,
        ...     receptor_id="1M17",
        ...     ligand_id="erlotinib",
        ...     engine="qvina",
        ...     pose_rank=1,
        ...     interaction_type="Hydrophobic",
        ...     chain_id="A",
        ...     residue_name="LEU",
        ...     residue_number=149,
        ...     residue_id="LEU149.A",
        ...     ligand_residue="LIG1",
        ...     occurrence_index=0,
        ...     ligand_atom_indices=[2],
        ...     protein_atom_indices=[9],
        ...     ligand_parent_atom_indices=[2],
        ...     protein_parent_atom_indices=[2392],
        ...     distance=4.49,
        ...     angle=None,
        ...     metadata={},
        ...     created_at="2026-04-02 10:00:00",
        ... )
        >>> record.pose_key
        '1M17__erlotinib__qvina__pose1'
    """

    interaction_id: int
    pose_db_id: int
    pose_id: Optional[str]
    receptor_id: str
    ligand_id: str
    engine: str
    pose_rank: int
    interaction_type: str
    chain_id: Optional[str]
    residue_name: Optional[str]
    residue_number: Optional[int]
    residue_id: Optional[str]
    ligand_residue: Optional[str]
    occurrence_index: int
    ligand_atom_indices: list[int]
    protein_atom_indices: list[int]
    ligand_parent_atom_indices: list[int]
    protein_parent_atom_indices: list[int]
    distance: Optional[float]
    angle: Optional[float]
    metadata: dict[str, Any]
    created_at: str

    @property
    def pose_key(self) -> str:
        """
        Return the best available human-readable pose key.

        This property prefers the stored external ``pose_id`` when available.
        Otherwise, it reconstructs a deterministic logical key from the pose
        identity fields.

        :returns:
            Stable pose key suitable for grouping, display, and record matching.
        :rtype: str
        """
        return self.pose_id or make_pose_key(
            self.receptor_id,
            self.ligand_id,
            self.engine,
            self.pose_rank,
        )
