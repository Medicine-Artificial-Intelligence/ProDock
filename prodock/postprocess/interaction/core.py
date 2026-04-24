from __future__ import annotations

"""High-level ProLIF-based interaction extraction for ProDock.

This module provides the main automation layer for protein-ligand interaction
analysis in ProDock. It wraps receptor loading, ligand preparation, ProLIF
fingerprint execution, event flattening, and pose-level summarization into a
small set of convenient entry points.

The main workflow is:

1. Load one receptor PDB with :func:`load_receptor_molecule`.
2. Prepare one or more ligands or docked poses with :func:`prepare_ligands`.
3. Run a configured ProLIF fingerprint calculation.
4. Convert raw ProLIF outputs into:
   - a wide fingerprint dataframe,
   - a long-form interaction-event dataframe,
   - a pose-centric summary table.

This module supports two usage styles:

- **single-run mode** with :meth:`InteractionProfiler.run` or
  :func:`extract_interactions`
- **pose-table automation mode** with :meth:`InteractionProfiler.run_pose_table`
  or :func:`extract_pose_table_interactions`

Example
-------
Simple single-receptor automation using docked poses:

.. code-block:: python

    from prodock.postprocess.interaction import extract_interactions

    result = extract_interactions(
        receptor_pdb="Data/testcase/Multi/1M17/filtered_protein/1M17.pdb",
        ligands="Data/testcase/post/1M17/erlotinib.sdf",
        progress=False,
        n_jobs=1,
    )

    fingerprint_df = result.fingerprint_df
    interaction_df = result.interaction_df

Example
-------
Pose-table automation using crawled molecules from multiple receptors:

.. code-block:: python

    from prodock.postprocess.pose.core import PoseCrawler
    from prodock.postprocess.interaction import extract_pose_table_interactions

    crawler = PoseCrawler(["./Data/testcase/post"])
    df = crawler.crawl_mols(backend="obabel")

    result = extract_pose_table_interactions(
        poses=df,
        receptor_pdb_by_id={
            "1M17": "Data/testcase/Multi/1M17/filtered_protein/1M17.pdb",
            "4WKQ": "Data/testcase/Multi/4WKQ/filtered_protein/4WKQ.pdb",
        },
        batch_size=10,
        progress=False,
        n_jobs=1,
        include_fingerprint_columns=True,
        include_interaction_events=True,
        include_bitvectors=False,
        include_countvectors=False,
        fail_fast=True,
    )

    merged_df = result.merged_df
    interaction_df = result.interaction_df
    summary_df = result.summary_df
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence
import gc
import logging
import traceback

import pandas as pd

from .exceptions import InteractionProcessingError
from .flatten import build_pose_interaction_table, build_pose_summary_table, flatten_ifp
from .io import _import_prolif, load_receptor_molecule, prepare_ligands
from .models import InteractionRunResult, PoseInteractionTableResult

PathLike = str | Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class InteractionProfiler:
    """
    High-level helper for protein-ligand interaction extraction using ProLIF.

    This class stores all interaction-analysis settings in one place and
    exposes two main execution methods:

    - :meth:`run` for one receptor plus one ligand source
    - :meth:`run_pose_table` for automated pose-table workflows across one or
      multiple receptors

    The class is designed for ProDock automation and notebook workflows where
    reproducible settings, pose-level summaries, and optional fingerprint
    vectors are useful.

    :param interactions:
        Optional subset of ProLIF interaction names to enable. If ``None``,
        ProLIF defaults are used.
    :type interactions: Optional[Sequence[str]]
    :param parameters:
        Optional parameter overrides passed directly to ProLIF interaction
        definitions.
    :type parameters: Optional[Dict[str, Dict[str, Any]]]
    :param count:
        Whether to generate count fingerprints instead of boolean fingerprints.
    :type count: bool
    :param vicinity_cutoff:
        Distance cutoff used by ProLIF when automatically selecting nearby
        receptor residues.
    :type vicinity_cutoff: float
    :param receptor_selection:
        Optional MDAnalysis selection string for the receptor. If ``None``, all
        atoms from the receptor structure are used.
    :type receptor_selection: Optional[str]
    :param receptor_use_segid:
        Whether ProLIF should use segment id instead of chain id for receptor
        residue identifiers.
    :type receptor_use_segid: Optional[bool]
    :param ligand_resname:
        Default ligand residue name used when an RDKit molecule has no residue
        metadata.
    :type ligand_resname: str
    :param ligand_resnumber:
        Default ligand residue number used when an RDKit molecule has no residue
        metadata.
    :type ligand_resnumber: int
    :param ligand_chain:
        Default ligand chain id used when an RDKit molecule has no residue
        metadata.
    :type ligand_chain: str
    :param ligand_use_segid:
        Whether ProLIF should use segment id instead of chain id for ligands.
    :type ligand_use_segid: bool
    :param sdf_sanitize:
        Whether RDKit should sanitize molecules when reading an SDF file.
    :type sdf_sanitize: bool
    :param receptor_guess_bonds:
        Whether to proactively guess receptor bond topology before ProLIF
        converts the receptor to RDKit.
    :type receptor_guess_bonds: bool
    :param receptor_vdwradii:
        Optional VdW radii mapping forwarded to MDAnalysis bond guessing.
    :type receptor_vdwradii: Optional[Mapping[str, float]]
    :param suppress_mdanalysis_warnings:
        Whether to suppress known non-actionable MDAnalysis warnings.
    :type suppress_mdanalysis_warnings: bool
    :param suppress_mdanalysis_info_logs:
        Whether to suppress repeated MDAnalysis info log messages.
    :type suppress_mdanalysis_info_logs: bool
    :param progress:
        Whether ProLIF should show a progress bar.
    :type progress: bool
    :param n_jobs:
        Number of parallel jobs used by ProLIF.
    :type n_jobs: Optional[int]
    :param drop_empty:
        Whether to drop empty columns in the wide fingerprint table.
    :type drop_empty: bool

    Example
    -------
    Create a profiler and run interaction extraction for one SDF file:

    .. code-block:: python

        from prodock.postprocess.interaction.core import InteractionProfiler

        profiler = InteractionProfiler(
            count=False,
            vicinity_cutoff=6.0,
            progress=False,
            n_jobs=1,
        )

        result = profiler.run(
            receptor_pdb="Data/testcase/Multi/1M17/filtered_protein/1M17.pdb",
            ligands="Data/testcase/post/1M17/erlotinib.sdf",
        )

        print(result.fingerprint_df.head())
        print(result.interaction_df.head())

    Example
    -------
    Run pose-table automation for multiple receptors:

    .. code-block:: python

        profiler = InteractionProfiler(progress=False, n_jobs=1)

        result = profiler.run_pose_table(
            poses=df,
            receptor_pdb_by_id={
                "1M17": "Data/testcase/Multi/1M17/filtered_protein/1M17.pdb",
                "4WKQ": "Data/testcase/Multi/4WKQ/filtered_protein/4WKQ.pdb",
            },
            batch_size=10,
            include_interaction_events=True,
            include_bitvectors=False,
            include_countvectors=False,
            fail_fast=True,
        )

        merged_df = result.merged_df
        interaction_df = result.interaction_df
        summary_df = result.summary_df
    """

    interactions: Optional[Sequence[str]] = None
    parameters: Optional[Dict[str, Dict[str, Any]]] = None
    count: bool = False
    vicinity_cutoff: float = 6.0
    receptor_selection: Optional[str] = None
    receptor_use_segid: Optional[bool] = None
    ligand_resname: str = "LIG"
    ligand_resnumber: int = 1
    ligand_chain: str = ""
    ligand_use_segid: bool = False
    sdf_sanitize: bool = True
    receptor_guess_bonds: bool = True
    receptor_vdwradii: Optional[Mapping[str, float]] = None
    suppress_mdanalysis_warnings: bool = True
    suppress_mdanalysis_info_logs: bool = True
    progress: bool = False
    n_jobs: Optional[int] = 1
    drop_empty: bool = True
    _settings_cache: Dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

    def _build_fingerprint(self) -> Any:
        """
        Construct a configured ProLIF fingerprint object.

        The fingerprint is built from the current profiler settings, including
        optional interaction subsets, parameter overrides, count mode, vicinity
        cutoff, and optional residue identifier behavior.

        :returns:
            Configured ProLIF fingerprint object.
        :rtype: Any
        """
        plf = _import_prolif()
        kwargs: Dict[str, Any] = {
            "count": self.count,
            "vicinity_cutoff": self.vicinity_cutoff,
        }
        if self.interactions is not None:
            kwargs["interactions"] = list(self.interactions)
        if self.parameters is not None:
            kwargs["parameters"] = dict(self.parameters)
        if self.receptor_use_segid is not None:
            kwargs["use_segid"] = self.receptor_use_segid
        return plf.Fingerprint(**kwargs)

    def available_interactions(
        self, show_hidden: bool = False, show_bridged: bool = False
    ) -> list[str]:
        """
        List interactions available in the installed ProLIF version.

        :param show_hidden:
            Whether hidden interactions should be included.
        :type show_hidden: bool
        :param show_bridged:
            Whether bridged interactions should be included.
        :type show_bridged: bool

        :returns:
            List of interaction names supported by the installed ProLIF version.
        :rtype: list[str]
        """
        plf = _import_prolif()
        return list(
            plf.Fingerprint.list_available(
                show_hidden=show_hidden, show_bridged=show_bridged
            )
        )

    def settings_snapshot(self) -> Dict[str, Any]:
        """
        Return a serializable snapshot of the current profiler settings.

        :returns:
            Serializable dictionary of profiler settings.
        :rtype: Dict[str, Any]
        """
        return {
            "interactions": (
                list(self.interactions) if self.interactions is not None else None
            ),
            "parameters": (
                dict(self.parameters) if self.parameters is not None else None
            ),
            "count": self.count,
            "vicinity_cutoff": self.vicinity_cutoff,
            "receptor_selection": self.receptor_selection,
            "receptor_use_segid": self.receptor_use_segid,
            "ligand_resname": self.ligand_resname,
            "ligand_resnumber": self.ligand_resnumber,
            "ligand_chain": self.ligand_chain,
            "ligand_use_segid": self.ligand_use_segid,
            "sdf_sanitize": self.sdf_sanitize,
            "receptor_guess_bonds": self.receptor_guess_bonds,
            "receptor_vdwradii": (
                dict(self.receptor_vdwradii)
                if self.receptor_vdwradii is not None
                else None
            ),
            "suppress_mdanalysis_warnings": self.suppress_mdanalysis_warnings,
            "suppress_mdanalysis_info_logs": self.suppress_mdanalysis_info_logs,
            "progress": self.progress,
            "n_jobs": self.n_jobs,
            "drop_empty": self.drop_empty,
        }

    def run(
        self,
        receptor_pdb: PathLike,
        ligands: PathLike | Any | Sequence[Any] | Iterable[Any] | Mapping[str, Any],
        residues: Sequence[str] | str | None = None,
    ) -> InteractionRunResult:
        """
        Extract protein-ligand interactions for one receptor and one ligand input source.

        :param receptor_pdb:
            Path to the receptor PDB file.
        :type receptor_pdb: str | pathlib.Path
        :param ligands:
            Ligand input source.
        :type ligands: str | pathlib.Path | Any | Sequence[Any] | Iterable[Any] | Mapping[str, Any]
        :param residues:
            Optional residue subset passed to ``Fingerprint.run_from_iterable``.
        :type residues: Optional[Sequence[str] | str]

        :returns:
            Structured interaction extraction result.
        :rtype: InteractionRunResult
        """
        receptor_path = Path(receptor_pdb)

        logger.info(
            "Starting interaction run: receptor=%s ligand_input_type=%s residues=%s "
            "progress=%s n_jobs=%s count=%s",
            receptor_path,
            type(ligands).__name__,
            residues,
            self.progress,
            self.n_jobs,
            self.count,
        )

        try:
            protein_molecule = load_receptor_molecule(
                receptor_path,
                selection=self.receptor_selection,
                use_segid=self.receptor_use_segid,
                guess_bonds=self.receptor_guess_bonds,
                vdwradii=self.receptor_vdwradii,
                suppress_mdanalysis_warnings=self.suppress_mdanalysis_warnings,
                suppress_mdanalysis_info_logs=self.suppress_mdanalysis_info_logs,
            )
            logger.debug("Receptor loaded successfully: %s", receptor_path)

            molecule_names, molecules, prolif_molecules = prepare_ligands(
                ligands,
                resname=self.ligand_resname,
                resnumber=self.ligand_resnumber,
                chain=self.ligand_chain,
                use_segid=self.ligand_use_segid,
                sdf_sanitize=self.sdf_sanitize,
            )
            logger.info(
                "Prepared ligands successfully: n_molecules=%d names=%s",
                len(molecule_names),
                molecule_names[:5],
            )

            fingerprint = self._build_fingerprint()
            logger.debug(
                "Fingerprint object created: interactions=%s parameters=%s vicinity_cutoff=%s",
                self.interactions,
                self.parameters,
                self.vicinity_cutoff,
            )

            fingerprint.run_from_iterable(
                prolif_molecules,
                protein_molecule,
                residues=residues,
                progress=self.progress,
                n_jobs=self.n_jobs,
            )
            logger.info(
                "Fingerprint run completed: receptor=%s n_ligands=%d",
                receptor_path,
                len(prolif_molecules),
            )

            fingerprint_df = fingerprint.to_dataframe(drop_empty=self.drop_empty)
            fingerprint_df = self._rename_fingerprint_index(
                fingerprint_df, molecule_names
            )

            interaction_df = flatten_ifp(
                fingerprint.ifp, mol_names=molecule_names, mols=molecules
            )

            logger.info(
                "Interaction tables built: fingerprint_shape=%s interaction_shape=%s",
                getattr(fingerprint_df, "shape", None),
                getattr(interaction_df, "shape", None),
            )

            bitvectors = None
            countvectors = None
            if hasattr(fingerprint, "to_bitvectors"):
                try:
                    bitvectors = list(fingerprint.to_bitvectors())
                    logger.debug("Collected bitvectors: n=%d", len(bitvectors))
                except Exception:
                    logger.exception("Failed to collect bitvectors.")

            if hasattr(fingerprint, "to_countvectors"):
                try:
                    countvectors = list(fingerprint.to_countvectors())
                    logger.debug("Collected countvectors: n=%d", len(countvectors))
                except Exception:
                    logger.exception("Failed to collect countvectors.")

            result = InteractionRunResult(
                receptor_path=receptor_path,
                molecule_names=molecule_names,
                molecules=molecules,
                prolif_molecules=prolif_molecules,
                fingerprint=fingerprint,
                fingerprint_df=fingerprint_df,
                interaction_df=interaction_df,
                bitvectors=bitvectors,
                countvectors=countvectors,
                protein_molecule=protein_molecule,
                settings=self.settings_snapshot(),
            )

            logger.info(
                "Interaction run finished successfully: receptor=%s n_molecules=%d",
                receptor_path,
                len(molecule_names),
            )
            return result

        except Exception as exc:
            logger.exception(
                "Interaction run failed: receptor=%s ligand_input_type=%s residues=%s",
                receptor_path,
                type(ligands).__name__,
                residues,
            )
            raise InteractionProcessingError(
                f"Single-run interaction processing failed for receptor {receptor_path!s}"
            ) from exc

    def run_pose_table(
        self,
        poses: pd.DataFrame,
        receptor_pdb_by_id: Mapping[str, PathLike],
        *,
        receptor_col: str = "receptor_id",
        ligand_col: str = "ligand_id",
        engine_col: str = "engine",
        pose_rank_col: str = "pose_rank",
        affinity_col: str = "affinity",
        mol_col: str = "mol",
        pose_id_col: str | None = None,
        residues: Sequence[str] | str | None = None,
        batch_size: int = 1,
        include_fingerprint_columns: bool = False,
        include_interaction_events: bool = True,
        include_bitvectors: bool = False,
        include_countvectors: bool = False,
        fingerprint_prefix: str = "ifp__",
        gc_collect: bool = True,
        fail_fast: bool = True,
        ultra_safe: bool = True,
    ) -> PoseInteractionTableResult:
        """
        Compute pose-centric interactions for a pose table.

        :param poses:
            Input pose table with at least receptor, ligand, engine, rank,
            affinity, and molecule columns.
        :type poses: pandas.DataFrame
        :param receptor_pdb_by_id:
            Mapping from receptor id to receptor PDB path.
        :type receptor_pdb_by_id: Mapping[str, str | pathlib.Path]
        :param receptor_col:
            Column containing receptor identifiers.
        :type receptor_col: str
        :param ligand_col:
            Column containing ligand identifiers.
        :type ligand_col: str
        :param engine_col:
            Column containing engine identifiers.
        :type engine_col: str
        :param pose_rank_col:
            Column containing pose rank.
        :type pose_rank_col: str
        :param affinity_col:
            Column containing affinity score.
        :type affinity_col: str
        :param mol_col:
            Column containing RDKit molecules.
        :type mol_col: str
        :param pose_id_col:
            Optional pre-existing pose id column.
        :type pose_id_col: Optional[str]
        :param residues:
            Optional ProLIF residue subset.
        :type residues: Optional[Sequence[str] | str]
        :param batch_size:
            Number of poses to process together when ``ultra_safe`` is ``False``.
        :type batch_size: int
        :param include_fingerprint_columns:
            Retained for API compatibility.
        :type include_fingerprint_columns: bool
        :param include_interaction_events:
            Whether to compute and store raw event payloads.
        :type include_interaction_events: bool
        :param include_bitvectors:
            Whether to collect ProLIF bitvectors aligned to pose order.
        :type include_bitvectors: bool
        :param include_countvectors:
            Whether to collect ProLIF countvectors aligned to pose order.
        :type include_countvectors: bool
        :param fingerprint_prefix:
            Retained for API compatibility.
        :type fingerprint_prefix: str
        :param gc_collect:
            Whether to call garbage collection between batches.
        :type gc_collect: bool
        :param fail_fast:
            Whether to stop immediately on the first failing batch.
        :type fail_fast: bool
        :param ultra_safe:
            Whether to force one-pose-at-a-time processing.
        :type ultra_safe: bool

        :returns:
            Pose-centric interaction result.
        :rtype: PoseInteractionTableResult
        """
        del (
            include_fingerprint_columns,
            fingerprint_prefix,
        )  # intentionally unused in pose-centric design

        required_columns = {
            receptor_col,
            ligand_col,
            engine_col,
            pose_rank_col,
            affinity_col,
            mol_col,
        }
        missing_columns = required_columns - set(poses.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required pose-table columns: {sorted(missing_columns)}"
            )
        if batch_size <= 0:
            batch_size = 1

        effective_batch_size = 1 if ultra_safe else batch_size
        effective_n_jobs = 1 if ultra_safe else self.n_jobs

        logger.info(
            "Starting pose-table interaction run: n_poses=%d n_receptors=%d "
            "effective_batch_size=%d effective_n_jobs=%s ultra_safe=%s fail_fast=%s",
            len(poses),
            poses[receptor_col].nunique() if receptor_col in poses.columns else -1,
            effective_batch_size,
            effective_n_jobs,
            ultra_safe,
            fail_fast,
        )

        working = poses.copy().reset_index(drop=False)
        original_index_name = working.columns[0]
        working = working.rename(columns={original_index_name: "_pose_order"})

        if pose_id_col is None or pose_id_col not in working.columns:
            working["pose_id"] = (
                working[receptor_col].astype(str)
                + "__"
                + working[ligand_col].astype(str)
                + "__"
                + working[engine_col].astype(str)
                + "__pose"
                + working[pose_rank_col].astype(str)
            )
            effective_pose_id_col = "pose_id"
        else:
            effective_pose_id_col = pose_id_col
            working[effective_pose_id_col] = working[effective_pose_id_col].astype(str)
            if effective_pose_id_col != "pose_id":
                working["pose_id"] = working[effective_pose_id_col]

        receptor_paths = {
            str(key): Path(value) for key, value in receptor_pdb_by_id.items()
        }
        merged_blocks: List[pd.DataFrame] = []
        interaction_blocks: List[pd.DataFrame] = []
        summary_blocks: List[pd.DataFrame] = []
        molecule_names: List[str] = []
        bitvectors: List[Any] = []
        countvectors: List[Any] = []
        errors: List[Dict[str, Any]] = []

        for receptor_value, receptor_df in working.groupby(receptor_col, sort=False):
            receptor_key = str(receptor_value)
            if receptor_key not in receptor_paths:
                raise KeyError(
                    f"No receptor PDB path provided for {receptor_col}={receptor_value!r}."
                )

            logger.info(
                "Processing receptor group: receptor=%r n_poses=%d receptor_pdb=%s",
                receptor_value,
                len(receptor_df),
                receptor_paths[receptor_key],
            )

            protein_molecule = load_receptor_molecule(
                receptor_paths[receptor_key],
                selection=self.receptor_selection,
                use_segid=self.receptor_use_segid,
                guess_bonds=self.receptor_guess_bonds,
                vdwradii=self.receptor_vdwradii,
                suppress_mdanalysis_warnings=self.suppress_mdanalysis_warnings,
                suppress_mdanalysis_info_logs=self.suppress_mdanalysis_info_logs,
            )

            receptor_df = receptor_df.reset_index(drop=True)
            for chunk_df in _iter_dataframe_chunks(
                receptor_df, batch_size=effective_batch_size
            ):
                pose_ids = chunk_df["pose_id"].astype(str).tolist()
                merged_chunk = chunk_df.copy()
                if (
                    effective_pose_id_col != "pose_id"
                    and effective_pose_id_col in merged_chunk.columns
                ):
                    merged_chunk = merged_chunk.drop(columns=[effective_pose_id_col])
                merged_blocks.append(merged_chunk)

                fingerprint = None
                names = None
                events_df = pd.DataFrame()
                try:
                    named_ligands = []
                    for pose_id, mol in zip(pose_ids, chunk_df[mol_col].tolist()):
                        if mol is None:
                            raise ValueError(
                                f"Encountered None molecule for pose {pose_id!r}."
                            )
                        named_ligands.append((pose_id, mol))

                    logger.debug(
                        "Preparing ligand chunk: receptor=%r pose_ids=%s",
                        receptor_value,
                        pose_ids,
                    )

                    names, _, prolif_molecules = prepare_ligands(
                        named_ligands,
                        resname=self.ligand_resname,
                        resnumber=self.ligand_resnumber,
                        chain=self.ligand_chain,
                        use_segid=self.ligand_use_segid,
                        sdf_sanitize=self.sdf_sanitize,
                    )
                    molecule_names.extend(names)

                    logger.debug(
                        "Prepared ligand chunk successfully: receptor=%r names=%s",
                        receptor_value,
                        names,
                    )

                    if include_interaction_events:
                        fingerprint = self._build_fingerprint()

                        logger.debug(
                            "Running ProLIF fingerprint: receptor=%r pose_ids=%s residues=%s n_jobs=%s",
                            receptor_value,
                            pose_ids,
                            residues,
                            effective_n_jobs,
                        )

                        fingerprint.run_from_iterable(
                            prolif_molecules,
                            protein_molecule,
                            residues=residues,
                            progress=self.progress,
                            n_jobs=effective_n_jobs,
                        )
                        events_df = flatten_ifp(
                            fingerprint.ifp, mol_names=names, mols=None
                        )
                        if not events_df.empty:
                            events_df = events_df.copy()
                            events_df["pose_id"] = events_df["mol_name"].astype(str)

                        logger.debug(
                            "Flattened interaction events: receptor=%r pose_ids=%s shape=%s",
                            receptor_value,
                            pose_ids,
                            getattr(events_df, "shape", None),
                        )

                    event_rows = build_pose_interaction_table(events_df, pose_ids)
                    summary_rows = build_pose_summary_table(events_df, pose_ids)
                    order_lookup = pd.DataFrame(
                        {
                            "pose_id": pose_ids,
                            "_pose_order": chunk_df["_pose_order"].tolist(),
                        }
                    )
                    event_rows = event_rows.merge(
                        order_lookup, on="pose_id", how="left"
                    )
                    summary_rows = summary_rows.merge(
                        order_lookup, on="pose_id", how="left"
                    )
                    interaction_blocks.append(event_rows)
                    summary_blocks.append(summary_rows)

                    if (
                        include_interaction_events
                        and include_bitvectors
                        and fingerprint is not None
                        and hasattr(fingerprint, "to_bitvectors")
                    ):
                        try:
                            bitvectors.extend(list(fingerprint.to_bitvectors()))
                        except Exception:
                            logger.exception(
                                "Failed collecting bitvectors for receptor=%r pose_ids=%s",
                                receptor_value,
                                pose_ids,
                            )
                    if (
                        include_interaction_events
                        and include_countvectors
                        and fingerprint is not None
                        and hasattr(fingerprint, "to_countvectors")
                    ):
                        try:
                            countvectors.extend(list(fingerprint.to_countvectors()))
                        except Exception:
                            logger.exception(
                                "Failed collecting countvectors for receptor=%r pose_ids=%s",
                                receptor_value,
                                pose_ids,
                            )
                except Exception as exc:
                    chunk_debug_rows = []
                    for pose_id, mol in zip(pose_ids, chunk_df[mol_col].tolist()):
                        mol_debug = {
                            "pose_id": pose_id,
                            "mol_is_none": mol is None,
                            "mol_type": type(mol).__name__ if mol is not None else None,
                        }
                        try:
                            mol_debug["has_conformers"] = (
                                mol.GetNumConformers()
                                if mol is not None and hasattr(mol, "GetNumConformers")
                                else None
                            )
                        except Exception:
                            mol_debug["has_conformers"] = "error"
                        try:
                            mol_debug["num_atoms"] = (
                                mol.GetNumAtoms()
                                if mol is not None and hasattr(mol, "GetNumAtoms")
                                else None
                            )
                        except Exception:
                            mol_debug["num_atoms"] = "error"
                        try:
                            mol_debug["mol_name"] = (
                                mol.GetProp("_Name")
                                if mol is not None
                                and hasattr(mol, "HasProp")
                                and mol.HasProp("_Name")
                                else None
                            )
                        except Exception:
                            mol_debug["mol_name"] = "error"
                        chunk_debug_rows.append(mol_debug)

                    error_record = {
                        receptor_col: receptor_value,
                        "batch_pose_ids": pose_ids,
                        "message": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                        "chunk_debug": chunk_debug_rows,
                    }
                    errors.append(error_record)

                    logger.exception(
                        "Interaction processing failed for receptor=%r poses=%s "
                        "chunk_size=%d effective_n_jobs=%s ultra_safe=%s",
                        receptor_value,
                        pose_ids,
                        len(pose_ids),
                        effective_n_jobs,
                        ultra_safe,
                    )
                    logger.error("Chunk debug info: %s", chunk_debug_rows)

                    if fail_fast:
                        raise InteractionProcessingError(
                            f"Interaction processing failed for receptor {receptor_value!r} "
                            f"and poses {pose_ids}"
                        ) from exc

                    fallback_events = pd.DataFrame(
                        {
                            "pose_id": pose_ids,
                            "interaction_events": [None] * len(pose_ids),
                            "interaction_events_json": [None] * len(pose_ids),
                            "has_interactions": [False] * len(pose_ids),
                            "_pose_order": chunk_df["_pose_order"].tolist(),
                        }
                    )
                    fallback_summary = pd.DataFrame(
                        {
                            "pose_id": pose_ids,
                            "interaction_compact": [None] * len(pose_ids),
                            "interaction_compact_json": [None] * len(pose_ids),
                            "interaction_detail": [None] * len(pose_ids),
                            "interaction_detail_json": [None] * len(pose_ids),
                            "has_interactions": [False] * len(pose_ids),
                            "_pose_order": chunk_df["_pose_order"].tolist(),
                        }
                    )
                    interaction_blocks.append(fallback_events)
                    summary_blocks.append(fallback_summary)
                finally:
                    if gc_collect:
                        gc.collect()

            del protein_molecule
            if gc_collect:
                gc.collect()

        merged_df = (
            pd.concat(merged_blocks, ignore_index=True)
            if merged_blocks
            else pd.DataFrame()
        )
        interaction_df = (
            pd.concat(interaction_blocks, ignore_index=True)
            if interaction_blocks
            else pd.DataFrame(
                columns=[
                    "pose_id",
                    "interaction_events",
                    "interaction_events_json",
                    "has_interactions",
                ]
            )
        )
        summary_df = (
            pd.concat(summary_blocks, ignore_index=True)
            if summary_blocks
            else pd.DataFrame(
                columns=[
                    "pose_id",
                    "interaction_compact",
                    "interaction_compact_json",
                    "interaction_detail",
                    "interaction_detail_json",
                    "has_interactions",
                ]
            )
        )

        if not merged_df.empty:
            merged_df = merged_df.sort_values("_pose_order").reset_index(drop=True)
            merged_df = merged_df.drop(columns=["_pose_order"], errors="ignore")
        if not interaction_df.empty:
            interaction_df = interaction_df.sort_values("_pose_order").reset_index(
                drop=True
            )
            interaction_df = interaction_df.drop(
                columns=["_pose_order"], errors="ignore"
            )
        if not summary_df.empty:
            summary_df = summary_df.sort_values("_pose_order").reset_index(drop=True)
            summary_df = summary_df.drop(columns=["_pose_order"], errors="ignore")

        logger.info(
            "Pose-table interaction run finished: merged_shape=%s interaction_shape=%s "
            "summary_shape=%s n_errors=%d",
            getattr(merged_df, "shape", None),
            getattr(interaction_df, "shape", None),
            getattr(summary_df, "shape", None),
            len(errors),
        )

        return PoseInteractionTableResult(
            merged_df=merged_df,
            interaction_df=interaction_df,
            summary_df=summary_df,
            receptor_pdb_by_id=receptor_paths,
            batch_size=effective_batch_size,
            settings={
                **self.settings_snapshot(),
                "receptor_col": receptor_col,
                "ligand_col": ligand_col,
                "engine_col": engine_col,
                "pose_rank_col": pose_rank_col,
                "affinity_col": affinity_col,
                "mol_col": mol_col,
                "pose_id_col": effective_pose_id_col,
                "include_interaction_events": include_interaction_events,
                "include_bitvectors": include_bitvectors,
                "include_countvectors": include_countvectors,
                "gc_collect": gc_collect,
                "fail_fast": fail_fast,
                "ultra_safe": ultra_safe,
                "effective_n_jobs": effective_n_jobs,
            },
            errors=errors,
            bitvectors=bitvectors or None,
            countvectors=countvectors or None,
            molecule_names=molecule_names,
        )

    @staticmethod
    def _rename_fingerprint_index(
        fingerprint_df: pd.DataFrame, molecule_names: Sequence[str]
    ) -> pd.DataFrame:
        """
        Replace the fingerprint dataframe index with molecule names when lengths match.

        :param fingerprint_df:
            Fingerprint dataframe returned by ProLIF.
        :type fingerprint_df: pandas.DataFrame
        :param molecule_names:
            Molecule names aligned with the fingerprint rows.
        :type molecule_names: Sequence[str]

        :returns:
            Copy of the fingerprint dataframe with renamed index when possible.
        :rtype: pandas.DataFrame
        """
        renamed = fingerprint_df.copy()
        if len(renamed.index) == len(molecule_names):
            renamed.index = list(molecule_names)
            renamed.index.name = "mol_name"
        return renamed


def _iter_dataframe_chunks(
    frame: pd.DataFrame, *, batch_size: int
) -> Iterator[pd.DataFrame]:
    """
    Yield dataframe slices of at most ``batch_size`` rows.

    :param frame:
        Input dataframe to split.
    :type frame: pandas.DataFrame
    :param batch_size:
        Maximum number of rows per yielded chunk.
    :type batch_size: int

    :returns:
        Iterator of dataframe chunks.
    :rtype: Iterator[pandas.DataFrame]
    """
    if batch_size <= 0:
        yield frame
        return
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size].copy()  # noqa


def extract_interactions(
    receptor_pdb: PathLike,
    ligands: PathLike | Any | Sequence[Any] | Iterable[Any] | Mapping[str, Any],
    *,
    interactions: Optional[Sequence[str]] = None,
    parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    count: bool = False,
    vicinity_cutoff: float = 6.0,
    receptor_selection: Optional[str] = None,
    receptor_use_segid: Optional[bool] = None,
    ligand_resname: str = "LIG",
    ligand_resnumber: int = 1,
    ligand_chain: str = "",
    ligand_use_segid: bool = False,
    sdf_sanitize: bool = True,
    receptor_guess_bonds: bool = True,
    receptor_vdwradii: Optional[Mapping[str, float]] = None,
    suppress_mdanalysis_warnings: bool = True,
    suppress_mdanalysis_info_logs: bool = True,
    progress: bool = False,
    n_jobs: Optional[int] = 1,
    residues: Sequence[str] | str | None = None,
    drop_empty: bool = True,
) -> InteractionRunResult:
    """
    Convenience wrapper around :class:`InteractionProfiler` for single-run extraction.

    :returns:
        Structured single-run interaction result.
    :rtype: InteractionRunResult
    """
    profiler = InteractionProfiler(
        interactions=interactions,
        parameters=parameters,
        count=count,
        vicinity_cutoff=vicinity_cutoff,
        receptor_selection=receptor_selection,
        receptor_use_segid=receptor_use_segid,
        ligand_resname=ligand_resname,
        ligand_resnumber=ligand_resnumber,
        ligand_chain=ligand_chain,
        ligand_use_segid=ligand_use_segid,
        sdf_sanitize=sdf_sanitize,
        receptor_guess_bonds=receptor_guess_bonds,
        receptor_vdwradii=receptor_vdwradii,
        suppress_mdanalysis_warnings=suppress_mdanalysis_warnings,
        suppress_mdanalysis_info_logs=suppress_mdanalysis_info_logs,
        progress=progress,
        n_jobs=n_jobs,
        drop_empty=drop_empty,
    )
    return profiler.run(receptor_pdb=receptor_pdb, ligands=ligands, residues=residues)


def extract_pose_table_interactions(
    poses: pd.DataFrame,
    receptor_pdb_by_id: Mapping[str, PathLike],
    *,
    interactions: Optional[Sequence[str]] = None,
    parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    count: bool = False,
    vicinity_cutoff: float = 6.0,
    receptor_selection: Optional[str] = None,
    receptor_use_segid: Optional[bool] = None,
    ligand_resname: str = "LIG",
    ligand_resnumber: int = 1,
    ligand_chain: str = "",
    ligand_use_segid: bool = False,
    sdf_sanitize: bool = True,
    receptor_guess_bonds: bool = True,
    receptor_vdwradii: Optional[Mapping[str, float]] = None,
    suppress_mdanalysis_warnings: bool = True,
    suppress_mdanalysis_info_logs: bool = True,
    progress: bool = False,
    n_jobs: Optional[int] = 1,
    receptor_col: str = "receptor_id",
    ligand_col: str = "ligand_id",
    engine_col: str = "engine",
    pose_rank_col: str = "pose_rank",
    affinity_col: str = "affinity",
    mol_col: str = "mol",
    pose_id_col: str | None = None,
    residues: Sequence[str] | str | None = None,
    batch_size: int = 1,
    include_fingerprint_columns: bool = False,
    include_interaction_events: bool = True,
    include_bitvectors: bool = False,
    include_countvectors: bool = False,
    fingerprint_prefix: str = "ifp__",
    gc_collect: bool = True,
    fail_fast: bool = True,
    ultra_safe: bool = True,
    drop_empty: bool = True,
) -> PoseInteractionTableResult:
    """
    Convenience wrapper for automated pose-table interaction extraction.

    :returns:
        Pose-centric interaction result containing ``merged_df``,
        ``interaction_df``, and ``summary_df``.
    :rtype: PoseInteractionTableResult
    """
    profiler = InteractionProfiler(
        interactions=interactions,
        parameters=parameters,
        count=count,
        vicinity_cutoff=vicinity_cutoff,
        receptor_selection=receptor_selection,
        receptor_use_segid=receptor_use_segid,
        ligand_resname=ligand_resname,
        ligand_resnumber=ligand_resnumber,
        ligand_chain=ligand_chain,
        ligand_use_segid=ligand_use_segid,
        sdf_sanitize=sdf_sanitize,
        receptor_guess_bonds=receptor_guess_bonds,
        receptor_vdwradii=receptor_vdwradii,
        suppress_mdanalysis_warnings=suppress_mdanalysis_warnings,
        suppress_mdanalysis_info_logs=suppress_mdanalysis_info_logs,
        progress=progress,
        n_jobs=n_jobs,
        drop_empty=drop_empty,
    )
    return profiler.run_pose_table(
        poses=poses,
        receptor_pdb_by_id=receptor_pdb_by_id,
        receptor_col=receptor_col,
        ligand_col=ligand_col,
        engine_col=engine_col,
        pose_rank_col=pose_rank_col,
        affinity_col=affinity_col,
        mol_col=mol_col,
        pose_id_col=pose_id_col,
        residues=residues,
        batch_size=batch_size,
        include_fingerprint_columns=include_fingerprint_columns,
        include_interaction_events=include_interaction_events,
        include_bitvectors=include_bitvectors,
        include_countvectors=include_countvectors,
        fingerprint_prefix=fingerprint_prefix,
        gc_collect=gc_collect,
        fail_fast=fail_fast,
        ultra_safe=ultra_safe,
    )
