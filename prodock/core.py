from __future__ import annotations

"""
High-level end-to-end automation pipeline for ProDock.

This module provides a single orchestration layer that can:

1. prepare receptors from raw PDB specifications or validate prebuilt receptors
2. prepare ligands from SMILES or reuse an existing ligand directory
3. build and save a docking campaign JSON file
4. run batch docking
5. crawl generated poses from the project directory
6. optionally extract protein-ligand interactions
7. optionally create and populate a project-local SQLite database

The default database location is::

    <project_dir>/prodock.db

The default workflow is therefore fully project-local and reproducible.

Example
-------
.. code-block:: python

    from prodock import prodock

    PROJECT = "Data/testcase/Multi"

    RECEPTORS = [
        {
            "pdb_id": "4WKQ",
            "receptor_name": "EGFR_4WKQ",
            "ligand_code": "IRE",
            "chains": ["A"],
            "cofactors": [],
        },
    ]

    LIGANDS = [
        {
            "id": "erlotinib",
            "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C",
        },
        {
            "id": "gefitinib",
            "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F",
        },
    ]

    result = prodock(
        PROJECT,
        receptors=RECEPTORS,
        ligands=LIGANDS,
        engines=["smina"],
        extract_interaction=True,
        db_name="prodock.db",
    )

    print(result.campaign_json)
    print(result.db_path)
    print(result.pose_df.head())
    print(result.merged_df.head())
"""

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from prodock.database import PoseDatabase
from prodock.dock import BatchConfig, BatchDock
from prodock.dock.campaign import Campaign
from prodock.io.logging import get_logger, setup_logging
from prodock.postprocess.interaction.core import (
    InteractionProfiler,
    extract_pose_table_interactions,
)
from prodock.postprocess.pose import PoseCrawler
from prodock.preprocess import LigandPrep, ReceptorPrep
from prodock.preprocess.gridbox import GridBox
from prodock.structure import PDBQuery

PathLike = Union[str, Path]
Vec3 = Tuple[float, float, float]

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedReceptorSpec:
    """
    Fully prepared receptor definition ready for campaign construction.

    :param receptor_id:
        Unique receptor identifier used inside the docking campaign.
    :type receptor_id: str
    :param receptor_pdbqt:
        Path to the final receptor ``.pdbqt`` file.
    :type receptor_pdbqt: Path
    :param center:
        Docking box center as ``(x, y, z)``.
    :type center: Vec3
    :param size:
        Docking box size as ``(sx, sy, sz)``.
    :type size: Vec3
    """

    receptor_id: str
    receptor_pdbqt: Path
    center: Vec3
    size: Vec3


@dataclass
class ProDockResult:
    """
    Structured result bundle returned by :class:`ProDockPipeline`.

    :param project_dir:
        Root working directory of the project.
    :type project_dir: Path
    :param ligand_dir:
        Directory containing final ligand files used by the campaign.
    :type ligand_dir: Path
    :param campaign_json:
        Path to the generated campaign JSON file.
    :type campaign_json: Path
    :param receptors:
        Prepared receptor specifications included in the campaign.
    :type receptors: List[PreparedReceptorSpec]
    :param receptor_pdb_by_id:
        Mapping from receptor identifier to receptor ``.pdb`` file used for
        interaction analysis.
    :type receptor_pdb_by_id: Dict[str, Path]
    :param campaign:
        In-memory campaign object.
    :type campaign: Campaign
    :param docking_results:
        Raw docking results returned by :class:`prodock.dock.BatchDock`.
    :type docking_results: Any
    :param pose_df:
        Pose table collected by :class:`prodock.postprocess.pose.PoseCrawler`.
    :type pose_df: pandas.DataFrame
    :param interaction_result:
        Raw interaction extraction result object, or ``None`` if interaction
        extraction was skipped.
    :type interaction_result: Any
    :param merged_df:
        Final dataframe chosen for downstream insertion into the database. This
        is the interaction-merged dataframe if interaction extraction is
        enabled; otherwise it is the crawled pose dataframe.
    :type merged_df: pandas.DataFrame
    :param interaction_df:
        Long-form interaction-event dataframe, or ``None``.
    :type interaction_df: Optional[pandas.DataFrame]
    :param summary_df:
        Pose-level interaction summary dataframe, or ``None``.
    :type summary_df: Optional[pandas.DataFrame]
    :param compact_interactions:
        Compact per-pose interaction dictionary, or ``None``.
    :type compact_interactions: Optional[Dict[str, Any]]
    :param db_path:
        SQLite database path if database writing was enabled.
    :type db_path: Optional[Path]
    """

    project_dir: Path
    ligand_dir: Path
    campaign_json: Path
    receptors: List[PreparedReceptorSpec]
    receptor_pdb_by_id: Dict[str, Path]
    campaign: Campaign
    docking_results: Any
    pose_df: pd.DataFrame
    interaction_result: Any
    merged_df: pd.DataFrame
    interaction_df: Optional[pd.DataFrame]
    summary_df: Optional[pd.DataFrame]
    compact_interactions: Optional[Dict[str, Any]]
    db_path: Optional[Path]


class ProDockPipeline:
    """
    High-level orchestration helper for ProDock projects.

    This class provides a single automation entry point for common workflows:

    1. raw receptor records + ligand SMILES records
    2. prebuilt receptor ``.pdbqt`` files + explicit docking box coordinates
    3. optional pose crawling after docking
    4. optional interaction extraction
    5. optional database creation and insertion

    All generated data are organized under ``project_dir``.

    :param project_dir:
        Root directory used for all generated project files.
    :type project_dir: PathLike
    :param engines:
        Docking engines to include in the campaign. Default is
        ``["smina", "qvina"]``.
    :type engines: Optional[Sequence[str]]
    :param cpu:
        Per-engine CPU setting stored in the campaign.
    :type cpu: int
    :param seed:
        Random seed stored in the campaign.
    :type seed: int
    :param exhaustiveness:
        Exhaustiveness stored in the campaign.
    :type exhaustiveness: int
    :param n_poses:
        Number of poses stored in the campaign.
    :type n_poses: int
    :param n_jobs:
        Number of parallel jobs used by :class:`BatchDock`. If ``None``, this
        defaults to ``cpu``.
    :type n_jobs: Optional[int]
    :param progress:
        Whether to enable progress reporting in :class:`BatchDock`.
    :type progress: bool
    :param receptor_use_meeko:
        Whether receptor preparation should use Meeko.
    :type receptor_use_meeko: bool
    :param ligand_output_format:
        Final ligand output format.
    :type ligand_output_format: str
    :param ligand_backend:
        Ligand conversion backend used by :class:`LigandPrep`.
    :type ligand_backend: str
    :param box_algorithm:
        Ligand-derived box algorithm. ``None`` selects the new default
        ``"pad"`` unless ``box_scale`` is supplied explicitly, in which case
        legacy ``"scale"`` behavior is preserved.
    :type box_algorithm: Optional[str]
    :param box_pad:
        Symmetric padding in Angstrom used by the ``"pad"`` algorithm.
    :type box_pad: float
    :param box_scale:
        Scale factor used by the ``"scale"`` algorithm. Supplying this without
        ``box_algorithm`` selects legacy scale behavior.
    :type box_scale: Optional[float]
    :param box_isotropic:
        Whether ligand-derived boxes should be isotropic.
    :type box_isotropic: bool
    :param campaign_name:
        Default campaign JSON file name.
    :type campaign_name: str

    Example
    -------
    .. code-block:: python

        pipeline = ProDockPipeline("Data/testcase/Multi")
        result = pipeline.run(
            receptors=RECEPTORS,
            ligands=LIGANDS,
            extract_interaction=True,
        )
        print(result.campaign_json)
        print(result.db_path)

    Example
    -------
    .. code-block:: python

        pipeline = ProDockPipeline("Data/testcase/Multi")
        result = pipeline.run(
            prepared_receptors=[
                {
                    "receptor_id": "4WKQ",
                    "receptor_pdbqt": "Data/testcase/Multi/4WKQ/filtered_protein/4WKQ.pdbqt",
                    "center": (2.865, 193.257, 21.367),
                    "size": (27.091, 27.091, 27.091),
                }
            ],
            ligand_dir="Data/testcase/Multi/ligands",
            extract_interaction=True,
        )
        print(result.receptor_pdb_by_id)
    """

    def __init__(
        self,
        project_dir: PathLike,
        *,
        engines: Optional[Sequence[str]] = None,
        cpu: int = 4,
        seed: int = 42,
        exhaustiveness: int = 8,
        n_poses: int = 10,
        n_jobs: Optional[int] = None,
        progress: bool = True,
        receptor_use_meeko: bool = False,
        ligand_output_format: str = "pdbqt",
        ligand_backend: str = "meeko",
        box_algorithm: Optional[str] = None,
        box_pad: float = 4.0,
        box_scale: Optional[float] = None,
        box_isotropic: bool = True,
        campaign_name: str = "campaign.json",
        log_file: str = "prodock.log",
        log_level: Union[str, int] = "INFO",
        log_colored: bool = True,
        log_json: bool = False,
    ) -> None:
        """
        Initialize a ProDock pipeline.

        :param project_dir:
            Root directory used for all generated files.
        :type project_dir: PathLike
        :param engines:
            Docking engines to include in the campaign. Default is
            ``["smina", "qvina"]``.
        :type engines: Optional[Sequence[str]]
        :param cpu:
            Per-engine CPU setting stored in the campaign.
        :type cpu: int
        :param seed:
            Random seed stored in the campaign.
        :type seed: int
        :param exhaustiveness:
            Exhaustiveness stored in the campaign.
        :type exhaustiveness: int
        :param n_poses:
            Number of poses stored in the campaign.
        :type n_poses: int
        :param n_jobs:
            Number of parallel jobs used by :class:`BatchDock`. If ``None``,
            this defaults to ``cpu``.
        :type n_jobs: Optional[int]
        :param progress:
            Whether to enable progress reporting in :class:`BatchDock`.
        :type progress: bool
        :param receptor_use_meeko:
            Whether receptor preparation should use Meeko.
        :type receptor_use_meeko: bool
        :param ligand_output_format:
            Final ligand output format.
        :type ligand_output_format: str
        :param ligand_backend:
            Ligand conversion backend used by :class:`LigandPrep`.
        :type ligand_backend: str
        :param box_algorithm:
            Ligand-derived box algorithm. ``None`` selects ``"pad"`` unless
            ``box_scale`` is supplied explicitly.
        :type box_algorithm: Optional[str]
        :param box_pad:
            Symmetric padding in Angstrom used by the ``"pad"`` algorithm.
        :type box_pad: float
        :param box_scale:
            Scale factor used by the ``"scale"`` algorithm. Supplying this
            without ``box_algorithm`` preserves the legacy scale behavior.
        :type box_scale: Optional[float]
        :param box_isotropic:
            Whether ligand-derived boxes should be isotropic.
        :type box_isotropic: bool
        :param campaign_name:
            Default campaign JSON file name.
        :type campaign_name: str
        """
        self.project_dir = Path(project_dir).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(
            log_dir=self.project_dir,
            log_file=log_file,
            level=log_level,
            colored=log_colored,
            json=log_json,
        )
        self.logger = get_logger(__name__)

        self.engines = list(engines or ["smina", "qvina"])
        self.cpu = cpu
        self.seed = seed
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.n_jobs = cpu if n_jobs is None else n_jobs
        self.progress = progress

        self.receptor_use_meeko = receptor_use_meeko
        self.ligand_output_format = ligand_output_format
        self.ligand_backend = ligand_backend

        if box_algorithm is None:
            resolved_box_algorithm = "scale" if box_scale is not None else "pad"
        else:
            resolved_box_algorithm = str(box_algorithm).strip().lower()
        if resolved_box_algorithm not in {"pad", "scale"}:
            raise ValueError(
                "box_algorithm must be one of {'pad', 'scale'}, "
                f"got {box_algorithm!r}."
            )
        if float(box_pad) < 0.0:
            raise ValueError("box_pad must be non-negative.")
        resolved_box_scale = 2.0 if box_scale is None else float(box_scale)
        if resolved_box_scale <= 0.0:
            raise ValueError("box_scale must be positive.")

        self.box_algorithm = resolved_box_algorithm
        self.box_pad = float(box_pad)
        self.box_scale = resolved_box_scale
        self.box_isotropic = box_isotropic
        self.campaign_name = campaign_name

        self.logger.info(
            "Initialized ProDockPipeline | project_dir=%s | engines=%s | cpu=%s | "
            "n_jobs=%s | seed=%s | exhaustiveness=%s | n_poses=%s | "
            "receptor_use_meeko=%s | ligand_output_format=%s | ligand_backend=%s | "
            "box_algorithm=%s | box_pad=%s | box_scale=%s | "
            "box_isotropic=%s | campaign_name=%s",
            self.project_dir,
            self.engines,
            self.cpu,
            self.n_jobs,
            self.seed,
            self.exhaustiveness,
            self.n_poses,
            self.receptor_use_meeko,
            self.ligand_output_format,
            self.ligand_backend,
            self.box_algorithm,
            self.box_pad,
            self.box_scale,
            self.box_isotropic,
            self.campaign_name,
        )

    @staticmethod
    def _as_vec3(value: Sequence[float], *, field_name: str) -> Vec3:
        """
        Normalize a 3D vector-like sequence to a fixed tuple.

        :param value:
            Sequence containing exactly three numeric values.
        :type value: Sequence[float]
        :param field_name:
            Field name used in error messages.
        :type field_name: str

        :returns:
            Normalized 3-tuple.
        :rtype: Vec3
        """
        if len(value) != 3:
            raise ValueError(f"{field_name!r} must contain exactly 3 values.")
        return (float(value[0]), float(value[1]), float(value[2]))

    def _resolve_path(self, path: PathLike) -> Path:
        """
        Resolve a possibly relative path.

        Resolution strategy:

        1. absolute path stays absolute
        2. existing relative path is resolved from the current working directory
        3. otherwise resolve relative to ``project_dir``

        :param path:
            Input path to normalize.
        :type path: PathLike

        :returns:
            Resolved absolute path.
        :rtype: Path
        """
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
            self.logger.debug("Resolved absolute path: %s -> %s", path, resolved)
            return resolved
        if p.exists():
            resolved = p.resolve()
            self.logger.debug(
                "Resolved existing relative path from cwd: %s -> %s", path, resolved
            )
            return resolved
        resolved = (self.project_dir / p).resolve()
        self.logger.debug("Resolved project-relative path: %s -> %s", path, resolved)
        return resolved

    def _resolve_db_path(self, db_name: PathLike) -> Path:
        """
        Resolve the SQLite database path.

        Relative database paths are interpreted relative to ``project_dir``.

        :param db_name:
            Database filename or path.
        :type db_name: PathLike

        :returns:
            Absolute database path.
        :rtype: Path
        """
        p = Path(db_name)
        if p.is_absolute():
            resolved = p.resolve()
            self.logger.debug("Resolved absolute db path: %s -> %s", db_name, resolved)
            return resolved
        resolved = (self.project_dir / p).resolve()
        self.logger.debug(
            "Resolved project-relative db path: %s -> %s", db_name, resolved
        )
        return resolved

    @staticmethod
    def _infer_receptor_id_from_path(path: Path) -> str:
        """
        Infer a receptor ID from a receptor file path.

        :param path:
            Receptor file path.
        :type path: Path

        :returns:
            Inferred receptor identifier.
        :rtype: str
        """
        return path.stem

    def _default_reference_ligand_path(self, record: Dict[str, Any]) -> Path:
        """
        Build the default co-crystal ligand path for a raw receptor record.

        The expected location is::

            <project_dir>/<pdb_id>/reference_ligand/<ligand_code>.sdf

        :param record:
            Raw receptor input record.
        :type record: Dict[str, Any]

        :returns:
            Expected reference ligand path.
        :rtype: Path
        """
        pdb_id = str(record["pdb_id"])
        ligand_code = str(record["ligand_code"])
        ref_path = self.project_dir / pdb_id / "reference_ligand" / f"{ligand_code}.sdf"
        self.logger.debug(
            "Default reference ligand path for receptor %s: %s",
            pdb_id,
            ref_path,
        )
        return ref_path

    def _box_from_record(self, record: Dict[str, Any]) -> Tuple[Vec3, Vec3]:
        """
        Derive docking box information from a receptor record.

        Supported modes are:

        - explicit ``center`` and ``size``
        - automatic box generation from a reference ligand file

        For automatic mode, the method first checks ``reference_ligand`` in the
        input record. If absent, it falls back to the default co-crystal ligand
        location inferred from ``pdb_id`` and ``ligand_code``.

        :param record:
            Raw receptor input record.
        :type record: Dict[str, Any]

        :returns:
            Pair ``(center, size)``.
        :rtype: Tuple[Vec3, Vec3]

        :raises ValueError:
            Raised if neither explicit box coordinates nor a usable reference
            ligand is available.
        :raises FileNotFoundError:
            Raised if the reference ligand path does not exist.

        Example
        -------
        .. code-block:: python

            center, size = pipeline._box_from_record(
                {
                    "pdb_id": "4WKQ",
                    "ligand_code": "IRE",
                }
            )
        """
        pdb_id = str(record.get("pdb_id", "<unknown>"))

        if record.get("center") is not None and record.get("size") is not None:
            center = self._as_vec3(record["center"], field_name="center")
            size = self._as_vec3(record["size"], field_name="size")
            self.logger.info(
                "Using explicit box for receptor %s | center=%s | size=%s",
                pdb_id,
                center,
                size,
            )
            return center, size

        ref_path = record.get("reference_ligand")
        if ref_path is None:
            if "ligand_code" not in record:
                raise ValueError(
                    "Raw receptor record must provide either "
                    "('center' and 'size') or a co-crystal reference ligand "
                    "via 'ligand_code' or 'reference_ligand'."
                )
            ref_path = self._default_reference_ligand_path(record)

        ref_path = self._resolve_path(ref_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference ligand file not found: {ref_path}")

        fmt = str(record.get("reference_ligand_format", ref_path.suffix.lstrip(".")))
        if not fmt:
            fmt = "sdf"

        algorithm = str(record.get("box_algorithm", self.box_algorithm)).lower()
        isotropic = bool(record.get("box_isotropic", self.box_isotropic))
        gb = GridBox().load_ligand(str(ref_path), fmt=fmt)

        if algorithm == "pad":
            pad = float(record.get("box_pad", self.box_pad))
            if pad < 0.0:
                raise ValueError("box_pad must be non-negative.")
            self.logger.info(
                "Deriving docking box from reference ligand for receptor %s | "
                "path=%s | fmt=%s | algorithm=pad | pad=%s | isotropic=%s",
                pdb_id,
                ref_path,
                fmt,
                pad,
                isotropic,
            )
            gb.from_ligand_pad(pad=pad, isotropic=isotropic)
        elif algorithm == "scale":
            scale = float(record.get("box_scale", self.box_scale))
            if scale <= 0.0:
                raise ValueError("box_scale must be positive.")
            self.logger.info(
                "Deriving docking box from reference ligand for receptor %s | "
                "path=%s | fmt=%s | algorithm=scale | scale=%s | isotropic=%s",
                pdb_id,
                ref_path,
                fmt,
                scale,
                isotropic,
            )
            gb.from_ligand_scale(scale=scale, isotropic=isotropic)
        else:
            raise ValueError(
                "box_algorithm must be one of {'pad', 'scale'}, "
                f"got {algorithm!r} for receptor {pdb_id!r}."
            )
        center = self._as_vec3(gb.center, field_name="center")
        size = self._as_vec3(gb.size, field_name="size")

        self.logger.info(
            "Computed box for receptor %s | center=%s | size=%s",
            pdb_id,
            center,
            size,
        )
        return center, size

    def _build_receptor_pdb_map(
        self,
        receptor_specs: Sequence[PreparedReceptorSpec],
    ) -> Dict[str, Path]:
        """
        Build the receptor ``.pdb`` mapping needed for interaction extraction.

        The mapping is derived directly from the prepared receptor path by
        replacing the suffix with ``.pdb``.

        :param receptor_specs:
            Prepared receptor specifications.
        :type receptor_specs: Sequence[PreparedReceptorSpec]

        :returns:
            Mapping from receptor ID to receptor ``.pdb`` path.
        :rtype: Dict[str, Path]

        :raises FileNotFoundError:
            Raised if the inferred receptor ``.pdb`` file does not exist.

        Example
        -------
        .. code-block:: python

            receptor_pdb_by_id = pipeline._build_receptor_pdb_map(receptor_specs)
            print(receptor_pdb_by_id["4WKQ"])
        """
        receptor_pdb_by_id: Dict[str, Path] = {}

        self.logger.info(
            "Building receptor PDB map for %d prepared receptors",
            len(receptor_specs),
        )

        for spec in receptor_specs:
            pdb_path = spec.receptor_pdbqt.with_suffix(".pdb")
            self.logger.debug(
                "Inferring receptor PDB path | receptor_id=%s | pdbqt=%s | pdb=%s",
                spec.receptor_id,
                spec.receptor_pdbqt,
                pdb_path,
            )
            if not pdb_path.exists():
                raise FileNotFoundError(
                    "Could not infer receptor PDB file from prepared receptor "
                    f"path: {pdb_path}"
                )
            receptor_pdb_by_id[spec.receptor_id] = pdb_path.resolve()

        self.logger.info(
            "Built receptor PDB map with %d entries", len(receptor_pdb_by_id)
        )
        return receptor_pdb_by_id

    def prepare_receptors(
        self,
        *,
        receptors: Optional[List[Dict[str, Any]]] = None,
        prepared_receptors: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PreparedReceptorSpec]:
        """
        Prepare receptor inputs for campaign construction.

        Exactly one receptor input mode must be supplied:

        - ``receptors`` for raw PDB-queryable receptors
        - ``prepared_receptors`` for already prepared receptor ``.pdbqt`` files

        Raw mode will:

        1. call :meth:`PDBQuery.process_batch`
        2. prepare receptor ``.pdb`` to ``.pdbqt``
        3. derive box coordinates from explicit box values or a reference ligand

        Prepared mode will only validate and normalize the provided paths and
        box definitions.

        :param receptors:
            Raw receptor records, typically compatible with
            :meth:`PDBQuery.process_batch`.
        :type receptors: Optional[List[Dict[str, Any]]]
        :param prepared_receptors:
            List of prebuilt receptor records. Each record must contain at least
            ``receptor_pdbqt`` (or ``receptor``), ``center``, and ``size``.
        :type prepared_receptors: Optional[List[Dict[str, Any]]]

        :returns:
            Normalized prepared receptor specifications.
        :rtype: List[PreparedReceptorSpec]

        :raises ValueError:
            Raised if both receptor modes are provided or if neither is
            provided.
        :raises FileNotFoundError:
            Raised if an expected receptor or reference-ligand file is missing.

        Example
        -------
        .. code-block:: python

            receptor_specs = pipeline.prepare_receptors(
                receptors=RECEPTORS
            )

        Example
        -------
        .. code-block:: python

            receptor_specs = pipeline.prepare_receptors(
                prepared_receptors=[
                    {
                        "receptor_id": "4WKQ",
                        "receptor_pdbqt": "Data/testcase/Multi/4WKQ/filtered_protein/4WKQ.pdbqt",
                        "center": (2.865, 193.257, 21.367),
                        "size": (27.091, 27.091, 27.091),
                    }
                ]
            )
        """
        has_raw = receptors is not None
        has_prepared = prepared_receptors is not None

        self.logger.info(
            "Preparing receptors | raw_mode=%s | prepared_mode=%s",
            has_raw,
            has_prepared,
        )

        if has_raw == has_prepared:
            raise ValueError(
                "Provide exactly one of 'receptors' or 'prepared_receptors'."
            )

        if prepared_receptors is not None:
            self.logger.info(
                "Using prepared receptor mode with %d receptor records",
                len(prepared_receptors),
            )
            specs: List[PreparedReceptorSpec] = []
            for idx, item in enumerate(prepared_receptors, start=1):
                receptor_path = item.get("receptor_pdbqt", item.get("receptor"))
                if receptor_path is None:
                    raise ValueError(
                        "Each prepared receptor record must contain "
                        "'receptor_pdbqt' or 'receptor'."
                    )

                receptor_path = self._resolve_path(receptor_path)
                receptor_id = str(
                    item.get("receptor_id")
                    or item.get("id")
                    or self._infer_receptor_id_from_path(receptor_path)
                )
                center = self._as_vec3(item["center"], field_name="center")
                size = self._as_vec3(item["size"], field_name="size")

                self.logger.debug(
                    "Prepared receptor record %d | receptor_id=%s | path=%s | center=%s | size=%s",
                    idx,
                    receptor_id,
                    receptor_path,
                    center,
                    size,
                )

                if not receptor_path.exists():
                    raise FileNotFoundError(
                        f"Prepared receptor file not found: {receptor_path}"
                    )

                specs.append(
                    PreparedReceptorSpec(
                        receptor_id=receptor_id,
                        receptor_pdbqt=receptor_path,
                        center=center,
                        size=size,
                    )
                )

            self.logger.info(
                "Prepared receptor mode complete | receptors=%d", len(specs)
            )
            return specs

        assert receptors is not None

        self.logger.info(
            "Using raw receptor mode with %d receptor records", len(receptors)
        )
        PDBQuery.process_batch(receptors, output_dir=str(self.project_dir))
        self.logger.info(
            "PDBQuery.process_batch completed | output_dir=%s", self.project_dir
        )

        specs: List[PreparedReceptorSpec] = []
        for idx, record in enumerate(receptors, start=1):
            pdb_id = str(record["pdb_id"])
            filtered_dir = self.project_dir / pdb_id / "filtered_protein"
            input_pdb = filtered_dir / f"{pdb_id}.pdb"
            output_pdbqt = filtered_dir / f"{pdb_id}.pdbqt"

            self.logger.info(
                "Preparing receptor %d/%d | pdb_id=%s | input_pdb=%s | output_pdbqt=%s",
                idx,
                len(receptors),
                pdb_id,
                input_pdb,
                output_pdbqt,
            )

            if not input_pdb.exists():
                raise FileNotFoundError(
                    f"Expected filtered receptor PDB not found: {input_pdb}"
                )

            rep = ReceptorPrep(use_meeko=self.receptor_use_meeko)
            rep.prep(
                input_pdb=str(input_pdb),
                output_dir=str(filtered_dir),
                out_fmt="pdbqt",
                add_prep_suffix=False,
            )

            self.logger.debug(
                "Receptor preparation finished | pdb_id=%s | output_dir=%s",
                pdb_id,
                filtered_dir,
            )

            center, size = self._box_from_record(record)

            specs.append(
                PreparedReceptorSpec(
                    receptor_id=pdb_id,
                    receptor_pdbqt=output_pdbqt.resolve(),
                    center=center,
                    size=size,
                )
            )

        self.logger.info("Raw receptor preparation complete | receptors=%d", len(specs))
        return specs

    def prepare_ligands(
        self,
        *,
        ligands: Optional[List[Dict[str, str]]] = None,
        ligand_dir: Optional[PathLike] = None,
    ) -> Path:
        """
        Prepare or resolve ligand inputs.

        Exactly one ligand input mode must be supplied:

        - ``ligands`` as a list of SMILES records
        - ``ligand_dir`` as an existing directory of prepared ligands

        When ``ligands`` is used, the output directory defaults to
        ``<project_dir>/ligands``.

        :param ligands:
            Ligand records, typically containing ``id`` and ``smiles``.
        :type ligands: Optional[List[Dict[str, str]]]
        :param ligand_dir:
            Existing directory containing prepared ligand files.
        :type ligand_dir: Optional[PathLike]

        :returns:
            Absolute path to the ligand directory used by the campaign.
        :rtype: Path

        :raises ValueError:
            Raised if both ligand modes are provided or if neither is provided.
        :raises FileNotFoundError:
            Raised if the provided ligand directory does not exist.
        :raises NotADirectoryError:
            Raised if the provided ligand path is not a directory.

        Example
        -------
        .. code-block:: python

            ligand_dir = pipeline.prepare_ligands(
                ligands=LIGANDS
            )

        Example
        -------
        .. code-block:: python

            ligand_dir = pipeline.prepare_ligands(
                ligand_dir="Data/testcase/Multi/ligands"
            )
        """
        has_ligands = ligands is not None
        has_ligand_dir = ligand_dir is not None

        self.logger.info(
            "Preparing ligands | smiles_mode=%s | directory_mode=%s",
            has_ligands,
            has_ligand_dir,
        )

        if has_ligands == has_ligand_dir:
            raise ValueError("Provide exactly one of 'ligands' or 'ligand_dir'.")

        if ligand_dir is not None:
            resolved = self._resolve_path(ligand_dir)
            self.logger.info("Using existing ligand directory: %s", resolved)
            if not resolved.exists():
                raise FileNotFoundError(f"Ligand directory not found: {resolved}")
            if not resolved.is_dir():
                raise NotADirectoryError(f"Not a directory: {resolved}")
            return resolved

        assert ligands is not None

        out_dir = self.project_dir / "ligands"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Preparing %d ligands from SMILES into %s | output_format=%s | backend=%s | seed=%s",
            len(ligands),
            out_dir,
            self.ligand_output_format,
            self.ligand_backend,
            self.seed,
        )

        (
            LigandPrep(
                output_dir=str(out_dir),
                smiles_key="smiles",
                name_key="id",
            )
            .set_conformer_seed(self.seed)
            .set_output_format(self.ligand_output_format)
            .set_converter_backend(self.ligand_backend)
            .from_list_of_dicts(ligands)
            .process_all()
        )

        resolved = out_dir.resolve()
        self.logger.info("Ligand preparation complete | ligand_dir=%s", resolved)
        return resolved

    def build_campaign(
        self,
        *,
        receptor_specs: Sequence[PreparedReceptorSpec],
        ligand_dir: PathLike,
    ) -> Campaign:
        """
        Build a :class:`Campaign` from prepared receptors and ligands.

        :param receptor_specs:
            Prepared receptor specifications.
        :type receptor_specs: Sequence[PreparedReceptorSpec]
        :param ligand_dir:
            Directory containing final ligand files.
        :type ligand_dir: PathLike

        :returns:
            Campaign object ready to be serialized or executed.
        :rtype: Campaign

        Example
        -------
        .. code-block:: python

            campaign = pipeline.build_campaign(
                receptor_specs=receptor_specs,
                ligand_dir=ligand_dir,
            )
            print(campaign)
        """
        ligand_dir = self._resolve_path(ligand_dir)

        pdb_ids = [spec.receptor_id for spec in receptor_specs]
        receptors = [str(spec.receptor_pdbqt) for spec in receptor_specs]
        boxes = [(spec.center, spec.size) for spec in receptor_specs]

        self.logger.info(
            "Building campaign | project_dir=%s | receptors=%d | ligand_dir=%s | engines=%s",
            self.project_dir,
            len(receptor_specs),
            ligand_dir,
            self.engines,
        )
        self.logger.debug("Campaign receptor_ids=%s", pdb_ids)

        campaign = Campaign.from_shared_ligand_dir(
            working_dir=str(self.project_dir),
            pdb_ids=pdb_ids,
            receptors=receptors,
            boxes=boxes,
            engines=self.engines,
            ligand_dir=str(ligand_dir),
            cpu=self.cpu,
            seed=self.seed,
            exhaustiveness=self.exhaustiveness,
            n_poses=self.n_poses,
        )

        self.logger.info("Campaign construction complete")
        return campaign

    def save_campaign(
        self,
        campaign: Campaign,
        *,
        campaign_name: Optional[str] = None,
    ) -> Path:
        """
        Save a campaign JSON file under the project directory.

        :param campaign:
            Campaign instance to serialize.
        :type campaign: Campaign
        :param campaign_name:
            Output JSON file name. If ``None``, the pipeline default is used.
        :type campaign_name: Optional[str]

        :returns:
            Path to the written JSON file.
        :rtype: Path

        Example
        -------
        .. code-block:: python

            campaign_json = pipeline.save_campaign(campaign)
            print(campaign_json)
        """
        name = campaign_name or self.campaign_name
        out_path = self.project_dir / name
        self.logger.info("Saving campaign JSON to %s", out_path)
        campaign.save_json(str(out_path))
        resolved = out_path.resolve()
        self.logger.info("Campaign JSON written: %s", resolved)
        return resolved

    def crawl_poses(self, *, backend: str = "auto") -> pd.DataFrame:
        """
        Crawl docked poses from the project directory.

        This wraps :class:`prodock.postprocess.pose.PoseCrawler` using the
        pipeline project directory as the crawl root.

        :param backend:
            Molecule loading backend passed to :meth:`PoseCrawler.crawl_mols`.
        :type backend: str

        :returns:
            Crawled pose dataframe.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            pose_df = pipeline.crawl_poses(backend="obabel")
            print(pose_df.head())
        """
        self.logger.info(
            "Crawling poses from project directory | root=%s | backend=%s",
            self.project_dir,
            backend,
        )
        crawler = PoseCrawler([str(self.project_dir)])
        pose_df = crawler.crawl_mols(backend=backend)

        self.logger.info(
            "Pose crawling complete | rows=%d | columns=%d",
            len(pose_df),
            len(pose_df.columns),
        )
        self.logger.debug("Pose dataframe columns=%s", list(pose_df.columns))
        return pose_df

    def extract_interactions(
        self,
        *,
        poses: pd.DataFrame,
        receptor_specs: Sequence[PreparedReceptorSpec],
        batch_size: int = 1,
        progress: bool = False,
        n_jobs: int = 1,
        include_fingerprint_columns: bool = True,
        include_interaction_events: bool = True,
        include_bitvectors: bool = False,
        include_countvectors: bool = False,
        fail_fast: bool = True,
        use_profiler: bool = False,
        receptor_guess_bonds: bool = False,
    ) -> Tuple[Any, Dict[str, Path]]:
        """
        Extract protein-ligand interactions from a crawled pose dataframe.

        The receptor ``.pdb`` mapping is automatically derived from the prepared
        receptor ``.pdbqt`` files by replacing ``.pdbqt`` with ``.pdb``.

        By default this method uses
        :func:`extract_pose_table_interactions`. Optionally, it can use
        :class:`InteractionProfiler`.

        :param poses:
            Pose dataframe returned by :meth:`crawl_poses`.
        :type poses: pandas.DataFrame
        :param receptor_specs:
            Prepared receptor specifications.
        :type receptor_specs: Sequence[PreparedReceptorSpec]
        :param batch_size:
            Batch size for interaction extraction.
        :type batch_size: int
        :param progress:
            Whether to show progress.
        :type progress: bool
        :param n_jobs:
            Number of parallel jobs.
        :type n_jobs: int
        :param include_fingerprint_columns:
            Whether to include fingerprint columns.
        :type include_fingerprint_columns: bool
        :param include_interaction_events:
            Whether to include the long-form interaction-event dataframe.
        :type include_interaction_events: bool
        :param include_bitvectors:
            Whether to include bitvectors.
        :type include_bitvectors: bool
        :param include_countvectors:
            Whether to include countvectors.
        :type include_countvectors: bool
        :param fail_fast:
            Whether to fail on the first extraction error.
        :type fail_fast: bool
        :param use_profiler:
            Whether to use :class:`InteractionProfiler` instead of the
            functional wrapper.
        :type use_profiler: bool

        :returns:
            Tuple ``(interaction_result, receptor_pdb_by_id)``.
        :rtype: Tuple[Any, Dict[str, Path]]

        Example
        -------
        .. code-block:: python

            interaction_result, receptor_pdb_by_id = pipeline.extract_interactions(
                poses=pose_df,
                receptor_specs=receptor_specs,
                batch_size=1,
                progress=False,
                n_jobs=1,
            )

            merged_df = interaction_result.merged_df
            interaction_df = interaction_result.interaction_df
            summary_df = interaction_result.summary_df
            compact = interaction_result.summary_dict(kind="compact")
        """
        self.logger.info(
            "Starting interaction extraction | poses=%d | receptors=%d | batch_size=%d | "
            "progress=%s | n_jobs=%d | use_profiler=%s | include_fingerprint_columns=%s | "
            "include_interaction_events=%s | include_bitvectors=%s | include_countvectors=%s | fail_fast=%s",
            len(poses),
            len(receptor_specs),
            batch_size,
            progress,
            n_jobs,
            use_profiler,
            include_fingerprint_columns,
            include_interaction_events,
            include_bitvectors,
            include_countvectors,
            fail_fast,
        )

        receptor_pdb_by_id = self._build_receptor_pdb_map(receptor_specs)
        receptor_pdb_by_id_str = {k: str(v) for k, v in receptor_pdb_by_id.items()}
        self.logger.debug(
            "Interaction receptor_pdb_by_id keys=%s",
            list(receptor_pdb_by_id_str.keys()),
        )

        if use_profiler:
            self.logger.info("Using InteractionProfiler.run_pose_table")
            profiler = InteractionProfiler(
                receptor_guess_bonds=receptor_guess_bonds,
            )
            result = profiler.run_pose_table(
                poses=poses,
                receptor_pdb_by_id=receptor_pdb_by_id_str,
                batch_size=batch_size,
                include_fingerprint_columns=include_fingerprint_columns,
                include_interaction_events=include_interaction_events,
                include_bitvectors=include_bitvectors,
                include_countvectors=include_countvectors,
                fail_fast=fail_fast,
            )
        else:
            self.logger.info("Using extract_pose_table_interactions")
            result = extract_pose_table_interactions(
                poses=poses,
                receptor_pdb_by_id=receptor_pdb_by_id_str,
                batch_size=batch_size,
                progress=progress,
                n_jobs=n_jobs,
                include_fingerprint_columns=include_fingerprint_columns,
                include_interaction_events=include_interaction_events,
                include_bitvectors=include_bitvectors,
                include_countvectors=include_countvectors,
                fail_fast=fail_fast,
                receptor_guess_bonds=receptor_guess_bonds,
            )

        merged_rows = (
            len(result.merged_df)
            if getattr(result, "merged_df", None) is not None
            else 0
        )
        interaction_rows = (
            len(result.interaction_df)
            if getattr(result, "interaction_df", None) is not None
            else 0
        )
        summary_rows = (
            len(result.summary_df)
            if getattr(result, "summary_df", None) is not None
            else 0
        )

        self.logger.info(
            "Interaction extraction complete | merged_rows=%d | interaction_rows=%d | summary_rows=%d",
            merged_rows,
            interaction_rows,
            summary_rows,
        )
        return result, receptor_pdb_by_id

    def save_database(
        self,
        *,
        df: pd.DataFrame,
        interactions_by_pose: Optional[Dict[str, Any]] = None,
        db_name: PathLike = "prodock.db",
        replace: bool = True,
        replace_interactions: bool = True,
    ) -> Path:
        """
        Create or update a project-local SQLite database and insert results.

        Relative database names are created inside ``project_dir``. With the
        default configuration, the database path is::

            <project_dir>/prodock.db

        :param df:
            Dataframe to insert.
        :type df: pandas.DataFrame
        :param interactions_by_pose:
            Optional compact interaction dictionary keyed by pose identifier.
        :type interactions_by_pose: Optional[Dict[str, Any]]
        :param db_name:
            Database filename or relative path under ``project_dir``.
        :type db_name: PathLike
        :param replace:
            Whether to replace existing pose rows.
        :type replace: bool
        :param replace_interactions:
            Whether to replace existing interaction rows.
        :type replace_interactions: bool

        :returns:
            Absolute path to the SQLite database.
        :rtype: Path

        Example
        -------
        .. code-block:: python

            db_path = pipeline.save_database(
                df=merged_df,
                interactions_by_pose=compact,
                db_name="prodock.db",
            )
            print(db_path)
        """
        db_path = self._resolve_db_path(db_name)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Saving results to database | db_path=%s | rows=%d | columns=%d | "
            "with_interactions=%s | replace=%s | replace_interactions=%s",
            db_path,
            len(df),
            len(df.columns),
            interactions_by_pose is not None,
            replace,
            replace_interactions,
        )
        self.logger.debug("Database dataframe columns=%s", list(df.columns))

        db = PoseDatabase(str(db_path), create=True)

        # Record run/campaign provenance and stamp inserted poses with its run_id.
        try:
            _prodock_version = importlib.metadata.version("prodock")
        except Exception:
            _prodock_version = None
        db.create_run(
            name=getattr(self, "campaign_name", None),
            config={
                "engines": list(self.engines),
                "cpu": self.cpu,
                "seed": self.seed,
                "exhaustiveness": self.exhaustiveness,
                "n_poses": self.n_poses,
                "n_jobs": self.n_jobs,
                "box_algorithm": self.box_algorithm,
                "box_pad": self.box_pad,
                "box_scale": self.box_scale,
                "box_isotropic": self.box_isotropic,
                "ligand_backend": self.ligand_backend,
                "ligand_output_format": self.ligand_output_format,
            },
            prodock_version=_prodock_version,
        )

        if interactions_by_pose is None:
            db.insert_dataframe(
                df,
                replace=replace,
            )
        else:
            db.insert_dataframe(
                df,
                interactions_by_pose=interactions_by_pose,
                replace=replace,
                replace_interactions=replace_interactions,
            )

        resolved = db_path.resolve()
        self.logger.info("Database write complete | db_path=%s", resolved)
        return resolved

    def run(
        self,
        *,
        receptors: Optional[List[Dict[str, Any]]] = None,
        prepared_receptors: Optional[List[Dict[str, Any]]] = None,
        ligands: Optional[List[Dict[str, str]]] = None,
        ligand_dir: Optional[PathLike] = None,
        campaign_name: Optional[str] = None,
        crawl_backend: str = "backend",
        extract_interaction: bool = False,
        interaction_batch_size: int = 1,
        interaction_progress: bool = False,
        interaction_n_jobs: int = 1,
        include_fingerprint_columns: bool = True,
        include_interaction_events: bool = True,
        include_bitvectors: bool = False,
        include_countvectors: bool = False,
        fail_fast: bool = True,
        use_interaction_profiler: bool = False,
        receptor_guess_bonds: bool = False,
        save_to_database: bool = True,
        db_name: PathLike = "prodock.db",
        replace: bool = True,
        replace_interactions: bool = True,
    ) -> ProDockResult:
        """
        Execute the full ProDock pipeline.

        This method performs the full end-to-end automation:

        1. prepare or validate receptors
        2. prepare or resolve ligands
        3. build and save the campaign JSON
        4. run batch docking
        5. run pose crawling on the project directory
        6. optionally extract interactions
        7. optionally create ``<project_dir>/prodock.db`` and insert results

        :param receptors:
            Raw receptor records for full receptor acquisition and preparation
            mode.
        :type receptors: Optional[List[Dict[str, Any]]]
        :param prepared_receptors:
            Already prepared receptor records for direct docking mode.
        :type prepared_receptors: Optional[List[Dict[str, Any]]]
        :param ligands:
            Ligand SMILES records for ligand preparation mode.
        :type ligands: Optional[List[Dict[str, str]]]
        :param ligand_dir:
            Existing prepared ligand directory for direct docking mode.
        :type ligand_dir: Optional[PathLike]
        :param campaign_name:
            Output campaign JSON file name.
        :type campaign_name: Optional[str]
        :param crawl_backend:
            Backend passed to :meth:`crawl_poses`.
        :type crawl_backend: str
        :param extract_interaction:
            Whether to run interaction extraction after pose crawling.
        :type extract_interaction: bool
        :param interaction_batch_size:
            Interaction extraction batch size.
        :type interaction_batch_size: int
        :param interaction_progress:
            Whether to show progress during interaction extraction.
        :type interaction_progress: bool
        :param interaction_n_jobs:
            Parallel jobs for interaction extraction.
        :type interaction_n_jobs: int
        :param include_fingerprint_columns:
            Whether to include fingerprint columns in interaction output.
        :type include_fingerprint_columns: bool
        :param include_interaction_events:
            Whether to include long-form interaction events.
        :type include_interaction_events: bool
        :param include_bitvectors:
            Whether to include bitvectors.
        :type include_bitvectors: bool
        :param include_countvectors:
            Whether to include countvectors.
        :type include_countvectors: bool
        :param fail_fast:
            Whether interaction extraction should fail immediately on errors.
        :type fail_fast: bool
        :param use_interaction_profiler:
            Whether to use :class:`InteractionProfiler` for interaction
            extraction.
        :type use_interaction_profiler: bool
        :param receptor_guess_bonds:
            Whether ProLIF should guess receptor bonds during interaction
            extraction. Disabled by default because enabling it can segfault
            on some receptor topologies; when disabled, MDAnalysis still infers
            bonds through its RDKit converter.
        :type receptor_guess_bonds: bool
        :param save_to_database:
            Whether to create or update the SQLite database and insert results.
        :type save_to_database: bool
        :param db_name:
            Database filename or relative path under ``project_dir``.
        :type db_name: PathLike
        :param replace:
            Whether database insertion should replace existing pose rows.
        :type replace: bool
        :param replace_interactions:
            Whether database insertion should replace existing interaction rows.
        :type replace_interactions: bool

        :returns:
            Structured pipeline result.
        :rtype: ProDockResult

        Example
        -------
        .. code-block:: python

            result = pipeline.run(
                receptors=RECEPTORS,
                ligands=LIGANDS,
                extract_interaction=True,
                db_name="prodock.db",
            )

            print(result.campaign_json)
            print(result.db_path)
            print(result.pose_df.head())
            print(result.merged_df.head())

        Example
        -------
        .. code-block:: python

            result = pipeline.run(
                prepared_receptors=[
                    {
                        "receptor_id": "4WKQ",
                        "receptor_pdbqt": "Data/testcase/Multi/4WKQ/filtered_protein/4WKQ.pdbqt",
                        "center": (2.865, 193.257, 21.367),
                        "size": (27.091, 27.091, 27.091),
                    }
                ],
                ligand_dir="Data/testcase/Multi/ligands",
                extract_interaction=True,
                interaction_batch_size=1,
                interaction_n_jobs=1,
            )
        """
        self.logger.info(
            "Starting ProDock pipeline run | project_dir=%s | extract_interaction=%s | "
            "save_to_database=%s | crawl_backend=%s | db_name=%s",
            self.project_dir,
            extract_interaction,
            save_to_database,
            crawl_backend,
            db_name,
        )

        receptor_specs = self.prepare_receptors(
            receptors=receptors,
            prepared_receptors=prepared_receptors,
        )
        self.logger.info("Receptor stage complete | receptors=%d", len(receptor_specs))

        final_ligand_dir = self.prepare_ligands(
            ligands=ligands,
            ligand_dir=ligand_dir,
        )
        self.logger.info("Ligand stage complete | ligand_dir=%s", final_ligand_dir)

        campaign = self.build_campaign(
            receptor_specs=receptor_specs,
            ligand_dir=final_ligand_dir,
        )

        campaign_json = self.save_campaign(
            campaign,
            campaign_name=campaign_name,
        )

        self.logger.info(
            "Launching batch docking | campaign_json=%s | n_jobs=%s | progress=%s",
            campaign_json,
            self.n_jobs,
            self.progress,
        )
        # ``BatchDock.run_from_config`` is a classmethod: calling it on an
        # instance does not use that instance's settings, it builds a fresh
        # BatchDock from whatever ``n_jobs``/``progress`` are recorded in the
        # campaign file itself (defaulting to n_jobs=1 when absent). Override
        # those fields on the loaded config so this pipeline's n_jobs/progress
        # actually take effect instead of silently falling back to serial
        # execution.
        batch_cfg = BatchConfig.from_file(str(campaign_json))
        batch_cfg.n_jobs = self.n_jobs
        batch_cfg.progress = self.progress
        docking_results = BatchDock.run_from_config(batch_cfg)
        self.logger.info("Batch docking complete")

        pose_df = self.crawl_poses(backend=crawl_backend)

        receptor_pdb_by_id: Dict[str, Path] = {}
        interaction_result: Any = None
        merged_df = pose_df
        interaction_df: Optional[pd.DataFrame] = None
        summary_df: Optional[pd.DataFrame] = None
        compact_interactions: Optional[Dict[str, Any]] = None

        if extract_interaction:
            self.logger.info("Interaction extraction enabled")
            interaction_result, receptor_pdb_by_id = self.extract_interactions(
                poses=pose_df,
                receptor_specs=receptor_specs,
                batch_size=interaction_batch_size,
                progress=interaction_progress,
                n_jobs=interaction_n_jobs,
                include_fingerprint_columns=include_fingerprint_columns,
                include_interaction_events=include_interaction_events,
                include_bitvectors=include_bitvectors,
                include_countvectors=include_countvectors,
                fail_fast=fail_fast,
                use_profiler=use_interaction_profiler,
                receptor_guess_bonds=receptor_guess_bonds,
            )

            merged_df = interaction_result.merged_df
            interaction_df = interaction_result.interaction_df
            summary_df = interaction_result.summary_df
            compact_interactions = interaction_result.summary_dict(kind="compact")

            self.logger.info(
                "Interaction stage complete | merged_rows=%d | interaction_rows=%s | summary_rows=%s",
                len(merged_df),
                None if interaction_df is None else len(interaction_df),
                None if summary_df is None else len(summary_df),
            )
        else:
            self.logger.info("Interaction extraction skipped")
            receptor_pdb_by_id = self._build_receptor_pdb_map(receptor_specs)

        db_path: Optional[Path] = None
        if save_to_database:
            db_path = self.save_database(
                df=merged_df,
                interactions_by_pose=compact_interactions,
                db_name=db_name,
                replace=replace,
                replace_interactions=replace_interactions,
            )
        else:
            self.logger.info("Database writing skipped")

        result = ProDockResult(
            project_dir=self.project_dir,
            ligand_dir=final_ligand_dir,
            campaign_json=campaign_json,
            receptors=list(receptor_specs),
            receptor_pdb_by_id=receptor_pdb_by_id,
            campaign=campaign,
            docking_results=docking_results,
            pose_df=pose_df,
            interaction_result=interaction_result,
            merged_df=merged_df,
            interaction_df=interaction_df,
            summary_df=summary_df,
            compact_interactions=compact_interactions,
            db_path=db_path,
        )

        self.logger.info(
            "ProDock pipeline run complete | project_dir=%s | campaign_json=%s | "
            "pose_rows=%d | merged_rows=%d | db_path=%s",
            result.project_dir,
            result.campaign_json,
            len(result.pose_df),
            len(result.merged_df),
            result.db_path,
        )
        return result


def prodock(
    project_dir: PathLike,
    *,
    receptors: Optional[List[Dict[str, Any]]] = None,
    prepared_receptors: Optional[List[Dict[str, Any]]] = None,
    ligands: Optional[List[Dict[str, str]]] = None,
    ligand_dir: Optional[PathLike] = None,
    engines: Optional[Sequence[str]] = None,
    cpu: int = 4,
    seed: int = 42,
    exhaustiveness: int = 8,
    n_poses: int = 10,
    n_jobs: Optional[int] = None,
    progress: bool = True,
    receptor_use_meeko: bool = False,
    ligand_output_format: str = "pdbqt",
    ligand_backend: str = "meeko",
    box_algorithm: Optional[str] = None,
    box_pad: float = 4.0,
    box_scale: Optional[float] = None,
    box_isotropic: bool = True,
    campaign_name: str = "campaign.json",
    crawl_backend: str = "auto",
    extract_interaction: bool = False,
    interaction_batch_size: int = 1,
    interaction_progress: bool = False,
    interaction_n_jobs: int = 1,
    include_fingerprint_columns: bool = True,
    include_interaction_events: bool = True,
    include_bitvectors: bool = False,
    include_countvectors: bool = False,
    fail_fast: bool = True,
    use_interaction_profiler: bool = False,
    receptor_guess_bonds: bool = False,
    save_to_database: bool = True,
    db_name: PathLike = "prodock.db",
    replace: bool = True,
    replace_interactions: bool = True,
    log_file: str = "prodock.log",
    log_level: Union[str, int] = "INFO",
    log_colored: bool = True,
    log_json: bool = False,
) -> ProDockResult:
    """
    Functional wrapper around :class:`ProDockPipeline`.

    This is the main convenience entry point for running a complete ProDock
    workflow in one call.

    :param project_dir:
        Root project directory.
    :type project_dir: PathLike
    :param receptors:
        Raw receptor records.
    :type receptors: Optional[List[Dict[str, Any]]]
    :param prepared_receptors:
        Already prepared receptor records.
    :type prepared_receptors: Optional[List[Dict[str, Any]]]
    :param ligands:
        Ligand SMILES records.
    :type ligands: Optional[List[Dict[str, str]]]
    :param ligand_dir:
        Existing ligand directory.
    :type ligand_dir: Optional[PathLike]
    :param engines:
        Docking engines to include. Default is ``["smina", "qvina"]``.
    :type engines: Optional[Sequence[str]]
    :param cpu:
        CPU value stored in the campaign.
    :type cpu: int
    :param seed:
        Random seed stored in the campaign.
    :type seed: int
    :param exhaustiveness:
        Exhaustiveness stored in the campaign.
    :type exhaustiveness: int
    :param n_poses:
        Number of poses stored in the campaign.
    :type n_poses: int
    :param n_jobs:
        Parallel jobs used by :class:`BatchDock`.
    :type n_jobs: Optional[int]
    :param progress:
        Whether to enable docking progress reporting.
    :type progress: bool
    :param receptor_use_meeko:
        Whether receptor preparation uses Meeko.
    :type receptor_use_meeko: bool
    :param ligand_output_format:
        Final ligand output format.
    :type ligand_output_format: str
    :param ligand_backend:
        Ligand conversion backend.
    :type ligand_backend: str
    :param box_algorithm:
        Ligand-derived box algorithm. ``None`` selects ``"pad"`` unless
        ``box_scale`` is supplied explicitly.
    :type box_algorithm: Optional[str]
    :param box_pad:
        Symmetric padding in Angstrom used by the ``"pad"`` algorithm.
    :type box_pad: float
    :param box_scale:
        Scale factor used by the ``"scale"`` algorithm. Supplying this without
        ``box_algorithm`` preserves legacy scale behavior.
    :type box_scale: Optional[float]
    :param box_isotropic:
        Whether ligand-derived boxes are isotropic.
    :type box_isotropic: bool
    :param campaign_name:
        Output campaign JSON file name.
    :type campaign_name: str
    :param crawl_backend:
        Pose crawler backend.
    :type crawl_backend: str
    :param extract_interaction:
        Whether to run interaction extraction.
    :type extract_interaction: bool
    :param interaction_batch_size:
        Interaction extraction batch size.
    :type interaction_batch_size: int
    :param interaction_progress:
        Whether to show interaction extraction progress.
    :type interaction_progress: bool
    :param interaction_n_jobs:
        Parallel jobs for interaction extraction.
    :type interaction_n_jobs: int
    :param include_fingerprint_columns:
        Whether to include fingerprint columns.
    :type include_fingerprint_columns: bool
    :param include_interaction_events:
        Whether to include long-form interaction events.
    :type include_interaction_events: bool
    :param include_bitvectors:
        Whether to include bitvectors.
    :type include_bitvectors: bool
    :param include_countvectors:
        Whether to include countvectors.
    :type include_countvectors: bool
    :param fail_fast:
        Whether interaction extraction fails immediately on errors.
    :type fail_fast: bool
    :param use_interaction_profiler:
        Whether to use :class:`InteractionProfiler`.
    :type use_interaction_profiler: bool
    :param receptor_guess_bonds:
        Whether ProLIF should guess receptor bonds during interaction
        extraction. Disabled by default because enabling it can segfault on
        some receptor topologies; when disabled, MDAnalysis still infers bonds
        through its RDKit converter.
    :type receptor_guess_bonds: bool
    :param save_to_database:
        Whether to write results into the SQLite database.
    :type save_to_database: bool
    :param db_name:
        Database filename or relative path under ``project_dir``.
    :type db_name: PathLike
    :param replace:
        Whether to replace existing pose rows in the database.
    :type replace: bool
    :param replace_interactions:
        Whether to replace existing interaction rows in the database.
    :type replace_interactions: bool

    :returns:
        Structured pipeline result.
    :rtype: ProDockResult

    Example
    -------
    .. code-block:: python

        from prodock import prodock

        PROJECT = "Data/testcase/Multi"

        RECEPTORS = [
            {
                "pdb_id": "4WKQ",
                "receptor_name": "EGFR_4WKQ",
                "ligand_code": "IRE",
                "chains": ["A"],
                "cofactors": [],
            },
        ]

        LIGANDS = [
            {
                "id": "erlotinib",
                "smiles": "COCCOc1cc2c(ncnc2cc1OCCOC)Nc1cccc(c1)C#C",
            },
            {
                "id": "gefitinib",
                "smiles": "COc1cc2ncnc(c2cc1OCCCN1CCOCC1)Nc1ccc(c(c1)Cl)F",
            },
        ]

        result = prodock(
            PROJECT,
            receptors=RECEPTORS,
            ligands=LIGANDS,
            engines=["smina"],
            extract_interaction=True,
            db_name="prodock.db",
        )

        print(result.campaign_json)
        print(result.db_path)
        print(result.receptor_pdb_by_id)

    Example
    -------
    .. code-block:: python

        from prodock import prodock

        PROJECT = "Data/testcase/Multi"

        result = prodock(
            PROJECT,
            prepared_receptors=[
                {
                    "receptor_id": "4WKQ",
                    "receptor_pdbqt": "Data/testcase/Multi/4WKQ/filtered_protein/4WKQ.pdbqt",
                    "center": (2.865, 193.257, 21.367),
                    "size": (27.091, 27.091, 27.091),
                }
            ],
            ligand_dir="Data/testcase/Multi/ligands",
            extract_interaction=True,
            db_name="prodock.db",
        )

        print(result.summary_df.head())
    """
    logger.info(
        "prodock() called | project_dir=%s | extract_interaction=%s | save_to_database=%s | engines=%s",
        project_dir,
        extract_interaction,
        save_to_database,
        engines,
    )

    pipeline = ProDockPipeline(
        project_dir=project_dir,
        engines=engines,
        cpu=cpu,
        seed=seed,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        n_jobs=n_jobs,
        progress=progress,
        receptor_use_meeko=receptor_use_meeko,
        ligand_output_format=ligand_output_format,
        ligand_backend=ligand_backend,
        box_algorithm=box_algorithm,
        box_pad=box_pad,
        box_scale=box_scale,
        box_isotropic=box_isotropic,
        campaign_name=campaign_name,
        log_file=log_file,
        log_level=log_level,
        log_colored=log_colored,
        log_json=log_json,
    )

    result = pipeline.run(
        receptors=receptors,
        prepared_receptors=prepared_receptors,
        ligands=ligands,
        ligand_dir=ligand_dir,
        campaign_name=campaign_name,
        crawl_backend=crawl_backend,
        extract_interaction=extract_interaction,
        interaction_batch_size=interaction_batch_size,
        interaction_progress=interaction_progress,
        interaction_n_jobs=interaction_n_jobs,
        include_fingerprint_columns=include_fingerprint_columns,
        include_interaction_events=include_interaction_events,
        include_bitvectors=include_bitvectors,
        include_countvectors=include_countvectors,
        fail_fast=fail_fast,
        use_interaction_profiler=use_interaction_profiler,
        receptor_guess_bonds=receptor_guess_bonds,
        save_to_database=save_to_database,
        db_name=db_name,
        replace=replace,
        replace_interactions=replace_interactions,
    )

    logger.info(
        "prodock() complete | campaign_json=%s | db_path=%s | pose_rows=%d",
        result.campaign_json,
        result.db_path,
        len(result.pose_df),
    )
    return result


run_prodock = prodock
