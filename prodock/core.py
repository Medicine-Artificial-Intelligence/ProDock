from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from prodock.dock import BatchDock
from prodock.dock.campaign import Campaign
from prodock.preprocess import LigandPrep, ReceptorPrep
from prodock.preprocess.gridbox import GridBox
from prodock.structure import PDBQuery

PathLike = Union[str, Path]
Vec3 = Tuple[float, float, float]


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
    Result bundle returned by :class:`ProDockPipeline`.

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
    :param results:
        Raw docking results returned by :class:`prodock.dock.BatchDock`.
    :type results: Any
    """

    project_dir: Path
    ligand_dir: Path
    campaign_json: Path
    receptors: List[PreparedReceptorSpec]
    results: Any


class ProDockPipeline:
    """
    High-level orchestration helper for ProDock projects.

    This helper unifies common workflows:

    1. raw receptor records + ligand SMILES records
    2. prebuilt receptor ``.pdbqt`` files + explicit box coordinates
    3. either ligand SMILES input or an existing ligand directory

    The pipeline can prepare receptors, prepare ligands, build a campaign JSON,
    and optionally run batch docking in a single call.

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
    :param box_scale:
        Scale factor used when computing a grid box from a reference ligand.
    :type box_scale: float
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
        )
        print(result.campaign_json)

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
        )
        print(result.campaign_json)

    Example
    -------
    .. code-block:: python

        from prodock import prodock

        PROJECT = "Data/testcase/Multi"

        RECEPTORS = [
            {
                "pdb_id": "1M17",
                "receptor_name": "EGFR_1M17",
                "ligand_code": "AQ4",
                "chains": ["A"],
                "cofactors": [],
            },
            {
                "pdb_id": "2ITY",
                "receptor_name": "EGFR_2ITY",
                "ligand_code": "IRE",
                "chains": ["A"],
                "cofactors": [],
            },
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
        )
        print(result.campaign_json)

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
                },
                {
                    "receptor_id": "1M17",
                    "receptor_pdbqt": "Data/testcase/Multi/1M17/filtered_protein/1M17.pdbqt",
                    "center": (21.623, 0.4, 52.467),
                    "size": (34.07, 34.07, 34.07),
                },
            ],
            ligand_dir="Data/testcase/Multi/ligands",
        )
        print(result.campaign_json)
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
        box_scale: float = 2.0,
        box_isotropic: bool = True,
        campaign_name: str = "campaign.json",
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
        :param box_scale:
            Scale factor used when computing a grid box from a reference ligand.
        :type box_scale: float
        :param box_isotropic:
            Whether ligand-derived boxes should be isotropic.
        :type box_isotropic: bool
        :param campaign_name:
            Default campaign JSON file name.
        :type campaign_name: str
        """
        self.project_dir = Path(project_dir).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)

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

        self.box_scale = box_scale
        self.box_isotropic = box_isotropic
        self.campaign_name = campaign_name

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
            return p.resolve()
        if p.exists():
            return p.resolve()
        return (self.project_dir / p).resolve()

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
        return self.project_dir / pdb_id / "reference_ligand" / f"{ligand_code}.sdf"

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
        if record.get("center") is not None and record.get("size") is not None:
            center = self._as_vec3(record["center"], field_name="center")
            size = self._as_vec3(record["size"], field_name="size")
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

        gb = (
            GridBox()
            .load_ligand(str(ref_path), fmt=fmt)
            .from_ligand_scale(
                scale=self.box_scale,
                isotropic=self.box_isotropic,
            )
        )
        center = self._as_vec3(gb.center, field_name="center")
        size = self._as_vec3(gb.size, field_name="size")
        return center, size

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

        if has_raw == has_prepared:
            raise ValueError(
                "Provide exactly one of 'receptors' or 'prepared_receptors'."
            )

        if prepared_receptors is not None:
            specs: List[PreparedReceptorSpec] = []
            for item in prepared_receptors:
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
            return specs

        assert receptors is not None

        PDBQuery.process_batch(receptors, output_dir=str(self.project_dir))

        specs: List[PreparedReceptorSpec] = []
        for record in receptors:
            pdb_id = str(record["pdb_id"])
            filtered_dir = self.project_dir / pdb_id / "filtered_protein"
            input_pdb = filtered_dir / f"{pdb_id}.pdb"
            output_pdbqt = filtered_dir / f"{pdb_id}.pdbqt"

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

            center, size = self._box_from_record(record)

            specs.append(
                PreparedReceptorSpec(
                    receptor_id=pdb_id,
                    receptor_pdbqt=output_pdbqt.resolve(),
                    center=center,
                    size=size,
                )
            )

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

        if has_ligands == has_ligand_dir:
            raise ValueError("Provide exactly one of 'ligands' or 'ligand_dir'.")

        if ligand_dir is not None:
            resolved = self._resolve_path(ligand_dir)
            if not resolved.exists():
                raise FileNotFoundError(f"Ligand directory not found: {resolved}")
            if not resolved.is_dir():
                raise NotADirectoryError(f"Not a directory: {resolved}")
            return resolved

        assert ligands is not None

        out_dir = self.project_dir / "ligands"
        out_dir.mkdir(parents=True, exist_ok=True)

        (
            LigandPrep(
                output_dir=str(out_dir),
                smiles_key="smiles",
                name_key="id",
            )
            .set_output_format(self.ligand_output_format)
            .set_converter_backend(self.ligand_backend)
            .from_list_of_dicts(ligands)
            .process_all()
        )

        return out_dir.resolve()

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
        """
        ligand_dir = self._resolve_path(ligand_dir)

        pdb_ids = [spec.receptor_id for spec in receptor_specs]
        receptors = [str(spec.receptor_pdbqt) for spec in receptor_specs]
        boxes = [(spec.center, spec.size) for spec in receptor_specs]

        return Campaign.from_shared_ligand_dir(
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
        """
        name = campaign_name or self.campaign_name
        out_path = self.project_dir / name
        campaign.save_json(str(out_path))
        return out_path.resolve()

    def run(
        self,
        *,
        receptors: Optional[List[Dict[str, Any]]] = None,
        prepared_receptors: Optional[List[Dict[str, Any]]] = None,
        ligands: Optional[List[Dict[str, str]]] = None,
        ligand_dir: Optional[PathLike] = None,
        campaign_name: Optional[str] = None,
    ) -> ProDockResult:
        """
        Execute the full ProDock pipeline.

        This method:

        1. prepares or validates receptors
        2. prepares or resolves ligands
        3. builds and saves the campaign JSON
        4. runs batch docking

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

        :returns:
            Structured pipeline result.
        :rtype: ProDockResult

        Example
        -------
        .. code-block:: python

            pipeline = ProDockPipeline("Data/testcase/Multi")
            result = pipeline.run(
                receptors=RECEPTORS,
                ligands=LIGANDS,
            )
            print(result.campaign_json)

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
                    },
                    {
                        "receptor_id": "1M17",
                        "receptor_pdbqt": "Data/testcase/Multi/1M17/filtered_protein/1M17.pdbqt",
                        "center": (21.623, 0.4, 52.467),
                        "size": (34.07, 34.07, 34.07),
                    },
                ],
                ligand_dir="Data/testcase/Multi/ligands",
            )
            print(result.campaign_json)
        """
        receptor_specs = self.prepare_receptors(
            receptors=receptors,
            prepared_receptors=prepared_receptors,
        )
        final_ligand_dir = self.prepare_ligands(
            ligands=ligands,
            ligand_dir=ligand_dir,
        )
        campaign = self.build_campaign(
            receptor_specs=receptor_specs,
            ligand_dir=final_ligand_dir,
        )
        campaign_json = self.save_campaign(
            campaign,
            campaign_name=campaign_name,
        )

        runner = BatchDock(n_jobs=self.n_jobs, progress=self.progress)
        results = runner.run_from_config(str(campaign_json))

        return ProDockResult(
            project_dir=self.project_dir,
            ligand_dir=final_ligand_dir,
            campaign_json=campaign_json,
            receptors=list(receptor_specs),
            results=results,
        )


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
    box_scale: float = 2.0,
    box_isotropic: bool = True,
    campaign_name: str = "campaign.json",
) -> ProDockResult:
    """
    Functional wrapper around :class:`ProDockPipeline`.

    This is the main convenience entry point for running a ProDock workflow in
    one call.

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
        Whether to enable progress reporting.
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
    :param box_scale:
        Box scale factor used for ligand-derived boxes.
    :type box_scale: float
    :param box_isotropic:
        Whether ligand-derived boxes are isotropic.
    :type box_isotropic: bool
    :param campaign_name:
        Output campaign JSON file name.
    :type campaign_name: str

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
                "pdb_id": "1M17",
                "receptor_name": "EGFR_1M17",
                "ligand_code": "AQ4",
                "chains": ["A"],
                "cofactors": [],
            },
            {
                "pdb_id": "2ITY",
                "receptor_name": "EGFR_2ITY",
                "ligand_code": "IRE",
                "chains": ["A"],
                "cofactors": [],
            },
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
        )
        print(result.campaign_json)

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
                },
                {
                    "receptor_id": "1M17",
                    "receptor_pdbqt": "Data/testcase/Multi/1M17/filtered_protein/1M17.pdbqt",
                    "center": (21.623, 0.4, 52.467),
                    "size": (34.07, 34.07, 34.07),
                },
            ],
            ligand_dir="Data/testcase/Multi/ligands",
        )
        print(result.campaign_json)
    """
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
        box_scale=box_scale,
        box_isotropic=box_isotropic,
        campaign_name=campaign_name,
    )
    return pipeline.run(
        receptors=receptors,
        prepared_receptors=prepared_receptors,
        ligands=ligands,
        ligand_dir=ligand_dir,
        campaign_name=campaign_name,
    )


# Optional backward-compatible alias.
run_prodock = prodock
