from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


JsonPath = Union[str, Path]


def _tuplize3(
    value: Optional[Union[List[float], Tuple[float, float, float]]],
) -> Optional[Tuple[float, float, float]]:
    """
    Normalize a 3-element vector-like object into a tuple of floats.

    This helper accepts either a list or tuple of length three and converts
    all entries to ``float``. Passing ``None`` returns ``None`` unchanged.

    :param value:
        Input 3-vector represented as a list or tuple, or ``None``.
    :type value:
        Optional[Union[List[float], Tuple[float, float, float]]]

    :returns:
        Normalized 3-vector as a tuple of floats, or ``None`` when the input
        is ``None``.
    :rtype:
        Optional[Tuple[float, float, float]]

    :raises ValueError:
        If the input is not ``None`` and does not contain exactly 3 elements.

    Example
    -------
    .. code-block:: python

        center = _tuplize3([1, 2, 3])
        # (1.0, 2.0, 3.0)

        missing = _tuplize3(None)
        # None
    """
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError("Expected a 3-vector")
    return tuple(float(x) for x in value)  # type: ignore[return-value]


@dataclass
class Box:
    """
    Docking box specification storing center and box size.

    :param center:
        Center coordinates of the docking box as ``(x, y, z)``.
    :type center: Tuple[float, float, float]

    :param size:
        Box dimensions along each axis as ``(sx, sy, sz)``.
    :type size: Tuple[float, float, float]
    """

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]

    @classmethod
    def from_mapping(
        cls, value: Union["Box", Dict[str, Any], List[Any], Tuple[Any, Any]]
    ) -> "Box":
        """
        Construct a :class:`Box` from a flexible mapping or pair representation.

        Accepted inputs are:

        - an existing :class:`Box` instance,
        - a dictionary with ``"center"`` and ``"size"``,
        - a two-item list or tuple of the form ``[center, size]``.

        :param value:
            Source object describing a docking box.
        :type value:
            Union[Box, Dict[str, Any], List[Any], Tuple[Any, Any]]

        :returns:
            Parsed docking box instance.
        :rtype: Box

        :raises TypeError:
            If the input is not a supported box representation.
        :raises ValueError:
            If either center or size is missing, or if either vector is not
            a valid 3-vector.

        Example
        -------
        .. code-block:: python

            box = Box.from_mapping(
                {
                    "center": [1.0, 2.0, 3.0],
                    "size": [20.0, 20.0, 20.0],
                }
            )
        """
        if isinstance(value, Box):
            return value
        if isinstance(value, dict):
            center = _tuplize3(value.get("center"))
            size = _tuplize3(value.get("size"))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            center = _tuplize3(value[0])
            size = _tuplize3(value[1])
        else:
            raise TypeError(
                "Box must be a Box, dict with center/size, or [center, size]"
            )
        if center is None or size is None:
            raise ValueError("Both center and size are required")
        return cls(center=center, size=size)

    def to_dict(self) -> Dict[str, List[float]]:
        """
        Convert the box to a JSON-serializable dictionary.

        :returns:
            Dictionary with ``"center"`` and ``"size"`` keys stored as lists.
        :rtype: Dict[str, List[float]]

        Example
        -------
        .. code-block:: python

            payload = Box(
                center=(1.0, 2.0, 3.0),
                size=(20.0, 20.0, 20.0),
            ).to_dict()
        """
        return {"center": list(self.center), "size": list(self.size)}


@dataclass
class SingleConfig:
    """
    Configuration for a single docking task.

    This compact form stores all parameters needed to launch one receptor-ligand
    docking job.

    :param engine:
        Docking engine name, for example ``"vina"``.
    :type engine: str

    :param receptor:
        Receptor input file path.
    :type receptor: Optional[str]

    :param ligand:
        Ligand input file path.
    :type ligand: Optional[str]

    :param box:
        Explicit docking box definition.
    :type box: Optional[Box]

    :param autobox_ref:
        Reference ligand or structure used for autoboxing.
    :type autobox_ref: Optional[str]

    :param autobox_pad:
        Padding added around the autobox reference.
    :type autobox_pad: Optional[float]

    :param exhaustiveness:
        Docking exhaustiveness parameter.
    :type exhaustiveness: Optional[int]

    :param n_poses:
        Number of poses to generate.
    :type n_poses: Optional[int]

    :param cpu:
        Number of CPU cores to use.
    :type cpu: Optional[int]

    :param seed:
        Random seed for reproducible runs.
    :type seed: Optional[int]

    :param out:
        Output pose file path.
    :type out: Optional[str]

    :param log:
        Log file path.
    :type log: Optional[str]

    :param executable:
        Explicit executable path for the engine binary.
    :type executable: Optional[str]

    :param engine_options:
        Extra engine-specific keyword options.
    :type engine_options: Dict[str, Any]

    :param validate_receptor:
        Whether to validate receptor format/content before docking.
    :type validate_receptor: bool
    """

    engine: str = "vina"
    receptor: Optional[str] = None
    ligand: Optional[str] = None
    box: Optional[Box] = None
    autobox_ref: Optional[str] = None
    autobox_pad: Optional[float] = None
    exhaustiveness: Optional[int] = None
    n_poses: Optional[int] = None
    cpu: Optional[int] = None
    seed: Optional[int] = None
    out: Optional[str] = None
    log: Optional[str] = None
    executable: Optional[str] = None
    engine_options: Dict[str, Any] = field(default_factory=dict)
    validate_receptor: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleConfig":
        """
        Build a :class:`SingleConfig` from a mapping.

        If a ``"box"`` entry is present, it is normalized through
        :meth:`Box.from_mapping`.

        :param data:
            Input configuration mapping.
        :type data: Dict[str, Any]

        :returns:
            Parsed single-task configuration.
        :rtype: SingleConfig

        Example
        -------
        .. code-block:: python

            cfg = SingleConfig.from_dict(
                {
                    "engine": "vina",
                    "receptor": "protein.pdbqt",
                    "ligand": "ligand.pdbqt",
                    "box": {
                        "center": [1.0, 2.0, 3.0],
                        "size": [20.0, 20.0, 20.0],
                    },
                }
            )
        """
        d = dict(data)
        if d.get("box") is not None:
            d["box"] = Box.from_mapping(d["box"])
        return cls(**d)

    @classmethod
    def from_file(cls, path: JsonPath) -> "SingleConfig":
        """
        Load a :class:`SingleConfig` from a JSON or YAML file.

        :param path:
            Configuration file path.
        :type path: JsonPath

        :returns:
            Parsed single-task configuration.
        :rtype: SingleConfig

        :raises FileNotFoundError:
            If the file does not exist.
        :raises RuntimeError:
            If a YAML file is provided but PyYAML is unavailable.
        :raises TypeError:
            If the file root is not a mapping.

        Example
        -------
        .. code-block:: python

            cfg = SingleConfig.from_file("config.json")
        """
        return cls.from_dict(_load_mapping(path))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration into a serializable dictionary.

        :returns:
            Dictionary representation suitable for JSON or YAML output.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            payload = cfg.to_dict()
        """
        out = asdict(self)
        if self.box is not None:
            out["box"] = self.box.to_dict()
        return out


@dataclass
class LigandSpec:
    id: str
    ligand: str
    out: Optional[str] = None
    log: Optional[str] = None
    exhaustiveness: Optional[int] = None
    n_poses: Optional[int] = None
    cpu: Optional[int] = None
    seed: Optional[int] = None
    engine_options: Dict[str, Any] = field(default_factory=dict)
    retries: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LigandSpec":
        """
        Build a :class:`LigandSpec` from a mapping.

        The method supports a few convenience aliases:

        - if ``"id"`` is missing, it is inferred from the ligand filename stem,
        - if ``"ligand"`` is missing but ``"path"`` is present, ``"path"`` is
          used as the ligand path.

        :param data:
            Ligand configuration mapping.
        :type data: Dict[str, Any]

        :returns:
            Parsed ligand specification.
        :rtype: LigandSpec

        Example
        -------
        .. code-block:: python

            lig = LigandSpec.from_dict(
                {
                    "path": "inputs/erlotinib.pdbqt",
                    "seed": 42,
                }
            )
        """
        d = dict(data)
        path_value = d.pop("path", None)
        if "ligand" not in d and path_value is not None:
            d["ligand"] = path_value
        if "id" not in d:
            ligand_path = str(d.get("ligand") or "ligand")
            d["id"] = Path(ligand_path).stem
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ligand specification into a serializable dictionary.

        :returns:
            Dictionary representation of the ligand specification.
        :rtype: Dict[str, Any]
        """
        return asdict(self)


@dataclass
class SoftwareSpec:
    """
    Per-engine specification nested under a receptor entry.

    :param name:
        Engine name, for example ``"vina"`` or ``"qvina"``.
    :type name: str

    :param executable:
        Optional explicit executable path.
    :type executable: Optional[str]

    :param exhaustiveness:
        Default exhaustiveness for ligands under this software block.
    :type exhaustiveness: Optional[int]

    :param n_poses:
        Default number of poses for ligands under this software block.
    :type n_poses: Optional[int]

    :param cpu:
        Default CPU count for ligands under this software block.
    :type cpu: Optional[int]

    :param seed:
        Default random seed for ligands under this software block.
    :type seed: Optional[int]

    :param out_dir:
        Output directory for generated docking poses.
    :type out_dir: Optional[str]

    :param log_dir:
        Directory for engine log files.
    :type log_dir: Optional[str]

    :param engine_options:
        Additional engine-specific options.
    :type engine_options: Dict[str, Any]

    :param ligands:
        Ligand specifications associated with this engine block.
    :type ligands: List[LigandSpec]
    """

    name: str
    executable: Optional[str] = None
    exhaustiveness: Optional[int] = None
    n_poses: Optional[int] = None
    cpu: Optional[int] = None
    seed: Optional[int] = None
    out_dir: Optional[str] = None
    log_dir: Optional[str] = None
    engine_options: Dict[str, Any] = field(default_factory=dict)
    ligands: List[LigandSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoftwareSpec":
        """
        Build a :class:`SoftwareSpec` from a mapping.

        Any entries in ``"ligands"`` are normalized into
        :class:`LigandSpec` instances.

        :param data:
            Software configuration mapping.
        :type data: Dict[str, Any]

        :returns:
            Parsed software specification.
        :rtype: SoftwareSpec

        Example
        -------
        .. code-block:: python

            sw = SoftwareSpec.from_dict(
                {
                    "name": "vina",
                    "ligands": [
                        {"id": "lig1", "ligand": "lig1.pdbqt"},
                    ],
                }
            )
        """
        d = dict(data)
        raw_ligands = d.get("ligands") or []
        d["ligands"] = [
            item if isinstance(item, LigandSpec) else LigandSpec.from_dict(item)
            for item in raw_ligands
        ]
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the software specification into a serializable dictionary.

        :returns:
            Dictionary form with serialized ligand blocks.
        :rtype: Dict[str, Any]
        """
        out = asdict(self)
        out["ligands"] = [lig.to_dict() for lig in self.ligands]
        return out


@dataclass
class ReceptorSpec:
    id: str
    receptor: str
    box: Optional[Box] = None
    out_dir: Optional[str] = None
    log_dir: Optional[str] = None
    autobox_ref: Optional[str] = None
    autobox_pad: Optional[float] = None
    validate_receptor: bool = False
    softwares: List[SoftwareSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReceptorSpec":
        """
        Build a :class:`ReceptorSpec` from a mapping.

        Supported conveniences:

        - if ``"id"`` is absent, it is inferred from the receptor file stem,
        - if ``"receptor"`` is absent but ``"path"`` exists, ``"path"`` is used,
        - ``"engines"`` is accepted as an alias for ``"softwares"``,
        - ``"box"`` is normalized through :meth:`Box.from_mapping`.

        :param data:
            Receptor configuration mapping.
        :type data: Dict[str, Any]

        :returns:
            Parsed receptor specification.
        :rtype: ReceptorSpec

        Example
        -------
        .. code-block:: python

            receptor = ReceptorSpec.from_dict(
                {
                    "path": "receptors/4WKQ.pdbqt",
                    "box": {
                        "center": [1.0, 2.0, 3.0],
                        "size": [20.0, 20.0, 20.0],
                    },
                    "engines": [
                        {"name": "vina", "ligands": []},
                    ],
                }
            )
        """
        d = dict(data)

        path_value = d.pop("path", None)
        if "receptor" not in d and path_value is not None:
            d["receptor"] = path_value

        if "id" not in d:
            receptor_path = str(d.get("receptor") or "receptor")
            d["id"] = Path(receptor_path).stem

        if d.get("box") is not None:
            d["box"] = Box.from_mapping(d["box"])

        raw_softwares = d.pop("softwares", None)
        if raw_softwares is None:
            raw_softwares = d.pop("engines", [])
        d["softwares"] = [
            item if isinstance(item, SoftwareSpec) else SoftwareSpec.from_dict(item)
            for item in (raw_softwares or [])
        ]
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the receptor specification into a serializable dictionary.

        :returns:
            Dictionary form with serialized software blocks and box.
        :rtype: Dict[str, Any]
        """
        out = asdict(self)
        out["softwares"] = [sw.to_dict() for sw in self.softwares]
        if self.box is not None:
            out["box"] = self.box.to_dict()
        return out


@dataclass
class DockRow:
    """
    Backward-compatible flat docking row representation.

    This structure is useful for legacy batch layouts where each row already
    combines receptor, ligand, and per-task engine parameters.

    :param id:
        Row or task identifier.
    :type id: str

    :param receptor:
        Receptor input file path.
    :type receptor: str

    :param ligand:
        Ligand input file path.
    :type ligand: str

    :param box:
        Optional pre-built box object.
    :type box: Optional[Box]

    :param center:
        Optional center triple used together with ``size``.
    :type center: Optional[Tuple[float, float, float]]

    :param size:
        Optional size triple used together with ``center``.
    :type size: Optional[Tuple[float, float, float]]

    :param autobox_ref:
        Optional autobox reference.
    :type autobox_ref: Optional[str]

    :param autobox_pad:
        Optional autobox padding.
    :type autobox_pad: Optional[float]

    :param out:
        Optional output file path.
    :type out: Optional[str]

    :param log:
        Optional log file path.
    :type log: Optional[str]

    :param exhaustiveness:
        Docking exhaustiveness.
    :type exhaustiveness: Optional[int]

    :param n_poses:
        Number of poses to generate.
    :type n_poses: Optional[int]

    :param cpu:
        CPU count.
    :type cpu: Optional[int]

    :param seed:
        Random seed.
    :type seed: Optional[int]

    :param engine_options:
        Extra engine-specific options.
    :type engine_options: Dict[str, Any]

    :param retries:
        Retry count for the row.
    :type retries: Optional[int]
    """

    id: str
    receptor: str
    ligand: str
    box: Optional[Box] = None
    center: Optional[Tuple[float, float, float]] = None
    size: Optional[Tuple[float, float, float]] = None
    autobox_ref: Optional[str] = None
    autobox_pad: Optional[float] = None
    out: Optional[str] = None
    log: Optional[str] = None
    exhaustiveness: Optional[int] = None
    n_poses: Optional[int] = None
    cpu: Optional[int] = None
    seed: Optional[int] = None
    engine_options: Dict[str, Any] = field(default_factory=dict)
    retries: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DockRow":
        """
        Build a :class:`DockRow` from a mapping.

        The method normalizes:

        - ``"box"`` via :meth:`Box.from_mapping`,
        - ``"center"`` and ``"size"`` via :func:`_tuplize3`.

        :param data:
            Flat row configuration mapping.
        :type data: Dict[str, Any]

        :returns:
            Parsed flat row instance.
        :rtype: DockRow

        Example
        -------
        .. code-block:: python

            row = DockRow.from_dict(
                {
                    "id": "job1",
                    "receptor": "protein.pdbqt",
                    "ligand": "ligand.pdbqt",
                    "center": [1, 2, 3],
                    "size": [20, 20, 20],
                }
            )
        """
        d = dict(data)
        if d.get("box") is not None:
            d["box"] = Box.from_mapping(d["box"])
        if d.get("center") is not None:
            d["center"] = _tuplize3(d["center"])
        if d.get("size") is not None:
            d["size"] = _tuplize3(d["size"])
        return cls(**d)

    def resolved_box(self) -> Optional[Box]:
        """
        Resolve the effective docking box for the row.

        Resolution follows this order:

        1. return ``self.box`` when already set,
        2. construct a new :class:`Box` from ``center`` and ``size`` when both
           are available,
        3. otherwise return ``None``.

        :returns:
            Resolved docking box or ``None`` if insufficient information exists.
        :rtype: Optional[Box]

        Example
        -------
        .. code-block:: python

            box = row.resolved_box()
        """
        if self.box is not None:
            return self.box
        if self.center is not None and self.size is not None:
            return Box(center=self.center, size=self.size)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the flat row into a serializable dictionary.

        If a box can be resolved, a serialized ``"box"`` entry is included.

        :returns:
            Dictionary representation of the docking row.
        :rtype: Dict[str, Any]
        """
        out = asdict(self)
        box = self.resolved_box()
        if box is not None:
            out["box"] = box.to_dict()
        return out


@dataclass
class BatchConfig:
    """Batch configuration supporting both flat and receptor-centric layouts."""

    engine: Optional[str] = None
    n_jobs: int = 1
    progress: bool = True
    default_retries: int = 1
    timeout: Optional[float] = None
    tmp_root: Optional[str] = None
    out_dir: Optional[str] = None
    log_dir: Optional[str] = None
    engine_options: Dict[str, Any] = field(default_factory=dict)
    exhaustiveness: Optional[int] = None
    n_poses: Optional[int] = None
    cpu: Optional[int] = None
    seed: Optional[int] = None
    autobox_ref_key: Optional[str] = None
    autobox_pad: Optional[float] = None
    retries: Optional[int] = None
    rows: List[DockRow] = field(default_factory=list)
    receptors: List[ReceptorSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchConfig":
        """
        Build a :class:`BatchConfig` from a mapping.

        Supported aliases:

        - ``"ligands"`` is accepted as an alias for ``"rows"``,
        - ``"receptors"`` contains receptor-centric blocks.

        :param data:
            Batch configuration mapping.
        :type data: Dict[str, Any]

        :returns:
            Parsed batch configuration.
        :rtype: BatchConfig

        Example
        -------
        .. code-block:: python

            batch = BatchConfig.from_dict(
                {
                    "engine": "vina",
                    "rows": [
                        {
                            "id": "job1",
                            "receptor": "protein.pdbqt",
                            "ligand": "ligand.pdbqt",
                        }
                    ],
                }
            )
        """
        d = dict(data)

        raw_rows = d.pop("rows", None)
        if raw_rows is None:
            raw_rows = d.pop("ligands", [])

        raw_receptors = d.pop("receptors", [])

        d["rows"] = [
            item if isinstance(item, DockRow) else DockRow.from_dict(item)
            for item in (raw_rows or [])
        ]
        d["receptors"] = [
            item if isinstance(item, ReceptorSpec) else ReceptorSpec.from_dict(item)
            for item in (raw_receptors or [])
        ]
        return cls(**d)

    @classmethod
    def from_file(cls, path: JsonPath) -> "BatchConfig":
        """
        Load a :class:`BatchConfig` from a JSON or YAML file.

        :param path:
            Configuration file path.
        :type path: JsonPath

        :returns:
            Parsed batch configuration.
        :rtype: BatchConfig

        :raises FileNotFoundError:
            If the file does not exist.
        :raises RuntimeError:
            If a YAML file is requested without PyYAML installed.
        :raises TypeError:
            If the configuration root is not a mapping.

        Example
        -------
        .. code-block:: python

            batch = BatchConfig.from_file("batch.json")
        """
        return cls.from_dict(_load_mapping(path))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the batch configuration into a serializable dictionary.

        :returns:
            Dictionary representation of the full batch configuration.
        :rtype: Dict[str, Any]
        """
        out = asdict(self)
        out["rows"] = [row.to_dict() for row in self.rows]
        out["receptors"] = [rec.to_dict() for rec in self.receptors]
        return out


CampaignConfig = BatchConfig
LigandTask = LigandSpec


def _load_mapping(path: JsonPath) -> Dict[str, Any]:
    """
    Load a configuration mapping from JSON or YAML.

    File format is inferred from the filename suffix. Files ending in
    ``.yaml`` or ``.yml`` are parsed with PyYAML; all other files are parsed
    as JSON.

    :param path:
        Path to the configuration file.
    :type path: JsonPath

    :returns:
        Parsed top-level mapping.
    :rtype: Dict[str, Any]

    :raises FileNotFoundError:
        If the file does not exist.
    :raises RuntimeError:
        If YAML input is requested but PyYAML is not installed.
    :raises TypeError:
        If the parsed root object is not a dictionary.

    Example
    -------
    .. code-block:: python

        data = _load_mapping("campaign.json")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is required to load YAML configuration files")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("Configuration root must be a mapping")
    return data
