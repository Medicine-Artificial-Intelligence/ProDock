"""
prodock.dock.campaign
=====================

Utilities to build, validate, serialize, and inspect receptor-centric docking
campaigns.

This module provides a lightweight hierarchy for multi-receptor,
multi-engine, multi-ligand docking workflows:

- :class:`LigandSpec` stores ligand identifiers and file paths.
- :class:`BoxSpec` stores docking box center and size.
- :class:`SoftwareSpec` stores per-engine parameters and ligand lists.
- :class:`ReceptorSpec` stores per-receptor receptor files, box metadata, and
  software blocks.
- :class:`Campaign` stores the full campaign plus a working directory used to
  derive receptor-local output locations at runtime.

The JSON representation is intentionally minimal and stable. It stores only
the core campaign inputs and does **not** persist derived runtime paths such as
``root_dir``, ``out_dir``, or ``log_dir``. Those are computed on demand from
``working_dir`` and receptor id so downstream code is not affected by extra
JSON fields.

Hierarchy
---------
The campaign is organized as::

    receptor -> software -> ligand

Runtime output layout
---------------------
For a receptor with ``id="4KWQ"`` and ``working_dir="demo"``, runtime paths are
derived as::

    demo/4KWQ/
    demo/4KWQ/results/docked/
    demo/4KWQ/results/logs/

Example
-------
Basic usage::

    from pathlib import Path
    from prodock.dock.campaign import Campaign

    workdir = Path("demo")

    campaign = Campaign.from_shared_ligand_dir(
        working_dir=workdir,
        pdb_ids=["4KWQ", "1M17"],
        receptors=[
            workdir / "4KWQ" / "filtered_protein" / "4KWQ.pdbqt",
            workdir / "1M17" / "filtered_protein" / "1M17.pdbqt",
        ],
        boxes=[
        ((2.865, 193.257, 21.367), (27.091, 27.091, 27.091)),
        ((21.623, 0.4, 52.467), (34.07, 34.07, 34.07)),
        ],
        engines=["vina", "smina", "qvina"],
        ligand_dir=workdir / "ligands",
        cpu=4,
        seed=42,
        exhaustiveness=16,
        n_poses=20,
    )

    campaign.ensure_receptor_dirs()
    campaign.save_json(workdir / "campaign.json")

    for job in campaign.iter_jobs():
        print(job)

A single yielded job has the form::

    (
        receptor_id,
        receptor_path,
        out_dir,
        log_dir,
        engine_name,
        ligand_id,
        ligand_path,
    )
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]
Vec3 = Tuple[float, float, float]
CenterSize = Tuple[Sequence[float], Sequence[float]]


@dataclass
class LigandSpec:
    """
    Ligand definition used in a docking campaign.

    :param id:
        Ligand identifier, usually derived from the filename stem.
    :type id: str

    :param ligand:
        Absolute or relative path to the ligand structure file.
    :type ligand: str
    """

    id: str
    ligand: str

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging.

        :returns:
            String representation.
        :rtype: str
        """
        return f"LigandSpec(id={self.id!r}, ligand={self.ligand!r})"


@dataclass
class BoxSpec:
    """
    Docking box definition.

    :param center:
        Box center as ``(x, y, z)``.
    :type center: Tuple[float, float, float]

    :param size:
        Box size as ``(sx, sy, sz)``.
    :type size: Tuple[float, float, float]
    """

    center: Vec3
    size: Vec3

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging.

        :returns:
            String representation.
        :rtype: str
        """
        return f"BoxSpec(center={self.center!r}, size={self.size!r})"


@dataclass
class SoftwareSpec:
    """
    Docking engine configuration for one receptor.

    :param name:
        Engine name, e.g. ``"vina"``, ``"smina"``, ``"qvina"``, ``"gnina"``.
    :type name: str

    :param cpu:
        Number of CPU cores to use.
    :type cpu: int

    :param seed:
        Random seed.
    :type seed: int

    :param exhaustiveness:
        Search exhaustiveness.
    :type exhaustiveness: int

    :param n_poses:
        Number of poses to write.
    :type n_poses: int

    :param ligands:
        Ligands to dock for this engine.
    :type ligands: List[LigandSpec]

    :param extra:
        Optional extra engine-specific parameters.
    :type extra: Dict[str, Any]
    """

    name: str
    cpu: int = 4
    seed: int = 42
    exhaustiveness: int = 16
    n_poses: int = 20
    ligands: List[LigandSpec] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging.

        :returns:
            String representation.
        :rtype: str
        """
        return (
            f"SoftwareSpec(name={self.name!r}, cpu={self.cpu!r}, seed={self.seed!r}, "
            f"exhaustiveness={self.exhaustiveness!r}, n_poses={self.n_poses!r}, "
            f"n_ligands={len(self.ligands)!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the software specification to a JSON-serializable dictionary.

        The ``extra`` field is flattened into the returned dictionary so that
        engine-specific parameters appear at the same level as the standard
        engine fields.

        :returns:
            Dictionary representation.
        :rtype: Dict[str, Any]
        """
        data: Dict[str, Any] = {
            "name": self.name,
            "cpu": self.cpu,
            "seed": self.seed,
            "exhaustiveness": self.exhaustiveness,
            "n_poses": self.n_poses,
            "ligands": [asdict(lig) for lig in self.ligands],
        }
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SoftwareSpec":
        """
        Build a :class:`SoftwareSpec` from a dictionary.

        All keys not part of the standard engine schema are placed into the
        ``extra`` field.

        :param data:
            Dictionary containing software fields.
        :type data: Mapping[str, Any]

        :returns:
            Parsed software specification.
        :rtype: SoftwareSpec
        """
        base_keys = {"name", "cpu", "seed", "exhaustiveness", "n_poses", "ligands"}
        ligands = [LigandSpec(**lig) for lig in data.get("ligands", [])]
        extra = {k: v for k, v in data.items() if k not in base_keys}
        return cls(
            name=str(data["name"]),
            cpu=int(data.get("cpu", 4)),
            seed=int(data.get("seed", 42)),
            exhaustiveness=int(data.get("exhaustiveness", 16)),
            n_poses=int(data.get("n_poses", 20)),
            ligands=ligands,
            extra=extra,
        )


@dataclass
class ReceptorSpec:
    """
    Receptor-level campaign configuration.

    :param id:
        Receptor identifier, e.g. ``"4KWQ"``.
    :type id: str

    :param receptor:
        Path to the prepared receptor structure, typically a PDBQT file.
    :type receptor: str

    :param box:
        Docking box definition.
    :type box: BoxSpec

    :param out_dir:
        Receptor-level docking output directory to persist in the campaign.
    :type out_dir: str

    :param log_dir:
        Receptor-level log directory to persist in the campaign.
    :type log_dir: str

    :param softwares:
        List of engine definitions to apply to this receptor.
    :type softwares: List[SoftwareSpec]
    """

    id: str
    receptor: str
    box: BoxSpec
    out_dir: str
    log_dir: str
    softwares: List[SoftwareSpec] = field(default_factory=list)

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging.

        :returns:
            String representation.
        :rtype: str
        """
        return (
            f"ReceptorSpec(id={self.id!r}, receptor={self.receptor!r}, "
            f"out_dir={self.out_dir!r}, log_dir={self.log_dir!r}, "
            f"n_softwares={len(self.softwares)!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the receptor specification to a JSON-serializable dictionary.

        :returns:
            Dictionary representation.
        :rtype: Dict[str, Any]
        """
        return {
            "id": self.id,
            "receptor": self.receptor,
            "box": asdict(self.box),
            "out_dir": self.out_dir,
            "log_dir": self.log_dir,
            "softwares": [sw.to_dict() for sw in self.softwares],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReceptorSpec":
        """
        Build a :class:`ReceptorSpec` from a dictionary.

        :param data:
            Dictionary containing receptor fields.
        :type data: Mapping[str, Any]

        :returns:
            Parsed receptor specification.
        :rtype: ReceptorSpec
        """
        box_data = data["box"]
        box = BoxSpec(
            center=tuple(float(x) for x in box_data["center"]),  # type: ignore[arg-type]
            size=tuple(float(x) for x in box_data["size"]),  # type: ignore[arg-type]
        )
        softwares = [SoftwareSpec.from_dict(sw) for sw in data.get("softwares", [])]
        return cls(
            id=str(data["id"]),
            receptor=str(data["receptor"]),
            box=box,
            out_dir=str(data["out_dir"]),
            log_dir=str(data["log_dir"]),
            softwares=softwares,
        )


@dataclass
class Campaign:
    """
    Full docking campaign container.

    The generated JSON follows the receptor-first hierarchy::

        receptor -> software -> ligand

    The persisted campaign stores stable receptor-level runtime directories
    ``out_dir`` and ``log_dir``. The optional ``working_dir`` remains runtime
    context only and is not written to JSON.

    Example JSON layout::

        {
          "receptors": [
            {
              "id": "4KWQ",
              "receptor": "/abs/path/demo/4KWQ/filtered_protein/4KWQ.pdbqt",
              "box": {
                "center": [1.0, 2.0, 3.0],
                "size": [20.0, 20.0, 20.0]
              },
              "out_dir": "/abs/path/demo/4KWQ/results/docked",
              "log_dir": "/abs/path/demo/4KWQ/results/logs",
              "softwares": [
                {
                  "name": "vina",
                  "cpu": 4,
                  "seed": 42,
                  "exhaustiveness": 16,
                  "n_poses": 20,
                  "ligands": [
                    {
                      "id": "erlotinib",
                      "ligand": "/abs/path/demo/ligands/erlotinib.pdbqt"
                    }
                  ]
                }
              ]
            }
          ]
        }

    :param receptors:
        Receptor-level specifications.
    :type receptors: List[ReceptorSpec]

    :param working_dir:
        Optional runtime working directory. This value is not written to JSON.
    :type working_dir: Optional[str]
    """

    receptors: List[ReceptorSpec] = field(default_factory=list)
    working_dir: Optional[str] = None

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging.

        :returns:
            String representation.
        :rtype: str
        """
        return (
            f"Campaign(working_dir={self.working_dir!r}, "
            f"n_receptors={len(self.receptors)!r})"
        )

    @property
    def working_path(self) -> Optional[Path]:
        """
        Return the runtime working directory as a :class:`pathlib.Path` if bound.

        :returns:
            Working directory path or ``None``.
        :rtype: Optional[Path]
        """
        return Path(self.working_dir) if self.working_dir else None

    @staticmethod
    def _resolve_path(path: PathLike, *, absolute: bool = True) -> str:
        """
        Normalize a filesystem path.

        :param path:
            Input path.
        :type path: PathLike

        :param absolute:
            Whether to resolve to an absolute path.
        :type absolute: bool

        :returns:
            Normalized path string.
        :rtype: str
        """
        p = Path(path)
        return str(p.resolve()) if absolute else str(p)

    @staticmethod
    def _normalize_vec3(values: Sequence[float], name: str) -> Vec3:
        """
        Normalize a 3D vector.

        :param values:
            Input coordinate triplet.
        :type values: Sequence[float]

        :param name:
            Field name for error messages.
        :type name: str

        :returns:
            Normalized 3-tuple of floats.
        :rtype: Tuple[float, float, float]

        :raises ValueError:
            If the input does not contain exactly three values.
        """
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly 3 values, got {values!r}")
        return (float(values[0]), float(values[1]), float(values[2]))

    def with_working_dir(self, working_dir: PathLike) -> "Campaign":
        """
        Bind a runtime working directory to the campaign.

        This does not alter persisted receptor-level ``out_dir`` and ``log_dir``.

        :param working_dir:
            Runtime working directory.
        :type working_dir: PathLike

        :returns:
            The same campaign instance.
        :rtype: Campaign
        """
        self.working_dir = str(Path(working_dir))
        return self

    @classmethod
    def scan_ligands(
        cls,
        ligand_dir: PathLike,
        *,
        pattern: str = "*.pdbqt",
        absolute: bool = True,
        recursive: bool = False,
    ) -> List[LigandSpec]:
        """
        Scan a folder and collect ligand files.

        The ligand identifier is derived from the filename stem.

        :param ligand_dir:
            Folder containing ligand files.
        :type ligand_dir: PathLike

        :param pattern:
            Glob pattern used to collect ligand files.
        :type pattern: str

        :param absolute:
            Whether to store absolute ligand paths.
        :type absolute: bool

        :param recursive:
            Whether to scan recursively.
        :type recursive: bool

        :returns:
            List of ligand specifications.
        :rtype: List[LigandSpec]

        :raises FileNotFoundError:
            If the ligand folder does not exist.

        :raises NotADirectoryError:
            If the supplied path is not a directory.

        :raises ValueError:
            If no matching ligand files are found.
        """
        root = Path(ligand_dir)
        if not root.exists():
            raise FileNotFoundError(f"Ligand folder does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Ligand path is not a directory: {root}")

        paths = root.rglob(pattern) if recursive else root.glob(pattern)

        ligands: List[LigandSpec] = []
        for fp in sorted(paths):
            if not fp.is_file():
                continue
            ligands.append(
                LigandSpec(
                    id=fp.stem,
                    ligand=cls._resolve_path(fp, absolute=absolute),
                )
            )

        if not ligands:
            raise ValueError(f"No ligand files matching '{pattern}' found in: {root}")

        return ligands

    @classmethod
    def from_lists(
        cls,
        working_dir: PathLike,
        pdb_ids: Sequence[str],
        receptors: Sequence[PathLike],
        boxes: Sequence[CenterSize],
        engines: Sequence[str],
        lig_paths: Sequence[PathLike],
        *,
        cpu: int = 4,
        seed: int = 42,
        exhaustiveness: int = 16,
        n_poses: int = 20,
        engine_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        ligand_pattern: str = "*.pdbqt",
        absolute_paths: bool = True,
        recursive_ligands: bool = False,
        create_receptor_dirs: bool = True,
        check_receptor_files: bool = True,
        check_ligand_files: bool = True,
    ) -> "Campaign":
        """
        Build a campaign from parallel receptor-level lists.

        Receptor-level ``out_dir`` and ``log_dir`` are computed from
        ``working_dir`` and persisted into the campaign.

        :param working_dir:
            Top-level working directory containing receptor folders.
        :type working_dir: PathLike

        :param pdb_ids:
            Receptor identifiers.
        :type pdb_ids: Sequence[str]

        :param receptors:
            Paths to receptor files.
        :type receptors: Sequence[PathLike]

        :param boxes:
            Per-receptor box definitions.
        :type boxes: Sequence[CenterSize]

        :param engines:
            Engines to assign to each receptor.
        :type engines: Sequence[str]

        :param lig_paths:
            Per-receptor ligand folders.
        :type lig_paths: Sequence[PathLike]

        :param cpu:
            Default CPU count for all engines.
        :type cpu: int

        :param seed:
            Default random seed for all engines.
        :type seed: int

        :param exhaustiveness:
            Default search exhaustiveness for all engines.
        :type exhaustiveness: int

        :param n_poses:
            Default number of poses for all engines.
        :type n_poses: int

        :param engine_overrides:
            Optional per-engine override dictionary.
        :type engine_overrides: Optional[Mapping[str, Mapping[str, Any]]]

        :param ligand_pattern:
            Glob pattern used to collect ligand files.
        :type ligand_pattern: str

        :param absolute_paths:
            Whether to resolve receptor and ligand paths to absolute paths.
        :type absolute_paths: bool

        :param recursive_ligands:
            Whether ligand folders should be scanned recursively.
        :type recursive_ligands: bool

        :param create_receptor_dirs:
            Whether to create missing receptor root directories under
            ``working_dir``.
        :type create_receptor_dirs: bool

        :param check_receptor_files:
            Whether receptor structure files must already exist.
        :type check_receptor_files: bool

        :param check_ligand_files:
            Whether collected ligand files must already exist.
        :type check_ligand_files: bool

        :returns:
            Constructed campaign object.
        :rtype: Campaign
        """
        n = len(pdb_ids)
        if len(receptors) != n:
            raise ValueError("pdb_ids and receptors must have the same length")
        if len(boxes) != n:
            raise ValueError("pdb_ids and boxes must have the same length")
        if len(lig_paths) != n:
            raise ValueError("pdb_ids and lig_paths must have the same length")
        if not engines:
            raise ValueError("At least one engine must be provided")

        workdir_path = Path(working_dir)
        campaign = cls(
            receptors=[],
            working_dir=str(workdir_path.resolve() if absolute_paths else workdir_path),
        )
        engine_overrides = dict(engine_overrides or {})
        receptor_specs: List[ReceptorSpec] = []

        for pdb_id, receptor_path, box_entry, lig_dir in zip(
            pdb_ids,
            receptors,
            boxes,
            lig_paths,
        ):
            receptor_id = str(pdb_id)
            receptor_root = workdir_path / receptor_id

            if receptor_root.exists():
                if not receptor_root.is_dir():
                    raise NotADirectoryError(
                        f"Expected receptor root to be a directory: {receptor_root}"
                    )
            elif create_receptor_dirs:
                receptor_root.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"Expected receptor folder does not exist: {receptor_root}"
                )

            receptor_file = Path(receptor_path)
            if check_receptor_files and not receptor_file.is_file():
                raise FileNotFoundError(
                    f"Receptor file not found for '{receptor_id}': {receptor_file}"
                )

            center_raw, size_raw = box_entry
            box = BoxSpec(
                center=cls._normalize_vec3(center_raw, f"box.center for {receptor_id}"),
                size=cls._normalize_vec3(size_raw, f"box.size for {receptor_id}"),
            )

            ligands = cls.scan_ligands(
                lig_dir,
                pattern=ligand_pattern,
                absolute=absolute_paths,
                recursive=recursive_ligands,
            )

            if check_ligand_files:
                missing_ligands = [
                    lig.ligand for lig in ligands if not Path(lig.ligand).is_file()
                ]
                if missing_ligands:
                    raise FileNotFoundError(
                        f"Some ligand files do not exist for receptor "
                        f"'{receptor_id}': {missing_ligands[:5]!r}"
                    )

            softwares: List[SoftwareSpec] = []
            for engine in engines:
                overrides = dict(engine_overrides.get(engine, {}))
                software = SoftwareSpec(
                    name=str(engine),
                    cpu=int(overrides.pop("cpu", cpu)),
                    seed=int(overrides.pop("seed", seed)),
                    exhaustiveness=int(overrides.pop("exhaustiveness", exhaustiveness)),
                    n_poses=int(overrides.pop("n_poses", n_poses)),
                    ligands=list(ligands),
                    extra=overrides,
                )
                softwares.append(software)

            out_path = receptor_root / "results" / "docked"
            log_path = receptor_root / "results" / "logs"

            receptor_specs.append(
                ReceptorSpec(
                    id=receptor_id,
                    receptor=cls._resolve_path(receptor_file, absolute=absolute_paths),
                    box=box,
                    out_dir=cls._resolve_path(out_path, absolute=absolute_paths),
                    log_dir=cls._resolve_path(log_path, absolute=absolute_paths),
                    softwares=softwares,
                )
            )

        campaign.receptors = receptor_specs
        campaign.validate(
            check_receptor_files=check_receptor_files,
            check_ligand_files=check_ligand_files,
        )
        return campaign

    @classmethod
    def from_shared_ligand_dir(
        cls,
        working_dir: PathLike,
        pdb_ids: Sequence[str],
        receptors: Sequence[PathLike],
        boxes: Sequence[CenterSize],
        engines: Sequence[str],
        ligand_dir: PathLike,
        **kwargs: Any,
    ) -> "Campaign":
        """
        Build a campaign when all receptors share the same ligand folder.

        :param working_dir:
            Top-level working directory containing receptor folders.
        :type working_dir: PathLike

        :param pdb_ids:
            Receptor identifiers.
        :type pdb_ids: Sequence[str]

        :param receptors:
            Receptor paths.
        :type receptors: Sequence[PathLike]

        :param boxes:
            Box definitions.
        :type boxes: Sequence[CenterSize]

        :param engines:
            Engine names.
        :type engines: Sequence[str]

        :param ligand_dir:
            Single ligand folder shared by all receptors.
        :type ligand_dir: PathLike

        :param kwargs:
            Additional keyword arguments forwarded to :meth:`from_lists`.
        :type kwargs: Any

        :returns:
            Constructed campaign object.
        :rtype: Campaign
        """
        lig_paths = [ligand_dir] * len(pdb_ids)
        return cls.from_lists(
            working_dir=working_dir,
            pdb_ids=pdb_ids,
            receptors=receptors,
            boxes=boxes,
            engines=engines,
            lig_paths=lig_paths,
            **kwargs,
        )

    def validate(
        self,
        *,
        check_receptor_files: bool = False,
        check_ligand_files: bool = False,
    ) -> None:
        """
        Validate the campaign contents.

        This validates the in-memory campaign structure. It can optionally check
        that receptor and ligand files exist.

        :param check_receptor_files:
            Whether receptor files must exist on disk.
        :type check_receptor_files: bool

        :param check_ligand_files:
            Whether ligand files must exist on disk.
        :type check_ligand_files: bool

        :raises ValueError:
            If required fields are missing or malformed.

        :raises FileNotFoundError:
            If requested file checks fail.
        """
        if not self.receptors:
            raise ValueError("Campaign contains no receptors")

        for receptor in self.receptors:
            if not receptor.id:
                raise ValueError("Receptor id cannot be empty")
            if not receptor.receptor:
                raise ValueError(f"Receptor path missing for receptor '{receptor.id}'")
            if not receptor.out_dir:
                raise ValueError(f"out_dir missing for receptor '{receptor.id}'")
            if not receptor.log_dir:
                raise ValueError(f"log_dir missing for receptor '{receptor.id}'")
            if len(receptor.softwares) == 0:
                raise ValueError(
                    f"Receptor '{receptor.id}' does not contain any software entries"
                )

            if check_receptor_files and not Path(receptor.receptor).is_file():
                raise FileNotFoundError(
                    f"Receptor file not found for '{receptor.id}': {receptor.receptor}"
                )

            for sw in receptor.softwares:
                if not sw.name:
                    raise ValueError(
                        f"Engine name cannot be empty in receptor '{receptor.id}'"
                    )
                if len(sw.ligands) == 0:
                    raise ValueError(
                        f"Receptor '{receptor.id}' / engine '{sw.name}' "
                        "does not contain any ligands"
                    )

                for lig in sw.ligands:
                    if not lig.id:
                        raise ValueError(
                            f"Empty ligand id in receptor '{receptor.id}' / "
                            f"engine '{sw.name}'"
                        )
                    if not lig.ligand:
                        raise ValueError(
                            f"Empty ligand path for ligand '{lig.id}' in receptor "
                            f"'{receptor.id}' / engine '{sw.name}'"
                        )
                    if check_ligand_files and not Path(lig.ligand).is_file():
                        raise FileNotFoundError(
                            f"Ligand file not found for receptor '{receptor.id}', "
                            f"engine '{sw.name}', ligand '{lig.id}': {lig.ligand}"
                        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the campaign to a JSON-serializable dictionary.

        ``working_dir`` is not persisted. Receptor-level ``out_dir`` and
        ``log_dir`` are persisted.

        :returns:
            Dictionary representation.
        :rtype: Dict[str, Any]
        """
        return {
            "receptors": [rec.to_dict() for rec in self.receptors],
        }

    def save_json(self, out_json: PathLike, *, indent: int = 2) -> Path:
        """
        Save the campaign to a JSON file.

        :param out_json:
            Output JSON path.
        :type out_json: PathLike

        :param indent:
            JSON indentation level.
        :type indent: int

        :returns:
            Written JSON path.
        :rtype: Path
        """
        self.validate()
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.to_dict(), indent=indent),
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        working_dir: Optional[PathLike] = None,
    ) -> "Campaign":
        """
        Build a campaign from a dictionary.

        :param data:
            Campaign dictionary.
        :type data: Mapping[str, Any]

        :param working_dir:
            Optional runtime working directory to bind after loading.
            This does not overwrite persisted receptor-level ``out_dir`` and
            ``log_dir``.
        :type working_dir: Optional[PathLike]

        :returns:
            Parsed campaign object.
        :rtype: Campaign
        """
        receptors = [ReceptorSpec.from_dict(rec) for rec in data.get("receptors", [])]
        campaign = cls(
            receptors=receptors,
            working_dir=str(Path(working_dir)) if working_dir is not None else None,
        )
        campaign.validate()
        return campaign

    @classmethod
    def load_json(
        cls,
        json_path: PathLike,
        *,
        working_dir: Optional[PathLike] = None,
    ) -> "Campaign":
        """
        Load a campaign from a JSON file.

        :param json_path:
            Path to the JSON file.
        :type json_path: PathLike

        :param working_dir:
            Optional runtime working directory to bind after loading.
        :type working_dir: Optional[PathLike]

        :returns:
            Parsed campaign object.
        :rtype: Campaign
        """
        path = Path(json_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, working_dir=working_dir)

    def ensure_receptor_dirs(
        self,
        *,
        exist_ok: bool = True,
        engine_subdirs: bool = False,
    ) -> None:
        """
        Ensure persisted receptor-local directories exist.

        This always ensures the receptor root directory itself exists, inferred
        from the parent hierarchy of ``out_dir`` and ``log_dir``. It then
        creates:

        - ``receptor.out_dir``
        - ``receptor.log_dir``

        If ``engine_subdirs=True``, it also creates per-engine subdirectories
        beneath persisted ``out_dir`` and ``log_dir``.

        :param exist_ok:
            Whether existing directories should be accepted.
        :type exist_ok: bool

        :param engine_subdirs:
            Whether to create per-engine directories beneath ``out_dir`` and
            ``log_dir``.
        :type engine_subdirs: bool
        """
        for receptor in self.receptors:
            out_dir = Path(receptor.out_dir)
            log_dir = Path(receptor.log_dir)

            out_dir.mkdir(parents=True, exist_ok=exist_ok)
            log_dir.mkdir(parents=True, exist_ok=exist_ok)

            receptor_root_candidates = [out_dir.parent.parent, log_dir.parent.parent]
            for root in receptor_root_candidates:
                root.mkdir(parents=True, exist_ok=exist_ok)

            if engine_subdirs:
                for sw in receptor.softwares:
                    (out_dir / sw.name).mkdir(parents=True, exist_ok=exist_ok)
                    (log_dir / sw.name).mkdir(parents=True, exist_ok=exist_ok)

    def iter_jobs(self) -> Iterator[Tuple[str, str, str, str, str, str, str]]:
        """
        Iterate over campaign jobs.

        Each yielded tuple is::

            (
                receptor_id,
                receptor_path,
                out_dir,
                log_dir,
                engine_name,
                ligand_id,
                ligand_path,
            )

        The output and log directories are read from persisted receptor-level
        fields rather than recomputed from ``working_dir``.

        :returns:
            Iterator over flattened jobs.
        :rtype: Iterator[Tuple[str, str, str, str, str, str, str]]
        """
        for receptor in self.receptors:
            for sw in receptor.softwares:
                for lig in sw.ligands:
                    yield (
                        receptor.id,
                        receptor.receptor,
                        receptor.out_dir,
                        receptor.log_dir,
                        sw.name,
                        lig.id,
                        lig.ligand,
                    )

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact campaign summary.

        :returns:
            Summary dictionary.
        :rtype: Dict[str, Any]
        """
        receptor_count = len(self.receptors)
        engine_count = sum(len(r.softwares) for r in self.receptors)
        ligand_jobs = sum(len(sw.ligands) for r in self.receptors for sw in r.softwares)

        return {
            "working_dir": self.working_dir,
            "n_receptors": receptor_count,
            "n_engine_blocks": engine_count,
            "n_jobs": ligand_jobs,
            "receptors": [r.id for r in self.receptors],
            "out_dirs": {r.id: r.out_dir for r in self.receptors},
            "log_dirs": {r.id: r.log_dir for r in self.receptors},
        }
