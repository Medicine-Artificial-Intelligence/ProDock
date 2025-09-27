from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def _tuplize(
    v: Optional[Union[List[float], Tuple[float, ...]]],
) -> Optional[Tuple[float, ...]]:
    if v is None:
        return None
    if isinstance(v, tuple):
        return v
    if isinstance(v, list):
        return tuple(float(x) for x in v)
    raise TypeError("Expected list/tuple for vector fields")


@dataclass
class Box:
    """
    Simple box container for center and size.
    """

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]

    @classmethod
    def from_mapping(
        cls, m: Union[Dict[str, Any], List[Any], Tuple[Any, ...]]
    ) -> "Box":
        if isinstance(m, dict):
            c = _tuplize(m.get("center"))
            s = _tuplize(m.get("size"))
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            c = _tuplize(m[0])
            s = _tuplize(m[1])
        else:
            raise TypeError(
                "Box must be dict with 'center' and 'size' or [center, size]"
            )
        if c is None or s is None:
            raise ValueError("Both center and size must be provided for Box")
        return cls(center=tuple(c), size=tuple(s))


@dataclass
class SingleConfig:
    """
    Dataclass configuration for a single docking job (SingleDock).
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
        d = dict(data)  # shallow copy
        box = d.get("box")
        if box is not None:
            d["box"] = Box.from_mapping(box)
        return cls(**d)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SingleConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML required to load YAML config files")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if self.box is not None:
            out["box"] = {"center": list(self.box.center), "size": list(self.box.size)}
        return out


@dataclass
class LigandTask:
    """
    Per-ligand entry for BatchConfig rows.
    """

    id: str
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
    engine_options: Dict[str, Any] = field(default_factory=dict)
    retries: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LigandTask":
        d = dict(data)
        box = d.get("box")
        if box is not None:
            d["box"] = Box.from_mapping(box)
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if self.box is not None:
            out["box"] = {"center": list(self.box.center), "size": list(self.box.size)}
        return out


@dataclass
class BatchConfig:
    """
    Dataclass configuration for BatchDock.
    """

    engine: str = "vina"
    engine_mode: Optional[str] = None
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
    rows: List[LigandTask] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchConfig":
        d = dict(data)
        rows = d.get("rows") or d.get("ligands") or []
        d["rows"] = [
            LigandTask.from_dict(r) if not isinstance(r, LigandTask) else r
            for r in rows
        ]
        return cls(**d)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BatchConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML required to load YAML config files")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["rows"] = [r.to_dict() for r in self.rows]
        return out
