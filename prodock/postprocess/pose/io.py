from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .convert import pdbqt_to_rdkit_mols, save_pose_sdf
from .record import PoseRecord

PathLike = str | Path

_MODEL_RE = re.compile(r"^MODEL\s+(\d+)\s*$", re.MULTILINE)
_VINA_RESULT_RE = re.compile(
    r"^REMARK\s+VINA\s+RESULT:\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)",
    re.MULTILINE,
)
_SMINA_AFF_RE = re.compile(
    r"^REMARK\s+minimizedAffinity\s+([-+]?\d*\.?\d+)",
    re.MULTILINE,
)
_MODE_TABLE_RE = re.compile(
    r"^\s*(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*$",
    re.MULTILINE,
)


def _as_path(value: PathLike) -> Path:
    """
    Convert a path-like value to :class:`pathlib.Path`.

    This helper normalizes user-supplied string paths and existing
    :class:`pathlib.Path` objects into a single concrete type used internally by
    the pose I/O utilities.

    :param value:
        Input path to normalize.
    :type value: str | pathlib.Path

    :returns:
        The normalized path object.
    :rtype: pathlib.Path

    Example
    -------
    .. code-block:: python

        path = _as_path("results/docked/vina/ligand.pdbqt")
        assert isinstance(path, Path)
    """
    return value if isinstance(value, Path) else Path(value)


def _to_float(value: str | None) -> float | None:
    """
    Safely convert numeric text into a floating-point value.

    Invalid numeric strings are tolerated and converted to ``None`` rather than
    raising an exception. This is useful when parsing partially malformed
    docking output.

    :param value:
        Numeric text to parse. ``None`` is accepted and returned unchanged.
    :type value: str | None

    :returns:
        Parsed floating-point value, or ``None`` when conversion fails.
    :rtype: float | None

    Example
    -------
    .. code-block:: python

        score = _to_float("-7.5")
        missing = _to_float("not-a-number")
    """
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _strip_ligand_suffix(stem: str) -> str:
    """
    Normalize a pose-file stem into a ligand identifier.

    Several common output suffixes produced by docking or postprocessing steps
    are removed so that downstream records use a stable ligand id.

    Supported suffixes include:

    - ``_docked``
    - ``_poses``
    - ``_pose``
    - ``_out``

    :param stem:
        File stem without extension.
    :type stem: str

    :returns:
        Normalized ligand identifier.
    :rtype: str

    Example
    -------
    .. code-block:: python

        ligand_id = _strip_ligand_suffix("erlotinib_docked")
        assert ligand_id == "erlotinib"
    """
    for suffix in ("_docked", "_poses", "_pose", "_out"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _infer_hierarchical_metadata(path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Infer receptor id and engine name from a hierarchical ProDock path.

    Supported path layouts include::

        <root>/<receptor>/results/docked/<engine>/<ligand>.pdbqt
        <root>/<receptor>/<engine>/<ligand>.pdbqt

    The first matching pattern is used. When no known hierarchical layout is
    recognized, both returned values are ``None``.

    :param path:
        Pose file path to inspect.
    :type path: pathlib.Path

    :returns:
        A tuple ``(receptor_id, engine)`` when metadata can be inferred;
        otherwise ``(None, None)``.
    :rtype: tuple[Optional[str], Optional[str]]

    Example
    -------
    .. code-block:: python

        receptor_id, engine = _infer_hierarchical_metadata(
            Path("demo/4WKQ/results/docked/vina/erlotinib.pdbqt")
        )
        assert receptor_id == "4WKQ"
        assert engine == "vina"
    """
    parts = path.parts

    for i, token in enumerate(parts):
        if token == "docked" and i >= 2 and i + 1 < len(parts):
            if parts[i - 1] == "results":
                return parts[i - 2], parts[i + 1]

    if len(parts) >= 3:
        return parts[-3], parts[-2]

    return None, None


def _build_records_for_path(
    path: Path,
    *,
    engine_name: str,
    receptor_id: Optional[str],
) -> List[PoseRecord]:
    """
    Build :class:`PoseRecord` objects for a single docked pose file.

    The ligand id is derived from the source filename stem after common docking
    suffix normalization. Pose-level rank and affinity values are obtained from
    :func:`parse_pdbqt_pose_scores`.

    :param path:
        Source ``.pdbqt`` file.
    :type path: pathlib.Path
    :param engine_name:
        Docking engine associated with the file, such as ``"vina"`` or
        ``"smina"``.
    :type engine_name: str
    :param receptor_id:
        Receptor identifier associated with the file. This may be ``None`` when
        the input layout does not encode receptor information.
    :type receptor_id: Optional[str]

    :returns:
        One :class:`PoseRecord` per parsed pose in the input file.
    :rtype: list[prodock.postprocess.pose.model.PoseRecord]

    Example
    -------
    .. code-block:: python

        records = _build_records_for_path(
            Path("demo/4WKQ/results/docked/vina/erlotinib.pdbqt"),
            engine_name="vina",
            receptor_id="4WKQ",
        )
    """
    ligand_id = _strip_ligand_suffix(path.stem)
    return [
        PoseRecord(
            receptor_id=receptor_id,
            ligand_id=ligand_id,
            engine=engine_name,
            pose_rank=int(row["pose_rank"]),
            affinity=row["affinity"],
            source_file=path.resolve(),
        )
        for row in parse_pdbqt_pose_scores(path)
    ]


def discover_pose_files(
    roots: Sequence[PathLike],
    *,
    recursive: bool = True,
    engine: Optional[str] = None,
) -> List[Path]:
    """
    Discover docked ``.pdbqt`` pose files from one or more input roots.

    Supported input scenarios are:

    1. Direct path to a single ``.pdbqt`` file.
    2. Direct path to a flat directory containing ``.pdbqt`` files.
    3. Higher-level ProDock directory trees, such as
       ``<root>/<receptor>/results/docked/<engine>/*.pdbqt``.

    For direct files and flat directories, ``engine`` is required because it
    cannot be inferred from the layout. For hierarchical ProDock layouts,
    receptor id and engine are inferred from the path, and ``engine`` acts as
    an optional filter.

    Duplicate files are removed after path resolution, and the final list is
    returned in deterministic sorted traversal order.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param recursive:
        Whether to recurse into nested directories when a root does not contain
        pose files directly.
    :type recursive: bool
    :param engine:
        Engine hint for direct-file or flat-directory layouts, or an optional
        engine filter for hierarchical layouts.
    :type engine: Optional[str]

    :returns:
        Sorted list of unique resolved ``.pdbqt`` paths.
    :rtype: list[pathlib.Path]

    :raises ValueError:
        If a direct ``.pdbqt`` file or flat directory of ``.pdbqt`` files is
        provided without ``engine=...``.

    Example
    -------
    .. code-block:: python

        files = discover_pose_files(
            ["demo/4WKQ/results/docked"],
            recursive=True,
            engine="vina",
        )
    """
    seen: set[Path] = set()
    ordered: List[Path] = []

    for root in roots:
        path = _as_path(root)

        if path.is_file():
            if path.suffix.lower() != ".pdbqt":
                continue
            if engine is None:
                raise ValueError("A direct .pdbqt file requires 'engine=...'.")
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                ordered.append(resolved)
            continue

        if not path.exists():
            continue

        direct_files = sorted(path.glob("*.pdbqt"))
        if direct_files:
            if engine is None:
                raise ValueError("A flat folder of .pdbqt files requires 'engine=...'.")
            for file_path in direct_files:
                resolved = file_path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    ordered.append(resolved)
            continue

        if not recursive:
            continue

        for file_path in sorted(path.rglob("*.pdbqt")):
            receptor_id, inferred_engine = _infer_hierarchical_metadata(file_path)
            if inferred_engine is None:
                continue
            if engine is not None and inferred_engine.lower() != engine.lower():
                continue
            resolved = file_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                ordered.append(resolved)

    return ordered


def parse_pdbqt_pose_scores(path: PathLike) -> List[Dict[str, Any]]:
    """
    Parse pose-level affinity data from a docked ``.pdbqt`` file.

    The parser supports several common output styles:

    - Vina per-model remarks of the form ``REMARK VINA RESULT: ...``
    - Smina per-model remarks of the form ``REMARK minimizedAffinity ...``
    - Vina-like mode tables without explicit ``MODEL`` blocks

    When explicit ``MODEL`` sections exist, affinity is searched independently
    within each model block. When no recognized score annotation is present, a
    fallback row is returned with ``pose_rank=1`` and ``affinity=None``.

    :param path:
        Input ``.pdbqt`` file to parse.
    :type path: str | pathlib.Path

    :returns:
        A list of dictionaries with keys ``pose_rank`` and ``affinity``.
    :rtype: list[dict[str, Any]]

    Example
    -------
    .. code-block:: python

        rows = parse_pdbqt_pose_scores("erlotinib_docked.pdbqt")
        for row in rows:
            print(row["pose_rank"], row["affinity"])
    """
    text = _as_path(path).read_text(encoding="utf-8", errors="replace")
    models = list(_MODEL_RE.finditer(text))

    if not models:
        vina_hits = list(_VINA_RESULT_RE.finditer(text))
        if vina_hits:
            return [
                {"pose_rank": i, "affinity": _to_float(hit.group(1))}
                for i, hit in enumerate(vina_hits, start=1)
            ]

        smina_hits = list(_SMINA_AFF_RE.finditer(text))
        if smina_hits:
            return [
                {"pose_rank": i, "affinity": _to_float(hit.group(1))}
                for i, hit in enumerate(smina_hits, start=1)
            ]

        table_hits = list(_MODE_TABLE_RE.finditer(text))
        if table_hits:
            return [
                {"pose_rank": int(hit.group(1)), "affinity": _to_float(hit.group(2))}
                for hit in table_hits
            ]

        return [{"pose_rank": 1, "affinity": None}]

    rows: List[Dict[str, Any]] = []
    for i, match in enumerate(models):
        pose_rank = int(match.group(1))
        start = match.start()
        end = models[i + 1].start() if i + 1 < len(models) else len(text)
        block = text[start:end]

        affinity: float | None = None
        vina_hit = _VINA_RESULT_RE.search(block)
        smina_hit = _SMINA_AFF_RE.search(block)

        if vina_hit is not None:
            affinity = _to_float(vina_hit.group(1))
        elif smina_hit is not None:
            affinity = _to_float(smina_hit.group(1))

        rows.append({"pose_rank": pose_rank, "affinity": affinity})

    return rows


def build_pose_records(
    roots: Sequence[PathLike],
    *,
    engine: Optional[str] = None,
    recursive: bool = True,
) -> List[PoseRecord]:
    """
    Discover pose files and build normalized :class:`PoseRecord` entries.

    This function combines file discovery with pose-score extraction and returns
    a flat list of record objects suitable for downstream tabular conversion or
    molecule loading.

    Input handling follows the same layout rules as
    :func:`discover_pose_files`.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engine:
        Engine hint for direct-file or flat-directory layouts, or an optional
        filter for hierarchical layouts.
    :type engine: Optional[str]
    :param recursive:
        Whether to recurse into nested directories when appropriate.
    :type recursive: bool

    :returns:
        Sorted pose records with source file paths preserved.
    :rtype: list[prodock.postprocess.pose.model.PoseRecord]

    :raises ValueError:
        If a direct ``.pdbqt`` file or flat directory of pose files is supplied
        without ``engine=...``.

    Example
    -------
    .. code-block:: python

        records = build_pose_records(
            ["demo/4WKQ/results/docked"],
            engine="vina",
            recursive=True,
        )
    """
    records: List[PoseRecord] = []

    for root in roots:
        path = _as_path(root)

        if path.is_file():
            if path.suffix.lower() != ".pdbqt":
                continue
            if engine is None:
                raise ValueError("A direct .pdbqt file requires 'engine=...'.")
            records.extend(
                _build_records_for_path(
                    path,
                    engine_name=engine,
                    receptor_id=None,
                )
            )
            continue

        if not path.exists():
            continue

        direct_files = sorted(path.glob("*.pdbqt"))
        if direct_files:
            if engine is None:
                raise ValueError("A flat folder of .pdbqt files requires 'engine=...'.")
            for file_path in direct_files:
                records.extend(
                    _build_records_for_path(
                        file_path,
                        engine_name=engine,
                        receptor_id=None,
                    )
                )
            continue

        if not recursive:
            continue

        for file_path in sorted(path.rglob("*.pdbqt")):
            receptor_id, inferred_engine = _infer_hierarchical_metadata(file_path)
            if inferred_engine is None:
                continue
            if engine is not None and inferred_engine.lower() != engine.lower():
                continue
            records.extend(
                _build_records_for_path(
                    file_path,
                    engine_name=inferred_engine,
                    receptor_id=receptor_id,
                )
            )

    records.sort(
        key=lambda rec: (
            rec.receptor_id or "",
            rec.ligand_id,
            rec.engine,
            rec.pose_rank,
            str(rec.source_file),
        )
    )
    return records


def build_pose_mol_rows(
    roots: Sequence[PathLike],
    *,
    engine: Optional[str] = None,
    recursive: bool = True,
    backend: str = "obabel",
    sanitize: bool = True,
    remove_hs: bool = False,
    save_sdf: bool = False,
    overwrite_sdf: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build row dictionaries containing pose metadata and RDKit molecule objects.

    The public row schema is intentionally minimal and stable:

    - ``receptor_id``
    - ``ligand_id``
    - ``engine``
    - ``pose_rank``
    - ``affinity``
    - ``mol``

    Molecules are loaded per source pose file and aligned to pose records by
    pose rank order. If fewer molecules are produced than parsed score rows, the
    unmatched ``mol`` entries are set to ``None``.

    Optionally, an SDF file can be written alongside each source ``.pdbqt`` file
    before RDKit molecule loading.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engine:
        Engine hint for direct-file or flat-directory layouts, or an optional
        filter for hierarchical layouts.
    :type engine: Optional[str]
    :param recursive:
        Whether to recurse into nested directories when appropriate.
    :type recursive: bool
    :param backend:
        Conversion backend passed to molecule-loading utilities.
    :type backend: str
    :param sanitize:
        Whether RDKit sanitization should be applied during import.
    :type sanitize: bool
    :param remove_hs:
        Whether hydrogens should be removed during SDF-based import.
    :type remove_hs: bool
    :param save_sdf:
        Whether to write an SDF file next to each source ``.pdbqt`` file before
        loading molecules.
    :type save_sdf: bool
    :param overwrite_sdf:
        Whether an existing neighboring SDF file may be overwritten.
    :type overwrite_sdf: bool

    :returns:
        A list of row dictionaries suitable for building a DataFrame.
    :rtype: list[dict[str, Any]]

    Example
    -------
    .. code-block:: python

        rows = build_pose_mol_rows(
            ["demo/4WKQ/results/docked/vina"],
            engine="vina",
            save_sdf=True,
            overwrite_sdf=False,
        )
    """
    records = build_pose_records(roots, engine=engine, recursive=recursive)

    grouped: Dict[Path, List[PoseRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.source_file, []).append(rec)

    rows: List[Dict[str, Any]] = []
    for source_file, source_records in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        source_records = sorted(source_records, key=lambda rec: rec.pose_rank)

        if save_sdf:
            save_pose_sdf(
                source_file,
                backend=backend,
                overwrite=overwrite_sdf,
            )

        mols = pdbqt_to_rdkit_mols(
            source_file,
            backend=backend,
            sanitize=sanitize,
            remove_hs=remove_hs,
        )

        for idx, rec in enumerate(source_records):
            mol = mols[idx] if idx < len(mols) else None
            rows.append(
                {
                    "receptor_id": rec.receptor_id,
                    "ligand_id": rec.ligand_id,
                    "engine": rec.engine,
                    "pose_rank": rec.pose_rank,
                    "affinity": rec.affinity,
                    "mol": mol,
                }
            )

    return rows
