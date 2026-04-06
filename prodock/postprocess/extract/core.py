from __future__ import annotations

"""
High-level score extraction utilities for docking log files and tables.

This module supports four main discovery layouts:

- ``auto``: generic recursive crawl with best-effort engine detection
- ``single_file``: parse one ``.log`` or ``.txt`` file with a required engine
- ``flat_dir``: parse a directory of ``.log`` or ``.txt`` files with a required
  engine
- ``engine_tree``: parse a high-level logs directory whose immediate subfolders
  are engine names such as ``vina`` or ``smina``; the folder name is used as the
  engine for all contained log files

The extracted dataframe uses canonical core columns:

- ``ligand_id``
- ``score``
- ``rank``
- ``engine``
- ``source_file``

Additional parsed columns are preserved when relevant, for example:

- ``rmsd_lb``
- ``rmsd_ub``
- ``cnn_pose``
- ``cnn_affinity``
"""

from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Sequence

import pandas as pd

from .engines import (
    canonicalize_engine_name,
    detect_engine,
    detect_engine_from_path,
)
from .normalize import read_text_flexible, safe_parse_file
from .reader import parse_log_text
from .utils import (
    build_engine_pattern,
    is_log_path,
    is_table_path,
    normalize_engine_token,
)

MatchMode = Literal["substring", "exact", "regex"]
LayoutMode = Literal["auto", "single_file", "flat_dir", "engine_tree"]


def _to_float_or_none(x) -> Optional[float]:
    """
    Convert a value to ``float`` when possible.

    :param x:
        Input value to convert.
    :type x: Any

    :returns:
        Floating-point representation of ``x`` when conversion succeeds,
        otherwise ``None``.
    :rtype: float | None

    Example
    -------
    .. code-block:: python

        value1 = _to_float_or_none("-7.5")
        value2 = _to_float_or_none(3)
        value3 = _to_float_or_none("bad")
    """
    try:
        return float(x)
    except Exception:
        return None


def _to_int_or_none(x) -> Optional[int]:
    """
    Convert a value to ``int`` when possible.

    The conversion first casts the input to ``float`` and then to ``int`` so
    values such as ``"2.0"`` can still be normalized to integer ranks.

    :param x:
        Input value to convert.
    :type x: Any

    :returns:
        Integer representation of ``x`` when conversion succeeds,
        otherwise ``None``.
    :rtype: int | None

    Example
    -------
    .. code-block:: python

        rank1 = _to_int_or_none("1")
        rank2 = _to_int_or_none("2.0")
        rank3 = _to_int_or_none(None)
    """
    try:
        return int(float(x))
    except Exception:
        return None


def _read_csv_flexible(path: Path) -> Optional[pd.DataFrame]:
    """
    Read a CSV- or TSV-like file using several fallback encodings.

    Files ending in ``.tsv`` or ``.tab`` are treated as tab-delimited.

    :param path:
        Path to the input table file.
    :type path: pathlib.Path

    :returns:
        Parsed dataframe when successful, otherwise ``None``.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        from pathlib import Path

        df = _read_csv_flexible(Path("scores.csv"))
        if df is not None:
            print(df.head())
    """
    suffix = path.suffix.lower()
    is_tsv = suffix in {".tsv", ".tab"}

    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            if is_tsv:
                return pd.read_csv(path, sep="\t", encoding=enc)
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    try:
        if is_tsv:
            return pd.read_csv(path, sep="\t", encoding="latin-1", engine="python")
        return pd.read_csv(path, encoding="latin-1", engine="python")
    except Exception:
        return None


def _normalize_table_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a score table to canonical column names.

    Supported canonical output columns include:

    - ``ligand_id``
    - ``score``
    - ``rank``
    - ``engine``

    This helper only renames columns. It does not coerce numeric types.

    :param df:
        Input dataframe loaded from a CSV- or TSV-like source.
    :type df: pandas.DataFrame

    :returns:
        Copy of the dataframe with recognized column aliases normalized.
    :rtype: pandas.DataFrame

    Example
    -------
    .. code-block:: python

        import pandas as pd

        raw = pd.DataFrame(
            {
                "Ligand": ["lig1", "lig2"],
                "Affinity": [-7.5, -6.8],
                "Engine": ["vina", "vina"],
            }
        )
        norm = _normalize_table_df(raw)
    """
    t = df.copy()
    colmap: dict[str, str] = {}

    for c in list(t.columns):
        lc = c.lower().strip()

        if (
            lc in {"affinity", "affinity_kcal_mol", "score"}
            and "score" not in t.columns
        ):
            colmap[c] = "score"

        if lc in {"ligand_id", "ligand", "id"} and "ligand_id" not in t.columns:
            colmap[c] = "ligand_id"

        if lc == "rank" and "rank" not in t.columns:
            colmap[c] = "rank"

        if lc == "engine" and "engine" not in t.columns:
            colmap[c] = "engine"

    if colmap:
        t = t.rename(columns=colmap)

    return t


def _finalize_parts(parts: list[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Concatenate parsed dataframe parts and enforce canonical core columns.

    The returned dataframe always contains at least the following columns:

    - ``ligand_id``
    - ``score``
    - ``rank``
    - ``engine``
    - ``source_file``

    :param parts:
        Parsed dataframe fragments collected during crawling.
    :type parts: list[pandas.DataFrame]

    :returns:
        Concatenated and normalized dataframe, or ``None`` when no parts are
        available.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        import pandas as pd

        part1 = pd.DataFrame({"ligand_id": ["lig1"], "score": [-7.5]})
        part2 = pd.DataFrame({"ligand_id": ["lig2"], "score": [-6.8]})
        df = _finalize_parts([part1, part2])
    """
    if not parts:
        return None

    df_all = pd.concat(parts, ignore_index=True, sort=False).astype(object)

    for col in ["ligand_id", "score", "rank", "engine", "source_file"]:
        if col not in df_all.columns:
            df_all[col] = None

    df_all["score"] = df_all["score"].apply(_to_float_or_none)
    df_all["rank"] = df_all["rank"].apply(_to_int_or_none)
    df_all["engine"] = df_all["engine"].fillna("").astype(str)
    df_all["source_file"] = df_all["source_file"].astype(str)

    return df_all.reset_index(drop=True)


def _rows_to_frame(
    path: Path,
    rows_parsed: list[dict],
    engine_used: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Convert parsed log rows into a normalized dataframe.

    The helper maps parser-native fields such as ``affinity_kcal_mol`` and
    ``mode`` to canonical columns ``score`` and ``rank``. Only a limited set of
    additional engine-specific fields are preserved.

    :param path:
        Source log file path.
    :type path: pathlib.Path
    :param rows_parsed:
        List of row dictionaries returned by the log parser.
    :type rows_parsed: list[dict]
    :param engine_used:
        Canonical engine name used for the parsed rows.
    :type engine_used: str | None

    :returns:
        Normalized dataframe for the parsed log rows, or ``None`` if no rows are
        available.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        from pathlib import Path

        rows = [
            {"mode": 1, "affinity_kcal_mol": -8.2, "rmsd_lb": 0.0, "rmsd_ub": 0.0},
            {"mode": 2, "affinity_kcal_mol": -7.5, "rmsd_lb": 1.2, "rmsd_ub": 2.3},
        ]
        df = _rows_to_frame(Path("lig1.log"), rows, "vina")
    """
    if not rows_parsed:
        return None

    stem = path.stem
    out_rows: list[dict] = []
    allowed_extra = {"rmsd_lb", "rmsd_ub", "cnn_pose", "cnn_affinity"}

    for r in rows_parsed:
        row = {
            "ligand_id": stem,
            "score": _to_float_or_none(r.get("affinity_kcal_mol")),
            "rank": _to_int_or_none(r.get("mode")),
            "engine": engine_used or "",
            "source_file": str(path),
        }

        for key, value in r.items():
            if key in allowed_extra:
                row[key] = value

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def _parse_log_file(
    path: Path,
    *,
    engine_hint: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Parse a single docking log file into a normalized dataframe.

    Parsing first uses the provided engine hint when available. If the engine is
    still unknown after parsing, a fallback detection step is applied using the
    file content and path.

    :param path:
        Path to the input log file.
    :type path: pathlib.Path
    :param engine_hint:
        Optional engine hint used during parsing.
    :type engine_hint: str | None

    :returns:
        Normalized dataframe for the parsed file, or ``None`` if no score rows
        are found.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        from pathlib import Path

        df = _parse_log_file(
            Path("vina/erlotinib.log"),
            engine_hint="vina",
        )
    """
    hint = canonicalize_engine_name(engine_hint)

    rows_parsed, engine_used = safe_parse_file(
        path,
        parse_fn=parse_log_text,
        engine_hint=hint,
        regex=None,
        normalize_on_failure=True,
    )

    if not engine_used:
        try:
            txt, _ = read_text_flexible(path)
            engine_used = (
                hint or detect_engine(txt) or detect_engine_from_path(path) or ""
            )
        except Exception:
            engine_used = hint or detect_engine_from_path(path) or ""

    return _rows_to_frame(path, rows_parsed, engine_used)


def _parse_table_file(path: Path) -> Optional[pd.DataFrame]:
    """
    Parse a single CSV- or TSV-like table file.

    The file is read with encoding fallbacks and its column names are normalized
    to canonical names when recognized.

    :param path:
        Path to the input table file.
    :type path: pathlib.Path

    :returns:
        Normalized dataframe for the table, or ``None`` if parsing fails.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        from pathlib import Path

        df = _parse_table_file(Path("scores.tsv"))
    """
    df = _read_csv_flexible(path)
    if df is None:
        return None

    t = _normalize_table_df(df)

    if "source_file" not in t.columns:
        t["source_file"] = str(path)

    return t


def _collect_log_files(root: Path, recursive: bool = True) -> list[Path]:
    """
    Collect valid log files from a file or directory root.

    Recognized log files include both ``.log`` and ``.txt``.

    :param root:
        File or directory to inspect.
    :type root: pathlib.Path
    :param recursive:
        Whether directory traversal should recurse into nested folders.
    :type recursive: bool

    :returns:
        Sorted list of discovered log-file paths.
    :rtype: list[pathlib.Path]

    Example
    -------
    .. code-block:: python

        from pathlib import Path

        files = _collect_log_files(Path("logs"), recursive=True)
    """
    if root.is_file():
        return [root] if is_log_path(root) else []

    if not root.exists():
        return []

    if recursive:
        return sorted(p for p in root.rglob("*") if is_log_path(p))

    return sorted(p for p in root.iterdir() if is_log_path(p))


def _require_engine(engine_hint: Optional[str], layout: str) -> str:
    """
    Validate that a required engine hint is available for a fixed-layout mode.

    This is used by layouts where automatic engine guessing is intentionally not
    allowed.

    :param engine_hint:
        User-provided engine hint.
    :type engine_hint: str | None
    :param layout:
        Layout name requesting a required engine.
    :type layout: str

    :returns:
        Canonical engine name.
    :rtype: str

    :raises ValueError:
        Raised when the engine is missing or not recognized.

    Example
    -------
    .. code-block:: python

        engine = _require_engine("vina", "single_file")
    """
    engine = canonicalize_engine_name(engine_hint)
    if engine is None:
        raise ValueError(
            f"engine_hint is required for layout={layout!r} and must be one of "
            "vina, vina-gpu, smina, qvina, qvina-gpu, gnina"
        )
    return engine


def _crawl_single_file(
    roots: Sequence[Path | str],
    *,
    engine_hint: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Parse one or more explicit log files using a required engine.

    All input paths must be valid ``.log`` or ``.txt`` files.

    :param roots:
        Sequence of file paths to parse.
    :type roots: Sequence[pathlib.Path | str]
    :param engine_hint:
        Required engine name applied to all files.
    :type engine_hint: str | None

    :returns:
        Combined parsed dataframe, or ``None`` when no rows are found.
    :rtype: pandas.DataFrame | None

    :raises ValueError:
        Raised when a provided path is not a valid log file or the engine is
        missing.

    Example
    -------
    .. code-block:: python

        df = _crawl_single_file(
            ["logs/erlotinib.log"],
            engine_hint="smina",
        )
    """
    engine = _require_engine(engine_hint, "single_file")
    parts: list[pd.DataFrame] = []

    for root in roots:
        p = Path(root)
        if not is_log_path(p):
            raise ValueError(f"single_file layout expects a .log or .txt file: {p}")

        part = _parse_log_file(p, engine_hint=engine)
        if part is not None:
            parts.append(part)

    return _finalize_parts(parts)


def _crawl_flat_dir(
    roots: Sequence[Path | str],
    *,
    engine_hint: Optional[str],
    recursive: bool,
) -> Optional[pd.DataFrame]:
    """
    Parse log files from one or more directories using a required engine.

    This layout is intended for directories that contain only one engine's log
    files.

    :param roots:
        Sequence of directory roots to inspect.
    :type roots: Sequence[pathlib.Path | str]
    :param engine_hint:
        Required engine name applied to all discovered log files.
    :type engine_hint: str | None
    :param recursive:
        Whether nested subdirectories should also be searched.
    :type recursive: bool

    :returns:
        Combined parsed dataframe, or ``None`` when no rows are found.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        df = _crawl_flat_dir(
            ["logs/smina_run"],
            engine_hint="smina",
            recursive=True,
        )
    """
    engine = _require_engine(engine_hint, "flat_dir")
    parts: list[pd.DataFrame] = []

    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue

        files = _collect_log_files(rp, recursive=recursive)
        for p in files:
            part = _parse_log_file(p, engine_hint=engine)
            if part is not None:
                parts.append(part)

    return _finalize_parts(parts)


def _crawl_engine_tree(
    roots: Sequence[Path | str],
    *,
    recursive: bool,
) -> Optional[pd.DataFrame]:
    """
    Parse a high-level logs directory whose immediate subfolders are engine names.

    Each immediate child directory is interpreted as an engine folder. Valid
    child names are canonicalized through the engine helper functions.

    :param roots:
        Sequence of top-level directories to inspect.
    :type roots: Sequence[pathlib.Path | str]
    :param recursive:
        Whether log discovery inside each engine folder should recurse into
        nested subdirectories.
    :type recursive: bool

    :returns:
        Combined parsed dataframe, or ``None`` when no rows are found.
    :rtype: pandas.DataFrame | None

    :raises ValueError:
        Raised when one of the provided roots is a file instead of a directory.

    Example
    -------
    .. code-block:: python

        df = _crawl_engine_tree(
            ["logs"],
            recursive=True,
        )
    """
    parts: list[pd.DataFrame] = []

    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue

        if rp.is_file():
            raise ValueError("engine_tree layout expects directories, not files")

        for child in sorted(p for p in rp.iterdir() if p.is_dir()):
            engine = canonicalize_engine_name(child.name)
            if engine is None:
                continue

            for p in _collect_log_files(child, recursive=recursive):
                part = _parse_log_file(p, engine_hint=engine)
                if part is not None:
                    parts.append(part)

    return _finalize_parts(parts)


def _crawl_auto(
    roots: Sequence[Path | str],
    *,
    include_logs: Sequence[str],
    include_tables: Sequence[str],
    engine_hint: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Perform generic recursive crawling with best-effort engine detection.

    Files are classified as log files or tabular files. Log files are parsed
    through the log parser stack. Table files are read through the flexible CSV
    reader stack.

    :param roots:
        Sequence of files or directories to inspect.
    :type roots: Sequence[pathlib.Path | str]
    :param include_logs:
        Glob patterns used to discover log files.
    :type include_logs: Sequence[str]
    :param include_tables:
        Glob patterns used to discover table files.
    :type include_tables: Sequence[str]
    :param engine_hint:
        Optional fallback engine hint used during log parsing.
    :type engine_hint: str | None

    :returns:
        Combined parsed dataframe, or ``None`` when no parsable inputs are found.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        df = _crawl_auto(
            ["results"],
            include_logs=("**/*.log", "**/*.txt"),
            include_tables=("**/*.csv", "**/*.tsv", "**/*.tab"),
            engine_hint=None,
        )
    """
    parts: list[pd.DataFrame] = []
    engine = canonicalize_engine_name(engine_hint)

    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue

        if rp.is_file():
            if is_log_path(rp):
                part = _parse_log_file(
                    rp,
                    engine_hint=engine or detect_engine_from_path(rp),
                )
                if part is not None:
                    parts.append(part)

            elif is_table_path(rp):
                part = _parse_table_file(rp)
                if part is not None:
                    parts.append(part)

            continue

        for pat in include_logs:
            for p in sorted(rp.rglob(pat)):
                if not is_log_path(p):
                    continue

                path_engine = detect_engine_from_path(p)
                part = _parse_log_file(
                    p,
                    engine_hint=engine or path_engine,
                )
                if part is not None:
                    parts.append(part)

        for pat in include_tables:
            for p in sorted(rp.rglob(pat)):
                if not is_table_path(p):
                    continue

                part = _parse_table_file(p)
                if part is not None:
                    parts.append(part)

    return _finalize_parts(parts)


def crawl_scores(
    roots: Sequence[Path | str],
    include_logs: Optional[Sequence[str]] = None,
    include_tables: Optional[Sequence[str]] = None,
    engine_hint: Optional[str] = None,
    *,
    layout: LayoutMode = "auto",
    recursive: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Discover and parse docking outputs under one or more filesystem roots.

    Supported layouts are:

    - ``auto``: generic crawling with best-effort engine detection
    - ``single_file``: explicit log file parsing with required engine
    - ``flat_dir``: log directory parsing with required engine
    - ``engine_tree``: high-level directory whose immediate child folders are
      engine names

    :param roots:
        Files or directories to inspect.
    :type roots: Sequence[pathlib.Path | str]
    :param include_logs:
        Optional log-file glob patterns used only in ``layout="auto"``.
    :type include_logs: Sequence[str] | None
    :param include_tables:
        Optional table-file glob patterns used only in ``layout="auto"``.
    :type include_tables: Sequence[str] | None
    :param engine_hint:
        Engine required for ``single_file`` and ``flat_dir``. In ``auto`` mode,
        it acts as a fallback hint.
    :type engine_hint: str | None
    :param layout:
        Extraction layout mode.
    :type layout: Literal["auto", "single_file", "flat_dir", "engine_tree"]
    :param recursive:
        Whether directory-based layouts should recurse into nested folders.
    :type recursive: bool

    :returns:
        Combined parsed dataframe, or ``None`` when no parsable data is found.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        df1 = crawl_scores(
            roots=["logs/erlotinib.log"],
            engine_hint="vina",
            layout="single_file",
        )

        df2 = crawl_scores(
            roots=["logs/smina_run"],
            engine_hint="smina",
            layout="flat_dir",
            recursive=True,
        )

        df3 = crawl_scores(
            roots=["logs"],
            layout="engine_tree",
        )

        df4 = crawl_scores(
            roots=["results"],
            layout="auto",
        )
    """
    include_logs = (
        tuple(include_logs) if include_logs is not None else ("**/*.log", "**/*.txt")
    )
    include_tables = (
        tuple(include_tables)
        if include_tables is not None
        else ("**/*.csv", "**/*.tsv", "**/*.tab")
    )

    if layout == "single_file":
        return _crawl_single_file(roots, engine_hint=engine_hint)

    if layout == "flat_dir":
        return _crawl_flat_dir(
            roots,
            engine_hint=engine_hint,
            recursive=recursive,
        )

    if layout == "engine_tree":
        return _crawl_engine_tree(
            roots,
            recursive=recursive,
        )

    return _crawl_auto(
        roots,
        include_logs=include_logs,
        include_tables=include_tables,
        engine_hint=engine_hint,
    )


class Extractor:
    """
    High-level score extractor with layout-aware helpers.

    The extractor wraps :func:`crawl_scores` and provides convenience methods for
    engine filtering, engine listing, and explicit layout-based extraction.

    :param include_logs:
        Optional custom log-file glob patterns.
    :type include_logs: Sequence[str] | None
    :param include_tables:
        Optional custom table-file glob patterns.
    :type include_tables: Sequence[str] | None
    :param match_mode:
        Matching mode used when filtering extracted rows by engine.
    :type match_mode: str
    :param crawl_func:
        Optional custom crawl function used instead of :func:`crawl_scores`.
    :type crawl_func: Callable | None
    :param engine_map:
        Optional mapping from logical engine groups to concrete engine tokens.
    :type engine_map: dict | None

    Example
    -------
    .. code-block:: python

        extractor = Extractor(
            match_mode="exact",
            engine_map={"vina-family": ["vina", "vina-gpu", "qvina", "qvina-gpu"]},
        )
    """

    def __init__(
        self,
        include_logs: Optional[Sequence[str]] = None,
        include_tables: Optional[Sequence[str]] = None,
        match_mode: str = "substring",
        crawl_func: Optional[Callable] = None,
        engine_map: Optional[dict] = None,
    ) -> None:
        """
        Initialize an extractor instance.

        :param include_logs:
            Optional custom log-file glob patterns.
        :type include_logs: Sequence[str] | None
        :param include_tables:
            Optional custom table-file glob patterns.
        :type include_tables: Sequence[str] | None
        :param match_mode:
            Matching mode used when filtering engine labels. Supported values are
            ``"substring"``, ``"exact"``, and ``"regex"``.
        :type match_mode: str
        :param crawl_func:
            Optional custom crawl function used instead of :func:`crawl_scores`.
        :type crawl_func: Callable | None
        :param engine_map:
            Optional mapping from logical engine groups to concrete engine
            tokens.
        :type engine_map: dict | None

        :returns:
            Configured extractor instance.
        :rtype: None

        Example
        -------
        .. code-block:: python

            extractor = Extractor(
                include_logs=("**/*.log", "**/*.txt"),
                match_mode="substring",
            )
        """
        self.include_logs = tuple(include_logs) if include_logs else None
        self.include_tables = tuple(include_tables) if include_tables else None
        self.match_mode: MatchMode = match_mode
        self._crawl_func = crawl_func if crawl_func is not None else crawl_scores
        self.engine_map = {
            k.lower(): [normalize_engine_token(tok) for tok in vals]
            for k, vals in (engine_map or {}).items()
        }

    def _call_crawl(
        self,
        roots,
        *,
        engine_hint=None,
        layout: LayoutMode = "auto",
        recursive: bool = True,
    ):
        """
        Call the configured crawl function with extractor defaults applied.

        :param roots:
            Input roots to inspect.
        :type roots: Sequence[pathlib.Path | str]
        :param engine_hint:
            Optional engine hint forwarded to the crawl function.
        :type engine_hint: str | None
        :param layout:
            Layout mode for crawling.
        :type layout: LayoutMode
        :param recursive:
            Whether directory-based layouts should recurse into nested folders.
        :type recursive: bool

        :returns:
            Parsed dataframe or ``None``.
        :rtype: pandas.DataFrame | None

        Example
        -------
        .. code-block:: python

            extractor = Extractor()
            df = extractor._call_crawl(["logs"], layout="engine_tree")
        """
        return self._crawl_func(
            roots,
            include_logs=(
                self.include_logs
                if self.include_logs is not None
                else ("**/*.log", "**/*.txt")
            ),
            include_tables=(
                self.include_tables
                if self.include_tables is not None
                else ("**/*.csv", "**/*.tsv", "**/*.tab")
            ),
            engine_hint=engine_hint,
            layout=layout,
            recursive=recursive,
        )

    def extract_scores(
        self,
        roots: Sequence[str | Path],
        engines: Optional[Iterable[str]] = None,
        engine_hint: Optional[str] = None,
        *,
        layout: LayoutMode = "auto",
        recursive: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Extract scores and optionally filter the results by engine.

        :param roots:
            Files or directories to inspect.
        :type roots: Sequence[str | pathlib.Path]
        :param engines:
            Optional iterable of engine filters.
        :type engines: Iterable[str] | None
        :param engine_hint:
            Optional engine hint used during parsing.
        :type engine_hint: str | None
        :param layout:
            Extraction layout mode.
        :type layout: LayoutMode
        :param recursive:
            Whether directory-based layouts should recurse into nested folders.
        :type recursive: bool

        :returns:
            Extracted dataframe, optionally filtered by engine, or ``None`` when
            no data is found.
        :rtype: pandas.DataFrame | None

        Example
        -------
        .. code-block:: python

            extractor = Extractor(match_mode="exact")

            df = extractor.extract_scores(
                roots=["logs"],
                engines=["vina", "smina"],
                layout="engine_tree",
            )
        """
        df = self._call_crawl(
            roots,
            engine_hint=engine_hint,
            layout=layout,
            recursive=recursive,
        )

        if df is None or df.empty:
            return df

        if engines is None:
            return df.reset_index(drop=True)

        requested: list[str] = []
        for e in engines:
            if e is None:
                continue
            en = normalize_engine_token(e)
            canon = canonicalize_engine_name(en) or en
            if canon in self.engine_map:
                requested.extend(self.engine_map[canon])
            else:
                requested.append(canon)

        requested = list(dict.fromkeys(requested))
        if not requested:
            return df.reset_index(drop=True)

        col = df["engine"].fillna("").astype(str).str.lower()

        if self.match_mode == "exact":
            mask = col.isin(set(requested))
        elif self.match_mode == "regex":
            pattern = "|".join(f"(?:{r})" for r in requested)
            mask = col.str.contains(pattern, regex=True, na=False)
        else:
            pattern = build_engine_pattern(requested)
            mask = (
                pd.Series([True] * len(df), index=df.index)
                if pattern == ""
                else col.str.contains(pattern, regex=True, na=False)
            )

        return df[mask].reset_index(drop=True)

    def list_engines(
        self,
        roots: Sequence[str | Path],
        engine_hint: Optional[str] = None,
        *,
        layout: LayoutMode = "auto",
        recursive: bool = True,
    ) -> set[str]:
        """
        List unique engine names discovered under the given roots.

        :param roots:
            Files or directories to inspect.
        :type roots: Sequence[str | pathlib.Path]
        :param engine_hint:
            Optional engine hint used during parsing.
        :type engine_hint: str | None
        :param layout:
            Extraction layout mode.
        :type layout: LayoutMode
        :param recursive:
            Whether directory-based layouts should recurse into nested folders.
        :type recursive: bool

        :returns:
            Set of lowercased engine names found in the extracted data.
        :rtype: set[str]

        Example
        -------
        .. code-block:: python

            extractor = Extractor()
            engines = extractor.list_engines(
                roots=["logs"],
                layout="engine_tree",
            )
        """
        df = self._call_crawl(
            roots,
            engine_hint=engine_hint,
            layout=layout,
            recursive=recursive,
        )

        if df is None or df.empty:
            return set()

        return set(df["engine"].dropna().astype(str).str.lower().unique().tolist())

    def extract_log_file(
        self,
        path: str | Path,
        *,
        engine: str,
    ) -> Optional[pd.DataFrame]:
        """
        Extract scores from a single explicit log file.

        :param path:
            Path to a ``.log`` or ``.txt`` file.
        :type path: str | pathlib.Path
        :param engine:
            Required engine name for the file.
        :type engine: str

        :returns:
            Extracted dataframe or ``None`` when no rows are found.
        :rtype: pandas.DataFrame | None

        Example
        -------
        .. code-block:: python

            extractor = Extractor()
            df = extractor.extract_log_file(
                "logs/erlotinib.log",
                engine="vina",
            )
        """
        return self.extract_scores(
            [path],
            engine_hint=engine,
            layout="single_file",
        )

    def extract_logs_dir(
        self,
        path: str | Path,
        *,
        engine: str,
        recursive: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Extract scores from a directory of log files belonging to one engine.

        :param path:
            Directory containing ``.log`` or ``.txt`` files.
        :type path: str | pathlib.Path
        :param engine:
            Required engine name applied to all discovered files.
        :type engine: str
        :param recursive:
            Whether nested subdirectories should also be searched.
        :type recursive: bool

        :returns:
            Extracted dataframe or ``None`` when no rows are found.
        :rtype: pandas.DataFrame | None

        Example
        -------
        .. code-block:: python

            extractor = Extractor()
            df = extractor.extract_logs_dir(
                "logs/smina_run",
                engine="smina",
                recursive=True,
            )
        """
        return self.extract_scores(
            [path],
            engine_hint=engine,
            layout="flat_dir",
            recursive=recursive,
        )

    def extract_engine_folders(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Extract scores from a high-level directory whose immediate subfolders are
        engine names.

        :param path:
            High-level logs directory.
        :type path: str | pathlib.Path
        :param recursive:
            Whether nested subdirectories inside each engine folder should also
            be searched.
        :type recursive: bool

        :returns:
            Extracted dataframe or ``None`` when no rows are found.
        :rtype: pandas.DataFrame | None

        Example
        -------
        .. code-block:: python

            extractor = Extractor()
            df = extractor.extract_engine_folders(
                "logs",
                recursive=True,
            )
        """
        return self.extract_scores(
            [path],
            layout="engine_tree",
            recursive=recursive,
        )


_default_extractor = Extractor()


def extract_scores(
    roots: Sequence[str | Path],
    engines: Optional[Iterable[str]] = None,
    engine_hint: Optional[str] = None,
    *,
    layout: LayoutMode = "auto",
    recursive: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Extract scores using the default :class:`Extractor` instance.

    :param roots:
        Files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engines:
        Optional iterable of engine filters.
    :type engines: Iterable[str] | None
    :param engine_hint:
        Optional engine hint used during parsing.
    :type engine_hint: str | None
    :param layout:
        Extraction layout mode.
    :type layout: LayoutMode
    :param recursive:
        Whether directory-based layouts should recurse into nested folders.
    :type recursive: bool

    :returns:
        Extracted dataframe or ``None`` when no data is found.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        df = extract_scores(
            roots=["logs"],
            layout="engine_tree",
        )
    """
    return _default_extractor.extract_scores(
        roots,
        engines=engines,
        engine_hint=engine_hint,
        layout=layout,
        recursive=recursive,
    )


def list_engines(
    roots: Sequence[str | Path],
    engine_hint: Optional[str] = None,
    *,
    layout: LayoutMode = "auto",
    recursive: bool = True,
) -> set[str]:
    """
    List unique engine names using the default :class:`Extractor` instance.

    :param roots:
        Files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engine_hint:
        Optional engine hint used during parsing.
    :type engine_hint: str | None
    :param layout:
        Extraction layout mode.
    :type layout: LayoutMode
    :param recursive:
        Whether directory-based layouts should recurse into nested folders.
    :type recursive: bool

    :returns:
        Set of lowercased engine names.
    :rtype: set[str]

    Example
    -------
    .. code-block:: python

        engines = list_engines(
            roots=["logs"],
            layout="engine_tree",
        )
    """
    return _default_extractor.list_engines(
        roots,
        engine_hint=engine_hint,
        layout=layout,
        recursive=recursive,
    )


def extract_log_file(
    path: str | Path,
    *,
    engine: str,
) -> Optional[pd.DataFrame]:
    """
    Extract scores from a single explicit log file using the default extractor.

    :param path:
        Path to a ``.log`` or ``.txt`` file.
    :type path: str | pathlib.Path
    :param engine:
        Required engine name for the file.
    :type engine: str

    :returns:
        Extracted dataframe or ``None`` when no rows are found.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        df = extract_log_file(
            "logs/erlotinib.log",
            engine="vina",
        )
    """
    return _default_extractor.extract_log_file(path, engine=engine)


def extract_logs_dir(
    path: str | Path,
    *,
    engine: str,
    recursive: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Extract scores from a flat log directory using the default extractor.

    :param path:
        Directory containing ``.log`` or ``.txt`` files.
    :type path: str | pathlib.Path
    :param engine:
        Required engine name applied to all discovered files.
    :type engine: str
    :param recursive:
        Whether nested subdirectories should also be searched.
    :type recursive: bool

    :returns:
        Extracted dataframe or ``None`` when no rows are found.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        df = extract_logs_dir(
            "logs/smina_run",
            engine="smina",
            recursive=True,
        )
    """
    return _default_extractor.extract_logs_dir(
        path,
        engine=engine,
        recursive=recursive,
    )


def extract_engine_folders(
    path: str | Path,
    *,
    recursive: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Extract scores from a high-level directory whose immediate subfolders are
    engine names.

    :param path:
        High-level logs directory.
    :type path: str | pathlib.Path
    :param recursive:
        Whether nested subdirectories inside each engine folder should also be
        searched.
    :type recursive: bool

    :returns:
        Extracted dataframe or ``None`` when no rows are found.
    :rtype: pandas.DataFrame | None

    Example
    -------
    .. code-block:: python

        df = extract_engine_folders(
            "logs",
            recursive=True,
        )
    """
    return _default_extractor.extract_engine_folders(
        path,
        recursive=recursive,
    )
