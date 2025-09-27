# prodock/postprocess/extract/core.py
"""
Core extraction helpers that wrap crawling + parsing into a convenient API.

This module provides:
- crawl_scores(roots, ...): discover log/table files, parse them (using
  reader.parse_log_text), and return a pandas DataFrame with standardized
  columns.
- Extractor: wrapper class around crawl_scores with engine-filtering utilities.
- functional wrappers extract_scores(...) and list_engines(...)

Examples
--------
>>> from prodock.postprocess.extract.core import crawl_scores
>>> df = crawl_scores(["/path/to/logs"])
>>> df.head()
The crawler is encoding-tolerant: for log files that fail to parse initially it
will attempt normalization via normalize.safe_parse_file (which will create a
.bak backup and write UTF-8 normalized content) and retry parsing.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Set, Literal
from pathlib import Path

import pandas as pd

from .utils import build_engine_pattern, normalize_engine_token
from .reader import parse_log_text
from .engines import detect_engine
from .normalize import safe_parse_file, read_text_flexible

# allowed match modes for extractor
MatchMode = Literal["substring", "exact", "regex"]


def _to_float_or_none(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _to_int_or_none(x) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _read_csv_flexible(path: Path) -> Optional[pd.DataFrame]:
    """
    Try reading a CSV/TSV with several encodings; return DataFrame or None.
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
    # last resort: try python engine with latin-1 replace
    try:
        if is_tsv:
            return pd.read_csv(path, sep="\t", encoding="latin-1", engine="python")
        return pd.read_csv(path, encoding="latin-1", engine="python")
    except Exception:
        return None


def crawl_scores(
    roots: Sequence[Path | str],
    include_logs: Optional[Sequence[str]] = None,
    include_tables: Optional[Sequence[str]] = None,
    engine_hint: Optional[str] = None,
    labels: Optional[dict] = None,
) -> Optional[pd.DataFrame]:
    """
    Discover and parse docking outputs under `roots` and return a combined
    :class:`pandas.DataFrame` with standardized columns.

    The function scans each path in ``roots`` (files or directories) for
    log files and table files, parses them using the package parsers and
    concatenates the results. For log files that cannot be parsed due to
    encoding issues, the function will attempt normalization and retry.

    :param roots: Sequence of filesystem roots (file paths or directories)
                  to search for logs and tables.
    :type roots: sequence of :class:`pathlib.Path` or :class:`str`
    :param include_logs: Glob patterns for identifying log files. If None,
                         defaults to ("**/*.log", "**/*.txt").
    :type include_logs: sequence of str, optional
    :param include_tables: Glob patterns for table files (CSV/TSV). If None,
                           defaults to ("**/*.csv", "**/*.tsv").
    :type include_tables: sequence of str, optional
    :param engine_hint: Optional engine hint forwarded to the parser when
                        automatic detection fails (e.g. "vina", "gnina").
    :type engine_hint: str or None
    :param labels: Optional mapping from ligand identifiers to labels
                   (e.g., ``{"LIG1": 1}``) to populate the 'label' column.
    :type labels: dict[str, int] or None

    :returns: A :class:`pandas.DataFrame` combining all parsed rows. Columns
              include at least ``["ligand_id", "score", "rank", "pose_path",
              "engine", "label"]``. Returns ``None`` if no files are found.
    :rtype: pandas.DataFrame or None

    :raises FileNotFoundError: If one of the provided root paths is invalid
                               and strict behavior is required.  (Note: the
                               default implementation silently ignores missing
                               roots, so this is rarely raised.)
    :raises ValueError: If an input table cannot be parsed as CSV/TSV.

    Example
    -------
    .. code-block:: python

        from pathlib import Path
        df = crawl_scores([Path("/data/docking")])
        # inspect top scoring poses
        top = df.sort_values("score").head(10)
    """
    include_logs = (
        tuple(include_logs) if include_logs is not None else ("**/*.log", "**/*.txt")
    )
    include_tables = (
        tuple(include_tables)
        if include_tables is not None
        else ("**/*.csv", "**/*.tsv")
    )
    labels_map = labels or {}

    parsed_rows: list[dict] = []
    csv_parts: list[pd.DataFrame] = []

    # helper to process a single log file robustly
    def _process_log_file(p: Path):
        # safe_parse_file will try multiple encodings and will normalize (create .bak)
        # and retry parsing if the initial parse returned no rows or failed.
        rows_parsed, eng_used = safe_parse_file(
            p,
            parse_fn=parse_log_text,
            engine_hint=engine_hint,
            regex=None,
            normalize_on_failure=True,
        )
        # If engine not set by engine_hint, try to detect from file content
        if not eng_used:
            try:
                txt, _ = read_text_flexible(p)
                eng_used = detect_engine(txt) or engine_hint or ""
            except Exception:
                eng_used = engine_hint or ""
        stem = p.stem
        for r in rows_parsed:
            affinity = r.get("affinity_kcal_mol")
            mode = r.get("mode")
            parsed_rows.append(
                {
                    "ligand_id": stem,
                    "score": _to_float_or_none(affinity),
                    "rank": _to_int_or_none(mode),
                    "pose_path": (
                        r.get("pose_path") if r.get("pose_path") is not None else None
                    ),
                    "engine": eng_used or "",
                    "label": labels_map.get(stem),
                }
            )

    # helper to process a table file (CSV/TSV)
    def _process_table_file(p: Path):
        df = _read_csv_flexible(p)
        if df is None:
            return
        # Normalize common column names to canonical ones where present
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
            if lc in {"rank"} and "rank" not in t.columns:
                colmap[c] = "rank"
            if lc in {"pose_path", "pose", "posepath"} and "pose_path" not in t.columns:
                colmap[c] = "pose_path"
            if lc in {"engine"} and "engine" not in t.columns:
                colmap[c] = "engine"
            if lc in {"label", "labels"} and "label" not in t.columns:
                colmap[c] = "label"
        if colmap:
            t = t.rename(columns=colmap)
        csv_parts.append(t)

    # Walk roots
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        if rp.is_file():
            # classify by suffix
            if rp.suffix.lower() in {".log", ".txt"}:
                _process_log_file(rp)
            elif rp.suffix.lower() in {".csv", ".tsv", ".tab"}:
                _process_table_file(rp)
            else:
                # try to match include patterns
                matched_log = any(rp.match(pat) for pat in include_logs)
                matched_table = any(rp.match(pat) for pat in include_tables)
                if matched_log:
                    _process_log_file(rp)
                elif matched_table:
                    _process_table_file(rp)
            continue

        # directory: rglob for given patterns
        for pat in include_logs:
            for p in rp.rglob(pat):
                if p.is_file():
                    _process_log_file(p)
        for pat in include_tables:
            for p in rp.rglob(pat):
                if p.is_file():
                    _process_table_file(p)

    parts: list[pd.DataFrame] = []
    if parsed_rows:
        parts.append(pd.DataFrame(parsed_rows))
    if csv_parts:
        parts.extend(csv_parts)

    if not parts:
        return None

    df_all = pd.concat(parts, ignore_index=True, sort=False).astype(object)

    # Ensure canonical columns exist
    for col in ["ligand_id", "score", "rank", "pose_path", "engine", "label"]:
        if col not in df_all.columns:
            df_all[col] = None

    # Normalize types
    df_all["score"] = df_all["score"].apply(_to_float_or_none)
    df_all["rank"] = df_all["rank"].apply(_to_int_or_none)
    df_all["engine"] = df_all["engine"].fillna("").astype(str)

    return df_all.reset_index(drop=True)


class Extractor:
    """
    High-level extractor that wraps crawling and provides filtering utilities.

    The :class:`Extractor` exposes convenience methods to discover engines,
    extract scores, and filter results by engine tokens.

    :ivar include_logs: Tuple of glob patterns used for finding log files.
    :vartype include_logs: tuple[str] | None
    :ivar include_tables: Tuple of glob patterns used for finding tables.
    :vartype include_tables: tuple[str] | None
    :ivar match_mode: Matching mode for engine filtering; one of
                      "substring", "exact", "regex".
    :vartype match_mode: str
    """

    def __init__(
        self,
        include_logs: Optional[Sequence[str]] = None,
        include_tables: Optional[Sequence[str]] = None,
        match_mode: str = "substring",
        crawl_func: Optional[Callable] = None,
        engine_map: Optional[dict] = None,
    ):
        """
        Create an :class:`Extractor` instance.

        :param include_logs: Optional glob patterns to restrict searched log
                             files (default: ``("**/*.log","**/*.txt")``).
        :type include_logs: sequence of str or None
        :param include_tables: Optional glob patterns to restrict table files
                               (default: ``("**/*.csv","**/*.tsv")``).
        :type include_tables: sequence of str or None
        :param match_mode: How engine filtering is performed:
                           - ``"substring"``: substring match (default)
                           - ``"exact"``: exact equality
                           - ``"regex"``: treat filter tokens as regex
        :type match_mode: str
        :param crawl_func: Optional callable used to crawl files. If provided,
                           it must have the same signature as :func:`crawl_scores`.
                           This is primarily intended for tests.
        :type crawl_func: callable or None
        :param engine_map: Optional mapping that expands logical engine groups
                           to concrete engine names, e.g.
                           ``{"vina-family": ["vina","vina-gpu"]}``.
        :type engine_map: dict[str, sequence[str]] or None
        """
        self.include_logs = tuple(include_logs) if include_logs else None
        self.include_tables = tuple(include_tables) if include_tables else None
        self.match_mode: MatchMode = match_mode
        self._crawl_func = crawl_func if crawl_func is not None else crawl_scores
        self.engine_map = {
            k.lower(): [normalize_engine_token(tok) for tok in vals]
            for k, vals in (engine_map or {}).items()
        }

    # internal wrapper to call crawl func with defaults
    def _call_crawl(self, roots, engine_hint=None, labels=None):
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
                else ("**/*.csv", "**/*.tsv")
            ),
            engine_hint=engine_hint,
            labels=labels,
        )

    def extract_scores(
        self,
        roots: Sequence[str | Path],
        engines: Optional[Iterable[str]] = None,
        engine_hint: Optional[str] = None,
        labels: Optional[dict] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Crawl the given ``roots`` and return parsed scores filtered by engine.

        :param roots: Sequence of directories or files to crawl.
        :type roots: sequence of :class:`pathlib.Path` or :class:`str`
        :param engines: If provided, only rows whose ``engine`` matches one
                        of these tokens will be returned. Matching respects
                        ``self.match_mode``.
        :type engines: sequence of str or None
        :param engine_hint: Optional engine hint forwarded to parsers when
                            detection is ambiguous.
        :type engine_hint: str or None
        :param labels: Optional mapping of ligand_id -> label to include in the
                       returned DataFrame.
        :type labels: dict[str, int] or None

        :returns: Filtered :class:`pandas.DataFrame` or ``None`` if no data.
        :rtype: pandas.DataFrame or None

        Example
        -------
        .. code-block:: python

            ex = Extractor(match_mode="exact")
            df_vina = ex.extract_scores(["/data/logs"], engines=["vina"])
        """
        df = self._call_crawl(roots, engine_hint=engine_hint, labels=labels)
        if df is None or df.empty:
            return df

        if engines is None:
            return df.reset_index(drop=True)

        # expand requested tokens via engine_map
        requested: list[str] = []
        for e in engines:
            if e is None:
                continue
            en = normalize_engine_token(e)
            if en in self.engine_map:
                requested.extend(self.engine_map[en])
            else:
                requested.append(en)
        requested = list(dict.fromkeys(requested))  # dedupe

        if not requested:
            return df.reset_index(drop=True)

        col = df["engine"].fillna("").astype(str).str.lower()

        if self.match_mode == "exact":
            tokens_set: Set[str] = set(requested)
            mask = col.isin(tokens_set)
        elif self.match_mode == "regex":
            pattern = "|".join(f"(?:{r})" for r in requested)
            mask = col.str.contains(pattern, regex=True, na=False)
        else:  # substring
            pattern = build_engine_pattern(requested)
            if pattern == "":
                mask = pd.Series([True] * len(df), index=df.index)
            else:
                mask = col.str.contains(pattern, regex=True, na=False)

        return df[mask].reset_index(drop=True)

    def list_engines(
        self,
        roots: Sequence[str | Path],
        engine_hint: Optional[str] = None,
        labels: Optional[dict] = None,
    ) -> set[str]:
        """
        Crawl `roots` and return the set of unique engine labels found (lowercased).
        """
        df = self._call_crawl(roots, engine_hint=engine_hint, labels=labels)
        if df is None or df.empty:
            return set()
        return set(df["engine"].dropna().astype(str).str.lower().unique().tolist())


# default instance + functional wrappers
_default_extractor = Extractor()


def extract_scores(
    roots: Sequence[str | Path],
    engines: Optional[Iterable[str]] = None,
    engine_hint: Optional[str] = None,
    labels: Optional[dict] = None,
) -> Optional[pd.DataFrame]:
    return _default_extractor.extract_scores(
        roots, engines=engines, engine_hint=engine_hint, labels=labels
    )


def list_engines(
    roots: Sequence[str | Path],
    engine_hint: Optional[str] = None,
    labels: Optional[dict] = None,
) -> set[str]:
    return _default_extractor.list_engines(
        roots, engine_hint=engine_hint, labels=labels
    )
