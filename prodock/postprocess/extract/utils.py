from __future__ import annotations

"""General utility helpers used by the extraction module."""

import re
from pathlib import Path
from typing import Iterable

LOG_SUFFIXES: tuple[str, ...] = (".log", ".txt")
TABLE_SUFFIXES: tuple[str, ...] = (".csv", ".tsv", ".tab")


def normalize_engine_token(token: str) -> str:
    """
    Normalize a docking-engine token for case-insensitive matching.

    This helper converts the input value to string, strips leading and trailing
    whitespace, and lowercases the result. It is useful when building
    normalized engine-name filters such as ``vina``, ``smina``, or ``qvina``.

    :param token:
        Raw engine token to normalize.
    :type token: str

    :returns:
        Normalized engine token in lowercase with surrounding whitespace
        removed.
    :rtype: str

    Example
    -------
    .. code-block:: python

        >>> normalize_engine_token("  Vina ")
        'vina'
    """
    return str(token).strip().lower()


def build_engine_pattern(engines: Iterable[str]) -> str:
    """
    Build a regex alternation pattern from a sequence of engine names.

    Each engine token is normalized with :func:`normalize_engine_token`.
    Empty or whitespace-only values are ignored. Returned tokens are escaped
    with :func:`re.escape` before being joined with ``|`` so special regex
    characters in engine names are treated literally.

    :param engines:
        Iterable of engine names to include in the regex alternation.
    :type engines: Iterable[str]

    :returns:
        Regex alternation string such as ``"vina|smina|qvina"``. Returns an
        empty string when no valid engine names are provided.
    :rtype: str

    Example
    -------
    .. code-block:: python

        >>> pattern = build_engine_pattern([" Vina ", "smina", ""])
        >>> pattern
        'vina|smina'
    """
    tokens = [normalize_engine_token(e) for e in engines if str(e).strip()]
    if not tokens:
        return ""
    return "|".join(re.escape(t) for t in tokens)


def engine_matches(engine_value: str, pattern: str) -> bool:
    """
    Check whether an engine value matches a regex engine-name pattern.

    The engine value is lowercased before matching. If ``pattern`` is empty,
    the function returns ``True`` and behaves as an unrestricted filter. If
    ``engine_value`` is empty while a pattern is provided, the function returns
    ``False``.

    :param engine_value:
        Engine value to test, for example ``"vina"`` or ``"qvina-w"``.
    :type engine_value: str
    :param pattern:
        Regex pattern string, typically produced by
        :func:`build_engine_pattern`.
    :type pattern: str

    :returns:
        ``True`` if the engine matches the pattern, otherwise ``False``.
    :rtype: bool

    Example
    -------
    .. code-block:: python

        >>> pattern = build_engine_pattern(["vina", "smina"])
        >>> engine_matches("VINA", pattern)
        True
        >>> engine_matches("gnina", pattern)
        False
    """
    if not pattern:
        return True
    if not engine_value:
        return False
    return bool(re.search(pattern, str(engine_value).lower()))


def is_log_path(path: Path) -> bool:
    """
    Return whether a path looks like a docking log file.

    A path is considered a log path only when it exists as a file and its
    suffix matches one of :data:`LOG_SUFFIXES`.

    :param path:
        Filesystem path to inspect.
    :type path: pathlib.Path

    :returns:
        ``True`` if ``path`` is an existing file with a supported log suffix,
        otherwise ``False``.
    :rtype: bool

    Example
    -------
    .. code-block:: python

        from pathlib import Path

        path = Path("results/run.log")
        ok = is_log_path(path)
    """
    return path.is_file() and path.suffix.lower() in LOG_SUFFIXES


def is_table_path(path: Path) -> bool:
    """
    Return whether a path looks like a tabular score file.

    A path is considered a table path only when it exists as a file and its
    suffix matches one of :data:`TABLE_SUFFIXES`.

    :param path:
        Filesystem path to inspect.
    :type path: pathlib.Path

    :returns:
        ``True`` if ``path`` is an existing file with a supported table suffix,
        otherwise ``False``.
    :rtype: bool

    Example
    -------
    .. code-block:: python

        from pathlib import Path

        path = Path("scores.csv")
        ok = is_table_path(path)
    """
    return path.is_file() and path.suffix.lower() in TABLE_SUFFIXES
