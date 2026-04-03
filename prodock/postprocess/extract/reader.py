from __future__ import annotations

"""Parsing helpers for Vina-family and GNINA docking logs."""

import re
from typing import Iterator, Optional

from .engines import (
    GNINA_ROW_RE,
    GNINA_TABLE_HEADER,
    VINA_ROW_RE,
    VINA_TABLE_HEADER,
    canonicalize_engine_name,
    detect_engine as _auto_detect_engine,
)


def _iter_lines(text: str) -> Iterator[str]:
    """
    Yield log lines with trailing newline characters removed.

    This helper splits the input text with :meth:`str.splitlines` and strips a
    trailing newline character from each produced line. It provides a compact
    iterator abstraction used by the table parsers.

    :param text:
        Raw log text to iterate line by line.
    :type text: str

    :returns:
        Iterator over normalized log lines.
    :rtype: Iterator[str]

    Example
    -------
    .. code-block:: python

        >>> list(_iter_lines("a\\nb\\n"))
        ['a', 'b']
    """
    for line in text.splitlines():
        yield line.rstrip("\n")


def _parse_vina_family(text: str) -> list[dict]:
    """
    Parse a Vina-family docking result table from log text.

    The parser searches for the standard Vina table header and then extracts
    rows matching :data:`VINA_ROW_RE`. Returned dictionaries contain the
    docking mode, affinity, lower RMSD bound, and upper RMSD bound.

    Supported inputs typically include logs produced by tools such as
    ``vina``, ``qvina``, or ``smina`` when they emit the standard Vina-style
    result table.

    :param text:
        Raw docking log text.
    :type text: str

    :returns:
        Parsed result rows. Each row contains the keys ``mode``,
        ``affinity_kcal_mol``, ``rmsd_lb``, and ``rmsd_ub``. Returns an empty
        list when no compatible table is found.
    :rtype: list[dict]

    Example
    -------
    .. code-block:: python

        text = '''
        -----+------------+----------+----------
           1       -7.5      0.000      0.000
           2       -7.1      1.200      2.400
        '''
        rows = _parse_vina_family(text)
    """
    lines = list(_iter_lines(text))
    rows: list[dict] = []
    header_idx = None
    for i, ln in enumerate(lines):
        if VINA_TABLE_HEADER.search(ln):
            header_idx = i
            break
    if header_idx is None:
        return rows
    for ln in lines[header_idx + 1 :]:  # noqa
        m = VINA_ROW_RE.match(ln)
        if not m:
            continue
        rows.append(
            {
                "mode": int(m.group(1)),
                "affinity_kcal_mol": float(m.group(2)),
                "rmsd_lb": float(m.group(3)),
                "rmsd_ub": float(m.group(4)),
            }
        )
    return rows


def _parse_gnina(text: str) -> list[dict]:
    """
    Parse a GNINA docking result table from log text.

    The parser searches for the standard GNINA table header and then extracts
    rows matching :data:`GNINA_ROW_RE`. Returned dictionaries contain the
    docking mode, affinity, CNN pose score, and CNN affinity score.

    :param text:
        Raw docking log text.
    :type text: str

    :returns:
        Parsed result rows. Each row contains the keys ``mode``,
        ``affinity_kcal_mol``, ``cnn_pose``, and ``cnn_affinity``. Returns an
        empty list when no GNINA-style table is found.
    :rtype: list[dict]

    Example
    -------
    .. code-block:: python

        text = '''
        mode | affinity | cnn_pose | cnn_affinity
           1     -8.2       0.71         7.45
           2     -7.8       0.66         7.10
        '''
        rows = _parse_gnina(text)
    """
    lines = list(_iter_lines(text))
    rows: list[dict] = []
    header_idx = None
    for i, ln in enumerate(lines):
        if GNINA_TABLE_HEADER.search(ln):
            header_idx = i
            break
    if header_idx is None:
        return rows
    for ln in lines[header_idx + 1 :]:  # noqa
        m = GNINA_ROW_RE.match(ln)
        if not m:
            continue
        rows.append(
            {
                "mode": int(m.group(1)),
                "affinity_kcal_mol": float(m.group(2)),
                "cnn_pose": float(m.group(3)),
                "cnn_affinity": float(m.group(4)),
            }
        )
    return rows


def parse_log_text(
    text: str,
    engine: Optional[str] = None,
    regex: Optional[dict[str, str]] = None,
) -> list[dict]:
    """
    Parse docking log text using built-in or custom regex rules.

    The parser first resolves the engine name using
    :func:`canonicalize_engine_name` and, if needed, automatic engine detection.
    When ``regex`` is provided, a custom row pattern is tried first. If custom
    parsing yields rows, those rows are returned immediately. Otherwise, the
    built-in engine-specific parsers are used.

    For GNINA logs, the parser will first try the GNINA table parser and then
    fall back to the Vina-family parser if no GNINA rows are found. This is
    useful because some GNINA outputs may also contain Vina-like score tables.

    :param text:
        Raw docking log text to parse.
    :type text: str
    :param engine:
        Optional engine name such as ``"vina"``, ``"smina"``, ``"qvina"``, or
        ``"gnina"``. If omitted, the engine is inferred automatically from the
        input text when possible.
    :type engine: Optional[str]
    :param regex:
        Optional mapping of custom regex patterns. Supported keys are
        ``"vina_row"`` and ``"gnina_row"``. The selected pattern must expose
        four capture groups in the same order as the built-in parser expects.
    :type regex: Optional[dict[str, str]]

    :returns:
        Parsed docking rows. For Vina-family logs, rows contain ``mode``,
        ``affinity_kcal_mol``, ``rmsd_lb``, and ``rmsd_ub``. For GNINA logs,
        rows contain ``mode``, ``affinity_kcal_mol``, ``cnn_pose``, and
        ``cnn_affinity``.
    :rtype: list[dict]

    Example
    -------
    .. code-block:: python

        text = '''
        -----+------------+----------+----------
           1       -7.5      0.000      0.000
           2       -7.1      1.200      2.400
        '''
        rows = parse_log_text(text, engine="vina")

    Example
    -------
    .. code-block:: python

        custom = {
            "vina_row": r"^\\s*(\\d+)\\s+([-+]?\\d*\\.?\\d+)\\s+([-+]?\\d*\\.?\\d+)\\s+([-+]?\\d*\\.?\\d+)$"
        }
        rows = parse_log_text(text, engine="vina", regex=custom)
    """
    eng = canonicalize_engine_name(engine) or _auto_detect_engine(text)

    if regex:
        patt_key = "gnina_row" if eng == "gnina" else "vina_row"
        patt = regex.get(patt_key)
        if patt:
            row_re = re.compile(patt)
            rows: list[dict] = []
            for ln in text.splitlines():
                m = row_re.match(ln)
                if not m:
                    continue
                if eng == "gnina":
                    rows.append(
                        {
                            "mode": int(m.group(1)),
                            "affinity_kcal_mol": float(m.group(2)),
                            "cnn_pose": float(m.group(3)),
                            "cnn_affinity": float(m.group(4)),
                        }
                    )
                else:
                    rows.append(
                        {
                            "mode": int(m.group(1)),
                            "affinity_kcal_mol": float(m.group(2)),
                            "rmsd_lb": float(m.group(3)),
                            "rmsd_ub": float(m.group(4)),
                        }
                    )
            if rows:
                return rows

    if eng == "gnina":
        rows = _parse_gnina(text)
        return rows or _parse_vina_family(text)

    return _parse_vina_family(text)
