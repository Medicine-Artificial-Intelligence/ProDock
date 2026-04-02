from __future__ import annotations

"""Utilities for robust reading and normalization of log files."""

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

from .engines import canonicalize_engine_name, detect_engine

_PREFERRED_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


def _try_decode(
    raw: bytes,
    encodings: Iterable[str] = _PREFERRED_ENCODINGS,
) -> Tuple[str, str]:
    """
    Try decoding raw bytes using a sequence of candidate encodings.

    The function attempts each encoding in order and returns the decoded text
    together with the encoding that succeeded. If all candidates fail, it
    falls back to ``latin-1`` with replacement semantics so decoding always
    returns a string.

    :param raw:
        Raw byte content to decode.
    :type raw: bytes
    :param encodings:
        Candidate encodings to try in order.
    :type encodings: Iterable[str]

    :returns:
        Tuple containing the decoded text and the encoding label that was used.
        When all candidate decoders fail, the encoding label is
        ``"latin-1-replace"``.
    :rtype: Tuple[str, str]

    Example
    -------
    .. code-block:: python

        raw = "café".encode("latin-1")
        text, encoding = _try_decode(raw, encodings=("utf-8", "latin-1"))
    """
    for enc in encodings:
        try:
            return raw.decode(enc), enc
        except Exception:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1-replace"


def read_text_flexible(path: Path) -> Tuple[str, str]:
    """
    Read a text file using multiple encoding fallbacks.

    This helper reads the file as raw bytes and decodes it using
    :func:`_try_decode` with the module's preferred encoding order.

    :param path:
        Path to the text file.
    :type path: pathlib.Path

    :returns:
        Tuple containing the decoded text and the detected encoding label.
    :rtype: Tuple[str, str]

    Example
    -------
    .. code-block:: python

        from pathlib import Path

        text, encoding = read_text_flexible(Path("dock.log"))
    """
    return _try_decode(path.read_bytes())


def normalize_file(
    path: Path,
    backup: bool = True,
    encodings: Optional[Iterable[str]] = None,
) -> str:
    """
    Normalize a text file to UTF-8 in place and return the detected encoding.

    The file is first decoded using :func:`_try_decode`. The decoded content is
    then written back as UTF-8. When ``backup`` is ``True``, the original file
    is moved to a sibling backup file before the normalized UTF-8 text is
    written. If the default backup name already exists, a numbered backup name
    such as ``.bak.1`` or ``.bak.2`` is chosen.

    :param path:
        Path to the text file to normalize.
    :type path: pathlib.Path
    :param backup:
        Whether to keep a backup of the original file before rewriting.
    :type backup: bool
    :param encodings:
        Optional candidate encodings to try. If omitted, the module-level
        preferred encodings are used.
    :type encodings: Optional[Iterable[str]]

    :returns:
        Detected source encoding label.
    :rtype: str

    Example
    -------
    .. code-block:: python

        from pathlib import Path

        detected = normalize_file(Path("dock.log"), backup=True)
    """
    encs = tuple(encodings) if encodings else _PREFERRED_ENCODINGS
    text, detected = _try_decode(path.read_bytes(), encodings=encs)
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if bak.exists():
            i = 1
            while True:
                cand = path.with_suffix(path.suffix + f".bak.{i}")
                if not cand.exists():
                    bak = cand
                    break
                i += 1
        path.replace(bak)
        path.write_text(text, encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")
    return str(detected)


def safe_parse_file(
    path: Path,
    parse_fn: Callable[[str, Optional[str], Optional[dict]], list],
    engine_hint: Optional[str] = None,
    regex: Optional[dict] = None,
    *,
    normalize_on_failure: bool = True,
) -> Tuple[list, Optional[str]]:
    """
    Parse a log file robustly and optionally retry after UTF-8 normalization.

    The file is first read using flexible decoding. If no explicit engine hint
    is provided, the engine is inferred from the decoded text. The supplied
    parser is then called. If parsing fails or returns no rows, the function
    can normalize the file to UTF-8 and retry parsing on the normalized text.

    :param path:
        Path to the log file to parse.
    :type path: pathlib.Path
    :param parse_fn:
        Parsing function with signature ``(text, engine, regex) -> list``.
    :type parse_fn: Callable[[str, Optional[str], Optional[dict]], list]
    :param engine_hint:
        Optional engine hint such as ``"vina"``, ``"smina"``, or ``"gnina"``.
    :type engine_hint: Optional[str]
    :param regex:
        Optional custom regex mapping passed through to the parser.
    :type regex: Optional[dict]
    :param normalize_on_failure:
        Whether to retry parsing after normalizing the file to UTF-8 when the
        first parsing attempt fails or returns no rows.
    :type normalize_on_failure: bool

    :returns:
        Tuple containing the parsed rows and the engine name used for parsing.
    :rtype: Tuple[list, Optional[str]]

    Example
    -------
    .. code-block:: python

        from pathlib import Path
        from prodock.postprocess.extract.reader import parse_log_text

        rows, engine = safe_parse_file(
            Path("dock.log"),
            parse_fn=parse_log_text,
            engine_hint=None,
            regex=None,
        )
    """
    engine_used = canonicalize_engine_name(engine_hint)
    text, _ = read_text_flexible(path)
    if engine_used is None:
        engine_used = detect_engine(text)

    try:
        rows = parse_fn(text, engine_used, regex)
    except Exception:
        rows = []
    if rows:
        return rows, engine_used

    if normalize_on_failure:
        try:
            normalize_file(path, backup=True)
        except Exception:
            return [], engine_used
        try:
            new_text = path.read_text(encoding="utf-8")
        except Exception:
            new_text, _ = _try_decode(path.read_bytes())
        if engine_used is None:
            engine_used = detect_engine(new_text)
        try:
            rows = parse_fn(new_text, engine_used, regex)
        except Exception:
            rows = []
        return rows, engine_used

    return rows, engine_used
