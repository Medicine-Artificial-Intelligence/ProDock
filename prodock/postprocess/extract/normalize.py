"""
Utilities for reading and normalizing log files before parsing.

If a parse fails due to encoding issues or unusual bytes, callers can:
- use read_text_flexible(path) to get a decoded string without modifying the file
- use normalize_file(path, backup=True) to convert the file to UTF-8 (with a .bak)
- use safe_parse_file(path, parse_fn, engine_hint=None, regex=None) which will
  attempt to parse, and if that yields no rows or raises an exception, normalize
  the file and retry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

# Encodings to try when reading unknown files (order matters)
_PREFERRED_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


def _try_decode(
    raw: bytes, encodings: Iterable[str] = _PREFERRED_ENCODINGS
) -> Tuple[str, str]:
    """
    Try to decode raw bytes using the provided encodings in order.
    Returns (decoded_text, encoding_used). If none succeed perfectly, returns
    a latin-1 "replace" decode and encoding tag 'latin-1-replace'.
    """
    for enc in encodings:
        try:
            text = raw.decode(enc)
            return text, enc
        except Exception:
            continue
    # fallback: decode with latin-1 but replace invalid sequences
    return raw.decode("latin-1", errors="replace"), "latin-1-replace"


def read_text_flexible(path: Path) -> Tuple[str, str]:
    """
    Read a text file trying multiple encodings. Returns (text, encoding_used).

    This never raises UnicodeDecodeError; in the worst case it returns a
    replacement-decoded latin-1 string.
    """
    raw = path.read_bytes()
    return _try_decode(raw)


def normalize_file(
    path: Path, backup: bool = True, encodings: Optional[Iterable[str]] = None
) -> str:
    """
    Normalize the given file to UTF-8 in-place.

    - Reads the file using the encodings list (falls back to latin-1 replace).
    - If `backup` is True a backup file at path + ".bak" is created (will not
      overwrite an existing .bak; it will append a numeric suffix if needed).
    - Writes the normalized UTF-8 text back to `path`.

    Returns the encoding that was detected / used for the source file.
    """
    encs = tuple(encodings) if encodings else _PREFERRED_ENCODINGS
    raw = path.read_bytes()
    text, detected = _try_decode(raw, encodings=encs)
    # backup
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        # avoid clobbering an existing .bak
        if bak.exists():
            # find a numbered backup
            i = 1
            while True:
                cand = path.with_suffix(path.suffix + f".bak.{i}")
                if not cand.exists():
                    bak = cand
                    break
                i += 1
        # create backup by writing decoded raw bytes (original bytes preserved)
        path.replace(bak)
        # write normalized content back
        path.write_text(text, encoding="utf-8")
    else:
        # just overwrite
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
    Try to parse a log file robustly.

    Parameters
    ----------
    path : Path
        Path to the log file.
    parse_fn : callable(text, engine, regex) -> list
        Parsing function (e.g., parse_log_text). It must accept arguments in
        the form (text, engine, regex) and return a list of parsed rows.
    engine_hint : str, optional
        Engine hint forwarded to `parse_fn` on first attempt.
    regex : dict, optional
        Custom regex mapping forwarded to `parse_fn`.
    normalize_on_failure : bool
        If True, when parsing fails (raises or returns empty) the file will be
        normalized to UTF-8 (with backup) and the parse retried.

    Returns
    -------
    (rows, engine_used)
        rows: list of parsed rows (may be empty)
        engine_used: engine_hint or detected engine string (if parse_fn uses detection),
                     else None
    """
    # first, read the text with flexible decoding (no file modifications)
    from pathlib import Path as _P  # local import to keep top-level light

    p = _P(path)
    text, enc = read_text_flexible(p)
    engine_used = engine_hint

    try:
        rows = parse_fn(text, engine_used, regex)
    except Exception:
        rows = []
    # if parse returned rows, return them
    if rows:
        return rows, engine_used

    # If parse produced no rows and normalization is enabled, attempt normalization
    if normalize_on_failure:
        # normalize file in-place (create backup)
        try:
            normalize_file(p, backup=True)
        except Exception:
            # if normalization fails, give up and return empty result
            return [], engine_used
        # re-read (now should be UTF-8)
        try:
            new_text = p.read_text(encoding="utf-8")
        except Exception:
            new_text, _ = _try_decode(p.read_bytes())
        # retry parse
        try:
            rows2 = parse_fn(new_text, engine_used, regex)
        except Exception:
            rows2 = []
        return rows2, engine_used

    return rows, engine_used
