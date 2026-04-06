from __future__ import annotations

"""Engine detection and canonicalization helpers for docking logs.

This module centralizes:

- banner-based engine detection from log text
- engine-name normalization from user input or directory names
- table-header and row regexes for Vina-family and GNINA logs

Canonical engine names used across the extractor are:

- ``vina``
- ``vina-gpu``
- ``smina``
- ``qvina``
- ``qvina-gpu``
- ``gnina``
"""

import re
from pathlib import PurePath
from typing import Iterable, List, Optional, Tuple

KNOWN_ENGINES: tuple[str, ...] = (
    "vina",
    "vina-gpu",
    "smina",
    "qvina",
    "qvina-gpu",
    "gnina",
)

_ENGINE_ALIASES: dict[str, str] = {
    "vina": "vina",
    "autodockvina": "vina",
    "autodock_vina": "vina",
    "autodock-vina": "vina",
    "vina1": "vina",
    "vina12": "vina",
    "vinagpu": "vina-gpu",
    "vina_gpu": "vina-gpu",
    "vina-gpu": "vina-gpu",
    "smina": "smina",
    "gnina": "gnina",
    "qvina": "qvina",
    "quickvina": "qvina",
    "quickvina2": "qvina",
    "quick_vina2": "qvina",
    "quick-vina2": "qvina",
    "quickvina2gpu": "qvina-gpu",
    "quickvina_gpu": "qvina-gpu",
    "quickvina2_gpu": "qvina-gpu",
    "quick-vina2-gpu": "qvina-gpu",
    "quick_vina2_gpu": "qvina-gpu",
    "qvinagpu": "qvina-gpu",
    "qvina_gpu": "qvina-gpu",
    "qvina-gpu": "qvina-gpu",
}

# Order matters: more specific patterns should appear earlier.
ENGINE_HINTS: List[Tuple[str, str]] = [
    (r"\bgnina\b", "gnina"),
    (r"\bsmina\b", "smina"),
    (r"\bquick\s*vina\s*2[-_\s]?gpu\b", "qvina-gpu"),
    (r"\bquickvina2[-_\s]?gpu\b", "qvina-gpu"),
    (r"\bqvina[-_\s]?gpu\b", "qvina-gpu"),
    (r"\bquick\s*vina\s*2\b", "qvina"),
    (r"\bquickvina2\b", "qvina"),
    (r"\bqvina\b", "qvina"),
    (r"\bvina[-_\s]?gpu\b", "vina-gpu"),
    (r"\bautodock\s+vina\b", "vina"),
    (r"\bvina\b", "vina"),
]

VINA_TABLE_HEADER = re.compile(
    r"mode\s*\|\s*affinity\s*\|\s*(?:dist|dist\s+from)\s*best\s*mode",
    re.IGNORECASE,
)

VINA_ROW_RE = re.compile(
    r"^\s*(\d+)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s*$"
)

GNINA_TABLE_HEADER = re.compile(
    r"mode\s*\|\s*affinity\s*\|\s*cnn\b.*\|\s*cnn", re.IGNORECASE
)

GNINA_ROW_RE = re.compile(
    r"^\s*(\d+)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s+("
    r"[+\-]?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s*$"
)


def canonicalize_engine_name(name: Optional[str]) -> Optional[str]:
    """Return a canonical engine token when ``name`` is recognized.

    The normalization is tolerant to case, whitespace, dashes, and underscores.
    Unknown values return ``None``.
    """
    if name is None:
        return None
    token = str(name).strip().lower()
    if not token:
        return None
    squashed = re.sub(r"[^a-z0-9]+", "", token)
    return _ENGINE_ALIASES.get(squashed) or _ENGINE_ALIASES.get(token)


def is_known_engine_name(name: Optional[str]) -> bool:
    """Return ``True`` when ``name`` can be canonicalized to a known engine."""
    return canonicalize_engine_name(name) is not None


def detect_engine(text: str) -> Optional[str]:
    """Best-effort engine detection from log text."""
    if text is None:
        return None

    for patt, name in ENGINE_HINTS:
        if re.search(patt, text, re.IGNORECASE):
            return name

    low = text.lower()
    if (
        "quickvina" in low
        or "quick vina" in low
        or "quick-vina" in low
        or "qvina" in low
    ):
        return "qvina-gpu" if "gpu" in low else "qvina"
    if "vina" in low and "gpu" in low:
        return "vina-gpu"
    if GNINA_TABLE_HEADER.search(text):
        return "gnina"
    if VINA_TABLE_HEADER.search(text):
        return "vina"
    return None


def detect_engine_from_path(path: PurePath | str | None) -> Optional[str]:
    """Infer an engine from any component of a filesystem path."""
    if path is None:
        return None
    p = PurePath(path)
    for part in reversed(p.parts):
        eng = canonicalize_engine_name(part)
        if eng is not None:
            return eng
    return None


def first_engine_from_parts(parts: Iterable[str]) -> Optional[str]:
    """Return the first canonical engine found in ``parts``."""
    for part in parts:
        eng = canonicalize_engine_name(part)
        if eng is not None:
            return eng
    return None
