# prodock/postprocess/extract/engines.py
"""Engine detection and table/row regexes for Vina-family and GNINA logs.

This module provides:
- ENGINE_HINTS: list of (pattern, canonical_name) used for banner-based detection
- VINA_TABLE_HEADER, VINA_ROW_RE: header/row regex for Vina-like tables
- GNINA_TABLE_HEADER, GNINA_ROW_RE: header/row regex for GNINA tables
- detect_engine(text): best-effort engine canonicalization
"""
from __future__ import annotations
import re
from typing import Optional, List, Tuple

# --- Engine banners / hints (case-insensitive) ---
# Order matters: more specific patterns should appear earlier
ENGINE_HINTS: List[Tuple[str, str]] = [
    (r"\bgnina\b", "gnina"),
    (r"\bsmina\b", "smina"),
    (r"\bquick\s*vina\s*2[-_\s]?gpu\b", "qvina-gpu"),
    (r"\bquick\s*vina\s*2\b", "qvina"),
    (r"\bquickvina2[-_\s]?gpu\b", "qvina-gpu"),
    (r"\bquickvina2\b", "qvina"),
    (r"\bqvina\b", "qvina"),
    (r"\bvina[-_\s]?gpu\b", "vina-gpu"),
    (r"\bautodock\s+vina\b", "vina"),
    (r"\bvina\b", "vina"),
]

# Vina-family header (vina/smina/qvina/vina-gpu/qvina-gpu)
VINA_TABLE_HEADER = re.compile(
    r"mode\s*\|\s*affinity\s*\|\s*(?:dist|dist\s+from)\s*best\s*mode", re.IGNORECASE
)

# Vina-family row pattern: mode, affinity, rmsd_lb, rmsd_ub
# numeric groups allow optional decimal and optional exponent (E+/-)
VINA_ROW_RE = re.compile(
    r"^\s*(\d+)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s*$"
)

# GNINA header (affinity + CNN columns)
GNINA_TABLE_HEADER = re.compile(
    r"mode\s*\|\s*affinity\s*\|\s*cnn\b.*\|\s*cnn", re.IGNORECASE
)
# GNINA row pattern: mode, affinity, cnn_pose, cnn_affinity
GNINA_ROW_RE = re.compile(
    r"^\s*(\d+)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s+("
    r"[+\-]?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s+("
    r"-?\d+(?:\.\d+)?(?:[Ee][\+\-]?\d+)?)\s*$"
)


def detect_engine(text: str) -> Optional[str]:
    """
    Best-effort engine detection from log banner/headers/content.

    Returns canonical engine strings:
      'gnina', 'smina', 'qvina', 'qvina-gpu', 'vina', 'vina-gpu'
    or None when no reasonable evidence is found.

    The function is intentionally permissive to cover common filename/header variants.
    """
    if text is None:
        return None

    # Try banner-based hints first (ordered)
    for patt, name in ENGINE_HINTS:
        if re.search(patt, text, re.IGNORECASE):
            return name

    # Additional heuristics for QuickVina/GPU variants (fallback checks)
    low = text.lower()
    # QuickVina variants not covered by ENGINE_HINTS (defensive)
    if (
        "quickvina" in low
        or "quick vina" in low
        or "quick-vina" in low
        or "qvina" in low
    ):
        return "qvina-gpu" if "gpu" in low else "qvina"

    # Vina GPU heuristics: if both 'vina' and 'gpu' appear anywhere -> vina-gpu
    if "vina" in low and "gpu" in low:
        return "vina-gpu"

    # Header-based fallback: GNINA and VINA table headers
    if GNINA_TABLE_HEADER.search(text):
        return "gnina"
    if VINA_TABLE_HEADER.search(text):
        return "vina"

    # No clear detection
    return None
