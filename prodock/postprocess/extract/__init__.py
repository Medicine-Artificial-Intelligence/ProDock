"""
Extract subpackage (regex-based log readers & wrappers).

Public API
----------
- detect_engine(text) -> str | None
- parse_log_text(text, engine: str | None = None, regex: dict[str, str] | None = None) -> list[dict]
- Extractor (wrapper around crawl_scores with engine filtering)
- extract_scores, list_engines (functional wrappers)

Examples
--------
>>> from prodock.postprocess.extract import detect_engine, parse_log_text
>>> eng = detect_engine(open("vina.log").read())
>>> rows = parse_log_text(open("vina.log").read(), engine=eng)
>>> rows[0]["affinity_kcal_mol"]
-10.0
"""

from .engines import detect_engine
from .reader import parse_log_text
from .core import Extractor, extract_scores, list_engines

__all__ = [
    "detect_engine",
    "parse_log_text",
    "Extractor",
    "extract_scores",
    "list_engines",
]
