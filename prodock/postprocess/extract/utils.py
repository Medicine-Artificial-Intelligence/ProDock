from __future__ import annotations
import re
from typing import Iterable


def normalize_engine_token(token: str) -> str:
    return str(token).strip().lower()


def build_engine_pattern(engines: Iterable[str]) -> str:
    tokens = [normalize_engine_token(e) for e in engines if str(e).strip()]
    if not tokens:
        return ""
    escaped = [re.escape(t) for t in tokens]
    return "|".join(escaped)


def engine_matches(engine_value: str, pattern: str) -> bool:
    if not pattern:
        return True
    if not engine_value:
        return False
    return bool(re.search(pattern, str(engine_value).lower()))
