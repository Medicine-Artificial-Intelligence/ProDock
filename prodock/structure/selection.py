# prodock/structure/selection.py
from typing import Sequence


def join_selection(prefix: str, tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    return " or ".join(f"{prefix} {t}" for t in tokens)


def chain_selection(chains: Sequence[str]) -> str:
    return join_selection("chain", chains)


def resn_selection(resnames: Sequence[str]) -> str:
    return join_selection("resn", resnames)
