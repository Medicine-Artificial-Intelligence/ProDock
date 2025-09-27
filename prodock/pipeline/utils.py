# prodock/pipeline/utils.py
from __future__ import annotations

from typing import Any, Iterable


def iter_progress(
    iterable: Iterable[Any], verbose: int = 0, desc: str = "", unit: str = ""
) -> Iterable[Any]:
    """
    Wrap an iterable with tqdm if available and ``verbose >= 2``. Returns the
    original iterable otherwise.

    :param iterable: Any iterable.
    :param verbose: 0=silent, 1=log-only, 2+=progress bars if tqdm available.
    :param desc: Optional description for the progress bar.
    :param unit: Optional unit label for the progress bar.
    :returns: An iterable (possibly wrapped by tqdm).
    """
    if verbose >= 2:
        try:
            from tqdm.auto import tqdm  # type: ignore

            return tqdm(iterable, desc=desc, unit=unit)
        except Exception:  # pragma: no cover - tqdm not installed
            return iterable
    return iterable
