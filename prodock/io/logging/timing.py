"""Timer context/decorator for logging elapsed times."""

from __future__ import annotations

import logging
import time
from contextlib import ContextDecorator
from typing import Optional


class Timer(ContextDecorator):
    """
    Measure elapsed time and optionally log a debug entry on exit.

    Usage:
    >>> with Timer("conformers", logger=get_logger("ProDock")):
    >>>     do_work()

    Or as decorator:
    >>> @Timer("step")
    >>> def f(...): ...
    """

    def __init__(
        self, label: Optional[str] = None, logger: Optional[logging.Logger] = None
    ):
        self.label = label or "timer"
        self.logger = logger or logging.getLogger("Timer")
        self._start: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed = round(
            (time.perf_counter() - (self._start or time.perf_counter())), 6
        )
        try:
            self.logger.debug(
                "timer.elapsed", extra={"label": self.label, "elapsed": self.elapsed}
            )
        except Exception:
            pass
        # do not suppress exceptions
        return False


__all__ = ["Timer"]
