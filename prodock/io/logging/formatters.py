"""Log formatters: colored simple formatter and JSON formatter."""

from __future__ import annotations

import json
import logging
from typing import Optional

from .constants import _LEVEL_TO_ANSI
from .vt_helpers import _console_supports_color


class SimpleColorFormatter(logging.Formatter):
    """
    Formatter that injects ANSI color codes into levelname when supported.

    :param fmt: format string
    :param datefmt: date format string
    :param force_color: if True, force color on even when not TTY (uses FORCE_COLOR env too)
    """

    def __init__(
        self,
        fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt: Optional[str] = "%Y-%m-%d %H:%M:%S",
        force_color: Optional[bool] = None,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = _console_supports_color(force_color)

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            color = _LEVEL_TO_ANSI.get(record.levelname, "")
            levelname = getattr(record, "levelname", "")
            # avoid mutating global record in place; create a shallow copy behavior by
            # temporarily setting attribute and restoring afterwards
            orig = record.levelname
            try:
                record.levelname = (
                    f"{color}{levelname}{_LEVEL_TO_ANSI.get('RESET', '')}"
                )
            except Exception:
                # fallback: just set color + level
                record.levelname = f"{color}{levelname}"
            out = super().format(record)
            record.levelname = orig
            return out
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """
    Minimal one-line JSON formatter for logs suitable for ingestion.

    The JSON contains: ts, level, logger, message and any extra fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # capture extras (anything not in the standard LogRecord attrs)
        excluded = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in excluded and not k.startswith("_")
        }
        if extras:
            base["extra"] = extras
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str, ensure_ascii=False)


__all__ = ["SimpleColorFormatter", "JSONFormatter"]
