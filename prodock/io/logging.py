# logging.py
"""
Central logging utilities for ProDock.

Features
--------
- LoggerManager: configure console + rotating file + JSON output
- Dependency-free colored console (ANSI escapes; attempts to enable VT on Windows)
- JSONFormatter: one-line JSON logs for log collectors
- StructuredAdapter: attach persistent context (run_id, pdb_id, ...)
- Timer: ContextDecorator to measure elapsed time (usable as decorator/context)
- log_step: decorator for pipeline methods (auto logs start/finish/error); works for sync and async methods

Example
-------
>>> from prodock.io.logging import setup_logging, get_logger, StructuredAdapter, log_step
>>> setup_logging(log_dir="logs", level="DEBUG", colored=True, json=False)
>>> log = StructuredAdapter(get_logger("ProDock"), {"run_id":"r1"})
>>> @log_step("validate")
... def validate(self, pdb_id): ...
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import logging.handlers
import os
import sys
import time
from contextlib import ContextDecorator
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

# ------------------------------
# ANSI / Windows VT helpers (no external deps)
# ------------------------------
ANSI = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
}

_LEVEL_TO_ANSI = {
    "DEBUG": ANSI["CYAN"],
    "INFO": ANSI["GREEN"],
    "WARNING": ANSI["YELLOW"],
    "ERROR": ANSI["RED"],
    "CRITICAL": ANSI["MAGENTA"],
}


def _is_a_tty(stream) -> bool:
    """
    Return True if the stream is a TTY-like object.
    """
    try:
        return stream.isatty()
    except Exception:
        return False


def _enable_windows_vt_mode() -> bool:
    """
    Try to enable Virtual Terminal Processing on Windows so ANSI escape sequences work.
    Returns True on non-Windows or on success, False on failure.
    """
    if os.name != "nt":
        return True
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle == 0:
            return False
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if not kernel32.SetConsoleMode(handle, new_mode):
            return False
        return True
    except Exception:
        return False


def _console_supports_color(force: Optional[bool] = None) -> bool:
    """
    Determine if console supports color. Force with env FORCE_COLOR=1/true.
    Honor NO_COLOR env var to disable color.
    """
    if os.getenv("NO_COLOR"):
        return False
    if force is None:
        forced = os.getenv("FORCE_COLOR", "").lower() in ("1", "true", "yes")
    else:
        forced = bool(force)
    if forced:
        return True
    # Must be a TTY and VT mode enabled (for Windows)
    return _is_a_tty(sys.stderr) and _enable_windows_vt_mode()


# ------------------------------
# Formatters
# ------------------------------
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
            record.levelname = f"{color}{levelname}{ANSI['RESET']}"
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


# ------------------------------
# LoggerManager
# ------------------------------
class LoggerManager:
    """
    Manager for configuring project-wide logging.

    :param log_dir: directory to put rotating log files. If None, no file handler.
    :param log_file: filename for log file.
    :param max_bytes: rotate file after this size (bytes).
    :param backup_count: number of rotated files to keep.
    :param level: root logging level (int or name).
    :param colored: enable colored console output (ANSI; no external deps).
    :param json: if True, use JSONFormatter for file (and optional console).
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = "logs",
        log_file: str = "prodock.log",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        level: Union[str, int] = logging.INFO,
        colored: bool = True,
        json: bool = False,
    ):
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_file = log_file
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)
        self._level_input = level
        self.colored = bool(colored)
        self.json = bool(json)
        self._configured = False

    def __repr__(self) -> str:
        return (
            f"LoggerManager(log_dir={self.log_dir}, log_file={self.log_file}, "
            f"level={self.level_name}, colored={self.colored}, json={self.json})"
        )

    # ------------------------------
    # fluent setup
    # ------------------------------
    def setup(self) -> "LoggerManager":
        """
        Configure the root logger and handlers. Safe to call multiple times.
        :return: self
        """
        if self._configured:
            return self

        # prepare log dir if needed
        if self.log_dir:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                # Best-effort; fall back to no file handler if creation fails
                self.log_dir = None

        root = logging.getLogger()
        root.setLevel(self._coerce_level(self._level_input))
        # remove old handlers to avoid duplication in interactive environments
        for h in list(root.handlers):
            root.removeHandler(h)

        # pick formatters
        text_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        # console formatter: color if requested and supported, else plain
        if self.json:
            console_formatter = JSONFormatter()
        else:
            console_formatter = SimpleColorFormatter(
                fmt=text_fmt, datefmt=datefmt, force_color=self.colored
            )

        ch = logging.StreamHandler()
        ch.setLevel(self._coerce_level(self._level_input))
        ch.setFormatter(console_formatter)
        root.addHandler(ch)

        # file handler: keep plain text unless json=True
        if self.log_dir:
            fp = self.log_dir / self.log_file
            try:
                fh = logging.handlers.RotatingFileHandler(
                    fp,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count,
                    encoding="utf-8",
                )
                fh.setLevel(self._coerce_level(self._level_input))
                fh.setFormatter(
                    JSONFormatter()
                    if self.json
                    else logging.Formatter(fmt=text_fmt, datefmt=datefmt)
                )
                root.addHandler(fh)
            except Exception:
                # if file handler fails (permissions, etc.), skip silently
                pass

        self._configured = True
        return self

    # ------------------------------
    # helpers / properties
    # ------------------------------
    @staticmethod
    def _coerce_level(level: Union[str, int]) -> int:
        if isinstance(level, int):
            return level
        return logging._nameToLevel.get(str(level).upper(), logging.INFO)

    @property
    def configured(self) -> bool:
        """
        :return: True if logging has been configured by this manager.
        """
        return self._configured

    @property
    def level_name(self) -> str:
        """
        :return: canonical level name (e.g., 'DEBUG', 'INFO').
        """
        return logging.getLevelName(self._coerce_level(self._level_input))

    def get_logger(self, name: str) -> logging.Logger:
        """
        Retrieve a logger instance. Ensures manager is setup.

        :param name: logger name
        :return: logger
        """
        if not self._configured:
            self.setup()
        return logging.getLogger(name)


# module-level default manager + helpers
_default_manager: LoggerManager = LoggerManager()


def setup_logging(**kwargs) -> LoggerManager:
    """
    Convenience entry to configure logging.

    Accepts the same kwargs as LoggerManager.__init__.
    Returns the configured LoggerManager.

    Example
    -------
    >>> setup_logging(log_dir="logs", level="DEBUG", colored=True, json=False)
    """
    global _default_manager
    _default_manager = LoggerManager(**kwargs).setup()
    return _default_manager


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger via the default manager.

    :param name: logger name
    :return: logging.Logger
    """
    if not _default_manager.configured:
        _default_manager.setup()
    return logging.getLogger(name)


# ------------------------------
# StructuredAdapter
# ------------------------------
class StructuredAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that merges adapter-level context with per-call 'extra' dicts.

    :param logger: base logger
    :param extra: persistent context dict (e.g., {'run_id': 'r1', 'pdb_id': '5N2F'})
    """

    def process(self, msg: str, kwargs: Dict[str, Any]):
        call_extra = kwargs.pop("extra", {}) or {}
        merged = {**(self.extra or {}), **(call_extra or {})}
        kwargs["extra"] = merged
        return msg, kwargs


# ------------------------------
# Timer context/decorator
# ------------------------------
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


# ------------------------------
# log_step decorator (sync + async)
# ------------------------------
def _try_summary_from(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Try to call obj.summarize_step() or obj.summarize() if present (best-effort).
    """
    try:
        if obj is None:
            return None
        if hasattr(obj, "summarize_step") and callable(getattr(obj, "summarize_step")):
            return getattr(obj, "summarize_step")()
        if hasattr(obj, "summarize") and callable(getattr(obj, "summarize")):
            return getattr(obj, "summarize")()
    except Exception:
        return None
    return None


def log_step(step_name: Optional[str] = None):
    """
    Decorator for pipeline methods representing a processing step.

    Logs:
      - step.start (INFO)
      - step.finish (INFO) with elapsed and optional summary
      - step.error (exception logged) if exception occurs

    Works for both sync and async methods.

    :param step_name: optional explicit step name (default: function name)
    """

    def decorator(func: Callable):
        is_coro = asyncio.iscoroutinefunction(func)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            self_obj = args[0] if args else None
            logger = (
                getattr(self_obj, "logger", get_logger(func.__module__))
                if self_obj
                else get_logger(func.__module__)
            )
            ctx = getattr(self_obj, "log_context", {}) if self_obj else {}
            adapter = StructuredAdapter(logger, ctx)
            name = step_name or func.__name__
            adapter.info("step.start", extra={"step": name})
            t0 = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                elapsed = round(time.perf_counter() - t0, 6)
                adapter.exception(
                    "step.error",
                    extra={"step": name, "elapsed": elapsed, "error": str(exc)},
                )
                raise
            elapsed = round(time.perf_counter() - t0, 6)
            summary = _try_summary_from(result) or _try_summary_from(self_obj)
            adapter.info(
                "step.finish",
                extra={"step": name, "elapsed": elapsed, "summary": summary},
            )
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            self_obj = args[0] if args else None
            logger = (
                getattr(self_obj, "logger", get_logger(func.__module__))
                if self_obj
                else get_logger(func.__module__)
            )
            ctx = getattr(self_obj, "log_context", {}) if self_obj else {}
            adapter = StructuredAdapter(logger, ctx)
            name = step_name or func.__name__
            adapter.info("step.start", extra={"step": name})
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                elapsed = round(time.perf_counter() - t0, 6)
                adapter.exception(
                    "step.error",
                    extra={"step": name, "elapsed": elapsed, "error": str(exc)},
                )
                raise
            elapsed = round(time.perf_counter() - t0, 6)
            summary = _try_summary_from(result) or _try_summary_from(self_obj)
            adapter.info(
                "step.finish",
                extra={"step": name, "elapsed": elapsed, "summary": summary},
            )
            return result

        return async_wrapper if is_coro else sync_wrapper

    return decorator
