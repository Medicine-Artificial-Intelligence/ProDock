"""LoggerManager and convenience helpers (setup_logging, get_logger)."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Union

from .formatters import SimpleColorFormatter, JSONFormatter


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


__all__ = ["LoggerManager", "setup_logging", "get_logger"]
