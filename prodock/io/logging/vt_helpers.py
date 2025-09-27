"""Terminal / VT helpers for cross-platform color support."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional


def _is_a_tty(stream) -> bool:
    """Return True if the stream is a TTY-like object."""
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


__all__ = ["_is_a_tty", "_enable_windows_vt_mode", "_console_supports_color"]
