"""log_step decorator (sync + async) and helper summary discovery."""

from __future__ import annotations

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .adapters import StructuredAdapter
from .manager import get_logger


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


__all__ = ["log_step"]
