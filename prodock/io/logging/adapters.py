"""Logger adapters for structured context."""

from __future__ import annotations

import logging
from typing import Any, Dict


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


__all__ = ["StructuredAdapter"]
