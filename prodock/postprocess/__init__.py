"""
Post-processing utilities for ProDock.

Subpackages
-----------
- :mod:`prodock.postprocess.extract` — crawling/parsing & extraction helpers.
- :mod:`prodock.postprocess.metrics` — structural and screening metrics.
"""

from __future__ import annotations

import importlib
from typing import Any

# public names we want to provide lazily
__all__ = [
    # extract/crawl helpers
    "crawl_scores",
    "parse_log_text",
    "parse_vina_log",
    "parse_gnina_log",
    "extract_scores",
    "list_engines",
    "Extractor",
    # metrics (classes + functions)
    "DockEvaluator",
    "ScreenEvaluator",
    "rmsd_aligned",
    "rmsd_min",
    "success_rate",
    "auc_roc",
    "pr_auc",
    "enrichment_factor",
    "bedroc",
    "topn_success",
]

# ---------------------------
# Lightweight lazy import map
# ---------------------------
# Map attribute -> (module_name, attr_name)
_lazy_map = {
    # extract / crawl
    "Extractor": ("prodock.postprocess.extract", "Extractor"),
    "extract_scores": ("prodock.postprocess.extract", "extract_scores"),
    "list_engines": ("prodock.postprocess.extract", "list_engines"),
    # crawl_scores lives in the extract.core implementation
    "crawl_scores": ("prodock.postprocess.extract.core", "crawl_scores"),
    # the generic parser (we expose it too)
    "parse_log_text": ("prodock.postprocess.extract.reader", "parse_log_text"),
    # metrics: delegate to the metrics package (we will import the package lazily)
    "DockEvaluator": ("prodock.postprocess.metrics", "DockEvaluator"),
    "ScreenEvaluator": ("prodock.postprocess.metrics", "ScreenEvaluator"),
    "rmsd_aligned": ("prodock.postprocess.metrics", "rmsd_aligned"),
    "rmsd_min": ("prodock.postprocess.metrics", "rmsd_min"),
    "success_rate": ("prodock.postprocess.metrics", "success_rate"),
    "auc_roc": ("prodock.postprocess.metrics", "auc_roc"),
    "pr_auc": ("prodock.postprocess.metrics", "pr_auc"),
    "enrichment_factor": ("prodock.postprocess.metrics", "enrichment_factor"),
    "bedroc": ("prodock.postprocess.metrics", "bedroc"),
    "topn_success": ("prodock.postprocess.metrics", "topn_success"),
}


def __getattr__(name: str) -> Any:
    """
    Lazy attribute loader for the package.

    When a user does `from prodock.postprocess import X` or `import prodock.postprocess as pp; pp.X`,
    this triggers and imports the underlying module only when needed.
    """
    # compatibility wrappers that aren't direct attributes in modules
    if name == "parse_vina_log":
        # wrapper calling the generic parse_log_text with engine hint
        def _parse_vina_log(text: str, regex: dict | None = None):
            mod = importlib.import_module("prodock.postprocess.extract.reader")
            parse_fn = getattr(mod, "parse_log_text")
            return parse_fn(text, engine="vina", regex=regex)

        globals()[name] = _parse_vina_log
        return _parse_vina_log

    if name == "parse_gnina_log":

        def _parse_gnina_log(text: str, regex: dict | None = None):
            mod = importlib.import_module("prodock.postprocess.extract.reader")
            parse_fn = getattr(mod, "parse_log_text")
            return parse_fn(text, engine="gnina", regex=regex)

        globals()[name] = _parse_gnina_log
        return _parse_gnina_log

    # If in lazy map, import module and return attribute
    if name in _lazy_map:
        module_name, attr = _lazy_map[name]
        try:
            mod = importlib.import_module(module_name)
            val = getattr(mod, attr)
            globals()[name] = val  # cache on module for subsequent access
            return val
        except Exception as exc:
            # Re-raise attribute error to surface a clean message for consumers
            raise AttributeError(
                f"Cannot import name {name!r} from {module_name!r}: {exc}"
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # make introspection list lazy attributes too
    return sorted(list(globals().keys()) + __all__)
