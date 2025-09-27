from __future__ import annotations

from .dock_power import DockEvaluator
from .screen_power import ScreenEvaluator

__all__ = [
    "DockEvaluator",
    "ScreenEvaluator",
    "rmsd_aligned",
    "rmsd_min",
    "auc_roc",
    "pr_auc",
    "enrichment_factor",
    "bedroc",
    "topn_success",
    "success_rate",
]

# Default ScreenEvaluator instance for functional wrappers (lower-is-better by default)
_default_screen = ScreenEvaluator(higher_is_better=False)


def rmsd_aligned(
    ref, probe, match_substructure: bool = True, engine: str = "rdkit"
) -> float:
    """
    Compute RMSD between reference and probe molecules using the chosen engine.

    Backwards-compatible wrapper: previously available as a free function.

    :param ref: RDKit Mol or path to molecule file (reference).
    :param probe: RDKit Mol or path to probe molecule.
    :param match_substructure: (ignored) kept for API compatibility.
    :param engine: backend engine to use ("rdkit", "openbabel", "pymol").
    :returns: RMSD (float)
    """
    de = DockEvaluator(engine=engine)
    return de.rmsd(ref, probe)


def rmsd_min(ref, probes, engine: str = "rdkit") -> float:
    """
    Minimum RMSD between ref and an iterable of probes.

    :param ref: reference molecule or path
    :param probes: iterable of probe molecules or paths
    :param engine: backend engine
    :returns: minimum RMSD (float) or nan if no probes
    """
    de = DockEvaluator(engine=engine)
    vals = [de.rmsd(ref, p) for p in probes]
    if not vals:
        return float("nan")
    return float(min(vals))


# Screening wrappers (use default ScreenEvaluator)
def auc_roc(scores, labels) -> float:
    """
    Compute ROC AUC (wrapper).
    """
    return _default_screen.auc_roc(scores, labels)


def pr_auc(scores, labels) -> float:
    """
    Compute PR AUC (wrapper).
    """
    return _default_screen.pr_auc(scores, labels)


def enrichment_factor(scores, labels, fraction: float = 0.01) -> float:
    """
    Compute Enrichment Factor (wrapper).
    """
    return _default_screen.enrichment_factor(scores, labels, fraction=fraction)


def bedroc(scores, labels, alpha: float = 20.0) -> float:
    """
    Compute BEDROC (wrapper).
    """
    return _default_screen.bedroc(scores, labels, alpha=alpha)


def topn_success(ligand_ids, scores, labels, n: int = 1) -> float:
    """
    Compute Top-N success (wrapper).
    """
    return _default_screen.topn_success(ligand_ids, scores, labels, n=n)


def success_rate(rmsds, cutoff: float = 2.0) -> float:
    """
    Pose success rate: fraction with RMSD <= cutoff.
    """
    import numpy as np

    arr = np.asarray(rmsds, dtype=float)
    return float(np.mean(arr <= cutoff)) if arr.size else float("nan")
