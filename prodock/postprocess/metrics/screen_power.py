"""
Screening performance utilities (ROC, PR, EF, BEDROC, enrichment curves).

All functions assume that **lower scores are better** (typical docking energies)
unless you create :class:`ScreenEvaluator` with ``higher_is_better=True``.

Sphinx-style documented class `ScreenEvaluator` is provided with convenience
instance methods for common metrics.

Examples
--------

>>> from prodock.postprocess.metrics.screen_power import ScreenEvaluator
>>> se = ScreenEvaluator(higher_is_better=False)  # docking scores: lower is better
>>> scores = [-10.0, -9.0, -8.0, -7.0, -2.0, -1.0]
>>> labels = [1, 0, 1, 0, 0, 0]
>>> auc = se.auc_roc(scores, labels)
>>> prec_at_top3 = se.precision_at_k(scores, labels, k=3)
>>> mcc_best = se.mcc(scores, labels)  # best MCC across thresholds
>>> # use a concrete threshold (interpreted on raw scores): predictions are score >= threshold
>>> mcc_at_thr = se.mcc(scores, labels, threshold=-5.0)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


class ScreenEvaluator:
    """
    Compute common virtual screening metrics.

    :param higher_is_better: If True, the input scores are treated as
                             higher-is-better. If False (default), scores
                             are treated as lower-is-better (e.g., docking).
    :type higher_is_better: bool
    """

    def __init__(self, higher_is_better: bool = False):
        self.higher_is_better = bool(higher_is_better)

    # ---------------------------
    # internal helpers
    # ---------------------------
    def _prepare(self, scores: Sequence[float], labels: Sequence[int]):
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=int)
        if s.shape[0] != y.shape[0]:
            raise ValueError("scores and labels must have the same length")
        # convert to higher-is-better for ranking and monotonic handling
        if not self.higher_is_better:
            s = -s
        return s, y

    def _rank_order(self, s: np.ndarray):
        """Return indices that sort scores descending (higher first)."""
        return np.argsort(-s, kind="mergesort")

    # ---------------------------
    # curve helpers
    # ---------------------------
    def roc_curve(
        self, scores: Sequence[float], labels: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve points.

        :param scores: array-like of scores (lower-is-better by default)
        :param labels: binary labels (1 active, 0 decoy)
        :returns: (fpr, tpr, thresholds) each as numpy arrays
        """
        s, y = self._prepare(scores, labels)
        P = (y == 1).sum()
        N = (y == 0).sum()
        if P == 0 or N == 0:
            return np.array([]), np.array([]), np.array([])
        order = self._rank_order(s)
        y_sorted = y[order]
        tp_cum = np.concatenate([[0], np.cumsum(y_sorted == 1)])
        fp_cum = np.concatenate([[0], np.cumsum(y_sorted == 0)])
        tpr = tp_cum / P
        fpr = fp_cum / N
        thresholds = np.concatenate([s[order], [s[order[-1]] - 1.0]])
        return fpr, tpr, thresholds

    def precision_recall_curve(
        self, scores: Sequence[float], labels: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve points.

        :param scores: array-like of scores
        :param labels: binary labels (1 active, 0 decoy)
        :returns: (precision, recall, thresholds)
        """
        s, y = self._prepare(scores, labels)
        P = (y == 1).sum()
        if P == 0:
            return np.array([]), np.array([]), np.array([])
        order = self._rank_order(s)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted == 1).astype(float)
        fp = np.cumsum(y_sorted == 0).astype(float)
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / P
        thresholds = s[order]
        return precision, recall, thresholds

    # ---------------------------
    # public metrics
    # ---------------------------
    def auc_roc(self, scores: Sequence[float], labels: Sequence[int]) -> float:
        """
        Compute ROC AUC.

        :param scores: array-like of scores (lower-is-better by default).
        :param labels: binary labels (1=active, 0=decoy).
        :returns: ROC AUC in [0,1] or nan for degenerate inputs.
        """
        fpr, tpr, _ = self.roc_curve(scores, labels)
        if fpr.size == 0:
            return float("nan")
        return float(np.trapz(tpr, fpr))

    def pr_auc(self, scores: Sequence[float], labels: Sequence[int]) -> float:
        """
        Compute precision-recall AUC.

        :param scores: array-like of scores
        :param labels: binary labels (1=active, 0=decoy)
        :returns: PR AUC (float) or nan
        """
        precision, recall, _ = self.precision_recall_curve(scores, labels)
        if precision.size == 0:
            return float("nan")
        r = np.concatenate([[0.0], recall])
        p = np.concatenate([[1.0], precision])
        return float(np.trapz(p, r))

    def average_precision(
        self, scores: Sequence[float], labels: Sequence[int]
    ) -> float:
        """
        Alias for PR AUC (Average Precision).
        """
        return self.pr_auc(scores, labels)

    def enrichment_factor(
        self, scores: Sequence[float], labels: Sequence[int], fraction: float = 0.01
    ) -> float:
        """
        Enrichment Factor at top fraction.

        Delegates to :meth:`enrichment_at_k` to avoid double-preparation.

        :param scores: array-like of scores
        :param labels: binary labels (1 active, 0 decoy)
        :param fraction: fraction of top-ranked list to consider (0< fraction <=1)
        :returns: EF value (float) or nan on degenerate inputs
        """
        n = len(labels)
        if n == 0:
            return float("nan")
        k = max(1, int(math.ceil(fraction * n)))
        return self.enrichment_at_k(scores, labels, k=k)

    def enrichment_at_k(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        k: Optional[int] = None,
        fraction: Optional[float] = None,
    ) -> float:
        """
        Enrichment Factor at absolute top-k (or fraction).

        :param scores: array-like of scores
        :param labels: binary labels (1 active, 0 decoy)
        :param k: top-k count (mutually exclusive with fraction)
        :param fraction: fraction of list to use as top (0< fraction <=1)
        :returns: EF value (float) or nan
        """
        s, y = self._prepare(scores, labels)
        n = len(y)
        if n == 0:
            return float("nan")
        if fraction is not None:
            k = max(1, int(math.ceil(fraction * n)))
        if k is None:
            raise ValueError("Either k or fraction must be provided")
        k = min(max(1, int(k)), n)
        order = self._rank_order(s)
        topk = y[order[:k]]
        hits_top = float(topk.sum())
        base_rate = float(y.mean()) if n > 0 else 0.0
        if base_rate == 0:
            return float("nan")
        ef = (hits_top / k) / base_rate
        return float(ef)

    def precision_at_k(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        k: Optional[int] = None,
        fraction: Optional[float] = None,
    ) -> float:
        """
        Precision at top-k.

        :param scores: array-like of scores
        :param labels: binary labels
        :param k: top-k count (exclusive with fraction)
        :param fraction: fraction of list to use as top
        :returns: precision value (float) or nan
        """
        s, y = self._prepare(scores, labels)
        n = len(y)
        if n == 0:
            return float("nan")
        if fraction is not None:
            k = max(1, int(math.ceil(fraction * n)))
        if k is None:
            raise ValueError("Either k or fraction must be provided")
        k = min(max(1, int(k)), n)
        order = self._rank_order(s)
        topk = y[order[:k]]
        return float(topk.sum() / k)

    def recall_at_k(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        k: Optional[int] = None,
        fraction: Optional[float] = None,
    ) -> float:
        """
        Recall at top-k: fraction of all actives retrieved in top-k.

        :returns: recall (float) or nan
        """
        s, y = self._prepare(scores, labels)
        P = (y == 1).sum()
        if P == 0:
            return float("nan")
        n = len(y)
        if fraction is not None:
            k = max(1, int(math.ceil(fraction * n)))
        if k is None:
            raise ValueError("Either k or fraction must be provided")
        k = min(max(1, int(k)), n)
        order = self._rank_order(s)
        topk = y[order[:k]]
        return float(topk.sum() / P)

    def topn_success(
        self,
        ligand_ids: Sequence[str],
        scores: Sequence[float],
        labels: Sequence[int],
        n: int = 1,
    ) -> float:
        """
        Top-N success rate computed per ligand.

        :param ligand_ids: sequence of ligand identifiers aligned with scores & labels
        :param scores: sequence of scores (lower better by default)
        :param labels: sequence of binary labels (1 active, 0 decoy)
        :param n: value of N for Top-N (default 1)
        :returns: fraction of ligands whose best-scored pose is active (float)
        """
        import pandas as pd

        if not (len(ligand_ids) == len(scores) == len(labels)):
            raise ValueError("ligand_ids, scores and labels must have the same length")
        df = pd.DataFrame({"ligand_id": ligand_ids, "score": scores, "label": labels})
        # best (lowest) score per ligand
        best = df.sort_values("score").groupby("ligand_id", sort=False).first()
        if best.empty:
            return float("nan")
        success = (best["label"] == 1).sum()
        total = len(best)
        return float(success / total)

    def bedroc(
        self, scores: Sequence[float], labels: Sequence[int], alpha: float = 20.0
    ) -> float:
        """
        BEDROC (Truchon & Bayly 2007) implementation.

        :param scores: array-like of scores
        :param labels: binary labels (1=active, 0=decoy)
        :param alpha: alpha parameter controlling early recognition emphasis
        :type alpha: float
        :returns: normalized BEDROC in [0,1] or nan
        """
        s, y = self._prepare(scores, labels)
        n = len(y)
        if n == 0:
            return float("nan")
        order = self._rank_order(s)
        y_sorted = y[order]
        m = int(y_sorted.sum())
        if m == 0:
            return float("nan")
        ri = np.flatnonzero(y_sorted == 1) + 1  # 1-based indices
        ka = float(alpha)
        exp_term = np.exp(-ka * (ri - 1) / n)
        sum_exp = float(exp_term.sum())
        fac = ka / (1.0 - math.exp(-ka))
        Ra = (sum_exp / m) * fac
        Ra_min = 1.0
        exp_best = float(np.exp(-ka * (np.arange(1, m + 1) - 1) / n).sum())
        Ra_max = (exp_best / m) * fac
        if Ra_max - Ra_min == 0:
            return float("nan")
        return float((Ra - Ra_min) / (Ra_max - Ra_min))

    def mcc(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        threshold: Optional[float] = None,
    ) -> float:
        """
        Compute Matthews Correlation Coefficient (MCC).

        If ``threshold`` is provided, predictions are computed on the original
        score scale using the conventional interpretation:

            pred = score >= threshold

        (this keeps threshold behavior intuitive for ML-style scores). If
        your scores are lower-is-better, choose thresholds accordingly.

        If ``threshold`` is None, the function searches across unique score
        thresholds in the prepared ranking space (higher-is-better) and returns
        the maximum MCC found.

        :param scores: array-like of scores
        :param labels: binary labels (1=active, 0=decoy)
        :param threshold: optional threshold on scores (in the same scale passed in)
        :returns: MCC value (float) or nan for degenerate inputs
        """
        s_prepared, y = self._prepare(scores, labels)
        P = (y == 1).sum()
        N = (y == 0).sum()
        if P == 0 or N == 0:
            return float("nan")

        def _mcc_from_preds(pred):
            tp = float(((pred == 1) & (y == 1)).sum())
            tn = float(((pred == 0) & (y == 0)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            fn = float(((pred == 0) & (y == 1)).sum())
            denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if denom == 0:
                return float("nan")
            return (tp * tn - fp * fn) / denom

        # If user provided a concrete threshold, compute predictions on original score scale:
        if threshold is not None:
            s_raw = np.asarray(scores, dtype=float)
            pred = (s_raw >= float(threshold)).astype(int)
            return float(_mcc_from_preds(pred))

        # Otherwise search thresholds in prepared ranking space
        thresholds = np.unique(s_prepared)[::-1]
        best = float("-inf")
        for thr in thresholds:
            pred = (s_prepared >= thr).astype(int)
            val = _mcc_from_preds(pred)
            if np.isnan(val):
                continue
            if val > best:
                best = float(val)
        return float(best) if best != float("-inf") else float("nan")
