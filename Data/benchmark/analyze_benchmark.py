#!/usr/bin/env python3
r"""Turn the benchmark databases into the paper's tables and figure.

Reads the redocking and screening databases produced by ``run_redocking.py``
and ``run_screening.py`` and writes, using only the packaged metrics API:

  * ``paper/tables/redock_rows.tex``  -- body rows of Table 2 (docking power)
  * ``paper/tables/screen_rows.tex``  -- body rows of Table 3 (screening power)
  * ``paper/Figure/fp_similarity.png`` -- interaction-fingerprint heatmap

The LaTeX tables in ``main.tex`` \input these row files when present and fall
back to red placeholders otherwise, so the paper always compiles. Everything
here is read-only with respect to the docking calculation.

Run (after the two campaigns finish):
    python paper/scripts/analyze_benchmark.py
"""

from __future__ import annotations

import argparse
import re

import egfr_config as cfg  # noqa: E402  (must precede prodock import)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prodock.database import PoseQuery  # noqa: E402
from prodock.postprocess.metrics import (  # noqa: E402
    DockEvaluator,
    ScreenEvaluator,
    success_rate,
)

log = cfg.get_logger("analyze")


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return r"\bmph{--}"
    return f"{x:.{nd}f}"


_VARIANT_SUFFIX_RE = re.compile(r"_\d+$")


def _is_active(ligand_id: str) -> bool:
    """True if ``ligand_id`` names an active compound.

    Each ligand was embedded as multiple conformer variants
    (``afatinib``, ``afatinib_1``, ``afatinib_2``, ...); strip that
    trailing ``_<n>`` suffix before matching against ``cfg.ACTIVES`` so
    every variant of an active compound is labeled consistently.
    """
    return _VARIANT_SUFFIX_RE.sub("", ligand_id) in cfg.ACTIVES


# --------------------------------------------------------------------------- #
# Table 2 : docking power (redocking RMSD vs. crystal pose)
# --------------------------------------------------------------------------- #
def docking_power() -> None:
    log.info("STEP docking power (redocking RMSD)")
    if not cfg.REDOCK_DB.exists():
        log.warning("%s missing; skipping docking power", cfg.REDOCK_DB)
        return
    q = PoseQuery(str(cfg.REDOCK_DB))
    de = DockEvaluator(engine="rdkit")

    # A receptor is part of the redocking benchmark only if its native ligand
    # yielded a valid (complete, bond-corrected) reference pose.
    included = [r for r in cfg.RECEPTORS if (cfg.REF_DIR / f"{r['pdb_id']}_native.sdf").exists()]
    excluded = [r for r in cfg.RECEPTORS if r not in included]
    if excluded:
        log.info("excluded from redocking (no valid reference): %s", ", ".join(r["pdb_id"] for r in excluded))

    rmsd = {e: {} for e in cfg.ENGINES}
    for rec in included:
        pdb = rec["pdb_id"]
        ref = cfg.REF_DIR / f"{pdb}_native.sdf"
        for eng in cfg.ENGINES:
            val = float("nan")
            try:
                pose = q.pose(receptor_id=pdb, ligand_id=f"native_{pdb}", engine=eng, pose_rank=1)
                if pose is not None and pose.mol is not None:
                    val = de.rmsd(str(ref), pose.mol)
            except Exception as exc:
                log.warning("%s/%s RMSD failed: %s", pdb, eng, exc)
            rmsd[eng][pdb] = val
            log.info("  %s / %-7s rank1 RMSD = %s", pdb, eng, _fmt(val))

    lines = []
    for rec in included:
        pdb, code = rec["pdb_id"], rec["ligand_code"]
        cells = " & ".join(_fmt(rmsd[e][pdb]) for e in cfg.ENGINES)
        lines.append(rf"\texttt{{{pdb}}} & \texttt{{{code}}} & {cells} \\")
    lines.append(r"\midrule")
    sr = " & ".join(_fmt(success_rate([rmsd[e][r["pdb_id"]] for r in included], 2.0)) for e in cfg.ENGINES)
    lines.append(rf"\multicolumn{{2}}{{@{{}}l}}{{\textbf{{Success rate ($\le$~2~\AA)}}}} & {sr} \\")
    lines.append(r"\bottomrule")

    out = cfg.TABLES_DIR / "redock_rows.tex"
    out.write_text("\n".join(lines) + "\n")
    log.info("wrote %s (%d receptors)", out, len(included))


# --------------------------------------------------------------------------- #
# Table 3 : screening power (ROC-AUC / EF / BEDROC) + consensus
# --------------------------------------------------------------------------- #
def _best_affinity_per_ligand(df):
    """Return one score per compound after removing restart duplicates.

    Interrupted benchmark restarts prepared the same compound under incremented
    names (``afatinib``, ``afatinib_1``, ``afatinib_2``, ...). Retain the
    highest suffix uniformly within each receptor, then pool receptors by the
    best affinity when more than one receptor is present.
    """
    df = df.copy()
    df["_compound_id"] = df["ligand_id"].str.replace(_VARIANT_SUFFIX_RE, "", regex=True)
    df["_variant"] = df["ligand_id"].str.extract(r"_(\d+)$", expand=False).fillna("0").astype(int)
    latest = df.groupby(["receptor_id", "_compound_id"], sort=False)["_variant"].transform("max")
    df = df[df["_variant"] == latest]
    best = df.sort_values("affinity").groupby("_compound_id", sort=True).first()
    return best.index.to_numpy(), best["affinity"].to_numpy(dtype=float)


def screening_power() -> None:
    log.info("STEP screening power (ROC/EF/BEDROC)")
    if not cfg.SCREEN_DB.exists():
        log.warning("%s missing; skipping screening power", cfg.SCREEN_DB)
        return
    q = PoseQuery(str(cfg.SCREEN_DB))
    se = ScreenEvaluator(higher_is_better=False)

    lines = []
    # per-ligand rank accumulator across every (engine, receptor) screen
    rank_sum: dict[str, list[float]] = {}

    for eng in cfg.ENGINES:
        # Pooled: each ligand is scored by its best pose across the configured
        # receptors. Filter explicitly so stale rows from a superseded receptor
        # panel cannot affect a re-analysis of an existing database.
        df = q.poses(engine=eng, pose_rank=1, include_mol=False, as_dataframe=True)
        receptor_ids = {r["pdb_id"] for r in cfg.RECEPTORS}
        df = df[df["receptor_id"].isin(receptor_ids)]
        ligs, scores = _best_affinity_per_ligand(df)
        labels = np.array([1 if _is_active(lg) else 0 for lg in ligs], int)
        row = " & ".join(
            _fmt(v)
            for v in (
                se.auc_roc(scores, labels),
                se.enrichment_factor(scores, labels, 0.01),
                se.enrichment_factor(scores, labels, 0.05),
                se.bedroc(scores, labels, alpha=20.0),
            )
        )
        lines.append(rf"\texttt{{{eng}}} & {row} \\")

        # accumulate per-receptor ranks for the consensus row
        for rec in cfg.RECEPTORS:
            sub = df[df["receptor_id"] == rec["pdb_id"]]
            if sub.empty:
                continue
            sl, ss = _best_affinity_per_ligand(sub)
            # Average ranks within tied affinity groups; enumerating a stable
            # sort would make consensus depend on ligand-id order.
            channel_ranks = pd.Series(ss).rank(method="average", ascending=True).to_numpy()
            for ligand, rank in zip(sl, channel_ranks):
                rank_sum.setdefault(ligand, []).append(float(rank))

    lines.append(r"\midrule")
    if rank_sum:
        ligs = list(rank_sum)
        consensus = np.array([np.mean(rank_sum[lg]) for lg in ligs])  # lower better
        clabels = np.array([1 if _is_active(lg) else 0 for lg in ligs], int)
        row = " & ".join(
            _fmt(v)
            for v in (
                se.auc_roc(consensus, clabels),
                se.enrichment_factor(consensus, clabels, 0.01),
                se.enrichment_factor(consensus, clabels, 0.05),
                se.bedroc(consensus, clabels, alpha=20.0),
            )
        )
    else:
        row = " & ".join([r"\bmph{--}"] * 4)
    lines.append(rf"\textbf{{Mean channel rank}} & {row} \\")
    lines.append(r"\bottomrule")

    out = cfg.TABLES_DIR / "screen_rows.tex"
    out.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", out)


# --------------------------------------------------------------------------- #
# Figure : interaction-fingerprint Tanimoto heatmap
# --------------------------------------------------------------------------- #
def fingerprint_heatmap(receptor: str = "4ZAU") -> None:
    if not cfg.SCREEN_DB.exists():
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log.warning("matplotlib unavailable; skipping heatmap")
        return

    q = PoseQuery(str(cfg.SCREEN_DB))
    # One row per ligand: fix a single representative engine so pose_id
    # ("<receptor>__<ligand>__<engine>__pose<rank>") maps 1:1 onto ligands,
    # then strip it back down to the ligand name for axis labels.
    fp = q.fingerprint(receptor_id=receptor, engine="vina", pose_rank=1, mode="binary", index_by="pose_id")
    if fp is None or fp.empty:
        log.warning("no fingerprint rows; skipping heatmap")
        return

    X = (fp.to_numpy() > 0).astype(float)
    inter = X @ X.T
    row = X.sum(1)
    union = row[:, None] + row[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    labels = [pid.split("__")[1] if "__" in pid else pid for pid in fp.index]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"Interaction-fingerprint Tanimoto similarity ({receptor})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Tanimoto")
    fig.tight_layout()
    out = cfg.FIGURE_DIR / "fp_similarity.png"
    fig.savefig(out, dpi=200)
    log.info("wrote %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--redocking-only",
        action="store_true",
        help="compute only the redocking RMSD table",
    )
    args = parser.parse_args()

    cfg.ensure_dirs()
    log.info("START analysis")
    docking_power()
    if not args.redocking_only:
        screening_power()
        fingerprint_heatmap()
    log.info("DONE analysis. Recompile the paper to pick up the tables/figure.")


if __name__ == "__main__":
    main()
