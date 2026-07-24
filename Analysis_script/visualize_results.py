"""
Publication-quality visualizations for Optuna Reranking Benchmark Results.
Generates figures suitable for academic publication from 9 JSON result files.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import seaborn as sns  # noqa: E402

# Config
BASE = Path(__file__).parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

SCORING = ["aff", "cnn", "combined"]
METRICS = ["roc", "pr", "log"]
SCORING_LABELS = {"aff": "Affinity", "cnn": "CNNaffinity", "combined": "CNN×Affinity"}
METRIC_LABELS = {"roc": "ROC-AUC",  "pr": "PR-AUC",      "log": "LogAUC"}
METRIC_KEYS = {
    "roc": ("test_optimized_roc_auc",  "test_baseline_roc_auc",  "roc_auc",  "roc-auc_improvement"),
    "pr":  ("test_optimized_pr-auc",   "test_baseline_pr-auc",   "pr-auc",   "pr-auc_improvement"),
    "log": ("test_optimized_logauc",   "test_baseline_logauc",   "logauc",   "logauc_improvement"),
}

PALETTE = {"aff": "#4878CF", "cnn": "#E05A2B", "combined": "#3BA553"}
LIGHTER = {"aff": "#A8C1EA", "cnn": "#F0B49A", "combined": "#9ED4AB"}

# Publication style
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype":     42,  # embeds fonts for vector export
    "svg.fonttype":     "none",
})

# Load data


def load_all():
    data = {}
    for s in SCORING:
        for m in METRICS:
            fname = BASE / f"{s}_{m}.json"
            with open(fname) as f:
                data[(s, m)] = json.load(f)
    return data


DATA = load_all()


def per_protein_df():
    """Build a flat DataFrame with one row per (protein, scoring, metric)."""
    rows = []
    for (s, m), d in DATA.items():
        opt_key, base_key, train_key, _ = METRIC_KEYS[m]
        for r in d["individual_results"]:
            row = {
                "protein":  r["protein"],
                "scoring":  s,
                "metric":   m,
                "train_opt": r.get(train_key, np.nan),
                "test_opt":  r.get(opt_key, np.nan),
                "test_base": r.get(base_key, np.nan),
                "test_delta": r.get(opt_key, np.nan) - r.get(base_key, np.nan),
                "train_delta": r.get("roc-auc_improvement") or r.get("roc_auc_improvement") or
                r.get("pr-auc_improvement") or r.get("logauc_improvement") or np.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows)


DF = per_protein_df()


# Figure 1 - Grouped bar chart: avg test performance across 9 configurations
def fig_grouped_bars():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=False)
    fig.suptitle("Average Test-Set Performance: Baseline vs. Optimised", fontsize=11, y=1.01)

    for ax, m in zip(axes, METRICS):
        opt_key, base_key, _, _ = METRIC_KEYS[m]
        x = np.arange(len(SCORING))
        w = 0.35

        bases = []
        opts = []
        for s in SCORING:
            d = DATA[(s, m)]
            results = d["individual_results"]
            bases.append(np.nanmean([r.get(base_key, np.nan) for r in results]))
            opts.append(np.nanmean([r.get(opt_key,  np.nan) for r in results]))

        bars_b = ax.bar(x - w/2, bases, w, label="Baseline (rank-1)",
                        color=[LIGHTER[s] for s in SCORING], edgecolor="white", linewidth=0.5)
        bars_o = ax.bar(x + w/2, opts,  w, label="Optimised",
                        color=[PALETTE[s] for s in SCORING], edgecolor="white", linewidth=0.5)

        # Value labels
        for bar in list(bars_b) + list(bars_o):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels([SCORING_LABELS[s] for s in SCORING], rotation=15, ha="right")
        ax.set_ylabel(METRIC_LABELS[m])
        # ax.set_title(f"Optimise: {METRIC_LABELS[m]}")
        ax.set_ylim(0, max(max(bases), max(opts)) * 1.12)

        legend_patches = [
            mpatches.Patch(color="#D0D0D0", label="Baseline (rank-1)"),
            mpatches.Patch(color="#555555", label="Optimised"),
        ]

        fig.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=2,
            frameon=False
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.tight_layout()
    fig.savefig(OUT / "fig1_grouped_bars.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_grouped_bars.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved fig1_grouped_bars")


# Figure 2 - Per-protein scatter: test baseline vs. test optimised (3×3 grid)
def fig_scatter_grid():
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle("Per-Protein Test Performance: Baseline vs. Optimised", fontsize=11, y=1.00)

    for row_i, m in enumerate(METRICS):
        for col_i, s in enumerate(SCORING):
            ax = axes[row_i][col_i]
            opt_key, base_key, _, _ = METRIC_KEYS[m]
            results = DATA[(s, m)]["individual_results"]
            bases = np.array([r.get(base_key, np.nan) for r in results])
            opts = np.array([r.get(opt_key,  np.nan) for r in results])

            lo = min(np.nanmin(bases), np.nanmin(opts)) - 0.02
            hi = max(np.nanmax(bases), np.nanmax(opts)) + 0.02
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=0)

            improved = opts > bases
            ax.scatter(bases[~improved], opts[~improved],
                       color=LIGHTER[s], edgecolors=PALETTE[s], linewidth=0.5, s=25, alpha=0.8, label="Decreased")
            ax.scatter(bases[improved],  opts[improved],
                       color=PALETTE[s],  edgecolors="white",     linewidth=0.3, s=25, alpha=0.9, label="Improved")

            # Pearson r
            mask = ~np.isnan(bases) & ~np.isnan(opts)
            r_val, p_val = stats.pearsonr(bases[mask], opts[mask])
            n_imp = improved.sum()
            ax.text(0.05, 0.95, f"r = {r_val:.2f}\nn↑ = {n_imp}/{len(bases)}",
                    transform=ax.transAxes, fontsize=7, va="top")

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            if col_i == 0:
                ax.set_ylabel(f"{METRIC_LABELS[m]}\n(Optimised)", fontsize=8)
            if row_i == 2:
                ax.set_xlabel("Baseline", fontsize=8)
            if row_i == 0:
                ax.set_title(SCORING_LABELS[s], fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "fig2_scatter_grid.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_scatter_grid.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved fig2_scatter_grid")


# Figure 3 - Δ heatmap: proteins × configurations
def fig_delta_heatmap():
    # Use ROC-AUC improvement for all configs, but show the actual test Δ
    configs = [(s, m) for m in METRICS for s in SCORING]
    col_labels = [f"{SCORING_LABELS[s]}\n{METRIC_LABELS[m]}" for s, m in configs]

    opt_key, base_key, _, _ = METRIC_KEYS["roc"]
    proteins = [r["protein"] for r in DATA[("aff", "roc")]["individual_results"]]

    matrix = np.zeros((len(proteins), len(configs)))
    for j, (s, m) in enumerate(configs):
        ok, bk = METRIC_KEYS[m][:2]
        results = DATA[(s, m)]["individual_results"]
        pmap = {r["protein"]: r for r in results}
        for i, p in enumerate(proteins):
            r = pmap[p]
            matrix[i, j] = r.get(ok, np.nan) - r.get(bk, np.nan)

    vmax = np.nanpercentile(np.abs(matrix), 95)
    fig, ax = plt.subplots(figsize=(11, 12))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins, fontsize=7)
    ax.set_title("Per-Protein Test-Set Δ (Optimised − Baseline)\nfor Each Scoring × Metric Configuration",
                 fontsize=10, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("Δ Performance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Annotate cells with value
    for i in range(len(proteins)):
        for j in range(len(configs)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=4.5, color="black" if abs(val) < vmax * 0.6 else "white")

    plt.tight_layout()
    fig.savefig(OUT / "fig3_delta_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_delta_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved fig3_delta_heatmap")


# Figure 4 - Train vs Test delta (overfitting diagnostic) - best 3 configs
def fig_train_vs_test():
    best_configs = [("aff", "roc"), ("cnn", "roc"), ("cnn", "log")]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    fig.suptitle("Train vs. Test Improvement (Δ ROC-AUC)\n— Overfitting Diagnostic", fontsize=10, y=1.01)

    for ax, (s, m) in zip(axes, best_configs):
        opt_key, base_key, train_key, impr_key = METRIC_KEYS[m]
        results = DATA[(s, m)]["individual_results"]

        train_delta = np.array([r.get(impr_key, np.nan) for r in results])
        test_delta = np.array([r.get(opt_key, np.nan) - r.get(base_key, np.nan) for r in results])
        proteins = [r["protein"] for r in results]

        mask = ~np.isnan(train_delta) & ~np.isnan(test_delta)
        r_val, p_val = stats.pearsonr(train_delta[mask], test_delta[mask])

        scatter_c = [PALETTE[s] if td >= 0 else LIGHTER[s] for td in test_delta]
        ax.scatter(train_delta, test_delta, c=scatter_c, s=30, edgecolors="white",
                   linewidth=0.3, alpha=0.85, zorder=3)

        # Add protein labels for outliers
        combined = train_delta - test_delta
        threshold = np.nanpercentile(np.abs(combined), 85)
        for i, (td, te, prot) in enumerate(zip(train_delta, test_delta, proteins)):
            if abs(td - te) > threshold:
                ax.annotate(prot, (td, te), fontsize=5.5, alpha=0.8,
                            xytext=(3, 3), textcoords="offset points")

        lo = min(np.nanmin(train_delta), np.nanmin(test_delta)) - 0.01
        hi = max(np.nanmax(train_delta), np.nanmax(test_delta)) + 0.01
        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax.axvline(0, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4, zorder=0)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Train Δ {METRIC_LABELS[m]}", fontsize=8)
        ax.set_ylabel(f"Test Δ {METRIC_LABELS[m]}", fontsize=8)
        ax.set_title(f"{SCORING_LABELS[s]} — {METRIC_LABELS[m]}", fontsize=9)
        ax.text(0.05, 0.95, f"r = {r_val:.2f}",
                transform=ax.transAxes, fontsize=7.5, va="top")

    plt.tight_layout()
    fig.savefig(OUT / "fig4_train_vs_test.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4_train_vs_test.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved fig4_train_vs_test")


# Figure 5 - Violin/box: distribution of per-protein test Δ across configs
def fig_delta_distribution():
    rows = []
    for s in SCORING:
        for m in METRICS:
            opt_key, base_key, _, _ = METRIC_KEYS[m]
            results = DATA[(s, m)]["individual_results"]
            for r in results:
                delta = r.get(opt_key, np.nan) - r.get(base_key, np.nan)
                rows.append({
                    "scoring": SCORING_LABELS[s],
                    "metric":  METRIC_LABELS[m],
                    "delta":   delta,
                })
    df = pd.DataFrame(rows).dropna()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=False)
    fig.suptitle("Distribution of Per-Protein Test-Set Improvement (Optimised − Baseline)",
                 fontsize=10, y=1.01)

    for ax, m_key in zip(axes, METRICS):
        m_label = METRIC_LABELS[m_key]
        sub = df[df["metric"] == m_label]
        palette = {SCORING_LABELS[s]: PALETTE[s] for s in SCORING}

        sns.violinplot(data=sub, x="scoring", y="delta", hue="scoring",
                       palette=palette, legend=False,
                       inner="box", ax=ax, linewidth=0.8, cut=0.5)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.7)

        # Per-scoring median annotation
        for i, s in enumerate(SCORING):
            vals = sub[sub["scoring"] == SCORING_LABELS[s]]["delta"].dropna()
            med = vals.median()
            ax.text(i, vals.max() + 0.005, f"med={med:+.3f}", ha="center",
                    fontsize=6.5, color=PALETTE[s])

        ax.set_xlabel("")
        ax.set_ylabel(f"Δ {m_label}" if ax == axes[0] else "")
        ax.set_title(f"Optimise: {m_label}", fontsize=9)
        ax.set_xticks(range(len(SCORING)))
        ax.set_xticklabels([SCORING_LABELS[s] for s in SCORING], rotation=12, ha="right")

    plt.tight_layout()
    fig.savefig(OUT / "fig5_delta_distribution.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig5_delta_distribution.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved fig5_delta_distribution")


# Figure 6 - Lollipop chart: per-protein test ROC-AUC for best configuration
def fig_lollipop_best():
    # Best config by avg test improvement: cnn_roc
    s, m = "cnn", "roc"
    opt_key, base_key, _, _ = METRIC_KEYS[m]
    results = DATA[(s, m)]["individual_results"]
    proteins = [r["protein"] for r in results]
    bases = np.array([r[base_key] for r in results])
    opts = np.array([r[opt_key] for r in results])
    deltas = opts - bases
    order = np.argsort(deltas)

    fig, ax = plt.subplots(figsize=(5, 9))
    y = np.arange(len(proteins))

    for i, idx in enumerate(order):
        color = PALETTE[s] if deltas[idx] >= 0 else LIGHTER[s]
        ax.plot([bases[idx], opts[idx]], [i, i], color=color, lw=1.2, zorder=2)
        ax.scatter(bases[idx], i, color="#888888", s=20, zorder=3, marker="o")
        ax.scatter(opts[idx],  i, color=color,     s=22, zorder=4, marker="D")

    ax.axvline(0.5, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([proteins[i] for i in order], fontsize=7)
    ax.set_xlabel("Test ROC-AUC", fontsize=9)
    ax.set_title(f"Per-Protein Test ROC-AUC\n{SCORING_LABELS[s]} scoring, optimised for {METRIC_LABELS[m]}",
                 fontsize=9)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="#888888", markersize=5, lw=0, label="Baseline (rank-1)"),
        plt.Line2D([0], [0], marker="D", color=PALETTE[s], markersize=5, lw=0, label="Optimised"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=7.5)
    plt.tight_layout()
    fig.savefig(OUT / "fig6_lollipop_best.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig6_lollipop_best.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved fig6_lollipop_best")


# Figure 7 - Summary statistics table figure
def fig_summary_table():
    rows = []
    for s in SCORING:
        for m in METRICS:
            opt_key, base_key, _, _ = METRIC_KEYS[m]
            results = DATA[(s, m)]["individual_results"]
            bases = [r.get(base_key, np.nan) for r in results]
            opts = [r.get(opt_key,  np.nan) for r in results]
            deltas = np.array(opts) - np.array(bases)
            mask = ~np.isnan(deltas)
            n_imp = (deltas[mask] > 0).sum()
            rows.append({
                "Scoring": SCORING_LABELS[s],
                "Opt. Metric": METRIC_LABELS[m],
                "Baseline (mean)": f"{np.nanmean(bases):.3f}",
                "Optimised (mean)": f"{np.nanmean(opts):.3f}",
                "Δ (mean)": f"{np.nanmean(deltas):+.3f}",
                "Δ (median)": f"{np.nanmedian(deltas):+.3f}",
                "# Improved": f"{n_imp}/{mask.sum()}",
            })
    df_table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color header
    for j in range(len(df_table.columns)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by scoring method
    color_map = {"Affinity": "#EBF3FB", "CNNaffinity": "#FDF2EE", "CNN×Affinity": "#EEF7F0"}
    for i, row in enumerate(df_table.itertuples()):
        bg = color_map.get(row[1], "white")
        for j in range(len(df_table.columns)):
            table[i + 1, j].set_facecolor(bg)

    # Highlight positive Δ
    delta_col = list(df_table.columns).index("Δ (mean)")
    for i, row in enumerate(df_table.itertuples()):
        val = float(row[delta_col + 1])
        if val > 0:
            table[i + 1, delta_col].set_facecolor("#C8EFC8")
        elif val < 0:
            table[i + 1, delta_col].set_facecolor("#F5C6C6")

    ax.set_title("Summary: Average Test-Set Performance Across All 9 Configurations",
                 fontsize=10, pad=20, y=0.95)
    plt.tight_layout()
    fig.savefig(OUT / "fig7_summary_table.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig7_summary_table.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved fig7_summary_table")


# Figure 8 - Threshold heatmap for best config (cnn_roc)
def fig_threshold_heatmap():
    results = DATA[("cnn", "roc")]["individual_results"]
    proteins = [r["protein"] for r in results]

    # Collect all threshold keys
    all_keys = set()
    for r in results:
        all_keys.update(r.get("thresholds", {}).keys())
    all_keys = sorted(all_keys)

    # Build matrix of z-scored thresholds (for visual comparison across different scales)
    matrix = np.full((len(proteins), len(all_keys)), np.nan)
    for i, r in enumerate(results):
        for j, k in enumerate(all_keys):
            matrix[i, j] = r.get("thresholds", {}).get(k, np.nan)

    # Z-score per column (threshold) for visual comparison
    matrix_z = np.full_like(matrix, np.nan)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        mask = ~np.isnan(col)
        if mask.sum() > 1:
            matrix_z[mask, j] = stats.zscore(col[mask])

    # Clean threshold key labels
    clean_keys = [k.replace("_gnina", "\n(GNINA)").replace("_diffdock", "\n(DiffDock)")
                  .replace("%", "Pct").replace("_", " ") for k in all_keys]

    fig, ax = plt.subplots(figsize=(8, 11))
    im = ax.imshow(matrix_z, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")

    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels(clean_keys, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins, fontsize=7)
    ax.set_title("Optimised Threshold Values per Protein\n(Z-scored per parameter; CNNaffinity, ROC-AUC)",
                 fontsize=9, pad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("Z-score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(OUT / "fig8_threshold_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig8_threshold_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved fig8_threshold_heatmap")


# Run all
if __name__ == "__main__":
    print(f"Saving figures to: {OUT}\n")
    fig_grouped_bars()
    fig_scatter_grid()
    fig_delta_heatmap()
    # fig_train_vs_test()
    # fig_delta_distribution()
    # fig_lollipop_best()
    # fig_summary_table()
    # fig_threshold_heatmap()
    print(f"\nDone. {len(list(OUT.glob('*.png')))} PNG + {len(list(OUT.glob('*.pdf')))} PDF files written.")
