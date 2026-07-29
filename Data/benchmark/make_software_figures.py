"""Generate publication figures that explain ProDock as workflow software."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "Figure"

COLORS = {
    "input": "#DCE7F2",
    "prepare": "#BFE3D5",
    "campaign": "#D6E5F4",
    "execute": "#D9CBE8",
    "analyse": "#F4D5B8",
    "store": "#C9D5EF",
    "ink": "#25313C",
    "muted": "#66727D",
    "arrow": "#7A8792",
}


def card(ax, x, y, w, h, title, body, color, *, title_size=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.0,
        edgecolor=COLORS["ink"],
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["muted"],
        linespacing=1.2,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=COLORS["arrow"],
        )
    )


def software_overview() -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.0))
    for ax in axes:
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3.6)
        ax.axis("off")

    ax = axes[0]
    ax.text(
        0,
        3.42,
        "A  Software and data flow",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    card(ax, 0.1, 1.45, 1.55, 1.25, "Inputs", "PDB ID / structure\nSMILES / SDF\nJSON configuration", COLORS["input"])
    stages = [
        (2.05, "Prepare", "ReceptorPrep\nLigandPrep\nGridBox", "prepare"),
        (4.02, "Compile", "validated campaign\nmany-to-many\ntask matrix", "campaign"),
        (5.99, "Execute", "engine adapters\nparallel jobs\nrestart-safe layout", "execute"),
        (7.96, "Postprocess", "poses and scores\ninteraction events\nquality metrics", "analyse"),
        (9.93, "Store & query", "normalized SQLite\nPoseQuery\nanalysis-ready tables", "store"),
    ]
    for x, title, body, key in stages:
        card(ax, x, 1.25, 1.62, 1.65, title, body, COLORS[key])
    arrow(ax, 1.68, 2.08, 2.02, 2.08)
    for left in [3.67, 5.64, 7.61, 9.58]:
        arrow(ax, left, 2.08, left + 0.32, 2.08)
    ax.add_patch(
        FancyBboxPatch(
            (2.05, 0.30),
            9.50,
            0.50,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#F3F5F7",
            edgecolor="#AAB3BB",
            linewidth=0.9,
        )
    )
    ax.text(
        6.8,
        0.55,
        "Machine-readable provenance: input paths · preparation choices · "
        "boxes · engines · seeds · parameters · software version",
        ha="center",
        va="center",
        fontsize=8.1,
        color=COLORS["muted"],
    )

    ax = axes[1]
    ax.text(
        0,
        3.42,
        "B  EGFR automation example",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    card(
        ax,
        0.15,
        1.32,
        1.65,
        1.55,
        "Campaign inputs",
        "3 receptors\n25 compounds\n4 engines",
        COLORS["input"],
        title_size=8.8,
    )
    card(
        ax,
        2.35,
        1.32,
        1.75,
        1.55,
        "Automatic expansion",
        "300 receptor–ligand–\nengine tasks\n+ 12 redocking tasks",
        COLORS["campaign"],
        title_size=8.8,
    )
    card(
        ax,
        4.65,
        1.32,
        1.75,
        1.55,
        "Execution",
        "uniform parameters\nparallel scheduling\nrestart-aware outputs",
        COLORS["execute"],
        title_size=8.8,
    )
    card(
        ax,
        6.95,
        1.32,
        1.75,
        1.55,
        "Structured results",
        "ranked poses\nscores · interactions\nSQLite relations",
        COLORS["store"],
        title_size=8.8,
    )
    arrow(ax, 1.83, 2.09, 2.32, 2.09)
    arrow(ax, 4.13, 2.09, 4.62, 2.09)
    arrow(ax, 6.43, 2.09, 6.92, 2.09)
    outputs = [
        (9.25, 2.42, "Integrity check", "12 top-pose RMSDs"),
        (9.25, 1.56, "Score query", "engine and ensemble summaries"),
        (9.25, 0.70, "Interaction query", "mean similarity ranking"),
    ]
    for x, y, title, body in outputs:
        card(ax, x, y, 2.35, 0.65, title, body, COLORS["analyse"], title_size=8.3)
    arrow(ax, 8.73, 2.09, 9.20, 2.74)
    arrow(ax, 8.73, 2.09, 9.20, 1.88)
    arrow(ax, 8.73, 2.09, 9.20, 1.02)
    ax.text(
        5.9,
        0.25,
        "Illustrative purpose: demonstrate orchestration, provenance, database "
        "reuse, and auditable analysis—not engine benchmarking.",
        ha="center",
        fontsize=8.3,
        color=COLORS["muted"],
        fontstyle="italic",
    )

    fig.tight_layout(h_pad=1.0)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "prodock_software_overview.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def graphical_abstract() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 3.0), dpi=100)
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    card(ax, 0.15, 0.80, 1.45, 1.45, "Molecular inputs", "receptors\nligands\nconfiguration", COLORS["input"])
    card(ax, 2.05, 0.80, 1.45, 1.45, "Prepare", "structures\nsearch boxes", COLORS["prepare"])
    card(ax, 3.95, 0.80, 1.45, 1.45, "Execute", "multi-engine\ncampaign", COLORS["execute"])
    card(ax, 5.85, 0.80, 1.45, 1.45, "Structure data", "poses · scores\ninteractions", COLORS["analyse"])
    card(ax, 7.75, 0.80, 1.30, 1.45, "SQLite", "query · compare\nreuse · audit", COLORS["store"])
    for left, right in [(1.62, 2.02), (3.52, 3.92), (5.42, 5.82), (7.32, 7.72)]:
        arrow(ax, left, 1.52, right, 1.52)
    ax.text(
        4.6,
        2.70,
        "ProDock: reproducible molecular docking as a connected software workflow",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    output = FIGURE_DIR / "graphical_abstract.png"
    fig.savefig(output, dpi=100, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(software_overview())
    print(graphical_abstract())
