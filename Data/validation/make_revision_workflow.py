"""LoLO Blue: draw the revised workflow directly as a vector figure."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402


def main():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    cards = [
        (
            0.15,
            2.9,
            "Inputs",
            "PDB code / local PDB\nSMILES / prepared PDBQT\nJSON configuration",
        ),
        (
            3.95,
            2.9,
            "Preparation",
            "Protein repair and conversion\nLigand conformer generation\nBinding-site search box",
        ),
        (
            7.75,
            2.9,
            "Docking",
            "Vina / Smina / QuickVina 2\nQuickVina-W / GNINA\nLocal parallel processes",
        ),
        (
            7.75,
            0.3,
            "Parsing and profiling",
            "Individual poses and ranks\nNative docking scores\nProLIF interaction events",
        ),
        (
            3.95,
            0.3,
            "SQLite and queries",
            "Pose coordinates and scores\nResidue interaction records\nRecorded campaign settings",
        ),
    ]
    for x, y, title, text in cards:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                3.1,
                1.65,
                boxstyle="round,pad=0.06",
                edgecolor="blue",
                facecolor="#f5f7ff",
                linewidth=1.1,
            )
        )
        ax.text(
            x + 1.55,
            y + 1.34,
            title,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="blue",
        )
        ax.text(
            x + 1.55,
            y + 0.62,
            text,
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.55,
            color="blue",
        )
    for start, end in [
        ((3.36, 3.7), (3.85, 3.7)),
        ((7.15, 3.7), (7.65, 3.7)),
        ((9.3, 2.8), (9.3, 2.08)),
        ((7.65, 1.1), (7.15, 1.1)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                color="blue",
                linewidth=1.3,
            )
        )
    ax.text(0.15, 1.4, "ProDock campaign", fontsize=13, color="blue", weight="bold")
    ax.text(
        0.15,
        0.65,
        "Protein structure × ligand × program\nExplicit paths, boxes, and settings",
        fontsize=10,
        color="blue",
        linespacing=1.6,
    )
    path = (
        Path(__file__).resolve().parents[2]
        / "paper/Figure/prodock_workflow_revision.pdf"
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
