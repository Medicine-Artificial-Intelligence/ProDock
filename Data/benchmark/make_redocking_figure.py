#!/usr/bin/env python3
"""Create the illustrative EGFR redocking figure from the stored pose database.

The script reads the first ten engine-ranked poses for every included
receptor--engine pair, computes symmetry-aware heavy-atom RMSD to the
corresponding cocrystal pose, and writes both the underlying values and a
two-panel figure. It does not modify or repeat the docking campaign.
"""

from __future__ import annotations

import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import egfr_config as cfg  # noqa: E402
from prodock.database import PoseQuery  # noqa: E402
from prodock.postprocess.metrics import DockEvaluator  # noqa: E402

ENGINE_LABELS = {
    "smina": "smina",
    "vina": "vina",
    "qvina": "qvina",
    "qvina-w": "qvina-w",
}
ENGINE_COLORS = {
    "smina": "#4C78A8",
    "vina": "#59A14F",
    "qvina": "#B279A2",
    "qvina-w": "#F28E2B",
}
RECEPTOR_COLORS = {
    "4WKQ": "#1F4E79",
    "1M17": "#A23B3B",
    "4ZAU": "#5B4B8A",
}
RECEPTOR_MARKERS = {"4WKQ": "o", "1M17": "s", "4ZAU": "^"}


def _stable_jitter(label: str, n: int, width: float = 0.10) -> np.ndarray:
    """Return deterministic, label-specific jitter for reproducible figures."""
    seed = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


def collect_rmsd(top_n: int = 10) -> pd.DataFrame:
    """Collect RMSD values for the first ``top_n`` poses in every channel."""
    query = PoseQuery(str(cfg.REDOCK_DB))
    evaluator = DockEvaluator(engine="rdkit")
    rows: list[dict[str, object]] = []

    for receptor in cfg.RECEPTORS:
        pdb_id = receptor["pdb_id"]
        ligand_code = receptor["ligand_code"]
        reference = cfg.REF_DIR / f"{pdb_id}_native.sdf"
        if not reference.exists():
            continue
        for engine in cfg.ENGINES:
            for pose_rank in range(1, top_n + 1):
                pose = query.pose(
                    receptor_id=pdb_id,
                    ligand_id=f"native_{pdb_id}",
                    engine=engine,
                    pose_rank=pose_rank,
                )
                if pose is None or pose.mol is None:
                    continue
                rows.append(
                    {
                        "receptor": pdb_id,
                        "ligand": ligand_code,
                        "engine": engine,
                        "pose_rank": pose_rank,
                        "rmsd_angstrom": evaluator.rmsd(str(reference), pose.mol),
                    }
                )

    frame = pd.DataFrame(rows)
    expected = len(cfg.RECEPTORS) * len(cfg.ENGINES) * top_n
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} RMSD values but recovered {len(frame)}.")
    return frame


def draw_figure(frame: pd.DataFrame) -> None:
    """Draw top-10 distributions and the rank-1 receptor comparison."""
    engines = cfg.ENGINES
    positions = np.arange(1, len(engines) + 1)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.25),
        gridspec_kw={"width_ratios": [1.12, 1.0]},
    )

    # Panel A: distribution across 3 receptors x 10 pose ranks per engine.
    ax = axes[0]
    values_by_engine = [frame.loc[frame["engine"] == engine, "rmsd_angstrom"].to_numpy() for engine in engines]
    violins = ax.violinplot(
        values_by_engine,
        positions=positions,
        widths=0.76,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.35,
    )
    for body, engine in zip(violins["bodies"], engines):
        body.set_facecolor(ENGINE_COLORS[engine])
        body.set_edgecolor(ENGINE_COLORS[engine])
        body.set_alpha(0.20)
        body.set_linewidth(1.0)

    box = ax.boxplot(
        values_by_engine,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F2933", "linewidth": 1.7},
        whiskerprops={"color": "#5D6873", "linewidth": 1.0},
        capprops={"color": "#5D6873", "linewidth": 1.0},
    )
    for patch, engine in zip(box["boxes"], engines):
        patch.set_facecolor(ENGINE_COLORS[engine])
        patch.set_edgecolor("#FFFFFF")
        patch.set_alpha(0.78)

    for position, engine in zip(positions, engines):
        subset = frame[frame["engine"] == engine].reset_index(drop=True)
        for receptor in cfg.RECEPTORS:
            pdb_id = receptor["pdb_id"]
            values = subset.loc[subset["receptor"] == pdb_id, "rmsd_angstrom"].to_numpy()
            ax.scatter(
                position + _stable_jitter(f"{engine}/{pdb_id}", len(values)),
                values,
                s=18,
                marker=RECEPTOR_MARKERS[pdb_id],
                facecolor=RECEPTOR_COLORS[pdb_id],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.78,
                zorder=3,
            )

    ax.axhline(2.0, color="#C43C39", linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(
        0.98,
        2.08,
        "2 Å reference",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color="#A52A27",
        fontsize=8,
    )
    ax.set_xticks(positions, [ENGINE_LABELS[e] for e in engines])
    ax.set_ylabel(r"Heavy-atom RMSD ($\AA$)")
    ax.set_xlabel("Docking engine")
    ax.text(
        0.01,
        0.98,
        "A",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: exact engine-ranked pose-1 values, connected within receptor.
    ax = axes[1]
    rank1 = frame[frame["pose_rank"] == 1]
    offsets = {"4WKQ": -0.11, "1M17": 0.0, "4ZAU": 0.11}
    for receptor in cfg.RECEPTORS:
        pdb_id = receptor["pdb_id"]
        subset = rank1[rank1["receptor"] == pdb_id].set_index("engine").reindex(engines)
        x = positions + offsets[pdb_id]
        y = subset["rmsd_angstrom"].to_numpy()
        ax.plot(
            x,
            y,
            color=RECEPTOR_COLORS[pdb_id],
            linewidth=1.15,
            alpha=0.72,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=54,
            marker=RECEPTOR_MARKERS[pdb_id],
            color=RECEPTOR_COLORS[pdb_id],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    ax.axhline(2.0, color="#C43C39", linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_xticks(positions, [ENGINE_LABELS[e] for e in engines])
    ax.set_ylabel(r"Pose-1 RMSD ($\AA$)")
    ax.set_xlabel("Docking engine")
    ax.set_ylim(1.30, 2.06)
    ax.text(
        0.01,
        0.98,
        "B",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=RECEPTOR_MARKERS[pdb_id],
                color=RECEPTOR_COLORS[pdb_id],
                label=f"{pdb_id}/{receptor['ligand_code']}",
                markerfacecolor=RECEPTOR_COLORS[pdb_id],
                linewidth=1.1,
                markersize=6,
            )
            for receptor in cfg.RECEPTORS
            for pdb_id in [receptor["pdb_id"]]
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )

    fig.tight_layout(w_pad=2.3)
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = cfg.FIGURE_DIR / "redocking_rmsd_summary.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame = collect_rmsd(top_n=10)
    output_csv = cfg.BENCH_DIR / "redocking" / "top10_pose_rmsd.csv"
    frame.to_csv(output_csv, index=False)
    draw_figure(frame)
    print(frame.groupby("engine")["rmsd_angstrom"].describe().round(3))
    print("\nRank-1 RMSD:")
    print(frame[frame["pose_rank"] == 1].pivot(index="receptor", columns="engine", values="rmsd_angstrom").round(3))
    print(f"\nWrote {output_csv}")
    print(f"Wrote {cfg.FIGURE_DIR / 'redocking_rmsd_summary.pdf'}")


if __name__ == "__main__":
    main()
