"""Plot cluster-generated native controls; never recompute docking or qualify models."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-directory", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    execution=False,
                    inputs=["summary.csv", "report.json"],
                    outputs=["native_quality.pdf", "native_quality.png"],
                    new_docking_tasks=0,
                )
            )
        )
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor

    folder = args.quality_directory
    table = pd.read_csv(folder / "summary.csv")
    report = json.loads((folder / "report.json").read_text())
    proteins = ["1M17", "4ZAU"]
    policies = ["default_minimized", "crystal_heavy_preserved"]
    colors = {"default_minimized": "#b35806", "crystal_heavy_preserved": "#2166ac"}
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = fig.add_subplot(grid[0, :])
    x = np.arange(4)
    labels = []
    for i, protein in enumerate(proteins):
        for j, engine in enumerate(["smina", "qvina"]):
            labels.append(protein + " / " + engine)
    for p, policy in enumerate(policies):
        values = []
        failed = []
        for protein in proteins:
            for engine in ["smina", "qvina"]:
                rows = table[
                    (table.pdb == protein)
                    & (table.engine == engine)
                    & (table.policy == policy)
                ]
                value = float(rows.rank1_RMSD_A.iloc[0]) if len(rows) else float("nan")
                values.append(value)
                failed.append(not np.isfinite(value))
        offsets = x + (p - 0.5) * 0.34
        ax.bar(
            offsets,
            values,
            width=0.32,
            label=policy.replace("_", " "),
            color=colors[policy],
        )
        for at, bad in zip(offsets, failed):
            if bad:
                ax.text(
                    at, 0.1, "failed", rotation=90, ha="center", va="bottom", fontsize=8
                )
    ax.axhline(2, color="black", ls="--", lw=1, label="2 Å reference")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Best-scored pose fixed-frame RMSD (Å)")
    ax.set_title(
        "A  Preparation policy and native-pose recognition; primary program: Smina"
    )
    ax.legend(fontsize=8)
    for i, protein in enumerate(proteins):
        ax = fig.add_subplot(grid[1, i])
        ax.axis("off")
        candidates = [
            r
            for r in report["prepared_specs"]
            if r["pdb"] == protein and r["policy"] == "crystal_heavy_preserved"
        ]
        reference = candidates[0].get("experimental_reference") if candidates else None
        mol = (
            Chem.SDMolSupplier(str(reference), removeHs=True)[0] if reference else None
        )
        if mol is not None:
            mol = Chem.Mol(mol)
            rdDepictor.Compute2DCoords(mol)
            ax.imshow(
                Draw.MolToImage(
                    mol,
                    size=(700, 300),
                    legend=(
                        "Erlotinib" if protein == "1M17" else "AZD9291 (osimertinib)"
                    ),
                )
            )
        counts = {
            r["policy"]: r.get("reference_pairs_below_2A")
            for r in report["prepared_specs"]
            if r["pdb"] == protein
        }
        ax.set_title(f"{'B' if i==0 else 'C'}  {protein}: experimental native ligand")
        ax.text(
            0.5,
            -0.1,
            "Native-ligand/protein atom pairs <2 Å: " + str(counts),
            transform=ax.transAxes,
            ha="center",
            wrap=True,
            fontsize=8,
        )
    fig.text(
        0.5,
        0.01,
        "No independent ligand fitting. Secondary-program/baseline outcomes are reported; "
        "qualification uses prespecified primary controls.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(folder / "native_quality.pdf")
    fig.savefig(folder / "native_quality.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
