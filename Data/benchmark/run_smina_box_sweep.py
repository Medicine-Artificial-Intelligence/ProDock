#!/usr/bin/env python3
"""Compare ligand-derived box strategies with smina on the EGFR redocking panel.

The sweep uses one engine to select a single, panel-wide box rule before the
four-engine benchmark is repeated. Every strategy uses the same cleaned,
single-fragment crystal ligand for box construction and RMSD evaluation.

Run through ``run_smina_box_sweep_one_cpu.sh`` so CPU affinity and numerical
library thread counts are restricted to one CPU.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Callable

import egfr_config as cfg  # noqa: E402  (must precede prodock import)

from prodock import prodock  # noqa: E402
from prodock.database import PoseQuery  # noqa: E402
from prodock.postprocess.metrics import DockEvaluator  # noqa: E402
from prodock.preprocess.gridbox import GridBox  # noqa: E402

SWEEP_DIR = cfg.BENCH_DIR / "smina_box_sweep"
REPORT_CSV = SWEEP_DIR / "results.csv"
REPORT_MD = SWEEP_DIR / "results.md"

BoxBuilder = Callable[[GridBox], GridBox]


CANDIDATES: dict[str, BoxBuilder] = {
    # Corrected baseline: single-fragment reference, 4 Å on every side.
    "pad4_iso": lambda gb: gb.from_ligand_pad(pad=4.0, isotropic=True),
    # Tests whether a slightly smaller cubic search space improves ranking.
    "pad3_iso": lambda gb: gb.from_ligand_pad(pad=3.0, isotropic=True),
    # Avoids empty cubic volume while retaining 4 Å per-axis padding.
    "pad4_aniso": lambda gb: gb.from_ligand_pad(pad=4.0, isotropic=False),
    # Common fixed-cube alternative centered at the ligand centroid.
    "fixed20": lambda gb: gb.from_centroid_fixed((20.0, 20.0, 20.0)),
    # Multiplicative alternative with proportional rather than fixed padding.
    "scale1p5_iso": lambda gb: gb.from_ligand_scale(
        scale=1.5,
        isotropic=True,
    ),
}


def _assert_one_cpu() -> None:
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        if len(affinity) != 1:
            raise RuntimeError(
                "The box sweep must run on exactly one CPU; "
                f"current affinity is {affinity}. "
                "Use run_smina_box_sweep_one_cpu.sh."
            )


def _reference(rec: dict) -> Path:
    path = cfg.REF_DIR / f"{rec['pdb_id']}_native.sdf"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned redocking reference: {path}. " "Run run_benchmark.py once to prepare references."
        )
    return path


def _box(candidate: str, reference: Path):
    gb = GridBox().load_ligand(reference)
    CANDIDATES[candidate](gb)
    return tuple(gb.center), tuple(gb.size)


def _is_complete(db_path: Path, pdb_id: str) -> bool:
    if not db_path.exists():
        return False
    try:
        pose = PoseQuery(str(db_path)).pose(
            receptor_id=pdb_id,
            ligand_id=f"native_{pdb_id}",
            engine="smina",
            pose_rank=1,
        )
        return pose is not None and pose.mol is not None
    except Exception:
        return False


def _run_candidate(candidate: str, *, dry_run: bool) -> None:
    candidate_dir = SWEEP_DIR / candidate
    db_path = candidate_dir / "redocking.db"

    for rec in cfg.RECEPTORS:
        pdb_id = rec["pdb_id"]
        reference = _reference(rec)
        center, size = _box(candidate, reference)
        print(
            f"{candidate:12s} {pdb_id} " f"center={center} size={size}",
            flush=True,
        )
        if dry_run:
            continue
        if _is_complete(db_path, pdb_id):
            print(f"  checkpoint: {pdb_id} already complete", flush=True)
            continue

        source_project = cfg.REDOCK_DIR / "projects" / pdb_id
        receptor_pdbqt = source_project / pdb_id / "filtered_protein" / f"{pdb_id}.pdbqt"
        ligand_dir = source_project / "ligands"
        if not receptor_pdbqt.exists():
            raise FileNotFoundError(f"Missing prepared receptor for sweep: {receptor_pdbqt}")
        if not (ligand_dir / f"native_{pdb_id}.pdbqt").exists():
            raise FileNotFoundError(f"Missing prepared native ligand in: {ligand_dir}")

        prepared_rec = dict(
            receptor_id=pdb_id,
            receptor_pdbqt=str(receptor_pdbqt),
            center=list(center),
            size=list(size),
        )
        prodock(
            project_dir=str(candidate_dir / "projects" / pdb_id),
            prepared_receptors=[prepared_rec],
            ligand_dir=str(ligand_dir),
            engines=["smina"],
            extract_interaction=False,
            save_to_database=True,
            db_name=str(db_path),
            replace=True,
            cpu=1,
            n_jobs=1,
            exhaustiveness=32,
            n_poses=20,
            seed=42,
        )


def _best_rmsd(
    query: PoseQuery,
    evaluator: DockEvaluator,
    *,
    pdb_id: str,
    reference: Path,
    top_n: int,
):
    values: list[tuple[int, float]] = []
    for rank in range(1, top_n + 1):
        pose = query.pose(
            receptor_id=pdb_id,
            ligand_id=f"native_{pdb_id}",
            engine="smina",
            pose_rank=rank,
        )
        if pose is None or pose.mol is None:
            continue
        values.append((rank, float(evaluator.rmsd(str(reference), pose.mol))))
    return min(values, key=lambda item: item[1]) if values else (None, None)


def _write_report(candidates: list[str]) -> None:
    evaluator = DockEvaluator(engine="rdkit")
    rows: list[dict] = []

    for candidate in candidates:
        db_path = SWEEP_DIR / candidate / "redocking.db"
        if not db_path.exists():
            continue
        query = PoseQuery(str(db_path))
        for rec in cfg.RECEPTORS:
            pdb_id = rec["pdb_id"]
            reference = _reference(rec)
            center, size = _box(candidate, reference)
            top1_rank, top1 = _best_rmsd(
                query,
                evaluator,
                pdb_id=pdb_id,
                reference=reference,
                top_n=1,
            )
            if top1 is None:
                continue
            top3_rank, top3 = _best_rmsd(
                query,
                evaluator,
                pdb_id=pdb_id,
                reference=reference,
                top_n=3,
            )
            top5_rank, top5 = _best_rmsd(
                query,
                evaluator,
                pdb_id=pdb_id,
                reference=reference,
                top_n=5,
            )
            top20_rank, top20 = _best_rmsd(
                query,
                evaluator,
                pdb_id=pdb_id,
                reference=reference,
                top_n=20,
            )
            rows.append(
                {
                    "strategy": candidate,
                    "receptor": pdb_id,
                    "center": " x ".join(f"{v:.3f}" for v in center),
                    "size": " x ".join(f"{v:.3f}" for v in size),
                    "top1_rmsd": top1,
                    "top3_rmsd": top3,
                    "top3_rank": top3_rank,
                    "top5_rmsd": top5,
                    "top5_rank": top5_rank,
                    "top20_rmsd": top20,
                    "top20_rank": top20_rank,
                }
            )

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "strategy",
        "receptor",
        "center",
        "size",
        "top1_rmsd",
        "top3_rmsd",
        "top3_rank",
        "top5_rmsd",
        "top5_rank",
        "top20_rmsd",
        "top20_rank",
    ]
    with REPORT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Smina box-strategy sweep",
        "",
        "Uniform settings: one CPU, exhaustiveness 32, seed 42, 20 poses.",
        "",
        "| Strategy | Receptor | Box size (Å) | Top 1 | Best top 3 | Best top 5 | Best top 20 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['receptor']} | {row['size']} | "
            f"{row['top1_rmsd']:.2f} | "
            f"{row['top3_rmsd']:.2f} (r{row['top3_rank']}) | "
            f"{row['top5_rmsd']:.2f} (r{row['top5_rank']}) | "
            f"{row['top20_rmsd']:.2f} (r{row['top20_rank']}) |"
        )

    lines.extend(["", "## Rank-1 success by strategy", ""])
    for candidate in candidates:
        vals = [row["top1_rmsd"] for row in rows if row["strategy"] == candidate]
        if vals:
            passed = sum(value <= 2.0 for value in vals)
            lines.append(f"- {candidate}: {passed}/{len(vals)} " f"({passed / len(vals):.0%})")
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"report: {REPORT_MD}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        choices=sorted(CANDIDATES),
        help="run only this candidate; repeat to select more than one",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the computed boxes without docking",
    )
    args = parser.parse_args()

    _assert_one_cpu()
    cfg.ensure_dirs()
    candidates = args.candidate or list(CANDIDATES)
    for candidate in candidates:
        _run_candidate(candidate, dry_run=args.dry_run)
    if not args.dry_run:
        _write_report(candidates)


if __name__ == "__main__":
    main()
