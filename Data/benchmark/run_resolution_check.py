#!/usr/bin/env python3
"""Smina-only pilot: do better-resolved, non-covalent EGFR structures redock?

2ITY (3.42 A) and 4G5J (afatinib, covalently bound to Cys797) are the two
receptors that fail rank-1 redocking in the main pad-4 panel. This script
checks two additional EGFR kinase-domain structures, chosen specifically to
separate "scoring function is bad" from "this receptor is a bad redocking
case":

- 4WKQ: 1.85 A, wild-type, non-covalent, same ligand (gefitinib / IRE) as
  2ITY. If this redocks cleanly where 2ITY does not, resolution (not
  chemistry) explains the 2ITY failure.
- 1XKK: 2.40 A, wild-type, non-covalent, lapatinib (FMM). A clean
  non-covalent alternative to 4G5J's covalent afatinib complex.

Uses only smina, the fixed pad-4 isotropic box (the panel default), and
cpu=3 (unlike the main panel's cpu=1) because this is a new, separate
exploratory check rather than part of the original controlled comparison.

Run:
    python Data/benchmark/run_resolution_check.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import egfr_config as cfg  # noqa: E402  (must precede prodock import)

from run_benchmark import prepare_native  # noqa: E402

from prodock import prodock  # noqa: E402
from prodock.database import PoseQuery  # noqa: E402
from prodock.postprocess.metrics import DockEvaluator  # noqa: E402

log = cfg.get_logger("rescheck")

CHECK_DIR = cfg.BENCH_DIR / "resolution_check"
DB_PATH = CHECK_DIR / "resolution_check.db"
REPORT_CSV = CHECK_DIR / "results.csv"
REPORT_MD = CHECK_DIR / "results.md"

NEW_RECEPTORS = [
    {"pdb_id": "4WKQ", "receptor_name": "EGFR_4WKQ", "ligand_code": "IRE", "chains": ["A"]},
    {"pdb_id": "1XKK", "receptor_name": "EGFR_1XKK", "ligand_code": "FMM", "chains": ["A"]},
    # 1.90 A, non-covalent, single-copy pyrrolopyrimidine/urea type-I ATP-competitive
    # inhibitor (T790M/L858R mutant). Candidate replacement for 4G5J.
    {"pdb_id": "3W2S", "receptor_name": "EGFR_3W2S", "ligand_code": "W2R", "chains": ["A"]},
]

CPU = 3
DOCK_PARAMS = dict(
    cpu=CPU,
    n_jobs=1,
    exhaustiveness=32,
    n_poses=20,
    seed=42,
    box_algorithm="pad",
    box_pad=4.0,
    box_isotropic=True,
)


def _is_complete(pdb_id: str) -> bool:
    if not DB_PATH.exists():
        return False
    try:
        pose = PoseQuery(str(DB_PATH)).pose(
            receptor_id=pdb_id,
            ligand_id=f"native_{pdb_id}",
            engine="smina",
            pose_rank=1,
        )
        return pose is not None and pose.mol is not None
    except Exception:
        return False


def _run_receptor(rec: dict) -> None:
    pdb, code = rec["pdb_id"], rec["ligand_code"]
    if _is_complete(pdb):
        log.info("[%s] checkpoint: already complete", pdb)
        return

    log.info("[%s] preparing native ligand %s", pdb, code)
    extract_dir = CHECK_DIR / "extract"
    smi, ref = prepare_native(pdb, code, rec["chains"], extract_dir)
    if not smi:
        log.warning("[%s] skipping (no usable native ligand)", pdb)
        return
    log.info("[%s] native SMILES (RCSB-corrected): %s", pdb, smi)

    dock_rec = dict(rec)
    dock_rec["reference_ligand"] = str(ref)

    log.info("[%s] redocking native_%s with smina (cpu=%d)", pdb, pdb, CPU)
    prodock(
        project_dir=str(CHECK_DIR / "projects" / pdb),
        receptors=[dock_rec],
        ligands=[{"id": f"native_{pdb}", "smiles": smi}],
        engines=["smina"],
        extract_interaction=False,
        save_to_database=True,
        db_name=str(DB_PATH),
        replace=True,
        **DOCK_PARAMS,
    )
    log.info("[%s] done", pdb)


def _best_rmsd(query: PoseQuery, evaluator: DockEvaluator, *, pdb_id: str, reference: Path, top_n: int):
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


def _write_report() -> None:
    evaluator = DockEvaluator(engine="rdkit")
    query = PoseQuery(str(DB_PATH))
    rows: list[dict] = []

    for rec in NEW_RECEPTORS:
        pdb_id = rec["pdb_id"]
        reference = cfg.REF_DIR / f"{pdb_id}_native.sdf"
        if not reference.exists():
            continue
        top1_rank, top1 = _best_rmsd(query, evaluator, pdb_id=pdb_id, reference=reference, top_n=1)
        if top1 is None:
            continue
        top3_rank, top3 = _best_rmsd(query, evaluator, pdb_id=pdb_id, reference=reference, top_n=3)
        top5_rank, top5 = _best_rmsd(query, evaluator, pdb_id=pdb_id, reference=reference, top_n=5)
        top20_rank, top20 = _best_rmsd(query, evaluator, pdb_id=pdb_id, reference=reference, top_n=20)
        rows.append(
            {
                "receptor": pdb_id,
                "top1_rmsd": top1,
                "top3_rmsd": top3,
                "top3_rank": top3_rank,
                "top5_rmsd": top5,
                "top5_rank": top5_rank,
                "top20_rmsd": top20,
                "top20_rank": top20_rank,
            }
        )

    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "receptor",
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
        "# Resolution/covalency check (smina only, cpu=3)",
        "",
        "pad-4 isotropic box, exhaustiveness 32, seed 42, 20 poses.",
        "",
        "| Receptor | Top 1 | Best top 3 | Best top 5 | Best top 20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['receptor']} | {row['top1_rmsd']:.2f} | "
            f"{row['top3_rmsd']:.2f} (r{row['top3_rank']}) | "
            f"{row['top5_rmsd']:.2f} (r{row['top5_rank']}) | "
            f"{row['top20_rmsd']:.2f} (r{row['top20_rank']}) |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")
    log.info("report: %s", REPORT_MD)


def main() -> None:
    cfg.ensure_dirs()
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    for rec in NEW_RECEPTORS:
        _run_receptor(rec)
    _write_report()


if __name__ == "__main__":
    main()
