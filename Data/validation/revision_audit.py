"""LoLO Blue revision evidence: measurements, never substitute for docking benchmarks.

Run with the prodock environment's Python. Synthetic tests use one repeated
3D molecule; measured capacity must not be described as screened-library size.
Existing project databases are inspected through read-only SQLite connections.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prodock.database import PoseDatabase, PoseQuery  # noqa: E402


def measure(n: int, output: Path) -> dict:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    rows = [
        dict(
            receptor_id=f"R{i % 2}",
            ligand_id=f"L{i // 20}",
            engine=("vina" if i % 4 < 2 else "smina"),
            pose_rank=(i % 20) // 4 + 1,
            affinity=-float(i % 11),
            mol=mol,
        )
        for i in range(n)
    ]
    frame = pd.DataFrame(rows)
    path = output / f"synthetic_{n}.db"
    if path.exists():
        raise FileExistsError(f"Use a fresh output directory: {path}")
    with PoseDatabase(path) as db:
        db.create_run(
            name="synthetic integrity fixture", config={"synthetic": True, "rows": n}
        )
        start = time.perf_counter()
        db.insert_dataframe(frame, replace=True)
        insertion = time.perf_counter() - start
        count = db.connection.execute("SELECT COUNT(*) FROM poses").fetchone()[0]
        start = time.perf_counter()
        db.insert_dataframe(frame.iloc[: min(n, 100)], replace=True)
        replay = time.perf_counter() - start
        assert (
            db.connection.execute("SELECT COUNT(*) FROM poses").fetchone()[0]
            == count
            == n
        )
        assert db.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not db.connection.execute("PRAGMA foreign_key_check").fetchall()
        payload = db.connection.execute(
            "SELECT SUM(LENGTH(mol_blob)) FROM poses"
        ).fetchone()[0]
    with PoseQuery(path) as query:
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = query.poses(
                receptor_id="R0", engine="vina", top_rank=1, include_mol=False
            )
            times.append(time.perf_counter() - start)
        mol2 = query.pose(
            receptor_id="R0", ligand_id="L0", engine="vina", pose_rank=1
        ).mol
        assert mol2.GetNumAtoms() == mol.GetNumAtoms()
        assert all(
            abs(
                mol.GetConformer().GetAtomPosition(i).x
                - mol2.GetConformer().GetAtomPosition(i).x
            )
            < 1e-5
            for i in range(mol.GetNumAtoms())
        )
    return dict(
        synthetic=True,
        poses=n,
        ligands=n // 20,
        receptors=2,
        engines=2,
        requested_poses_per_combination=5,
        insert_seconds=insertion,
        replay_100_seconds=replay,
        filtered_query_seconds=times,
        filtered_query_rows=len(result),
        database_bytes=path.stat().st_size,
        molecule_blob_bytes=payload,
        coordinate_roundtrip=True,
        integrity="ok",
    )


def inspect_existing(path: Path) -> dict:
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        counts = {
            t: conn.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]
            for t in ["receptors", "ligands", "poses", "interactions", "runs"]
            if t in tables
        }
        return dict(
            path=str(path.relative_to(ROOT)),
            bytes=path.stat().st_size,
            counts=counts,
            integrity=conn.execute("PRAGMA integrity_check").fetchone()[0],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        python=sys.executable,
        python_version=platform.python_version(),
        rdkit=rdBase.rdkitVersion,
        sqlite=sqlite3.sqlite_version,
        platform=platform.platform(),
        caveat="Synthetic storage test; no docking, interaction profiling, or screening accuracy measured.",
        synthetic=[measure(n, args.output) for n in args.sizes],
        existing=[
            inspect_existing(p)
            for p in [
                ROOT / "dev/Testcase/testcase.db",
                ROOT / "Data/benchmark/runs/redocking/redocking.db",
            ]
            if p.exists()
        ],
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
