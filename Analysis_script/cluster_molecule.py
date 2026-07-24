import argparse
import csv
import pathlib
import shutil
import subprocess
import sys
from typing import List
import os
import pandas as pd
from natsort import natsorted

from joblib import Parallel, delayed


def rmsd_obabel(conformation_1: str, conformation_2: str) -> float:

    """Return heavy‑atom RMSD (Å) between conformation_1 and conformation_2 using Open Babel."""
    cmd: List[str] = [
        "obrms",
        str(conformation_1),
        str(conformation_2),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "obabel failed")

    for line in proc.stdout.splitlines():
        try:
            return float(line.split(" ")[2])
        except (IndexError, ValueError):
            continue
    raise ValueError(f"RMSD not found in obabel output for {conformation_1.name} vs {conformation_2.name}\n{proc.stdout}")

def cluster_sdfs(
    source_dir: str,
    threshold: float,
    outdir: str,
    njobs: int,
) -> str:
    """Cluster every .sdf in *source_dir using Joblib‑parallel RMSD comparisons."""
    os.makedirs(outdir, exist_ok=True)
    csv_name = os.path.basename(source_dir)
    sdf_files = []
    for name in os.listdir(source_dir):
        if name.endswith(".sdf"):
            sdf_files.append(os.path.join(name))
    if not sdf_files:
        sys.exit(f"No .sdf files found in {source_dir}")

    clusters: List[dict] = []  # {"rep": Path, "members": List[Path]}
    current_dir = os.getcwd()
    os.chdir(source_dir)
    csv_list = []

    for sdf in natsorted(sdf_files):

        if not clusters:  # first molecule seeds cluster 1
            clusters.append({"rep": sdf, "members": sdf, "rmsd_to_rep": 0.0})
            csv_list.append([sdf[:-4], sdf[:-4], 0.0])
            continue

        # Compute RMSDs to all representatives *in parallel*
        rmsds: List[float] = Parallel(n_jobs=njobs, backend="loky")(
            delayed(rmsd_obabel)(cl["rep"], sdf) for cl in clusters
        )

        placed = False
        for cl, rmsd in zip(clusters, rmsds):
            if rmsd <= threshold:
                csv_list.append([cl["rep"][:-4], sdf[:-4], rmsd])
                placed = True
                
        if not placed:
            clusters.append({"rep": sdf, "members": [sdf], "rmsd_to_rep": 0.0})
            csv_list.append([sdf[:-4], sdf[:-4], 0.0])
            continue
    os.chdir(current_dir)

    df = pd.DataFrame(csv_list, columns=["representative", "member", "rmsd"])
    df_sorted = df.loc[natsorted(df.index, key=lambda i: df.loc[i, "representative"])].reset_index(drop=True)
    if df_sorted['member'].duplicated().any():
        df_sorted = df_sorted.sort_values(by='rmsd', ascending=True).drop_duplicates(subset=['member'], keep='first').reset_index(drop=True)
    df_sorted["representative"].value_counts(dropna=True).reset_index(name="frequency").rename(columns={"index": "representative"}).to_csv(os.path.join(outdir, f"{csv_name}_threshold{threshold}_frequency.csv"), index=False)
    df_sorted.to_csv(os.path.join(outdir, f"{csv_name}_threshold{threshold}_cluster.csv"), index=False)

def main() -> None:
    p = argparse.ArgumentParser(description="Cluster SDF files by RMSD with Joblib parallelism.")
    p.add_argument("--source_dir", type=str, help="Directory containing .sdf files")
    p.add_argument("--threshold", type=float, default=2.0,
                   help=f"RMSD threshold in Å (default 2.0)")
    p.add_argument("--njobs", type=int, default=-1,
                   help="Number of parallel njobs (default: all cores)")
    p.add_argument("--output_dir", type=str, default=os.path.join(args.source_dir, "clusters"),
                   help="Output directory (default: same as source_dir/clusters)")
    args = p.parse_args()