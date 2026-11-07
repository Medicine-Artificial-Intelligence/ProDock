"""Small predefined natural structure panel for bounded CPU receptor preparation.

The manifest is frozen before execution. This evaluates locally chain-filtered
PDB inputs (not retrieval) with max_minimization_steps=200 rather than the
5000-iteration default; results are not a population-wide failure-rate estimate.
Each case runs in a separate timed subprocess. Malformed controls are separate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import platform
import shutil
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, count, steps):
    rows = list(csv.DictReader((ROOT / "Project/benchmark/Target_ID.csv").open()))
    selected = []
    for row in sorted(rows, key=lambda x: x["Protein"])[:count]:
        source = (
            ROOT / "Project/benchmark/filtered_protein" / f"{row['pdb_id'].lower()}.pdb"
        )
        selected.append(
            dict(
                target=row["Protein"],
                pdb_id=row["pdb_id"],
                chains=row["chains"],
                source_path=str(source.relative_to(ROOT)),
                sha256=sha(source),
                original_declared_cofactors=row["cofactors"],
            )
        )
    manifest = dict(
        selection=(
            f"First {count} target names in lexical order from committed Target_ID.csv, "
            "selected before observing preparation outcomes. Locally chain-filtered coordinate inputs; "
            "not retrieved or selected using success."
        ),
        count=len(selected),
        source_table_sha256=sha(ROOT / "Project/benchmark/Target_ID.csv"),
        parameters=dict(
            cpu_threads=1,
            max_minimization_steps=steps,
            default_max_minimization_steps=5000,
            energy_diff=10.0,
            backbone_k_kcal_per_A2=5.0,
            minimize_in_water=False,
            use_meeko=False,
            obabel_steps=500,
        ),
        cases=selected,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text())
        assert (
            previous == manifest
        ), "Manifest already frozen with different inputs/settings"
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def count_heterogens(path):
    found = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("HETATM"):
            key = (line[17:20].strip(), line[21:22], line[22:27].strip())
            found[key] = found.get(key, 0) + 1
    return [
        dict(residue=k[0], chain=k[1], residue_id=k[2], atoms=v)
        for k, v in found.items()
    ]


def worker(source, output, max_steps):
    # Hardware selection is pinned by the experiment; production source unchanged.
    sys.path.insert(0, str(ROOT))
    import prodock.preprocess.receptor.prep as prep
    import prodock.preprocess.receptor.minimizers as minimizers
    from openmm import Platform

    minimizers._choose_platform = lambda: (
        Platform.getPlatformByName("CPU"),
        {"Threads": "1"},
    )
    stages = []
    for name in ["fix_pdb", "minimize_with_openmm", "minimize_with_obabel"]:
        original = getattr(prep, name)

        def timed(*args, _name=name, _original=original, **kwargs):
            start = time.perf_counter()
            item = dict(stage=_name)
            try:
                result = _original(*args, **kwargs)
                item["success"] = True
                return result
            except Exception as e:
                item.update(success=False, error_type=type(e).__name__, error=str(e))
                raise
            finally:
                item["seconds"] = time.perf_counter() - start
                stages.append(item)

        setattr(prep, name, timed)
    start = time.perf_counter()
    result = dict(
        input=str(source), stages=stages, input_heterogens=count_heterogens(source)
    )
    try:
        obj = prep.ReceptorPrep(use_meeko=False)
        obj.prep(
            str(source),
            str(output),
            out_fmt="pdbqt",
            max_minimization_steps=max_steps,
            obabel_steps=500,
        )
        report = (
            obj.last_simulation_report
            if hasattr(obj, "last_simulation_report")
            else obj._last_simulation_report
        )
        artifact = Path(obj._final_artifact)
        lines = artifact.read_text().splitlines()
        atoms = [s for s in lines if s.startswith(("ATOM", "HETATM"))]
        if not atoms:
            raise ValueError("No atoms in resulting PDBQT")
        result.update(
            success=True,
            artifact=str(artifact),
            artifact_sha256=sha(artifact),
            output_atom_records=len(atoms),
            preparation_report=report,
            output_heterogens=count_heterogens(artifact),
        )
    except Exception as e:
        result.update(success=False, error_type=type(e).__name__, error=str(e))
    result["seconds"] = time.perf_counter() - start
    (output / "result.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )


def run_one(case, out, steps, timeout):
    folder = out / case["target"]
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(ROOT / case["source_path"]),
        "--worker-output",
        str(folder),
        "--steps",
        str(steps),
    ]
    env = os.environ.copy()
    env.update(OPENMM_CPU_THREADS="1", OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        (folder / "stdout.log").write_text(completed.stdout)
        (folder / "stderr.log").write_text(completed.stderr)
        result = (
            json.loads((folder / "result.json").read_text())
            if (folder / "result.json").exists()
            else dict(
                success=False,
                error_type="WorkerFailure",
                error="No result.json",
                returncode=completed.returncode,
            )
        )
    except subprocess.TimeoutExpired:
        result = dict(
            success=False,
            error_type="Timeout",
            error=f"Exceeded {timeout}s bounded per-case wall-time",
            seconds=time.perf_counter() - started,
        )
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return dict(**case, **result)


def enrich_report(report):
    report["environment"] = dict(
        python=sys.executable,
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_versions={
            name: importlib.metadata.version(name)
            for name in ["openmm", "pdbfixer", "rdkit"]
        },
        obabel_executable=shutil.which("obabel"),
    )
    rows = report["natural_results"]
    report["natural_timeout_count"] = sum(
        r.get("error_type") == "Timeout" for r in rows
    )
    report["natural_non_timeout_error_count"] = sum(
        not r["success"] and r.get("error_type") != "Timeout" for r in rows
    )
    report["cofactor_retention"] = []
    for row in rows:
        for requested in row.get("original_declared_cofactors", "").split("+"):
            if not requested:
                continue
            present_input = any(
                x["residue"] == requested for x in row.get("input_heterogens", [])
            )
            present_output = any(
                x["residue"] == requested for x in row.get("output_heterogens", [])
            )
            report["cofactor_retention"].append(
                dict(
                    target=row["target"],
                    pdb_id=row["pdb_id"],
                    requested_cofactor=requested,
                    present_input=present_input,
                    present_output=present_output,
                    interpretation=(
                        "File-generation success does not establish "
                        "a chemically complete cofactor environment."
                    ),
                )
            )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "Data/validation/inputs/structure_panel.json",
    )
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--timeout", type=float, default=90)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--worker", type=Path)
    p.add_argument("--worker-output", type=Path)
    p.add_argument("--summarize-existing", action="store_true")
    args = p.parse_args()
    if args.worker:
        worker(args.worker, args.worker_output, args.steps)
        return
    assert args.output is not None
    if args.summarize_existing:
        report_path = args.output / "report.json"
        report = enrich_report(json.loads(report_path.read_text()))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in [
                        "natural_attempted",
                        "natural_successes",
                        "natural_timeout_count",
                        "natural_non_timeout_error_count",
                    ]
                },
                indent=2,
            )
        )
        return
    manifest = freeze(args.manifest, args.count, args.steps)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(
            pool.map(
                lambda x: run_one(x, args.output, args.steps, args.timeout),
                manifest["cases"],
            )
        )
    # Separate malformed-input controls, excluded from natural-panel denominators.
    controls = []
    for label, content in [
        ("empty", ""),
        (
            "invalid_coordinates",
            "ATOM      1  CA  ALA A   1         BAD     BAD     BAD\nEND\n",
        ),
    ]:
        path = args.output / f"{label}.pdb"
        path.write_text(content)
        controls.append(
            run_one(
                dict(
                    target="CONTROL_" + label,
                    source_path=str(path.resolve()),
                    control=True,
                ),
                args.output,
                args.steps,
                args.timeout,
            )
        )
    report = dict(
        environment=dict(
            python=sys.executable,
            python_version=platform.python_version(),
            platform=platform.platform(),
            package_versions={
                name: importlib.metadata.version(name)
                for name in ["openmm", "pdbfixer", "rdkit"]
            },
            obabel_executable=shutil.which("obabel"),
        ),
        protocol=manifest,
        wall_seconds=time.perf_counter() - start,
        natural_attempted=len(results),
        natural_successes=sum(bool(r["success"]) for r in results),
        natural_failures=sum(not r["success"] for r in results),
        natural_results=results,
        malformed_controls=controls,
        limitations=[
            "Small predefined convenience panel, not a population-wide rate.",
            "Locally chain-filtered inputs exclude network retrieval and chain-selection stages.",
            (
                "Bounded200-iteration CPU protocol differs from default5000-iteration maximum; "
                "timeouts are explicit failures under this budget."
            ),
            "Default repair removes heterogens and may drop biologically important nonstandard residues/cofactors.",
        ],
    )
    report = enrich_report(report)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for r in results + controls:
        rows.append(
            {
                k: r.get(k)
                for k in [
                    "target",
                    "pdb_id",
                    "source_path",
                    "success",
                    "seconds",
                    "error_type",
                    "error",
                    "output_atom_records",
                ]
            }
        )
    with (args.output / "results.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "natural_attempted",
                    "natural_successes",
                    "natural_failures",
                    "wall_seconds",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
