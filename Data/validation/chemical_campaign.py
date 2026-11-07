"""Run a bounded, chemically annotated EGFR campaign with public ProDock stages.

The staged CLI reuses one frozen prepared library. Resume here is a validation
runner policy (hash-checked output replay), not an undocumented ProDock feature.
No compound is selected by docking outcome; activity annotations are not fitted.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import tomllib
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENMM_CPU_THREADS", "1")
# Subprocess CLI tools must come from the same conda environment as Python.
os.environ["PATH"] = (
    str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from rdkit import rdBase  # noqa: E402

from prodock.preprocess import LigandPrep  # noqa: E402
from prodock.dock.batch import (  # noqa: E402
    DockTask,
    worker_process_job_using_singledock,
)  # noqa: E402
from prodock.database import PoseDatabase  # noqa: E402
from prodock.postprocess.pose import PoseCrawler  # noqa: E402
from prodock.postprocess.interaction.core import (  # noqa: E402
    extract_pose_table_interactions,
)  # noqa: E402


def source_version():
    try:
        return importlib.metadata.version("prodock")
    except importlib.metadata.PackageNotFoundError:
        return (
            tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
            + "+working-tree"
        )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


@contextmanager
def resource_stage(root, name):
    start = time.perf_counter()
    stop = threading.Event()
    peaks = [0]

    def sample():
        process = psutil.Process()
        while not stop.is_set():
            rss = 0
            for p in [process, *process.children(recursive=True)]:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            peaks[0] = max(peaks[0], rss)
            stop.wait(0.1)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        rec = dict(
            stage=name,
            seconds=time.perf_counter() - start,
            sampled_aggregate_peak_rss_bytes=peaks[0],
            sampling_seconds=0.1,
            memory_method="Sum of process-tree RSS sampled at 0.1 s; shared pages may be counted more than once.",
        )
        with (root / "resources.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")


def prepare_one(payload):
    record, directory = payload
    start = time.perf_counter()
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    manifest = p / (record["id"] + ".preparation.json")
    expected = hashlib.sha256(
        (record["smiles"] + "|ETKDGv3|42|MMFF94|200|meeko").encode()
    ).hexdigest()
    if manifest.exists():
        old = json.loads(manifest.read_text())
        out = Path(old.get("out_path") or p / "absent")
        if (
            old.get("input_hash") == expected
            and old.get("success")
            and out.exists()
            and old.get("output_sha256") == digest(out)
        ):
            return dict(old, reused=True)
    try:
        prep = (
            LigandPrep(output_dir=p, smiles_key="smiles", name_key="id")
            .set_keep_intermediate(True)
            .set_conformer_seed(42)
            .set_output_format("pdbqt")
            .set_converter_backend("meeko")
            .from_list_of_dicts([record])
            .process_all()
        )
        r = prep.records[0]
        out = Path(r["out_path"]) if r.get("out_path") else None
        result = dict(
            id=record["id"],
            success=r.get("status") == "ok",
            out_path=str(out) if out else None,
            error=r.get("error"),
            input_hash=expected,
            reused=False,
            elapsed=time.perf_counter() - start,
            output_sha256=digest(out) if out and out.exists() else None,
        )
        # Preserve the original prepared chemical graph for later identity checks.
        if r.get("molblock"):
            (p / (record["id"] + ".mol")).write_text(r["molblock"])
    except Exception as e:
        result = dict(
            id=record["id"],
            success=False,
            error=str(e),
            input_hash=expected,
            reused=False,
            elapsed=time.perf_counter() - start,
        )
    write_json(manifest, result)
    return result


def prepare(args, library):
    selected = library["ligands"][: args.limit]
    directory = args.prepared_root or args.output / "prepared_ligands"
    jobs = [(r, str(directory / r["id"])) for r in selected]
    results = []
    with resource_stage(args.output, "ligand_preparation"):
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {pool.submit(prepare_one, j): j[0]["id"] for j in jobs}
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 20 == 0:
                    print("Prepared", len(results), "/", len(jobs), flush=True)
    pd.DataFrame(results).sort_values("id").to_csv(
        args.output / "preparation.csv", index=False
    )
    print(
        "Preparation complete",
        sum(r["success"] for r in results),
        "/",
        len(results),
        flush=True,
    )


def receptor_specs(args):
    if args.receptors:
        payload = json.loads(args.receptors.read_text())
        records = payload.get("receptors", payload.get("prepared_receptors", []))
        if not records:
            raise ValueError("Prepared receptor configuration contains no entries")
        specs = []
        for r in records:

            def resolve(value):
                p = Path(value)
                return str(
                    (args.receptors.parent / p).resolve() if not p.is_absolute() else p
                )

            protein = resolve(
                r.get("pdbqt")
                or r.get("receptor_pdbqt")
                or r.get("prepared_pdbqt")
                or r.get("receptor")
            )
            pdb_value = r.get("prepared_pdb") or r.get("receptor_pdb")
            if not pdb_value and str(r.get("pdb", "")).lower().endswith(".pdb"):
                pdb_value = r["pdb"]
            paired = resolve(pdb_value or str(Path(protein).with_suffix(".pdb")))
            box = r.get("box", {})
            specs.append(
                dict(
                    id=r.get("id") or r.get("receptor_id"),
                    pdbqt=protein,
                    pdb=paired,
                    center=r.get("center", box.get("center")),
                    size=r.get("size", box.get("size")),
                    caveat=r.get(
                        "caveat", "Explicit externally prepared protein model."
                    ),
                )
            )
    else:
        specs = []
        for rid in ["1M17", "4ZAU"]:
            root = ROOT / "Data/benchmark/runs/redocking/projects" / rid
            cfg = json.loads((root / "campaign.json").read_text())["receptors"][0]
            specs.append(
                dict(
                    id=rid,
                    pdbqt=cfg["receptor"],
                    pdb=str(Path(cfg["receptor"]).with_suffix(".pdb")),
                    center=cfg["box"]["center"],
                    size=cfg["box"]["size"],
                    caveat="Existing prepared input; chemical/state and protein-frame audit recorded separately.",
                )
            )
    if args.proteins:
        specs = [s for s in specs if s["id"] in args.proteins]
    return specs


def dock(args, library):
    records = library["ligands"][: args.limit]
    specs = receptor_specs(args)
    tasks = []
    plan = []
    skipped = []
    for engine in args.engines:
        binary = ROOT / "prodock/dock/binary" / engine
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"Missing/nonexecutable docking program: {binary}")
    from dataclasses import asdict

    for record in records:
        mpath = (
            (args.prepared_root or args.output / "prepared_ligands")
            / record["id"]
            / (record["id"] + ".preparation.json")
        )
        if not mpath.exists() or not (prep := json.loads(mpath.read_text())).get(
            "success"
        ):
            skipped.append(
                dict(id=record["id"], reason="ligand_preparation_missing_or_failed")
            )
            continue
        expected = hashlib.sha256(
            (record["smiles"] + "|ETKDGv3|42|MMFF94|200|meeko").encode()
        ).hexdigest()
        ligand_file = Path(prep["out_path"])
        if (
            prep.get("input_hash") != expected
            or not ligand_file.exists()
            or prep.get("output_sha256") != digest(ligand_file)
        ):
            skipped.append(
                dict(
                    id=record["id"], reason="prepared_ligand_identity_or_hash_mismatch"
                )
            )
            continue
        for spec in specs:
            for engine in args.engines:
                folder = args.output / spec["id"] / "results"
                out = folder / "docked" / engine / (record["id"] + "_docked.pdbqt")
                log = folder / "logs" / engine / (record["id"] + ".log")
                task = DockTask(
                    job_id=f"{spec['id']}:{engine}:{record['id']}",
                    receptor_id=spec["id"],
                    ligand_id=record["id"],
                    engine_name=engine,
                    receptor=spec["pdbqt"],
                    ligand=prep["out_path"],
                    center=tuple(spec["center"]),
                    size=tuple(spec["size"]),
                    cpu=1,
                    seed=42,
                    exhaustiveness=args.exhaustiveness,
                    n_poses=args.poses,
                    out_path=str(out),
                    log_path=str(log),
                    retries=1,
                    engine_options={"timeout": args.timeout},
                )
                binary = ROOT / "prodock/dock/binary" / engine
                definition = dict(
                    task=asdict(task),
                    receptor_sha256=digest(spec["pdbqt"]),
                    ligand_sha256=digest(prep["out_path"]),
                    binary_sha256=digest(binary),
                )
                jobhash = hashlib.sha256(
                    json.dumps(definition, sort_keys=True).encode()
                ).hexdigest()
                meta = out.with_suffix(".task.json")
                cached = False
                if meta.exists() and out.exists():
                    old = json.loads(meta.read_text())
                    cached = bool(
                        old.get("success")
                        and old.get("task_hash") == jobhash
                        and old.get("output_sha256") == digest(out)
                    )
                plan.append(dict(**asdict(task), task_hash=jobhash, reused=cached))
                if not cached:
                    tasks.append((task, meta, jobhash))
    pd.DataFrame(plan).to_csv(
        args.output / f"task_plan_{args.run_name}.csv", index=False
    )
    results = []
    with resource_stage(args.output, "docking_" + args.run_name):
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(worker_process_job_using_singledock, asdict(t)): (t, m, h)
                for t, m, h in tasks
            }
            for future in as_completed(futures):
                task, meta, jobhash = futures[future]
                try:
                    r = future.result()
                except Exception as exc:
                    r = dict(job_id=task.job_id, success=False, error=str(exc))
                out = Path(task.out_path)
                r.update(
                    task_hash=jobhash,
                    output_sha256=digest(out) if out.exists() else None,
                    reused=False,
                )
                # A positive exit alone is insufficient to accept a checkpoint.
                text = out.read_text() if out.exists() else ""
                scored = (
                    "REMARK VINA RESULT:" in text or "REMARK minimizedAffinity " in text
                )
                r["success"] = bool(
                    r.get("success") and "MODEL" in text and "ATOM" in text and scored
                )
                write_json(meta, r)
                results.append(r)
                with (args.output / "docking_attempts.jsonl").open("a") as f:
                    f.write(json.dumps(r, default=str) + "\n")
                if len(results) % 10 == 0:
                    print("Docked", len(results), "/", len(tasks), flush=True)
    summary = dict(
        requested_compounds=len(records),
        prepared_compounds=len(records) - len(skipped),
        planned_tasks=len(plan),
        reused=sum(p["reused"] for p in plan),
        executed=len(results),
        successful_new=sum(bool(r["success"]) for r in results),
        failures=[r for r in results if not r["success"]],
        preparation_skips=skipped,
        proteins=[s["id"] for s in specs],
        engines=args.engines,
        exhaustiveness=args.exhaustiveness,
        poses_requested=args.poses,
        runner_resume_policy=(
            "Only reuse matching input/binary/settings and full output hash; this is an "
            "experiment runner policy, not generic ProDock checkpoint/resume."
        ),
    )
    write_json(args.output / f"docking_{args.run_name}.json", summary)
    print(
        "DOCKING SUMMARY",
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ["failures", "preparation_skips"]
            }
        ),
        flush=True,
    )


def analyze(args, library):
    metadata = {r["id"]: r for r in library["ligands"]}
    specs = receptor_specs(args)
    with resource_stage(args.output, "pose_parsing"):
        frames = []
        allowed = (
            {r["id"] for r in library["ligands"][: args.analysis_limit]}
            if args.analysis_limit
            else None
        )
        for sidecar in sorted(args.output.glob("*/results/docked/*/*.task.json")):
            record = json.loads(sidecar.read_text())
            if not record.get("success"):
                continue
            source = Path(record["out_path"])
            if not source.exists() or digest(source) != record.get("output_sha256"):
                continue
            if allowed and record["ligand_id"] not in allowed:
                continue
            frame = PoseCrawler([source], engine=record["engine_name"]).crawl_mols(
                backend="meeko"
            )
            frame["receptor_id"] = record["receptor_id"]
            frame["ligand_id"] = record["ligand_id"]
            frame["engine"] = record["engine_name"]
            frames.append(frame)
        poses = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if poses.empty:
        raise RuntimeError("No poses parsed; inspect task outputs")
    poses["pose_id"] = [
        f"{r.receptor_id}__{r.ligand_id}__{r.engine}__{r.pose_rank}"
        for r in poses.itertuples()
    ]
    if args.analysis_limit:
        allowed = {r["id"] for r in library["ligands"][: args.analysis_limit]}
        poses = poses[poses.ligand_id.isin(allowed)].copy()
    with resource_stage(args.output, "database_insertion"):
        with PoseDatabase(args.output / "chemical_campaign.db") as db:
            db.create_run(
                name=args.run_name,
                config=json.loads((args.output / "manifest.json").read_text()),
                prodock_version=source_version(),
            )
            db.insert_dataframe(poses, replace=True)
    top = poses[poses.pose_rank == 1].copy()
    interaction_errors = []
    # Only rank1 is profiled in the main cost-bounded campaign; all returned
    # coordinate ranks remain stored and an explicit top-k subset can be added.
    if args.profile:
        with resource_stage(args.output, "rank1_interaction_profiling"):
            result = extract_pose_table_interactions(
                poses=top,
                receptor_pdb_by_id={s["id"]: s["pdb"] for s in specs},
                receptor_guess_bonds=False,
                n_jobs=1,
                batch_size=16,
                progress=False,
                fail_fast=False,
                include_interaction_events=True,
                include_fingerprint_columns=False,
            )
            events = result.serializable_interaction_df()
            events.to_csv(args.output / "interaction_events.csv", index=False)
            result.serializable_summary_df().to_csv(
                args.output / "interaction_summary.csv", index=False
            )
            with PoseDatabase(args.output / "chemical_campaign.db") as db:
                db.insert_dataframe(
                    result.merged_df,
                    interactions_by_pose=result.summary_dict(kind="detail"),
                    replace=True,
                    replace_interactions=True,
                )
            interaction_errors = getattr(result, "errors", [])
    table = top.drop(columns=["mol"], errors="ignore").copy()
    for key in [
        "median_pic50",
        "activity_stratum",
        "heavy_atoms",
        "rotatable_bonds",
        "molecular_weight",
        "logp",
        "scaffold",
        "activity_pic50_range",
    ]:
        table[key] = table.ligand_id.map(lambda i: metadata.get(i, {}).get(key))
    table.to_csv(args.output / "annotated_rank1.csv", index=False)
    write_json(
        args.output / "analysis_counts.json",
        dict(
            parsed_poses=len(poses),
            rank1_poses=len(top),
            compounds=poses.ligand_id.nunique(),
            proteins=sorted(poses.receptor_id.unique()),
            engines=sorted(poses.engine.unique()),
            interaction_errors=interaction_errors,
            rank1_only_interactions=bool(args.profile),
            limitations=(
                "IC50 annotations aggregate heterogeneous binding assays; no fitted activity "
                "weights, no homogeneous potency benchmark, no prospective validation."
            ),
        ),
    )
    print("ANALYSIS", len(poses), "poses", len(top), "rank1", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "dock", "analyze", "all"])
    ap.add_argument(
        "--library",
        type=Path,
        default=ROOT / "Data/validation/inputs/chembl_egfr/library.json",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--proteins", nargs="+")
    ap.add_argument("--engines", nargs="+", default=["smina"])
    ap.add_argument("--receptors", type=Path)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument(
        "--prepared-root",
        type=Path,
        help="Reuse a frozen, hash-checked prepared library in another protocol-specific campaign.",
    )
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--run-name", default="pilot")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--analysis-limit", type=int)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Read metadata and print the stage plan only; no preparation, docking, profiling or database writes.",
    )
    args = ap.parse_args()
    args.output = args.output.resolve()
    if args.dry_run:
        data = json.loads(args.library.read_text())
        n = min(args.limit, len(data["ligands"]))
        tasks = (
            n * len(args.proteins or ["1M17", "4ZAU"]) * len(args.engines)
            if args.stage in ["dock", "all"]
            else 0
        )
        print(
            json.dumps(
                dict(
                    dry_run=True,
                    execution=False,
                    stage=args.stage,
                    compounds=n,
                    planned_docking_tasks=tasks,
                    maximum_requested_poses=tasks * args.poses,
                    proteins=args.proteins or ["1M17", "4ZAU"],
                    engines=args.engines,
                    receptor_configuration=(
                        str(args.receptors) if args.receptors else None
                    ),
                    receptor_configuration_exists=bool(
                        args.receptors and args.receptors.is_file()
                    ),
                    prepared_library=str(
                        args.prepared_root or args.output / "prepared_ligands"
                    ),
                    library_sha256=digest(args.library),
                    output=str(args.output),
                    note="This plan does not qualify a protein preparation or report completed scientific results.",
                ),
                indent=2,
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "invocations.jsonl").open("a") as record:
        record.write(
            json.dumps(
                dict(
                    command=sys.argv,
                    python=sys.executable,
                    unix_time=time.time(),
                    source_sha256=digest(Path(__file__)),
                    core_sha256=digest(ROOT / "prodock/core.py"),
                    database_sha256=digest(ROOT / "prodock/database/pose_db.py"),
                    mk_prepare_ligand=shutil.which("mk_prepare_ligand.py"),
                    obabel=shutil.which("obabel"),
                )
            )
            + "\n"
        )
    library = json.loads(args.library.read_text())
    manifest = args.output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["library_sha256"] != digest(args.library):
            raise RuntimeError("Library changed: use a fresh output directory")
    else:
        write_json(
            manifest,
            dict(
                library=str(args.library.resolve()),
                library_sha256=digest(args.library),
                source_version=library.get("chembl_status"),
                chemical_question=library.get("chemical_question"),
                python=sys.executable,
                python_version=platform.python_version(),
                rdkit=rdBase.rdkitVersion,
                proDock=source_version(),
                platform=platform.platform(),
                source_files_sha256={
                    str(p.relative_to(ROOT)): digest(p)
                    for p in [
                        Path(__file__),
                        ROOT / "prodock/core.py",
                        ROOT / "prodock/database/pose_db.py",
                    ]
                },
                ligand_preparation=(
                    "Supplied SMILES state, explicit H, one ETKDGv3 conformer seed42, MMFF94 "
                    "max200, Meeko PDBQT; no protonation/tautomer enumeration."
                ),
                output_policy=(
                    "Task attempts and outputs have input/settings/binary hashes. All returned "
                    "valid pose ranks stored, rank1 interaction profiling."
                ),
                activity_caveat=library.get("interpretation"),
            ),
        )
    if args.stage in ["prepare", "all"]:
        prepare(args, library)
    if args.stage in ["dock", "all"]:
        dock(args, library)
    if args.stage in ["analyze", "all"]:
        analyze(args, library)


if __name__ == "__main__":
    main()
