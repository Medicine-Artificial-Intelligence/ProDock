"""LoLO Blue: plan and run the smallest declared reviewer experiment package.

Default behavior only prints JSON. Add --execute to run the selected suite.
Chemical runs use chemical_campaign.py's explicit input/binary/settings/output
hash checks; --resume does not add generic checkpoint/resume to ProDock.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "Data/validation"
EVIDENCE = ROOT / "paper/revision/evidence"
SUITE_LIMITS = {"pilot": 32, "minimum": 250, "full": 1000}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def observed_evidence(args) -> dict:
    """Describe actual files; a filename alone never establishes completion."""
    result = {}
    redocking = read_json(EVIDENCE / "redocking_expanded/report.json")
    if redocking:
        result["E1_redocking"] = dict(
            status="available_artifact_audit_with_explicit_missing_cases",
            report=str(EVIDENCE / "redocking_expanded/report.json"),
            expected_rank1=redocking.get("expected_configured_rank1"),
            evaluated_rank1=redocking.get("evaluated_configured_rank1"),
            below_2A=redocking.get("configured_rank1_below_2A"),
            new_complete_redocking_campaign=False,
        )
    else:
        result["E1_redocking"] = {"status": "no_readable_report"}
    preparation = read_json(EVIDENCE / "preparation_panel/report.json")
    if preparation:
        result["E3_preparation"] = dict(
            status="completed_bounded_panel_only",
            report=str(EVIDENCE / "preparation_panel/report.json"),
            attempted=preparation.get("natural_attempted"),
            successes=preparation.get("natural_successes"),
            timeouts=preparation.get("natural_timeout_count"),
            other_errors=preparation.get("natural_non_timeout_error_count"),
            population_failure_rate=False,
        )
    else:
        result["E3_preparation"] = {"status": "no_readable_report"}
    integrity_path = EVIDENCE / "expanded_integrity/final/report.json"
    integrity = read_json(integrity_path)
    result["E6_integrity"] = dict(
        status=(
            "controlled_checks_completed"
            if integrity and integrity.get("failed") == 0
            else "not_verified"
        ),
        report=str(integrity_path),
        passed=integrity.get("passed") if integrity else None,
        failed=integrity.get("failed") if integrity else None,
        documented_limitations=(
            integrity.get("documented_limitations") if integrity else None
        ),
        verified_workflow_checkpoint_resume=False,
    )
    comparison_path = EVIDENCE / "comparison_native/final/report.json"
    comparison = read_json(comparison_path)
    result["E4_comparison"] = dict(
        status="native_helpers_executed" if comparison else "no_readable_report",
        report=str(comparison_path),
        checks=(
            [
                {k: c.get(k) for k in ["task", "status"]}
                for c in comparison.get("checks", [])
            ]
            if comparison
            else []
        ),
        full_platform_comparison=False,
    )
    sidecars = []
    for protein in ["1M17", "4ZAU"]:
        for engine in ["smina", "qvina"]:
            for path in (
                args.campaign_output / protein / "results/docked" / engine
            ).glob("*.task.json"):
                record = read_json(path)
                if record and record.get("success"):
                    output = Path(record.get("out_path", "missing"))
                    verified = output.is_file() and record.get(
                        "output_sha256"
                    ) == sha256(output)
                    sidecars.append(
                        dict(
                            protein=protein,
                            engine=engine,
                            hash_verified=verified,
                            elapsed=record.get("elapsed"),
                        )
                    )
    summaries = {}
    for path in args.campaign_output.glob("docking_*.json"):
        report = read_json(path)
        if report:
            planned = report.get("planned_tasks", 0)
            summaries[path.name] = dict(
                planned_tasks=planned,
                executed=report.get("executed"),
                reused=report.get("reused"),
                successful_new=report.get("successful_new"),
                failures=len(report.get("failures", [])),
                nonzero_accounted_task_matrix=bool(
                    planned
                    and report.get("executed", 0) + report.get("reused", 0) == planned
                ),
            )
    times = sorted(
        float(r["elapsed"])
        for r in sidecars
        if r["hash_verified"] and r.get("elapsed") is not None
    )
    median = (
        (
            times[len(times) // 2]
            if len(times) % 2
            else sum(times[len(times) // 2 - 1 : len(times) // 2 + 1]) / 2
        )
        if times
        else None
    )
    result["E2_chemical_campaign"] = dict(
        status=(
            "some_hash_verified_task_outputs"
            if sidecars
            else "no_verified_task_outputs"
        ),
        output=str(args.campaign_output),
        observed_successful_sidecars=len(sidecars),
        hash_verified_successes=sum(r["hash_verified"] for r in sidecars),
        summaries=summaries,
        measured_task_elapsed_median_seconds=median,
        analysis_counts=read_json(args.campaign_output / "analysis_counts.json"),
        caveat=(
            "A stale zero-task summary is not completion; task hashes must be "
            "checked by the chemical driver on execution."
        ),
    )
    result["E5_restart"] = {
        "status": "database_rollback_checked_runner_recovery_study_separate",
        "parameter_history_policy": "isolate parameter variants in separate databases/directories",
        "generic_prodock_checkpoint_resume": False,
    }
    result["E7_contacts"] = {
        "status": "read_only_protocol_implemented_results_depend_on_valid_profiled_campaign",
        "existing_result": read_json(EVIDENCE / "chemical_usecase/results/report.json"),
    }
    return result


def command_step(
    name: str, script: str, arguments: list, report: Path, workstream: str, **extra
):
    command = [sys.executable, str(VALIDATION / script), *map(str, arguments)]
    return dict(
        name=name,
        command=command,
        shell_display=shlex.join(command),
        report=str(report),
        workstream=workstream,
        script=str(VALIDATION / script),
        **extra,
    )


def protein_quality_gate(args) -> dict:
    if args.suite == "cheap":
        return {
            "status": "not_applicable_to_read_only_or_storage_checks",
            "ready": True,
        }
    if args.suite == "quality":
        return {
            "status": "generate_quality_report_before_large_campaign",
            "ready": True,
            "large_campaign_qualified": False,
            "note": "This permits the quality suite itself, not a large chemical campaign.",
        }
    config = read_json(args.receptors) if args.receptors else None
    report = read_json(args.quality_report) if args.quality_report else None
    entries = (
        (config.get("receptors") or config.get("prepared_receptors") or [])
        if config
        else []
    )
    prepared_checks = []
    recorded_paths = set()
    prepared_hashes = report.get("prepared_inputs_sha256", {}) if report else {}
    for name, expected in prepared_hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = args.quality_report.parent / path
        path = path.resolve()
        actual = sha256(path) if path.is_file() else None
        recorded_paths.add(str(path))
        prepared_checks.append(
            dict(
                path=str(path),
                expected_sha256=expected,
                actual_sha256=actual,
                matches=bool(actual and actual == expected),
            )
        )
    required_paths = []
    missing_fields = []
    for entry in entries:
        for label, aliases in [
            ("prepared_pdbqt", ["receptor_pdbqt", "pdbqt", "receptor"]),
            ("prepared_pdb", ["prepared_pdb", "pdb"]),
        ]:
            value = next((entry.get(key) for key in aliases if entry.get(key)), None)
            if value is None:
                missing_fields.append(
                    {
                        "structure": entry.get("receptor_id", entry.get("id")),
                        "field": label,
                    }
                )
                continue
            path = Path(value)
            if not path.is_absolute():
                path = args.receptors.parent / path
            required_paths.append(str(path.resolve()))
    uncovered = sorted(set(required_paths) - recorded_paths)
    prepared_verified = bool(
        prepared_checks
        and required_paths
        and not uncovered
        and not missing_fields
        and all(row["matches"] for row in prepared_checks)
    )
    protocol_checks = []
    for name, expected in (
        report.get("source_protocol_sha256", {}) if report else {}
    ).items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        actual = sha256(path) if path.is_file() else None
        protocol_checks.append(
            dict(
                path=str(path),
                expected_sha256=expected,
                actual_sha256=actual,
                matches=bool(actual and actual == expected),
            )
        )
    config_hash = (
        sha256(args.receptors) if args.receptors and args.receptors.is_file() else None
    )
    context_hash = sha256(args.context) if args.context.is_file() else None
    bound = bool(
        config
        and entries
        and report
        and report.get("receptor_config_sha256") == config_hash
        and report.get("chemical_context_sha256") == context_hash
        and context_hash is not None
        and prepared_verified
        and all(row["matches"] for row in protocol_checks)
    )
    qualified = bool(bound and report.get("qualified_for_new_campaign") is True)
    acknowledged = bool(bound and args.acknowledge_unqualified_receptors)
    ready = qualified or acknowledged or args.suite == "pilot"
    return dict(
        status=(
            "qualified_config_and_context"
            if qualified
            else (
                "explicit_unqualified_failure_illustration"
                if acknowledged
                else (
                    "legacy_pilot_diagnostic_only"
                    if args.suite == "pilot"
                    else "blocked_pending_protein_quality_gate"
                )
            )
        ),
        ready=ready,
        receptor_config=str(args.receptors) if args.receptors else None,
        receptor_config_sha256=config_hash,
        chemical_context_sha256=context_hash,
        prepared_file_checks=prepared_checks,
        missing_prepared_config_fields=missing_fields,
        prepared_paths_missing_from_quality_report=uncovered,
        all_prepared_bytes_match_qualification=prepared_verified,
        source_protocol_checks=protocol_checks,
        quality_report=str(args.quality_report) if args.quality_report else None,
        acknowledged_unqualified_receptors=acknowledged,
        requirement=(
            "Minimum/full require an explicit receptor config and a quality report "
            "with qualified_for_new_campaign=true and matching receptor_config_sha256 "
            "and chemical_context_sha256, plus matching prepared_inputs_sha256 for "
            "every prepared PDB and PDBQT file. An explicit acknowledgement only permits "
            "a documented failure illustration, not routine quality/accuracy claims."
        ),
        reason=(
            "Existing ligand-free minimization created extra clashes in the experimental "
            "bound-ligand volume. A small controlled native-ligand protocol comparison "
            "must precede spending hundreds or thousands of additional docking calls."
        ),
        context_caveat=(
            "A new protein preparation protocol requires its matching "
            "atom mapping and coordinate-frame context."
        ),
    )


def make_plan(args) -> dict:
    library = read_json(args.library)
    steps = []
    cheap_output = args.output / "cheap"
    if args.suite == "quality":
        native_output = args.output / "quality"
        arguments = [
            "--execute",
            "--input-manifest",
            args.structure_manifest,
            "--output",
            native_output,
            "--jobs",
            args.workers,
            "--timeout",
            180,
        ]
        if args.acknowledge_unqualified_receptors:
            arguments.append("--acknowledge-native-failures")
        steps.append(
            command_step(
                "native_preparation_quality_controls",
                "preparation_sensitivity.py",
                arguments,
                native_output / "quality_report.json",
                "E1/E3",
                requested_docking_tasks=8,
                requires_existing=str(args.structure_manifest),
                note=(
                    "Only --execute on this coordinator invokes the child "
                    "--execute command; dryrun never launches it."
                ),
            )
        )
        requested_tasks = 8
        selected = []
    elif args.suite == "cheap":
        steps.append(
            command_step(
                "redocking_available_artifacts",
                "redocking_audit.py",
                ["--output", cheap_output / "redocking", "--top-k", 20],
                cheap_output / "redocking/report.json",
                "E1",
                new_docking_tasks=0,
                read_only_campaign_inputs=True,
                requires_existing=str(
                    ROOT / "Data/benchmark/runs/redocking/redocking.db"
                ),
                optional=True,
                always_recompute=True,
            )
        )
        prep_report = EVIDENCE / "preparation_panel/report.json"
        steps.append(
            command_step(
                "summarize_bounded_preparation_panel",
                "preparation_panel.py",
                ["--output", prep_report.parent, "--summarize-existing"],
                prep_report,
                "E3",
                requires_existing=str(prep_report),
                new_preparation_attempts=0,
                optional=True,
            )
        )
        steps.append(
            command_step(
                "full_database_integrity",
                "integrity_experiments.py",
                ["--output", cheap_output / "integrity"],
                cheap_output / "integrity/report.json",
                "E6",
                new_docking_tasks=0,
            )
        )
        steps.append(
            command_step(
                "pinned_native_helper_comparison",
                "compare_stored_queries.py",
                ["--output", cheap_output / "comparison"],
                cheap_output / "comparison/report.json",
                "E4",
                full_platform_comparison=False,
                new_docking_tasks=0,
            )
        )
        requested_tasks = 0
        selected = []
    else:
        limit = SUITE_LIMITS[args.suite]
        selected = library.get("ligands", [])[:limit] if library else []
        common = [
            "--library",
            args.library,
            "--output",
            args.campaign_output,
            "--workers",
            args.workers,
            "--proteins",
            "1M17",
            "4ZAU",
            "--exhaustiveness",
            8,
            "--poses",
            5,
            "--timeout",
            180,
        ]
        if args.receptors:
            common += ["--receptors", args.receptors]
        if args.prepared_root:
            common += ["--prepared-root", args.prepared_root]
        steps.append(
            command_step(
                "prepare_shared_library",
                "chemical_campaign.py",
                ["prepare", *common, "--limit", limit],
                args.campaign_output / "preparation.csv",
                "E2",
                chemical_driver=True,
                requested_compounds=limit,
            )
        )
        if args.suite == "pilot":
            requested_tasks = 32 * 2 * 2
            steps.append(
                command_step(
                    "pilot32_two_engines",
                    "chemical_campaign.py",
                    [
                        "dock",
                        *common,
                        "--limit",
                        32,
                        "--engines",
                        "smina",
                        "qvina",
                        "--run-name",
                        "pilot32",
                    ],
                    args.campaign_output / "docking_pilot32.json",
                    "E2",
                    chemical_driver=True,
                    requested_docking_tasks=requested_tasks,
                )
            )
        else:
            requested_tasks = limit * 2 + 32 * 2
            steps.append(
                command_step(
                    f"{args.suite}_primary_smina",
                    "chemical_campaign.py",
                    [
                        "dock",
                        *common,
                        "--limit",
                        limit,
                        "--engines",
                        "smina",
                        "--run-name",
                        f"{args.suite}{limit}_primary",
                    ],
                    args.campaign_output / f"docking_{args.suite}{limit}_primary.json",
                    "E2",
                    chemical_driver=True,
                    requested_docking_tasks=limit * 2,
                )
            )
            steps.append(
                command_step(
                    f"{args.suite}_secondary_qvina32",
                    "chemical_campaign.py",
                    [
                        "dock",
                        *common,
                        "--limit",
                        32,
                        "--engines",
                        "qvina",
                        "--run-name",
                        f"{args.suite}32_secondary",
                    ],
                    args.campaign_output / f"docking_{args.suite}32_secondary.json",
                    "E2",
                    chemical_driver=True,
                    requested_docking_tasks=64,
                )
            )
        steps.append(
            command_step(
                "store_all_ranks_profile_rank1",
                "chemical_campaign.py",
                [
                    "analyze",
                    *common,
                    "--limit",
                    limit,
                    "--analysis-limit",
                    limit,
                    "--profile",
                    "--run-name",
                    f"{args.suite}_analysis",
                ],
                args.campaign_output / "analysis_counts.json",
                "E2/E6",
                chemical_driver=True,
                analysis_limit=limit,
            )
        )
        case_output = args.output / "chemical_usecase"
        steps.append(
            command_step(
                "read_only_chemical_and_contact_analysis",
                "analyze_chemical_usecase.py",
                [
                    "--campaign",
                    args.campaign_output,
                    "--library",
                    args.library,
                    "--context",
                    args.context,
                    "--output",
                    case_output,
                ],
                case_output / "report.json",
                "E7",
                requires_existing=str(
                    ROOT / "Data/validation/protocols/chemical_usecase.txt"
                ),
            )
        )
    observed = observed_evidence(args)
    measured = observed["E2_chemical_campaign"]["measured_task_elapsed_median_seconds"]
    assumed_per_task = measured if measured is not None else 30.0
    counts = library.get("counts", {}) if library else {}
    if args.only_step:
        requested = set(args.only_step)
        missing = requested - {s["name"] for s in steps}
        if missing:
            raise ValueError(
                f"Unknown selected steps for {args.suite}: {sorted(missing)}"
            )
        steps = [s for s in steps if s["name"] in requested]
    return dict(
        mode="execute" if args.execute else "plan_only_no_commands_executed",
        suite=args.suite,
        interpreter=sys.executable,
        generated_utc=stamp(),
        steps=steps,
        selected_step_names=[s["name"] for s in steps],
        selected_step_docking_tasks=sum(
            s.get("requested_docking_tasks", 0) for s in steps
        ),
        protein_quality_gate=protein_quality_gate(args),
        library=dict(
            path=str(args.library),
            exists=args.library.exists(),
            sha256=sha256(args.library) if args.library.exists() else None,
            release=(
                library.get("chembl_status", {}).get("chembl_db_version")
                if library
                else None
            ),
            frozen_compounds=counts.get("selected"),
            full_library_strata=counts.get("selected_strata"),
            full_library_scaffolds=counts.get("selected_unique_scaffolds"),
            selected_prefix=len(selected),
            prefix_strata=dict(Counter(r["activity_stratum"] for r in selected)),
            selection=(
                "Frozen ChEMBL37 supplied-state library; predetermined prefix, "
                "no docking-outcome selection or fitted activity weights."
            ),
        ),
        requested_docking_tasks=requested_tasks,
        requested_pose_upper_bound=(
            None if args.suite == "quality" else requested_tasks * 5
        ),
        core_package_docking_calls={
            "quality_controls": 8,
            "full_real_chemical_campaign": 2064,
            "total_before_reuse_failures_or_retries": 2072,
        },
        scope_decision="Four selected core packages; this does not implement every reviewer expectation.",
        resource_estimate=dict(
            per_task_seconds=assumed_per_task,
            basis=(
                "observed successful-task elapsed median"
                if measured is not None
                else "provisional 30-second planning assumption"
            ),
            serial_docking_seconds=requested_tasks * assumed_per_task,
            idealized_parallel_docking_seconds=requested_tasks
            * assumed_per_task
            / args.workers,
            workers=args.workers,
            caveat=(
                "Not a promised wall time: preparation, profiling, setup, heterogeneity "
                "and timeout tails are additional; resumed valid tasks are reused."
            ),
        ),
        resume_policy=(
            "Chemical commands always run their hash-checking driver. "
            "This coordinator adds no library-wide checkpoint guarantee. "
            "Cheap completed steps can be reused only with unchanged source/command/result hashes."
        ),
        essential=[
            "E1/E3 eight native preparation-quality controls and correct frame/RMSD interpretation",
            "E2 a realistic chemical use case and measured resource/failure denominators",
            "E5/E6 chemical/coordinate/interaction/provenance and failed-write integrity",
            "E4 small executed native-helper checks plus an honest version-pinned capability discussion",
        ],
        optional=[
            "The archived 12-structure bounded preparation panel and any larger population panel",
            "Complete-platform installations and matched workflow benchmarks beyond the native helpers",
            "Full all-engine accuracy benchmark, learned rescoring, prospective enrichment or cluster scaling",
        ],
        claims_to_avoid_instead_of_large_studies=[
            "full engine accuracy benchmark",
            "fitted ML rescoring superiority",
            "prospective enrichment or hit discovery",
            "cluster/multi-node scalability",
            "untested maximum capacity",
            "complete-platform superiority from helper timings",
        ],
        residual_reviewer_risk=(
            "The bounded package and native-helper comparison may still "
            "prompt requests for a larger library or full workflow comparison. "
            "Acceptance cannot be guaranteed."
        ),
        observed_evidence=observed,
    )


def script_fingerprints(step):
    paths = [
        Path(__file__),
        Path(step["script"]),
        ROOT / "prodock/database/pose_db.py",
        ROOT / "prodock/core.py",
    ]
    paths += list(VALIDATION.glob("*.py"))
    paths += list((ROOT / "prodock").rglob("*.py"))
    return {
        str(path.relative_to(ROOT)): sha256(path) for path in paths if path.exists()
    }


def execute_plan(args, plan):
    if not plan["protein_quality_gate"]["ready"]:
        raise RuntimeError(plan["protein_quality_gate"]["requirement"])
    required = ["rdkit", "prolif", "meeko", "openmm", "pdbfixer"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"Selected interpreter lacks {missing}. Run this command using the prodock environment's Python."
        )
    if not args.library.exists() and args.suite not in ["cheap", "quality"]:
        raise FileNotFoundError(
            "Frozen library missing. Run fetch_egfr_library.py once, then preserve its manifest and hash."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "execution_manifest.json"
    old = read_json(manifest_path)
    if old and not args.resume:
        raise FileExistsError(
            "Execution manifest exists; use --resume or a fresh --output"
        )
    if old and old.get("suite") != args.suite:
        raise ValueError(
            "Use a different orchestration --output for a different suite; "
            "--campaign-output may be shared for growing prefixes"
        )
    if old and old.get("library_sha256") != plan["library"]["sha256"]:
        raise RuntimeError(
            "Frozen library changed; use a new campaign and orchestration directory"
        )
    if old and old.get("receptor_config_sha256") != plan["protein_quality_gate"].get(
        "receptor_config_sha256"
    ):
        raise RuntimeError(
            "Protein configuration changed; use separate campaign and orchestration outputs"
        )
    if args.suite not in ["cheap", "quality"]:
        if (
            args.prepared_root
            and "--prepared-root"
            not in (VALIDATION / "chemical_campaign.py").read_text()
        ):
            raise RuntimeError(
                "This chemical driver does not expose --prepared-root; update it before execution"
            )
        prior = read_json(args.campaign_output / "analysis_counts.json")
        if prior and prior.get("compounds", 0) > SUITE_LIMITS[args.suite]:
            raise RuntimeError(
                "Cannot analyze a smaller prefix into an existing larger database; choose a fresh --campaign-output"
            )
        # Prevent accidental concurrent writes into the actively running pilot.
        import psutil

        for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
            if process.pid == os.getpid():
                continue
            command = process.info.get("cmdline") or []
            if not any(
                Path(part).name == "chemical_campaign.py" for part in command[:3]
            ):
                continue
            if "--output" in command:
                current = Path(command[command.index("--output") + 1])
                current = (
                    current
                    if current.is_absolute()
                    else Path(process.info.get("cwd") or ROOT) / current
                )
                if current.resolve() == args.campaign_output:
                    raise RuntimeError(
                        f"Campaign writer PID {process.pid} is active; wait before executing "
                        "against the same campaign output"
                    )
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    environment.setdefault("OPENMM_CPU_THREADS", "1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    state = old or dict(
        suite=args.suite,
        library_sha256=plan["library"]["sha256"],
        receptor_config_sha256=plan["protein_quality_gate"].get(
            "receptor_config_sha256"
        ),
        started_utc=stamp(),
        steps=[],
        scope="Command execution status, not reviewer acceptance/completion",
    )
    previous = {step["name"]: step for step in state["steps"]}
    state["last_invocation"] = dict(
        utc=stamp(),
        command=sys.argv,
        interpreter=sys.executable,
        runner_sha256=sha256(Path(__file__)),
        plan=plan,
    )
    write_json(manifest_path, state)
    for step in plan["steps"]:
        fingerprints = script_fingerprints(step)
        command_hash = hashlib.sha256(json.dumps(step["command"]).encode()).hexdigest()
        report = Path(step["report"])
        prior = previous.get(step["name"])
        if (
            args.resume
            and not step.get("chemical_driver")
            and not step.get("always_recompute")
            and prior
            and prior.get("returncode") == 0
        ):
            if (
                prior.get("source_sha256") == fingerprints
                and prior.get("command_sha256") == command_hash
                and report.exists()
                and prior.get("report_sha256") == sha256(report)
            ):
                prior["reused_verified_at"] = stamp()
                write_json(manifest_path, state)
                continue
            if report.exists():
                raise RuntimeError(
                    f"Cheap evidence inputs/source changed for {step['name']}; "
                    "choose a fresh --output to preserve old evidence"
                )
        requirement = step.get("requires_existing")
        if requirement and not Path(requirement).exists():
            record = dict(
                name=step["name"],
                status="skipped_missing_prerequisite",
                prerequisite=requirement,
                source_sha256=fingerprints,
                command_sha256=command_hash,
            )
            state["steps"].append(record)
            write_json(manifest_path, state)
            if step.get("optional"):
                continue
            raise FileNotFoundError(
                f"Missing prerequisite for {step['name']}: {requirement}"
            )
        log = args.output / "logs" / (step["name"] + ".log")
        log.parent.mkdir(exist_ok=True)
        record = dict(
            name=step["name"],
            command=step["command"],
            command_sha256=command_hash,
            source_sha256=fingerprints,
            started_utc=stamp(),
            status="running",
            log=str(log),
        )
        state["steps"] = [r for r in state["steps"] if r["name"] != step["name"]] + [
            record
        ]
        write_json(manifest_path, state)
        started = time.perf_counter()
        with log.open("a") as handle:
            handle.write("\n" + stamp() + " " + step["shell_display"] + "\n")
            completed = subprocess.run(
                step["command"],
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        record.update(
            returncode=completed.returncode,
            seconds=time.perf_counter() - started,
            completed_utc=stamp(),
            status=(
                "command_succeeded" if completed.returncode == 0 else "command_failed"
            ),
            report_sha256=sha256(report) if report.exists() else None,
        )
        write_json(manifest_path, state)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Step {step['name']} failed with code {completed.returncode}; inspect {log}"
            )
        if not report.exists():
            raise RuntimeError(
                f"Step {step['name']} exited successfully but did not produce {report}"
            )
    state["completed_utc"] = stamp()
    state["observed_after"] = observed_evidence(args)
    write_json(manifest_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        "--dry-run",
        action="store_true",
        help="Print plan JSON only; this is the default",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually run selected commands sequentially",
    )
    parser.add_argument(
        "--suite",
        choices=["quality", "cheap", "pilot", "minimum", "full"],
        default="full",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=EVIDENCE / "organized_package")
    parser.add_argument("--campaign-output", type=Path)
    parser.add_argument(
        "--library", type=Path, default=VALIDATION / "inputs/chembl_egfr/library.json"
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=EVIDENCE / "redocking_expanded/chemical_context.json",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--structure-manifest",
        type=Path,
        default=VALIDATION / "inputs/egfr_structures/manifest.json",
    )
    parser.add_argument(
        "--receptors",
        type=Path,
        help="Explicit protocol-specific prepared-receptor JSON",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="Reuse hash-verified prepared ligands from another run",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        help="Real protocol qualification report; never autogenerated as passed",
    )
    parser.add_argument(
        "--acknowledge-unqualified-receptors",
        action="store_true",
        help="Explicitly run a known-unqualified failure illustration; not a routine-screening claim",
    )
    parser.add_argument(
        "--only-step",
        action="append",
        help="Advanced explicit subset; omitted steps are not marked complete",
    )
    parser.add_argument("--write-plan", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.campaign_output is None:
        folder = (
            "chemical_campaign_v1"
            if args.suite in ["cheap", "pilot"]
            else "qualified_campaign_v1"
        )
        args.campaign_output = VALIDATION / "runs" / folder
    for field in [
        "output",
        "campaign_output",
        "library",
        "context",
        "receptors",
        "prepared_root",
        "quality_report",
        "structure_manifest",
    ]:
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())
    try:
        plan = make_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.write_plan:
        write_json(args.write_plan, plan)
    if args.execute:
        try:
            state = execute_plan(args, plan)
        except (RuntimeError, ValueError, FileExistsError, FileNotFoundError) as exc:
            parser.exit(2, f"Execution stopped: {exc}\n")
        print(
            json.dumps(
                dict(
                    suite=args.suite,
                    manifest=str(args.output / "execution_manifest.json"),
                    completed_utc=state.get("completed_utc"),
                ),
                indent=2,
            )
        )
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
