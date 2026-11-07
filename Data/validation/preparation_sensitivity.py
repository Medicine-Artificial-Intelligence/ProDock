"""Prespecified native-pose diagnostic for two chemically distinct preparation policies.

2 EGFR structures x 2 preparation policies x 2 programs = 8 tasks. The design is
frozen before outcomes. Both policies are generated from tracked raw inputs on the cluster. The
crystal-preserving policy adds missing atoms/hydrogens while preserving observed
heavy-atom coordinates, omits missing-residue modeling, and skips final receptor
heavy-atom minimization. Newly added atoms may be relaxed by PDBFixer. The contrast
is a preparation-policy comparison, not a claimed isolated minimizer ablation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "Data/validation/inputs/egfr_structures/manifest.json"


def scientific_imports():
    """Scientific dependencies are loaded only for explicit execution, never dry-run."""
    global np, Chem, AllChem, ChemicalFeatures, rdMolAlign, RDConfig
    global cas, digest, heavy_atoms, sequence_pairs, transform, SingleDock, LigandPrep
    global raw_ligand, coordinate_distance
    global PoseCrawler, convert_with_obabel
    import numpy as np
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem, ChemicalFeatures, rdMolAlign

    sys.path.insert(0, str(ROOT))
    from Data.validation.redocking_audit import (
        cas,
        digest,
        heavy_atoms,
        sequence_pairs,
        transform,
        raw_ligand,
        coordinate_distance,
    )
    from prodock.dock import SingleDock
    from prodock.preprocess import LigandPrep
    from prodock.postprocess.pose import PoseCrawler
    from prodock.structure.conversion import convert_with_obabel


PROTEINS = ["1M17", "4ZAU"]
POLICIES = ["default_minimized", "crystal_heavy_preserved"]
ENGINES = ["smina", "qvina"]
PRIMARY_PROGRAMS = ["smina"]
PARAMETERS = dict(seed=42, cpu=1, exhaustiveness=8, n_poses=5)


def controlled_environment():
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [
            str(Path(sys.executable).parent),
            str(ROOT / "prodock/dock/binary"),
            env.get("PATH", ""),
        ]
    )
    env.update(OPENMM_CPU_THREADS="1", OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
    return env


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n")


def coordinate_preserving_preparation(raw, output):
    from pdbfixer import PDBFixer
    from openmm import Platform
    from openmm.app import PDBFile

    output.mkdir(parents=True, exist_ok=True)
    selected = output / "selected_crystal_chain_A.pdb"
    lines = [
        line
        for line in Path(raw).read_text().splitlines()
        if line.startswith("ATOM") and line[21:22] == "A" and line[16:17] in {" ", "A"}
    ]
    selected.write_text("\n".join(lines) + "\nTER\nEND\n")
    observed = {
        (a["residue_id"], a["residue"], a["atom"]): a for a in heavy_atoms(selected)
    }
    start = time.perf_counter()
    fixer = PDBFixer(filename=str(selected), platform=Platform.getPlatformByName("CPU"))
    fixer.findMissingResidues()
    omitted_missing_residues = {str(k): v for k, v in fixer.missingResidues.items()}
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)
    pdb = output / "protein.pdb"
    with pdb.open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    prepared = {(a["residue_id"], a["residue"], a["atom"]): a for a in heavy_atoms(pdb)}
    missing = sorted(set(observed) - set(prepared))
    deltas = [
        float(np.linalg.norm(np.array(v["xyz"]) - prepared[k]["xyz"]))
        for k, v in observed.items()
        if k in prepared
    ]
    maximum = max(deltas) if deltas else None
    if missing or maximum is None or maximum > 0.002:
        raise ValueError(
            f"Observed heavy atoms not preserved: missing={missing}, max_difference={maximum}"
        )
    pdbqt = pdb.with_suffix(".pdbqt")
    convert_with_obabel(
        pdb,
        pdbqt,
        sanitize_rebuild=False,
        sanitize_aggressive=False,
        sanitize_backup=False,
        validate_receptor=True,
    )
    converted = {
        (a["residue_id"], a["residue"], a["atom"]): a for a in heavy_atoms(pdbqt)
    }
    if set(converted) != set(prepared):
        raise ValueError("PDBQT conversion changed the named heavy-atom set")
    conversion_delta = max(
        float(np.linalg.norm(np.array(v["xyz"]) - converted[k]["xyz"]))
        for k, v in prepared.items()
    )
    if conversion_delta > 0.002:
        raise ValueError("PDBQT conversion displaced heavy atoms")
    return dict(
        prepared_pdb=str(pdb),
        prepared_pdbqt=str(pdbqt),
        seconds=time.perf_counter() - start,
        original_heavy_atoms=len(observed),
        resulting_heavy_atoms=len(prepared),
        added_heavy_atoms=[list(k) for k in sorted(set(prepared) - set(observed))],
        max_observed_heavy_atom_displacement_A=maximum,
        max_conversion_heavy_atom_displacement_A=conversion_delta,
        missing_residue_modeling_omitted=omitted_missing_residues,
        pH=7.4,
        receptor_minimization=False,
        coordinate_ids_preserved=True,
        selected_input=str(selected),
    )


def frame_and_clashes(raw, prepared, reference_raw, output):
    mobile = cas(prepared)
    original = cas(raw)
    pairs, mismatches = sequence_pairs(mobile, original)
    x = np.asarray([mobile[i]["xyz"] for i, j in pairs])
    y = np.asarray([original[j]["xyz"] for i, j in pairs])
    rotation, translation, residual = transform(x, y)
    reference = Chem.Mol(reference_raw)
    coords = (reference.GetConformer().GetPositions() - translation) @ rotation.T
    for i, xyz in enumerate(coords):
        reference.GetConformer().SetAtomPosition(i, xyz)
    refpath = output / "native_reference_prepared_frame.sdf"
    writer = Chem.SDWriter(str(refpath))
    writer.write(reference)
    writer.close()
    protein = heavy_atoms(prepared)
    protein_xyz = np.asarray([a["xyz"] for a in protein])
    distances = np.linalg.norm(protein_xyz[:, None, :] - coords[None, :, :], axis=2)
    severe = [
        dict(
            residue=protein[i]["residue"],
            residue_id=protein[i]["residue_id"],
            atom=protein[i]["atom"],
            ligand_atom=int(j),
            distance_A=float(distances[i, j]),
        )
        for i, j in np.argwhere(distances < 2.0)
    ]
    pdb = (
        "1M17"
        if any(a["residue_id"] == "769" and a["residue"] == "MET" for a in original)
        else "4ZAU"
    )
    raw_hinge = "769" if pdb == "1M17" else "793"
    hinge_index = next(
        i
        for i, j in pairs
        if original[j]["residue_id"] == raw_hinge and original[j]["residue"] == "MET"
    )
    hinge = mobile[hinge_index]["residue_id"]
    return dict(
        reference_prepared_frame=str(refpath),
        rotation=rotation.tolist(),
        translation=translation.tolist(),
        atom_mapping=[
            dict(
                prepared_residue=mobile[i]["residue_id"],
                raw_residue=original[j]["residue_id"],
                residue=original[j]["residue"],
            )
            for i, j in pairs
        ],
        matched_ca=len(pairs),
        sequence_mismatches=mismatches,
        protein_fit_residual_A=residual,
        reference_minimum_protein_distance_A=float(distances.min()),
        reference_pairs_below_1p5A=int(np.sum(distances < 1.5)),
        reference_pairs_below_2A=len(severe),
        reference_clashes=severe,
        hinge_raw=raw_hinge,
        hinge_prepared=hinge,
    )


def dock_worker(taskpath, resultpath):
    task = json.loads(Path(taskpath).read_text())
    start = time.perf_counter()
    result = dict(task_id=task["task_id"])
    try:
        dock = (
            SingleDock(task["engine"])
            .set_receptor(task["receptor"], validate=True)
            .set_ligand(task["ligand"])
            .set_box(tuple(task["center"]), tuple(task["size"]))
        )
        dock.set_executable(task["executable"]).set_cpu(1).set_seed(
            42
        ).set_exhaustiveness(8).set_num_modes(5)
        dock.set_out(task["output"]).set_log(task["log"])
        run = dock.run(exhaustiveness=8, n_poses=5)
        result.update(
            success=True, called=run.artifacts.called, metadata=run.artifacts.metadata
        )
    except Exception as exc:
        result.update(success=False, error_type=type(exc).__name__, error=str(exc))
    result.update(
        seconds=time.perf_counter() - start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    write_json(resultpath, result)


def execute(task, timeout):
    taskpath = Path(task["task_path"])
    resultpath = taskpath.parent / "execution.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--execute",
        "--dock-task",
        str(taskpath),
        "--task-result",
        str(resultpath),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=controlled_environment(),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        write_json(
            resultpath,
            dict(
                success=False, error_type="Timeout", error=f"Exceeded {timeout} seconds"
            ),
        )
    (taskpath.parent / "worker_stdout.log").write_text(stdout)
    (taskpath.parent / "worker_stderr.log").write_text(stderr)
    result = (
        json.loads(resultpath.read_text())
        if resultpath.exists()
        else dict(
            success=False,
            error_type="WorkerFailure",
            error=f"Exit {process.returncode}",
        )
    )
    return dict(**task, execution=result)


def evaluate(task):
    if not task["execution"]["success"]:
        return []
    reference = Chem.SDMolSupplier(task["reference"], removeHs=True)[0]
    table = PoseCrawler([task["output"]]).crawl_mols(backend="auto")
    protein = heavy_atoms(task["receptor"])
    hinge = next(
        a
        for a in protein
        if a["residue_id"] == task["hinge"]
        and a["residue"] == "MET"
        and a["atom"] == "N"
    )
    factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    results = []
    for _, record in table.iterrows():
        row = dict(
            task_id=task["task_id"],
            pdb=task["pdb"],
            policy=task["policy"],
            engine=task["engine"],
            pose_rank=int(record["pose_rank"]),
            score=float(record["affinity"]),
        )
        try:
            mol = Chem.RemoveHs(record["mol"], sanitize=False)
            if mol.GetNumAtoms() != reference.GetNumAtoms():
                raise ValueError("Heavy-atom count differs from reference")
            mol = AllChem.AssignBondOrdersFromTemplate(reference, mol)
            Chem.SanitizeMol(mol)
            row["fixed_frame_RMSD_A"] = float(rdMolAlign.CalcRMS(mol, reference))
            row["centroid_displacement_A"] = float(
                np.linalg.norm(
                    mol.GetConformer().GetPositions().mean(0)
                    - reference.GetConformer().GetPositions().mean(0)
                )
            )
            acceptors = {
                i
                for f in factory.GetFeaturesForMol(mol)
                if f.GetFamily() == "Acceptor"
                for i in f.GetAtomIds()
            }
            row["hinge_N_to_nearest_acceptor_A"] = min(
                float(
                    np.linalg.norm(
                        np.array(hinge["xyz"]) - mol.GetConformer().GetPositions()[i]
                    )
                )
                for i in acceptors
            )
            row["success"] = True
        except Exception as e:
            row.update(success=False, error=str(e))
        results.append(row)
    return results


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_plan(args):
    manifest_path = args.input_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    structures = manifest.get("structures", manifest.get("cases", []))
    if len(structures) != 2 or {row["pdb"] for row in structures} != set(PROTEINS):
        raise ValueError(
            "This prespecified diagnostic requires exactly two manifest structures"
        )
    resolved = []
    for item in structures:
        row = dict(item)
        if item.get("chain", "A") != "A":
            raise ValueError("This prespecified diagnostic selects chain A")
        for key in ["raw_pdb", "native_reference_sdf", "ccd_ideal_sdf"]:
            path = (manifest_path.parent / item[key]).resolve()
            expected = item.get("sha256", {}).get(key)
            actual = file_digest(path)
            if not expected or actual != expected:
                raise ValueError(
                    f"Missing/mismatched manifest digest for {item['pdb']}:{key}"
                )
            row[key] = str(path)
        resolved.append(row)
    return dict(
        schema_version=1,
        status="planned",
        qualified_for_new_campaign=False,
        input_manifest=str(manifest_path),
        input_manifest_sha256=file_digest(manifest_path),
        structures=resolved,
        policies=POLICIES,
        engines=ENGINES,
        primary_programs=PRIMARY_PROGRAMS,
        secondary_programs=[
            engine for engine in ENGINES if engine not in PRIMARY_PROGRAMS
        ],
        parameters=PARAMETERS,
        task_count=len(resolved) * len(POLICIES) * len(ENGINES),
        default_minimization_max_steps=args.max_minimization_steps,
        max_minimization_steps=args.max_minimization_steps,
        native_quality_threshold_A=2.0,
        actions=[
            "Prepare both policies from identical chain-selected raw coordinates",
            "Verify observed heavy-atom preservation and full PDB/PDBQT coordinate equality",
            "Run eight fresh native controls with fixed seed/settings",
            "Compute quality gates from actual coordinates and native outcomes",
        ],
        note="Metadata-only plan: no molecular preparation, scoring, docking, or scientific analysis is executed.",
    )


def prepare_default(selected, output, max_steps):
    from prodock.preprocess.receptor import prep
    from prodock.preprocess.receptor import minimizers
    from openmm import Platform

    output.mkdir(parents=True, exist_ok=True)
    minimizers._choose_platform = lambda: (
        Platform.getPlatformByName("CPU"),
        {"Threads": "1"},
    )
    start = time.perf_counter()
    obj = prep.ReceptorPrep(use_meeko=False)
    obj.prep(
        str(selected),
        str(output),
        out_fmt="pdbqt",
        max_minimization_steps=max_steps,
        add_prep_suffix=False,
    )
    artifact = Path(obj._final_artifact)
    return dict(
        prepared_pdb=str(artifact.with_suffix(".pdb")),
        prepared_pdbqt=str(artifact),
        seconds=time.perf_counter() - start,
        max_minimization_steps=max_steps,
        preparation_report=obj._last_simulation_report,
    )


def build_quality(plan, prepared_specs, summary, output, acknowledged):
    selected = [
        row for row in prepared_specs if row["policy"] == "crystal_heavy_preserved"
    ]
    primary = [
        row
        for row in summary
        if row["policy"] == "crystal_heavy_preserved"
        and row["engine"] in PRIMARY_PROGRAMS
    ]
    counts = dict(
        expected_tasks=plan["task_count"],
        executed_tasks=len(summary),
        successful_tasks=sum(row["task_success"] for row in summary),
        expected_primary_native_tasks=len(plan["structures"]) * len(PRIMARY_PROGRAMS),
        primary_native_tasks=len(primary),
    )
    expected_ids = {
        f"{entry['pdb']}:{policy}:{engine}"
        for entry in plan["structures"]
        for policy in POLICIES
        for engine in ENGINES
    }
    all_controls_accounted = (
        len(summary) == plan["task_count"]
        and {row["task_id"] for row in summary} == expected_ids
        and all(
            row.get("terminal_state") in {"completed", "task_failed", "analysis_failed"}
            for row in summary
        )
    )
    all_controls_successful = all_controls_accounted and all(
        row["task_success"] and row["valid_poses"] > 0 for row in summary
    )
    primary_complete = len(primary) == counts["expected_primary_native_tasks"] and all(
        row["task_success"] and row["valid_poses"] > 0 for row in primary
    )
    native_pass = len(primary) == counts["expected_primary_native_tasks"] and all(
        row["task_success"]
        and row["rank1_RMSD_A"] is not None
        and row["rank1_RMSD_A"] <= 2.0
        for row in primary
    )
    preservation = len(selected) == len(plan["structures"]) and all(
        row.get("observed_heavy_atoms_preserved", False) for row in selected
    )
    no_clashes = len(selected) == len(plan["structures"]) and all(
        row["reference_pairs_below_2A"] <= row["raw_reference_pairs_below_2A"]
        for row in selected
    )
    input_unchanged = all(
        file_digest(row[key]) == row["sha256"][key]
        for row in plan["structures"]
        for key in ["raw_pdb", "native_reference_sdf", "ccd_ideal_sdf"]
    )
    source_unchanged = all(
        file_digest(ROOT / name) == expected
        for name, expected in plan["source_protocol_sha256"].items()
    )
    prepared_inputs = {}
    prepared_unchanged = True
    for row in prepared_specs:
        for name, digest_name in [
            ("prepared_pdb", "prepared_pdb_sha256"),
            ("prepared_pdbqt", "prepared_pdbqt_sha256"),
            ("reference_prepared_frame", "reference_prepared_frame_sha256"),
        ]:
            path = Path(row[name]).resolve()
            prepared_inputs[str(path.relative_to(output.resolve()))] = row[digest_name]
            prepared_unchanged = (
                prepared_unchanged and file_digest(path) == row[digest_name]
            )
    gates = dict(
        input_integrity=input_unchanged,
        source_protocol_unchanged=source_unchanged,
        prepared_inputs_unchanged=prepared_unchanged,
        crystal_heavy_preservation=preservation,
        no_new_severe_clashes=no_clashes,
        native_controls_accounted_for=all_controls_accounted,
        primary_native_controls_complete=primary_complete,
        native_pose_recovery=native_pass,
    )
    qualified = all(gates.values())
    ready = output / "ready_receptors.json"
    context = output / "chemical_context.json"
    status = (
        "passed"
        if qualified
        else ("incomplete" if not all_controls_accounted else "failed")
    )
    return dict(
        schema_version=1,
        status=status,
        qualified_for_new_campaign=qualified,
        gates=gates,
        counts=counts,
        selected_policy="crystal_heavy_preserved",
        primary_programs=PRIMARY_PROGRAMS,
        secondary_programs=[
            engine for engine in ENGINES if engine not in PRIMARY_PROGRAMS
        ],
        all_controls_successful=all_controls_successful,
        source_protocol_base="repository_root",
        source_protocol_sha256=plan["source_protocol_sha256"],
        prepared_inputs_base="quality_report_directory",
        prepared_inputs_sha256=prepared_inputs,
        native_quality_threshold_A=2.0,
        native_control_summaries=summary,
        receptor_config_sha256=file_digest(ready),
        chemical_context_sha256=file_digest(context),
        ready_receptors_path=str(ready),
        chemical_context_path=str(context),
        decision=dict(
            proceed_large_campaign=qualified or bool(acknowledged),
            requires_explicit_acknowledgement=not qualified,
            acknowledgement_provided=bool(acknowledged),
        ),
        caveat=(
            "Prespecified workflow quality control on two native complexes, "
            "not independent docking validation. Acknowledgement never changes "
            "failed scientific gates to passed."
        ),
    )


def execute_experiment(args, plan):
    scientific_imports()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preregistration = output / "prespecified_design.json"
    if preregistration.exists():
        raise FileExistsError(
            "Use a fresh output directory; do not overwrite prior outcomes"
        )
    source_files = [
        "Data/validation/preparation_sensitivity.py",
        "Data/validation/redocking_audit.py",
        "prodock/preprocess/receptor/prep.py",
        "prodock/preprocess/receptor/minimizers.py",
        "prodock/preprocess/ligand/prep.py",
        "prodock/structure/conversion.py",
        "prodock/dock/single.py",
        "prodock/dock/common_binary.py",
        "prodock/postprocess/pose/core.py",
    ]
    plan["source_protocol_sha256"] = {
        name: file_digest(ROOT / name) for name in source_files
    }
    plan.update(
        status="execution_started",
        created_UTC=datetime.now(timezone.utc).isoformat(),
        script_sha256=file_digest(__file__),
    )
    write_json(preregistration, plan)
    started = time.perf_counter()
    prepared_specs = []
    preparation = []
    tasks = []
    chemical_cases = []
    for entry in plan["structures"]:
        pdb = entry["pdb"]
        raw = Path(entry["raw_pdb"])
        reference = Chem.SDMolSupplier(entry["native_reference_sdf"], removeHs=True)[0]
        template = Chem.SDMolSupplier(entry["ccd_ideal_sdf"], removeHs=True)[0]
        if (
            reference is None
            or template is None
            or reference.GetNumAtoms() != template.GetNumAtoms()
        ):
            raise ValueError(
                f"{pdb}: incomplete/invalid experimental ligand or CCD chemistry"
            )
        reference = AllChem.AssignBondOrdersFromTemplate(template, reference)
        Chem.SanitizeMol(reference)
        raw_copies = raw_ligand(raw, entry["ligand_code"], entry.get("chain", "A"))
        matching_copies = [
            (residue, coordinate_distance(reference, molecule))
            for residue, molecule in raw_copies
        ]
        matching_copies = [
            (residue, difference)
            for residue, difference in matching_copies
            if difference is not None and difference <= 0.01
        ]
        if not matching_copies:
            raise ValueError(
                f"{pdb}: reference does not match an experimental bound ligand copy"
            )
        folder = output / pdb
        folder.mkdir(exist_ok=True)
        fresh = coordinate_preserving_preparation(
            raw, folder / "crystal_heavy_preserved"
        )
        minimized = prepare_default(
            fresh["selected_input"],
            folder / "default_minimized",
            args.max_minimization_steps,
        )
        preparation.extend(
            [
                dict(pdb=pdb, policy="crystal_heavy_preserved", **fresh),
                dict(pdb=pdb, policy="default_minimized", **minimized),
            ]
        )
        ligand_dir = folder / "ligands"
        ligand_dir.mkdir(exist_ok=True)
        smiles = Chem.MolToSmiles(template, isomericSmiles=True)
        ligand = (
            LigandPrep(output_dir=ligand_dir, name_key="id")
            .set_conformer_seed(42)
            .set_output_format("pdbqt")
            .set_converter_backend("meeko")
            .from_list_of_dicts([dict(id=f"native_{pdb}", smiles=smiles)])
            .process_all()
        )
        paths = [path for path in ligand.output_paths if path is not None]
        if len(paths) != 1:
            raise ValueError(
                f"{pdb}: fresh ligand preparation failed: {ligand.records}"
            )
        write_json(
            folder / "ligand_manifest.json",
            dict(
                records=ligand.records,
                sha256=digest(paths[0]),
                seed=42,
                ETKDG="v3",
                MMFF94_max_iterations=200,
                backend="meeko",
            ),
        )
        raw_protein = [
            a
            for a in heavy_atoms(raw)
            if a["residue"]
            in {a["residue"] for a in heavy_atoms(fresh["selected_input"])}
        ]
        raw_xyz = np.asarray([a["xyz"] for a in raw_protein])
        ref_xyz = reference.GetConformer().GetPositions()
        raw_pairs = int(
            np.sum(
                np.linalg.norm(raw_xyz[:, None, :] - ref_xyz[None, :, :], axis=2) < 2.0
            )
        )
        # Box definitions are frozen from experimental coordinates, before docking outcomes.
        center = ref_xyz.mean(0).tolist()
        size = [float(np.max(ref_xyz.max(0) - ref_xyz.min(0)) + 8.0)] * 3
        for policy, prep_result in [
            ("default_minimized", minimized),
            ("crystal_heavy_preserved", fresh),
        ]:
            directory = folder / policy
            protein = Path(prep_result["prepared_pdb"])
            protein_qt = Path(prep_result["prepared_pdbqt"])
            framed = frame_and_clashes(raw, protein, reference, directory)
            ready = dict(
                pdb=pdb,
                policy=policy,
                receptor_id=pdb,
                receptor_pdbqt=str(protein_qt),
                prepared_pdb=str(protein),
                prepared_pdbqt=str(protein_qt),
                experimental_reference=entry["native_reference_sdf"],
                box_reference_prepared_frame=framed["reference_prepared_frame"],
                center=center,
                size=size,
                prepared_to_experimental_rotation=framed["rotation"],
                prepared_to_experimental_translation=framed["translation"],
                experimental_to_prepared_formula="(experimental_xyz - translation) @ rotation.T",
                selected_bound_copy=matching_copies[0][0],
                reference_coordinate_identity_max_difference_A=matching_copies[0][1],
                raw_reference_pairs_below_2A=raw_pairs,
                observed_heavy_atoms_preserved=(
                    prep_result.get("max_observed_heavy_atom_displacement_A", 1e9)
                    <= 0.002
                ),
                prepared_pdb_sha256=file_digest(protein),
                prepared_pdbqt_sha256=file_digest(protein_qt),
                reference_prepared_frame_sha256=file_digest(
                    framed["reference_prepared_frame"]
                ),
                **framed,
            )
            prepared_specs.append(ready)
            write_json(directory / "prepared_spec.json", ready)
            if policy == "crystal_heavy_preserved":
                chemical_cases.append(
                    dict(
                        pdb=pdb,
                        hinge_raw=entry["hinge_raw"],
                        hinge_prepared=framed["hinge_prepared"],
                        prepared_to_experimental_rotation=framed["rotation"],
                        prepared_to_experimental_translation=framed["translation"],
                        protein_fit_residual_A=framed["protein_fit_residual_A"],
                        atom_mapping=framed["atom_mapping"],
                        model_scope=entry.get("model_scope"),
                    )
                )
            for engine in ENGINES:
                task_folder = directory / engine
                task_folder.mkdir(exist_ok=True)
                executable = (
                    (args.engine_bin_dir / engine).resolve()
                    if args.engine_bin_dir
                    else ROOT / "prodock/dock/binary" / engine
                )
                if not executable.is_file():
                    raise FileNotFoundError(executable)
                task = dict(
                    task_id=f"{pdb}:{policy}:{engine}",
                    pdb=pdb,
                    policy=policy,
                    engine=engine,
                    receptor=str(protein_qt),
                    receptor_sha256=digest(protein_qt),
                    ligand=str(paths[0]),
                    ligand_sha256=digest(paths[0]),
                    reference=framed["reference_prepared_frame"],
                    hinge=framed["hinge_prepared"],
                    executable=str(executable),
                    executable_sha256=digest(executable),
                    center=center,
                    size=size,
                    output=str(task_folder / f"native_{pdb}_docked.pdbqt"),
                    log=str(task_folder / "dock.log"),
                    task_path=str(task_folder / "task.json"),
                    parameters=PARAMETERS,
                )
                write_json(task["task_path"], task)
                tasks.append(task)
    selected = [
        row for row in prepared_specs if row["policy"] == "crystal_heavy_preserved"
    ]
    ready_json = dict(
        schema_version=1,
        policy="crystal_heavy_preserved",
        prepared_receptors=selected,
        provenance=dict(input_manifest_sha256=plan["input_manifest_sha256"]),
    )
    context_json = dict(
        schema_version=1,
        policy="crystal_heavy_preserved",
        cases=chemical_cases,
        ready_prepared_specs=selected,
        input_manifest_sha256=plan["input_manifest_sha256"],
    )
    write_json(output / "ready_receptors.json", ready_json)
    write_json(output / "chemical_context.json", context_json)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        completed = list(pool.map(lambda task: execute(task, args.timeout), tasks))
    poses = []
    summary = []
    for task in completed:
        try:
            measurements = evaluate(task)
        except Exception as exc:
            measurements = []
            task["analysis_error"] = str(exc)
        poses.extend(measurements)
        valid = [r for r in measurements if r.get("success")]
        summary.append(
            dict(
                task_id=task["task_id"],
                pdb=task["pdb"],
                policy=task["policy"],
                engine=task["engine"],
                task_success=task["execution"]["success"],
                terminal_state=(
                    "completed"
                    if task["execution"]["success"] and valid
                    else (
                        "task_failed"
                        if not task["execution"]["success"]
                        else "analysis_failed"
                    )
                ),
                seconds=task["execution"].get("seconds"),
                valid_poses=len(valid),
                rank1_RMSD_A=next(
                    (r["fixed_frame_RMSD_A"] for r in valid if r["pose_rank"] == 1),
                    None,
                ),
                best_of_returned_RMSD_A=min(
                    (r["fixed_frame_RMSD_A"] for r in valid), default=None
                ),
                rank1_hinge_acceptor_distance_A=next(
                    (
                        r["hinge_N_to_nearest_acceptor_A"]
                        for r in valid
                        if r["pose_rank"] == 1
                    ),
                    None,
                ),
                error=task.get("analysis_error", task["execution"].get("error")),
            )
        )
    for filename, rows in [("summary.csv", summary), ("poses.csv", poses)]:
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (output / filename).open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    quality = build_quality(
        plan, prepared_specs, summary, output, args.acknowledge_native_failures
    )
    write_json(output / "quality_report.json", quality)
    write_json(
        output / "report.json",
        dict(
            prespecified_design=plan,
            wall_seconds=time.perf_counter() - started,
            preparation=preparation,
            prepared_specs=prepared_specs,
            tasks=completed,
            pose_metrics=poses,
            summary=summary,
            quality_report=quality,
        ),
    )
    print(json.dumps(quality, indent=2))
    return 0 if quality["decision"]["proceed_large_campaign"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest", "--manifest", type=Path, default=DEFAULT_MANIFEST
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/preparation_sensitivity")
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true")
    execution.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-minimization-steps", type=int, default=5000)
    parser.add_argument("--engine-bin-dir", type=Path)
    parser.add_argument("--acknowledge-native-failures", action="store_true")
    parser.add_argument("--dock-task", type=Path)
    parser.add_argument("--task-result", type=Path)
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout <= 0 or args.max_minimization_steps < 1:
        parser.error("Positive resource limits required")
    if args.dock_task:
        if not args.execute:
            parser.error("Docking worker requires explicit --execute")
        os.environ.update(controlled_environment())
        scientific_imports()
        dock_worker(args.dock_task, args.task_result)
        return 0
    plan = manifest_plan(args)
    if not args.execute:
        if args.plan_json:
            args.plan_json.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.plan_json, plan)
        print(json.dumps(plan, indent=2))
        return 0
    if (args.output / "prespecified_design.json").exists():
        parser.error(
            "Use a fresh output directory; prior experimental outcomes are preserved"
        )
    os.environ.update(controlled_environment())
    try:
        return execute_experiment(args, plan)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        # An interrupted/incomplete protocol can never qualify solely because files exist.
        write_json(
            args.output / "quality_report.json",
            dict(
                schema_version=1,
                status="incomplete",
                qualified_for_new_campaign=False,
                gates={},
                counts=dict(expected_tasks=plan["task_count"]),
                error_type=type(exc).__name__,
                error=str(exc),
                decision=dict(
                    proceed_large_campaign=False, requires_explicit_acknowledgement=True
                ),
                receptor_config_sha256=None,
                chemical_context_sha256=None,
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
