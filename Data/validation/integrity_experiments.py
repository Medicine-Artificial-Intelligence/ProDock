"""LoLO Blue: inexpensive database integrity and failure-injection experiments.

These are controlled storage/API fixtures, not an end-to-end docking restart
study or a virtual-screening accuracy benchmark. Run with ProDock's Python and
use a fresh output directory; reports include failed checks rather than hiding
unsupported behavior. Interaction events are constructed fixtures, not ProLIF
predictions. Protein and engine names in these fixtures are logical identifiers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prodock.core import ProDockPipeline  # noqa: E402
from prodock.database import PoseDatabase, PoseQuery  # noqa: E402

SMILES = [
    "CCO",
    "CC(=O)[O-]",
    "C[NH3+]",
    "c1ccncc1",
    "c1ccccc1O",
    "CC(=O)Nc1ccc(O)cc1",
    "C[C@H](O)C(=O)O",
    "[13CH3]CO",
    "O=C1NCCN1",
    "FC(F)(F)c1ccc(Cl)cc1",
    "CCS(=O)(=O)N",
    "C1CCOC1",
]
COORDINATE_TOLERANCE_ANGSTROM = 1e-5


def fresh_path(output: Path, name: str) -> Path:
    path = output / name
    if path.exists():
        raise FileExistsError(f"Use a fresh output directory: {path}")
    return path


def make_fixtures() -> tuple[list[dict], dict]:
    """Twelve real chemical graphs, two distinct coordinate sets per record."""
    rows, payloads = [], {}
    for molecule_index, smiles in enumerate(SMILES):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 + molecule_index
        assert len(AllChem.EmbedMultipleConfs(mol, numConfs=2, params=params)) == 2
        mol.SetProp("source_id", f"fixture_{molecule_index}")
        mol.SetProp("_Name", f"chemical_fixture_{molecule_index}")
        for atom in mol.GetAtoms():
            atom.SetIntProp("fixture_atom_index", atom.GetIdx())
        for bond in mol.GetBonds():
            bond.SetDoubleProp("fixture_bond_tag", bond.GetIdx() + 0.125)
        for protein in ["R0", "R1"]:
            for engine in ["vina", "smina"]:
                for rank in [1, 2]:
                    molecule = Chem.Mol(mol)
                    offset = molecule_index * 0.17 + rank * 0.31 + (protein == "R1")
                    for conf in molecule.GetConformers():
                        for atom_index in range(molecule.GetNumAtoms()):
                            pos = conf.GetAtomPosition(atom_index)
                            conf.SetAtomPosition(
                                atom_index,
                                (pos.x + offset, pos.y - offset / 2, pos.z + rank),
                            )
                    ligand = f"compound_{molecule_index}"  # Real IDs ending in digits must remain distinct.
                    pose_id = f"{protein}__{ligand}__{engine}__pose{rank}"
                    rows.append(
                        dict(
                            receptor_id=protein,
                            ligand_id=ligand,
                            engine=engine,
                            pose_rank=rank,
                            pose_id=pose_id,
                            affinity=-5.0 - molecule_index * 0.1 - rank * 0.01,
                            mol=molecule,
                            pose_metadata={
                                "input_id": ligand,
                                "rank": rank,
                                "unicode": "Å",
                            },
                            score_data={
                                "raw_native": -5.0 - molecule_index * 0.1 - rank * 0.01,
                                "components": {"fixture_term": molecule_index + 0.5},
                            },
                            score_metadata={
                                "units": "fixture native score",
                                "not_experimental_affinity": True,
                            },
                            ligand_metadata={
                                "source": "constructed chemical integrity fixture",
                                "input_id": ligand,
                            },
                            receptor_metadata={
                                "source": "logical fixture",
                                "pdb_id": protein,
                            },
                            engine_metadata={
                                "version": "logical fixture; no executable invoked"
                            },
                        )
                    )
                    if molecule_index % 3 == 0:
                        payloads[pose_id] = None
                    else:
                        payloads[pose_id] = {
                            "Hydrophobic": {
                                "LEU23.A": [
                                    {
                                        "ligand_residue": "LIG1",
                                        "indices": {"ligand": [0], "protein": [12]},
                                        "parent_indices": {
                                            "ligand": [0],
                                            "protein": [1002],
                                        },
                                        "distance": 3.25,
                                        "angle": None,
                                        "metadata": {"fixture": True},
                                    },
                                    {
                                        "ligand_residue": "LIG1",
                                        "indices": {"ligand": [1], "protein": [13]},
                                        "parent_indices": {
                                            "ligand": [1],
                                            "protein": [1003],
                                        },
                                        "distance": 3.75,
                                        "angle": None,
                                        "metadata": {"fixture": True},
                                    },
                                ]
                            },
                            "HBDonor": {
                                "ASP45.B": [
                                    {
                                        "ligand_residue": "LIG1",
                                        "indices": {"ligand": [0], "protein": [21]},
                                        "parent_indices": {
                                            "ligand": [0],
                                            "protein": [1021],
                                        },
                                        "distance": 2.81,
                                        "angle": 159.25,
                                        "metadata": {
                                            "fixture": True,
                                            "note": "constructed event, not predicted",
                                        },
                                    },
                                ]
                            },
                        }
    return rows, payloads


def atom_signature(mol):
    return [
        (
            a.GetAtomicNum(),
            a.GetFormalCharge(),
            a.GetIsotope(),
            str(a.GetChiralTag()),
            a.GetIsAromatic(),
        )
        for a in mol.GetAtoms()
    ]


def bond_signature(mol):
    return [
        (
            b.GetBeginAtomIdx(),
            b.GetEndAtomIdx(),
            str(b.GetBondType()),
            str(b.GetStereo()),
            tuple(b.GetStereoAtoms()),
            b.GetIsAromatic(),
        )
        for b in mol.GetBonds()
    ]


def roundtrip_case(output: Path, compressed: bool) -> dict:
    rows, payloads = make_fixtures()
    path = fresh_path(output, f"roundtrip_compressed_{compressed}.db")
    expected = {r["pose_id"]: r for r in rows}
    with PoseDatabase(path, compress_mol=compressed) as db:
        db.create_run(
            name="chemical and interaction fixture",
            config={"seed": 42, "fixture": True},
        )
        start = time.perf_counter()
        db.insert_dataframe(pd.DataFrame(rows), interactions_by_pose=payloads)
        insert_seconds = time.perf_counter() - start
        count = db.count_poses()
        interaction_count = db.connection.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]
        db.insert_dataframe(pd.DataFrame(rows), interactions_by_pose=payloads)
        assert db.count_poses() == count == len(rows)
        assert (
            db.connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            == interaction_count
            == 192
        )
        assert {
            r[0] for r in db.connection.execute("SELECT mol_is_compressed FROM poses")
        } == {int(compressed)}
        assert db.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not db.connection.execute("PRAGMA foreign_key_check").fetchall()
    max_error, conformers, atoms = 0.0, 0, 0
    with PoseQuery(path) as query:
        records = query.poses()
        assert len(records) == len(expected)
        for record in records:
            original = expected[record.pose_id]
            before, after = original["mol"], record.mol
            assert atom_signature(before) == atom_signature(after)
            assert bond_signature(before) == bond_signature(after)
            assert Chem.MolToSmiles(before, isomericSmiles=True) == Chem.MolToSmiles(
                after, isomericSmiles=True
            )
            for key in ["source_id", "_Name"]:
                assert before.GetProp(key) == after.GetProp(key)
            for atom in after.GetAtoms():
                assert atom.GetIntProp("fixture_atom_index") == atom.GetIdx()
            for bond in after.GetBonds():
                assert bond.GetDoubleProp("fixture_bond_tag") == bond.GetIdx() + 0.125
            assert before.GetNumConformers() == after.GetNumConformers() == 2
            for conf_before, conf_after in zip(
                before.GetConformers(), after.GetConformers()
            ):
                error = float(
                    np.max(
                        np.abs(conf_before.GetPositions() - conf_after.GetPositions())
                    )
                )
                max_error = max(max_error, error)
                assert error <= COORDINATE_TOLERANCE_ANGSTROM
                conformers += 1
                atoms += before.GetNumAtoms()
            assert record.affinity == original["affinity"]
            assert record.pose_metadata == original["pose_metadata"]
            assert record.score_data == original["score_data"]
            assert record.score_metadata == original["score_metadata"]
        interaction_records = query.interactions()
        assert len(interaction_records) == interaction_count
        for event in interaction_records:
            original = payloads[event.pose_id][event.interaction_type][
                event.residue_id
            ][event.occurrence_index]
            assert event.ligand_residue == original["ligand_residue"]
            assert event.ligand_atom_indices == original["indices"]["ligand"]
            assert event.protein_atom_indices == original["indices"]["protein"]
            assert (
                event.ligand_parent_atom_indices == original["parent_indices"]["ligand"]
            )
            assert (
                event.protein_parent_atom_indices
                == original["parent_indices"]["protein"]
            )
            assert event.distance == original["distance"]
            assert event.angle == original["angle"]
            assert event.metadata == dict(
                original["metadata"], source_format="detailed"
            )
            assert event.chain_id == event.residue_id.rsplit(".", 1)[1]
            assert event.residue_number in [23, 45]
        hits = query.poses(
            receptor_id="R0",
            engine="vina",
            top_rank=1,
            interaction_type="HBDonor",
            residue_id="ASP45.B",
            include_mol=False,
        )
        expected_hits = {
            r["pose_id"]
            for r in rows
            if r["receptor_id"] == "R0"
            and r["engine"] == "vina"
            and r["pose_rank"] == 1
            and payloads[r["pose_id"]]
        }
        assert {r.pose_id for r in hits} == expected_hits
        score_hits = query.scores(
            receptor_id="R1", engine="smina", top_rank=1, affinity_threshold=-5.5
        )
        expected_scores = {
            r["pose_id"]
            for r in rows
            if r["receptor_id"] == "R1"
            and r["engine"] == "smina"
            and r["pose_rank"] == 1
            and r["affinity"] <= -5.5
        }
        assert {r.pose_id for r in score_hits} == expected_scores
        assert set(query.ligands()) == {f"compound_{i}" for i in range(len(SMILES))}
    return dict(
        status="passed",
        compressed=compressed,
        molecules=12,
        pose_records=len(rows),
        conformers_checked=conformers,
        atom_coordinate_triplets_checked=atoms,
        coordinate_axes_checked=["x", "y", "z"],
        maximum_coordinate_error_angstrom=max_error,
        tolerance_angstrom=COORDINATE_TOLERANCE_ANGSTROM,
        interaction_events=interaction_count,
        combined_filter_hits=len(expected_hits),
        native_scores_and_metadata_preserved=True,
        stable_numeric_suffix_ids=True,
        database_bytes=path.stat().st_size,
        insertion_seconds=insert_seconds,
        caveat="Constructed graphs/events and logical protein/engine IDs; no docking or ProLIF run.",
    )


def simple_row(ligand="seed", pose_id=None):
    return dict(
        receptor_id="R",
        ligand_id=ligand,
        engine="vina",
        pose_rank=1,
        affinity=-7.25,
        mol=Chem.MolFromSmiles("CCO"),
        pose_id=pose_id or f"R__{ligand}__vina__pose1",
    )


def atomicity_case(output: Path, operation: str, replace=True, interrupt=False) -> dict:
    path = fresh_path(output, f"atomic_{operation}_{replace}_{interrupt}.db")
    exception_type = KeyboardInterrupt if interrupt else ValueError
    with PoseDatabase(path) as db:
        db.insert_many([simple_row("existing")])
        prior = db.count_poses()
        try:
            if operation in ["insert_many", "insert_dataframe"]:
                valid, invalid = simple_row("new"), simple_row("invalid")
                invalid["pose_rank"] = 0
                if interrupt:

                    def interrupted_rows():
                        yield valid
                        raise KeyboardInterrupt("controlled interruption after one row")

                    db.insert_many(interrupted_rows(), replace=replace)
                elif operation == "insert_many":
                    db.insert_many([valid, invalid], replace=replace)
                else:
                    db.insert_dataframe(pd.DataFrame([valid, invalid]), replace=replace)
            elif operation == "insert_interactions":
                db.insert_interactions(
                    [
                        dict(
                            pose_id=simple_row("existing")["pose_id"],
                            interaction_type="Hydrophobic",
                            residue_id="LEU23.A",
                        ),
                        dict(
                            pose_id="missing_pose",
                            interaction_type="Hydrophobic",
                            residue_id="VAL24.A",
                        ),
                    ]
                )
            elif operation == "dataframe_and_interactions":
                row = simple_row("new")
                db.insert_dataframe(
                    pd.DataFrame([row]),
                    interactions_by_pose={
                        row["pose_id"]: {"Hydrophobic": ["LEU23.A"]},
                        "missing_pose": {"Hydrophobic": ["VAL24.A"]},
                    },
                )
            elif operation == "interaction_payload":
                db.upsert_interaction_payload(
                    {
                        simple_row("existing")["pose_id"]: {"Hydrophobic": ["LEU23.A"]},
                        "missing_pose": {"Hydrophobic": ["VAL24.A"]},
                    }
                )
            elif operation == "caller_transaction":
                db.connection.execute("BEGIN")
                db.upsert_pose(**{k: v for k, v in simple_row("new").items()})
                db.connection.rollback()
            else:
                raise ValueError(operation)
        except (ValueError, KeyError, KeyboardInterrupt) as exc:
            caught = type(exc).__name__
        else:
            caught = None
        poses = db.count_poses()
        events = db.connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[
            0
        ]
        refs = [
            r[0]
            for r in db.connection.execute(
                "SELECT ligand_id FROM ligands ORDER BY ligand_id"
            )
        ]
        all_atomic = poses == prior and events == 0 and refs == ["existing"]
        if operation == "caller_transaction":
            assert caught is None
        else:
            assert caught in [exception_type.__name__, "KeyError"]
        return dict(
            status="passed" if all_atomic else "failed",
            operation=operation,
            replace=replace,
            controlled_keyboard_interrupt=interrupt,
            exception=caught,
            poses_before=prior,
            poses_after=poses,
            interaction_rows_after=events,
            ligand_reference_rows_after=refs,
            atomic_scope="single public write call, or caller-owned SQLite transaction",
            verified_workflow_checkpoint_resume=False,
        )


def parameter_history_case(output: Path) -> dict:
    path = fresh_path(output, "parameter_history.db")
    row = simple_row("compound_2026")
    with PoseDatabase(path) as db:
        first = db.create_run(name="seed42", config={"seed": 42})
        db.insert_many([row])
        second = db.create_run(name="seed43", config={"seed": 43})
        changed = dict(row, affinity=-8.5)
        db.insert_many([changed])
        record = db.get_pose(pose_id=row["pose_id"])
        run_id = db.connection.execute("SELECT run_id FROM poses").fetchone()[0]
        assert db.count_poses() == 1 and record.affinity == -8.5 and run_id == second
        assert first != second
    isolated = []
    for seed, score in [(42, -7.25), (43, -8.5)]:
        p = fresh_path(output, f"isolated_seed_{seed}.db")
        with PoseDatabase(p) as db:
            db.create_run(name=f"seed{seed}", config={"seed": seed})
            db.insert_many([dict(row, affinity=score)])
        with PoseQuery(p) as query:
            isolated.append(query.pose(pose_id=row["pose_id"]).affinity)
    assert isolated == [-7.25, -8.5]
    return dict(
        status="documented_limitation",
        shared_database_pose_records=1,
        shared_database_latest_score=-8.5,
        both_run_metadata_rows_retained=True,
        independent_pose_histories_retained=False,
        isolated_databases_preserve_variants=True,
        policy="Parameter variants must use isolated databases or explicit distinct namespaces under current key.",
    )


def provenance_case(output: Path) -> dict:
    project = fresh_path(output, "provenance_project")
    project.mkdir()
    pipeline = ProDockPipeline(project, progress=False, log_colored=False)
    original = {
        "receptors": [
            {"id": "R", "receptor": "prepared.pdbqt", "box": {"center": [1, 2, 3]}}
        ],
        "fixture": True,
        "seed": 42,
    }
    payload = json.dumps(original).encode()
    campaign = SimpleNamespace(save_json=lambda path: Path(path).write_bytes(payload))
    saved = pipeline.save_campaign(campaign)
    saved.write_text(json.dumps({"edited_before_database_save": True}))
    database = pipeline.save_database(df=pd.DataFrame([simple_row()]))
    saved.write_text(json.dumps({"edited_after_database_save": True}))
    with sqlite3.connect(database) as connection:
        config = json.loads(
            connection.execute("SELECT config_json FROM runs").fetchone()[0]
        )
    frozen = config.get("campaign") == original
    digest_matches_original = (
        config.get("campaign_sha256") == hashlib.sha256(payload).hexdigest()
    )
    return dict(
        status="passed" if frozen and digest_matches_original else "failed",
        saved_campaign_bytes_survive_edit_before_database_save=frozen,
        campaign_digest_matches_original=digest_matches_original,
        later_file_edit_does_not_mutate_database=True,
        captured_fields=sorted(config),
        caveat=(
            "This tests saved-campaign bytes, not hashes/versions of every "
            "structure, external executable, or full environment."
        ),
    )


def failed_pipeline_run_metadata_case(output: Path) -> dict:
    project = fresh_path(output, "failed_pipeline_project")
    project.mkdir()
    pipeline = ProDockPipeline(project, progress=False, log_colored=False)
    invalid = simple_row()
    invalid["pose_rank"] = 0
    try:
        pipeline.save_database(df=pd.DataFrame([invalid]))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid pose was not rejected")
    with sqlite3.connect(project / "prodock.db") as connection:
        runs = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        poses = connection.execute("SELECT COUNT(*) FROM poses").fetchone()[0]
    assert runs == 1 and poses == 0
    return dict(
        status="documented_limitation",
        failed_pipeline_write_pose_rows=poses,
        committed_run_metadata_rows=runs,
        full_campaign_transaction=False,
        caveat=(
            "The insertion operation rolls back, but create_run commits earlier. "
            "An empty run metadata record remains after failed pipeline insertion."
        ),
    )


def run_experiments(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("Use a fresh output directory")
    checks = []
    cases = [
        ("roundtrip_compressed", lambda: roundtrip_case(output, True)),
        ("roundtrip_uncompressed", lambda: roundtrip_case(output, False)),
    ]
    for operation in [
        "insert_many",
        "insert_dataframe",
        "insert_interactions",
        "dataframe_and_interactions",
        "interaction_payload",
        "caller_transaction",
    ]:
        for replace in (
            [False, True]
            if operation in ["insert_many", "insert_dataframe"]
            else [True]
        ):
            cases.append(
                (
                    f"atomic_{operation}_{replace}",
                    lambda op=operation, rep=replace: atomicity_case(
                        output, op, replace=rep
                    ),
                )
            )
    cases.extend(
        [
            (
                "controlled_interrupt",
                lambda: atomicity_case(output, "insert_many", interrupt=True),
            ),
            ("parameter_history_policy", lambda: parameter_history_case(output)),
            ("saved_campaign_provenance", lambda: provenance_case(output)),
            (
                "failed_pipeline_run_metadata",
                lambda: failed_pipeline_run_metadata_case(output),
            ),
        ]
    )
    for name, function in cases:
        start = time.perf_counter()
        try:
            result = function()
        except Exception as exc:
            result = dict(status="failed", exception=type(exc).__name__, error=str(exc))
        result.update(check=name, seconds=time.perf_counter() - start)
        checks.append(result)
    report = dict(
        interpreter=sys.executable,
        python=platform.python_version(),
        rdkit=rdBase.rdkitVersion,
        sqlite=sqlite3.sqlite_version,
        cases=checks,
        passed=sum(c["status"] == "passed" for c in checks),
        failed=sum(c["status"] == "failed" for c in checks),
        documented_limitations=sum(
            c["status"] == "documented_limitation" for c in checks
        ),
        scope="Controlled database/API fixtures, not docking or persisted workflow checkpoint/resume.",
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "prodock/database/pose_db.py",
                ROOT / "prodock/core.py",
                Path(__file__).resolve(),
            ]
        },
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "checks.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["check", "status", "seconds", "details_json"]
        )
        writer.writeheader()
        for check in checks:
            writer.writerow(
                dict(
                    check=check["check"],
                    status=check["status"],
                    seconds=check["seconds"],
                    details_json=json.dumps(check, sort_keys=True),
                )
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiments(args.output)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["passed", "failed", "documented_limitations", "scope"]
            },
            indent=2,
        )
    )
    raise SystemExit(bool(report["failed"]))


if __name__ == "__main__":
    main()
