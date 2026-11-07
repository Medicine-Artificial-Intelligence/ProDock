"""LoLO Blue: execute pinned native reanalysis helpers on a shared fixture.

This is a module-level functional comparison of score loading and Z-score
aggregation. It does not execute complete DockM8/MolDockLab workflows, docking,
preparation, interaction detection, or virtual-screening accuracy evaluation.
Every comparator result comes from the downloaded unmodified native function.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prodock.database import PoseDatabase, PoseQuery  # noqa: E402

SOURCE_FILES = {
    "DockM8": {
        "repository": "DrugBud-Suite/DockM8",
        "commit": "3b765ea1122d01b6d3484ea25ec789f4456d8f8a",
        "sha256": {
            "LICENSE": "5e52d44b167607af7c06a34ed692376f7e7d7e2949d3b3a8d2bd0e3d16905d8c",
            "scripts/utilities/fast_sdf_loader.py": "c3c4010674ab54c0af5c4b25ce5f148589ed4546ed448668932aa4d196b39b15",
            "scripts/consensus/methods/zscore.py": "2a42383ac6f83029640dbc1132d04261cb19a289f55ec5858d1fb0a146e411e9",
            "scripts/consensus/methods/rbr.py": "4def5df0c781ac34f121000a70e26add56cfe53c4f24319a782e7ffdcb340438",
        },
        "paths": [
            "LICENSE",
            "scripts/utilities/fast_sdf_loader.py",
            "scripts/consensus/methods/zscore.py",
            "scripts/consensus/methods/rbr.py",
        ],
    },
    "MolDockLab": {
        "repository": "volkamerlab/MolDockLab",
        "commit": "1cceb4f099ee02c2884272edde3c10881cb8fdc6",
        "sha256": {
            "LICENSE.txt": "5e52d44b167607af7c06a34ed692376f7e7d7e2949d3b3a8d2bd0e3d16905d8c",
            "src/consensus_rank.py": "33d1ecec73a6db6f913ea0c17f0d2abfba7599c939aeded68fce18155cb53770",
            "src/preprocessing.py": "27e2475ea7c8d348149c766995657c1ce4727df42682209862d11a6cd186e3c2",
            "src/utilities.py": "8c59e36eaa7a86b5afee8bfe855874e24de71a18d03fbae93c1dbfb6e6f0daef",
        },
        "paths": [
            "LICENSE.txt",
            "src/consensus_rank.py",
            "src/preprocessing.py",
            "src/utilities.py",
        ],
    },
}


def load_native(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_sources(output: Path, source_cache: Path | None = None) -> dict:
    manifest = {}
    cache_manifest = None
    if source_cache is not None and (source_cache / "source_manifest.json").is_file():
        cache_manifest = json.loads((source_cache / "source_manifest.json").read_text())
    for name, config in SOURCE_FILES.items():
        commit = config["commit"]
        if cache_manifest is not None:
            declared = cache_manifest.get(name, {})
            if declared.get("commit") != commit:
                raise RuntimeError(
                    f"Offline source manifest has a different {name} commit"
                )
            declared_hashes = {
                r["path"]: r["sha256"] for r in declared.get("files", [])
            }
            if any(
                declared_hashes.get(p) != expected
                for p, expected in config["sha256"].items()
            ):
                raise RuntimeError(
                    f"Offline source manifest has different {name} file digests"
                )
        records = []
        for relative in config["paths"]:
            path = output / "sources" / name / relative
            url = f"https://raw.githubusercontent.com/{config['repository']}/{commit}/{relative}"
            origin = "existing_output"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                if source_cache is not None:
                    cached = source_cache / "sources" / name / relative
                    if not cached.is_file():
                        raise FileNotFoundError(
                            f"Offline native source cache is incomplete: {cached}"
                        )
                    shutil.copyfile(cached, path)
                    origin = "source_cache"
                else:
                    path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
                    origin = "pinned_network_source"
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != config["sha256"][relative]
            ):
                raise RuntimeError(f"Pinned native source digest mismatch: {path}")
            imports = []
            if path.suffix == ".py":
                for node in ast.parse(path.read_text()).body:
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        imports.append(ast.unparse(node))
            records.append(
                dict(
                    path=relative,
                    url=url,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    bytes=path.stat().st_size,
                    top_level_imports=imports,
                    retrieved_from=origin,
                )
            )
        manifest[name] = dict(
            repository=config["repository"], commit=commit, files=records
        )
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def timed(call, repetitions=5):
    elapsed = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = call()
        elapsed.append(time.perf_counter() - start)
    return result, elapsed


def run_comparison(output: Path, source_cache: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError(
            "Use a fresh output directory (native sources may be copied as a cache)"
        )
    manifest = ensure_sources(output, source_cache)
    fixtures = load_native(
        "prodock_comparison_integrity_fixtures",
        ROOT / "Data/validation/integrity_experiments.py",
    )
    rows, _ = fixtures.make_fixtures()
    # Different channel distributions avoid a trivial all-channels-identical ensemble.
    rng = np.random.default_rng(20261104)
    generated_scores = rng.normal(loc=-7.0, scale=1.2, size=(12, 4))
    proteins, engines = ["R0", "R1"], ["vina", "smina"]
    for row in rows:
        i = int(row["ligand_id"].rsplit("_", 1)[1])
        c = proteins.index(row["receptor_id"]) * 2 + engines.index(row["engine"])
        row["affinity"] = float(generated_scores[i, c] + (row["pose_rank"] - 1) * 0.27)
        row["score_data"] = {"fixture_native_score": row["affinity"]}
    database = output / "shared_fixture.db"
    if database.exists():
        raise FileExistsError(
            "Use a fresh output directory; existing fixture database will not be reused"
        )
    with PoseDatabase(database) as db:
        db.create_run(
            name="shared comparator fixture", config={"fixture": True, "seed": 20261104}
        )
        db.insert_dataframe(pd.DataFrame(rows))
    sdf = output / "shared_fixture.sdf"
    writer = Chem.SDWriter(str(sdf))
    for row in rows:
        mol = Chem.Mol(row["mol"])
        mol.SetProp("_Name", row["pose_id"])
        for key in ["receptor_id", "ligand_id", "engine", "pose_rank"]:
            mol.SetProp(key, str(row[key]))
        mol.SetProp("native_score", repr(row["affinity"]))
        writer.write(mol, confId=0)
    writer.close()
    native_loader = load_native(
        "dockm8_pinned_fast_sdf_loader",
        output / "sources/DockM8/scripts/utilities/fast_sdf_loader.py",
    )
    dockm8_z = load_native(
        "dockm8_pinned_zscore",
        output / "sources/DockM8/scripts/consensus/methods/zscore.py",
    )
    moldock_z = load_native(
        "moldocklab_pinned_consensus",
        output / "sources/MolDockLab/src/consensus_rank.py",
    )
    sys.path.insert(0, str(ROOT / "Data/benchmark"))
    prodock_analysis = load_native(
        "prodock_native_ensemble_analysis", ROOT / "Data/benchmark/analyze_ensemble.py"
    )

    def prodock_scores():
        with PoseQuery(database) as query:
            return query.scores(as_dataframe=True)

    def dockm8_scores():
        return native_loader.fast_load_sdf(
            sdf,
            molColName=None,
            idName="ID",
            n_cpus=1,
            required_props={
                "native_score",
                "receptor_id",
                "ligand_id",
                "engine",
                "pose_rank",
            },
        )

    sql_scores, sql_times = timed(prodock_scores)
    sdf_scores, sdf_times = timed(dockm8_scores)
    sql_values = sql_scores.set_index("pose_id")["affinity"].astype(float).sort_index()
    sdf_values = sdf_scores.set_index("ID")["native_score"].astype(float).sort_index()
    assert list(sql_values.index) == list(sdf_values.index)
    np.testing.assert_allclose(
        sql_values.to_numpy(), sdf_values.to_numpy(), atol=1e-12, rtol=0
    )
    expected = {r["pose_id"]: r for r in rows}
    for record in sdf_scores.to_dict("records"):
        original = expected[record["ID"]]
        assert record["receptor_id"] == original["receptor_id"]
        assert record["ligand_id"] == original["ligand_id"]
        assert record["engine"] == original["engine"]
        assert int(record["pose_rank"]) == original["pose_rank"]
    full_sdf, full_sdf_times = timed(
        lambda: native_loader.fast_load_sdf(
            sdf,
            molColName="Molecule",
            idName="ID",
            n_cpus=1,
            required_props={
                "native_score",
                "receptor_id",
                "ligand_id",
                "engine",
                "pose_rank",
            },
        ),
        repetitions=3,
    )
    coordinate_errors = []
    for record in full_sdf.to_dict("records"):
        reference = Chem.RemoveHs(expected[record["ID"]]["mol"])
        recovered = record["Molecule"]
        assert recovered is not None
        assert Chem.MolToSmiles(reference, isomericSmiles=True) == Chem.MolToSmiles(
            recovered, isomericSmiles=True
        )
        coordinate_errors.append(
            float(
                np.max(
                    np.abs(
                        reference.GetConformer(0).GetPositions()
                        - recovered.GetConformer(0).GetPositions()
                    )
                )
            )
        )
    assert max(coordinate_errors) <= 0.000051

    rank1 = sql_scores[sql_scores["pose_rank"] == 1].copy()
    rank1["channel"] = rank1["receptor_id"] + "/" + rank1["engine"]
    matrix = rank1.pivot(
        index="ligand_id", columns="channel", values="affinity"
    ).sort_index()
    assert matrix.shape == (12, 4)
    (output / "shared_rank1_native_scores.csv").write_text(matrix.to_csv())
    # Native consensus modules document higher-is-better inputs; docking energies
    # are negated explicitly for all methods. No activity labels are supplied.
    higher = (-matrix).reset_index().rename(columns={"ligand_id": "ID"})
    columns = list(matrix.columns)
    prodock_consensus, prodock_consensus_times = timed(
        lambda: -prodock_analysis._zscore(matrix).mean(axis=1)
    )
    dockm8_consensus, dockm8_consensus_times = timed(
        lambda: dockm8_z.zscore_consensus(higher.copy(), columns)
    )
    moldock_consensus, moldock_consensus_times = timed(
        lambda: moldock_z.Zscore(higher.copy(), 0.05, columns, "ID")
    )
    dockm8_values = dockm8_consensus.set_index("ID")["Zscore"].reindex(matrix.index)
    moldock_values = moldock_consensus.set_index("ID")["Zscore"].reindex(matrix.index)
    np.testing.assert_allclose(
        prodock_consensus.to_numpy(), dockm8_values.to_numpy(), atol=1e-12, rtol=0
    )
    # MolDockLab uses pandas sample std (ddof=1); the other two use ddof=0.
    expected_scale = float(np.sqrt((len(matrix) - 1) / len(matrix)))
    np.testing.assert_allclose(
        moldock_values.to_numpy(),
        prodock_consensus.to_numpy() * expected_scale,
        atol=1e-12,
        rtol=0,
    )
    reference_ranks = prodock_consensus.rank(ascending=False, method="average")
    assert np.array_equal(
        reference_ranks, dockm8_values.rank(ascending=False, method="average")
    )
    assert np.array_equal(
        reference_ranks, moldock_values.rank(ascending=False, method="average")
    )
    result_table = pd.DataFrame(
        {
            "ProDock_analysis_z": prodock_consensus,
            "DockM8_native_z": dockm8_values,
            "MolDockLab_native_z": moldock_values,
            "common_rank": reference_ranks,
        }
    )
    result_table.to_csv(output / "consensus_outputs.csv")
    sql_scores.drop(
        columns=[c for c in ["score_data", "score_metadata"] if c in sql_scores]
    ).to_csv(output / "prodock_native_score_query.csv", index=False)
    sdf_scores.to_csv(output / "dockm8_native_sdf_score_load.csv", index=False)
    checks = [
        dict(
            task="recover_all_shared_pose_ids_and_native_scores",
            tools=["ProDock PoseQuery.scores", "DockM8 fast_load_sdf"],
            status="passed",
            records=len(rows),
            tolerance=1e-12,
        ),
        dict(
            task="recover_heavy_atom_graphs_and_exported_coordinates",
            tools=["DockM8 fast_load_sdf"],
            status="passed",
            molecules=len(full_sdf),
            maximum_coordinate_error_angstrom=max(coordinate_errors),
            note="SDF stores first conformer at four decimal places; loader returns heavy atoms by default.",
        ),
        dict(
            task="reaggregate_shared_rank1_scores_without_docking",
            tools=[
                "ProDock analysis _zscore",
                "DockM8 zscore_consensus",
                "MolDockLab Zscore",
            ],
            status="passed",
            compounds=12,
            channels=4,
            same_ordering=True,
            exact_z_scores_agree=False,
            note="DockM8 and ProDock ddof=0 values agree; MolDockLab ddof=1 differs by known common scale.",
            moldocklab_to_population_z_scale=expected_scale,
        ),
    ]
    report = dict(
        scope="Executed native helper-module score-loading and reanalysis comparison; not complete-platform execution.",
        dataset="Constructed 12-chemical-graph fixture, 96 pose records, artificial native scores, no activity labels.",
        full_workflow_comparison_complete=False,
        accuracy_benchmark=False,
        new_docking_runs=0,
        source_manifest=manifest,
        source_cache=str(source_cache) if source_cache else None,
        comparator_license_policy=(
            "Original GPL license files are preserved; "
            "comparator source is not relicensed by this runner."
        ),
        checks=checks,
        timings_seconds=dict(
            ProDock_score_query=sql_times,
            DockM8_score_SDF_load=sdf_times,
            DockM8_molecule_SDF_load=full_sdf_times,
            ProDock_analysis_z=prodock_consensus_times,
            DockM8_native_z=dockm8_consensus_times,
            MolDockLab_native_z=moldock_consensus_times,
        ),
        timing_caveats=[
            "Small fixture; repeated warm filesystem observations, no cold-cache guarantee.",
            "Score SQL lookup and full SDF parsing are different storage operations; no platform-speed ranking.",
            "DockM8 loader explicitly uses one worker; process startup is included.",
            "Molecule SDF load has three repetitions; score/aggregation operations have five.",
        ],
        storage_bytes=dict(
            ProDock_database=database.stat().st_size, shared_SDF=sdf.stat().st_size
        ),
        limitations=[
            "MolDockLab SDF/preprocessing modules import Bio and hdbscan, "
            "unavailable in this environment; no custom substitute was used.",
            "MolDockLab comparison therefore covers its native consensus "
            "function, not its full SDF/preprocessing path.",
            "No docking/preparation/rescoring executables, restart policies, "
            "complete-platform interactions or prospective hit finding compared.",
            "Registry/component counts and a full matched workflow study remain separate tasks.",
        ],
        environment=dict(
            interpreter=sys.executable,
            python=platform.python_version(),
            rdkit=rdBase.rdkitVersion,
            pandas=pd.__version__,
            numpy=np.__version__,
            sqlite=sqlite3.sqlite_version,
        ),
        own_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prodock_analysis_sha256=hashlib.sha256(
            (ROOT / "Data/benchmark/analyze_ensemble.py").read_bytes()
        ).hexdigest(),
        input_sha256=dict(
            SDF=hashlib.sha256(sdf.read_bytes()).hexdigest(),
            database=hashlib.sha256(database.read_bytes()).hexdigest(),
        ),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="Offline source cache containing sources/DockM8 and sources/MolDockLab; pinned hashes verified",
    )
    args = parser.parse_args()
    cache = args.source_cache
    if cache is None:
        default_cache = ROOT / "Data/validation/inputs/comparison_native"
        cache = default_cache if default_cache.is_dir() else None
    report = run_comparison(args.output, cache)
    print(
        json.dumps(
            dict(
                scope=report["scope"],
                checks=report["checks"],
                full_workflow_comparison_complete=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
