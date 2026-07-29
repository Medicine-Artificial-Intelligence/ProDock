from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


optimizer_module = _load_module(
    "prodock_optuna_reranking",
    "Project/Optimization_script/optuna_combine_all_structure.py",
)
batch_module = _load_module(
    "prodock_optuna_batch",
    "Project/Optimization_script/run_all_proteins_optimization.py",
)


def _write_flat_inputs(root: Path, target: str = "TEST") -> None:
    gnina_dir = root / "result_gnina"
    diffdock_dir = root / "result_diffdock"
    gnina_dir.mkdir(parents=True)
    diffdock_dir.mkdir(parents=True)

    compounds = ["CHEMBL1", "named-active", "ZINC0001"]
    pd.DataFrame(
        {
            "Compounds": compounds,
            "Affinity_rank1": [-9.0, -8.0, -7.0],
            "CNNpose_rank1": [0.9, 0.8, 0.7],
            "CNNaffinity_rank1": [8.0, 7.0, 6.0],
            "Similarity-type1_rank1": [0.8, 0.7, 0.2],
        }
    ).to_csv(gnina_dir / f"{target}_final.csv", index=False)
    pd.DataFrame(
        {
            "Compounds": compounds,
            "Confidence_score_rank1": [-3.0, -2.0, -1.0],
            "Occupancy_rank1": [90.0, 80.0, 20.0],
            "%atoms_rank1": [100.0, 95.0, 60.0],
        }
    ).to_csv(diffdock_dir / f"{target}_final.csv", index=False)


def test_flat_repository_layout_and_explicit_dude_labels(tmp_path: Path) -> None:
    _write_flat_inputs(tmp_path)

    reranker = optimizer_module.AllStructureReranker(
        protein="TEST",
        scoring_metric="affinity",
        base_dir=str(tmp_path),
        dude_labels=True,
    )

    assert reranker.gnina_csv_path == (tmp_path / "result_gnina" / "TEST_final.csv")
    assert reranker.diffdock_csv_path == (tmp_path / "result_diffdock" / "TEST_final.csv")
    labels = reranker.data.drop_duplicates("molecule").set_index("molecule")["is_active"].to_dict()
    assert labels == {"CHEMBL1": 1, "named-active": 1, "ZINC0001": 0}


def test_flat_repository_layout_is_discovered(tmp_path: Path) -> None:
    _write_flat_inputs(tmp_path)

    assert batch_module.discover_proteins(str(tmp_path)) == ["TEST"]
