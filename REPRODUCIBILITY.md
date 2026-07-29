# Reproducibility

This file lists the inputs and commands needed to reproduce the included
workflows and distinguishes them from generated campaign artifacts.

## Reproduction inputs and scripts

### Package and tests

- `prodock/`
- `Test/`
- `pyproject.toml`, environment files, and the root documentation

The tests cover the default ligand-derived grid box, batch receptor-name
handling, tied-score metrics, and the committed reranking data contract.

### EGFR workflow example

- `Data/case/config.json`
- `Data/case/receptor.json`
- `Data/case/ligand.json`
- `Data/benchmark/*.py`
- `Data/benchmark/*.sh`
- `Data/benchmark/README.md`

Run the full campaign and analysis with:

```bash
pip install -e .
Data/benchmark/run_all.sh --no-paper
```

The four configured docking executables (`smina`, `vina`, `qvina`, and
`qvina-w`) must be on `PATH`. Initial structure retrieval also requires network
access. The full screening campaign is intentionally not part of the unit-test
suite.

### GNINA + DiffDock reranking

The immutable reranking inputs are already tracked:

- `Project/benchmark/Target_ID.csv`
- `Project/benchmark/result_gnina/*_final.csv`
- `Project/benchmark/result_diffdock/*_final.csv`
- `Project/benchmark/{fetched,filtered,processed}_protein/`
- `Project/benchmark/{cocrystal,reference_ligand}/`

The executable workflow is:

- `Project/Optimization_script/optuna_combine_all_structure.py`
- `Project/Optimization_script/run_all_proteins_optimization.py`
- `Project/Analysis_script/extract_optimized_thresholds.py`
- `Project/Analysis_script/visualize_results.py`

Smoke-test one committed target without running Optuna:

```bash
python Project/Optimization_script/optuna_combine_all_structure.py \
  --protein ABL1 \
  --base-dir Project/benchmark \
  --dude-labels \
  --combine-only
```

Run optimization:

```bash
pip install -e ".[reranking]"
python Project/Optimization_script/optuna_combine_all_structure.py \
  --protein ABL1 \
  --base-dir Project/benchmark \
  --dude-labels \
  --metric roc-auc \
  --n-trials 200
```

The DUD-E label rule is opt-in and explicit: `ZINC*` identifiers are decoys and
all remaining identifiers are actives. The `MT1` files contain `NONE` as the
only non-ZINC identifier, so exclude `MT1` from aggregate optimization until
the source active record is recovered.

## Generated files that must not be tracked

- Python caches and `.pytest_cache/`
- `Data/benchmark/runs/`
- `results_all_structure/`, `Combined/`, and `batch_optimization_results.json`
- Optuna `*.pkl` studies
- `threshold_csv/` and analysis `figures/`
- built Sphinx documentation under `docs/`
- local manuscript and submission bundles under `paper/`

For a fast repository check:

```bash
python -m compileall -q prodock Project Data/benchmark
pytest -q
git diff --check
```
