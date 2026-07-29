# Paper benchmark scripts

Reproduces the quantitative results in the *Validation and benchmarking*
section of `paper/main.tex`. All numbers are computed from ProDock's own
`prodock.postprocess.metrics` API over the SQLite databases the campaigns
produce, so the analysis step is read-only and cheap to re-run.

## Files

| File | Role |
|------|------|
| `egfr_config.py` | Shared config: receptor panel, engines, actives, paths, logging. Edit here. |
| `run_benchmark.py` | Checkpointed redocking and screening for 25 ligands × 3 receptors × 4 engines. |
| `run_benchmark_3cpu.sh` | Pins the benchmark runner and numerical libraries to three CPUs. |
| `run_smina_box_sweep.py` | Checkpointed smina-only comparison of panel-wide box strategies before the four-engine rerun. |
| `run_smina_box_sweep_one_cpu.sh` | Pins the box sweep and numerical libraries to one CPU. |
| `analyze_benchmark.py` | Reads both DBs → writes machine-readable table fragments and `paper/Figure/fp_similarity.png`. |
| `analyze_ensemble.py` | Compares label-free affinity, size-corrected, cocrystal-contact, and all-pose reranking ensembles → `runs/ensemble/`. |
| `run_all.sh` | One-shot wrapper: chains the three scripts + `latexmk`. |

All run artifacts (databases, references, logs, and derived tables) land under
`Data/benchmark/runs/` or the local `paper/` worktree. They are generated
artifacts and are intentionally excluded from Git. The committed tree contains
the configuration, ligand inputs, and drivers required to regenerate them.

## Quick start (one command)

```bash
cd <repo root>            # the directory containing prodock/ and paper/
pip install -e .          # ProDock + deps; docking engines must be on PATH

Data/benchmark/run_all.sh
```

Options:

```bash
Data/benchmark/run_all.sh --skip-dock   # analysis only; both SQLite DBs must exist
Data/benchmark/run_all.sh --no-paper    # run everything except latexmk
Data/benchmark/run_all.sh --help
```

## Step by step (equivalent)

```bash
python Data/benchmark/run_benchmark.py     # checkpointed redocking + screening
Data/benchmark/run_smina_box_sweep_one_cpu.sh  # box strategy selection
python Data/benchmark/analyze_benchmark.py # fills the paper tables + figure
python Data/benchmark/analyze_ensemble.py  # post-processes existing screen
cd paper && latexmk -pdf main.tex          # tables now show real numbers
```

## Logging

Every script logs timestamped, step-level progress to the console **and**
appends to `Data/benchmark/runs/benchmark.log`, e.g.

```
14:02:11 | redock    | INFO  | [3/5] 4G5J : extracting native ligand 0WM
14:02:19 | redock    | INFO  | [4G5J] redocking native_4G5J across smina,vina,qvina,qvina-w
```

so a finished run leaves an auditable record of exactly which steps ran.

## Requirements

- Docking engines on `PATH` with these exact names: `smina`, `vina`, `qvina`,
  `qvina-w`. Adjust `ENGINES` in `egfr_config.py` to match your binaries; the
  paper's table columns key off the same list.
- A working ProDock install (`pip install -e .` from the repo). The scripts
  prepend the repo root to `sys.path`, so the **repo** package is used even if
  an older `prodock` wheel is installed elsewhere.
- Network access for the initial PDB fetches.
- `matplotlib` for the fingerprint heatmap (optional; skipped if absent).

## Notes

- `--skip-dock` requires both
  `runs/redocking/redocking.db` and `runs/screening/screening.db`. These
  databases are not source inputs and are not committed.
- The screening campaign is 300 docking jobs at exhaustiveness 32; run it on
  a workstation. Re-runs reuse prepared receptors and replace database rows.
- The **consensus** row averages each ligand's rank across all three receptors
  and four engines, then recomputes the same metrics.
