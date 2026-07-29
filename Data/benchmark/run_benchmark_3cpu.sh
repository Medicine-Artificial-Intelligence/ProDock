#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Pin the whole process to 6 physical CPUs. Each individual docking call
# stays single-threaded (cpu=1 in egfr_config.DOCK_PARAMS); n_jobs=6 lets
# BatchDock run 6 independent engine/ligand/receptor jobs concurrently
# across them, so numerical libraries are kept single-threaded to avoid
# oversubscription.
exec env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  taskset -c 0-5 \
  conda run --no-capture-output -n prodock \
  python Data/benchmark/run_benchmark.py "$@"
