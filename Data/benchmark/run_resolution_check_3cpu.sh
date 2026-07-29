#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Pin the whole process to 3 physical CPUs. Numerical libraries stay
# single-threaded so they don't oversubscribe on top of smina's own
# --cpu 3 search parallelism.
exec env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  taskset -c 0-2 \
  conda run --no-capture-output -n prodock \
  python Data/benchmark/run_resolution_check.py "$@"
