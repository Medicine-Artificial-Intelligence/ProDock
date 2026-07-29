#!/usr/bin/env bash
#
# One-shot driver for the ProDock paper benchmark.
#
# Chains: redocking campaign -> screening campaign -> analysis (tables+figure)
# -> paper rebuild. Safe to re-run; campaigns reuse prepared receptors and
# replace database rows.
#
# Usage:
#   Data/benchmark/run_all.sh              # full pipeline
#   Data/benchmark/run_all.sh --skip-dock  # only re-run analysis + paper build
#   Data/benchmark/run_all.sh --no-paper   # run everything except latexmk
#
set -euo pipefail

# --- locate paths (works from any cwd) ------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAPER_DIR="${REPO_ROOT}/paper"
PY="${PYTHON:-python3}"

SKIP_DOCK=0
BUILD_PAPER=1
for arg in "$@"; do
  case "$arg" in
    --skip-dock) SKIP_DOCK=1 ;;
    --no-paper)  BUILD_PAPER=0 ;;
    -h|--help)
      tail -n +2 "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

LOG_DIR="${SCRIPT_DIR}/runs"
LOG_FILE="${LOG_DIR}/benchmark.log"
mkdir -p "${LOG_DIR}"

# Timestamped step banner, echoed to console and appended to the shared log.
step() { printf '%s | run_all   | STEP  | %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"; }

step "repo root : ${REPO_ROOT}"
step "python    : $(${PY} --version 2>&1)"
step "log file  : ${LOG_FILE}"

# --- sanity: engines on PATH ----------------------------------------------
if [ "${SKIP_DOCK}" -eq 0 ]; then
  missing=""
  for eng in smina vina qvina qvina-w; do
    command -v "$eng" >/dev/null 2>&1 || missing="${missing} ${eng}"
  done
  if [ -n "${missing}" ]; then
    echo "WARNING: docking engine(s) not found on PATH:${missing}" >&2
    echo "         adjust ENGINES in Data/benchmark/egfr_config.py or fix PATH." >&2
  fi
fi

# --- 1. redocking + screening (docking power) ------------------------------
if [ "${SKIP_DOCK}" -eq 0 ]; then
  step "[1-2/4] redocking + screening campaign"
  "${PY}" "${SCRIPT_DIR}/run_benchmark.py"
else
  step "[1-2/4] skipping docking campaigns (--skip-dock)"
fi

# --- 3. analysis: tables + figure -----------------------------------------
step "[3/4] analysis -> paper tables + figure"
"${PY}" "${SCRIPT_DIR}/analyze_benchmark.py"

# --- 4. rebuild paper ------------------------------------------------------
if [ "${BUILD_PAPER}" -eq 1 ]; then
  if [ ! -f "${PAPER_DIR}/main.tex" ]; then
    step "[4/4] paper/main.tex not present; skipping paper build"
  elif command -v latexmk >/dev/null 2>&1; then
    step "[4/4] rebuilding paper"
    ( cd "${PAPER_DIR}" && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null )
    step "done: ${PAPER_DIR}/main.pdf"
  else
    step "[4/4] latexmk not found; skipping paper build"
  fi
else
  step "[4/4] skipping paper build (--no-paper)"
fi

step "benchmark pipeline complete."
