#!/usr/bin/env bash
# A dry-run is the default. Scientific execution requires --execute explicitly.
set -euo pipefail
MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
  printf 'Usage: %s [--dry-run|--execute]\n' "$0" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRODOCK_REPO="${PRODOCK_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PRODOCK_PYTHON="${PRODOCK_PYTHON:-python}"
PRODOCK_WORKDIR="${PRODOCK_WORKDIR:-$PRODOCK_REPO/Data/validation/runs/cluster_revision_v1}"
PRODOCK_WORKERS="${PRODOCK_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
if ! [[ "$PRODOCK_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PRODOCK_WORKERS must be a positive integer.\n' >&2; exit 2
fi
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "$PRODOCK_WORKERS" -gt "$SLURM_CPUS_PER_TASK" ]]; then
  PRODOCK_WORKERS="$SLURM_CPUS_PER_TASK"
fi
export OMP_NUM_THREADS=1 OPENMM_CPU_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd -- "$PRODOCK_REPO"
QUALITY_DIR="$PRODOCK_WORKDIR/preparation_quality"
EVIDENCE_DIR="$PRODOCK_WORKDIR/evidence"
CAMPAIGN_DIR="$PRODOCK_WORKDIR/chemical_campaign"
RESUME=()
if [[ "${PRODOCK_RESUME:-0}" == "1" ]]; then RESUME=(--resume); fi
COMMON=(--suite full --output "$EVIDENCE_DIR/full" --campaign-output "$CAMPAIGN_DIR"
        --library "$PRODOCK_REPO/Data/validation/inputs/chembl_egfr/library.json"
        --receptors "$QUALITY_DIR/ready_receptors.json" --quality-report "$QUALITY_DIR/quality_report.json"
        --context "$QUALITY_DIR/chemical_context.json" --workers "$PRODOCK_WORKERS")
if [[ -n "${PRODOCK_PREPARED_ROOT:-}" ]]; then COMMON+=(--prepared-root "$PRODOCK_PREPARED_ROOT"); fi
if [[ "$MODE" == "--dry-run" ]]; then
  printf 'PLAN ONLY: four core packages; no preparation, docking, profiling, or database experiments will run.\n'
  printf 'Work directory (not created by this dry-run): %s\n' "$PRODOCK_WORKDIR"
  "$PRODOCK_PYTHON" Data/validation/check_cluster_readiness.py
  if [[ -f paper/main.tex ]]; then
    "$PRODOCK_PYTHON" Data/validation/audit_minor_text.py
  else
    printf 'Manuscript text audit omitted: paper/main.tex is not in this cluster checkout.\n'
  fi
  "$PRODOCK_PYTHON" Data/validation/preparation_sensitivity.py \
    --input-manifest Data/validation/inputs/egfr_structures/manifest.json \
    --output "$QUALITY_DIR" --jobs "$PRODOCK_WORKERS" --dry-run
  "$PRODOCK_PYTHON" Data/validation/run_revision_experiments.py --plan --suite cheap \
    --output "$EVIDENCE_DIR/cheap" --workers "$PRODOCK_WORKERS" \
    --only-step full_database_integrity --only-step pinned_native_helper_comparison
  "$PRODOCK_PYTHON" Data/validation/run_revision_experiments.py --plan "${COMMON[@]}" "${RESUME[@]}"
  printf 'Expected docking workload: 8 preparation/native controls + 2,000 primary + 64 secondary = 2,072 tasks.\n'
  printf 'The large campaign is correctly BLOCKED until the cluster-generated protein quality gate passes.\n'
  exit 0
fi
mkdir -p -- "$PRODOCK_WORKDIR"
"$PRODOCK_PYTHON" Data/validation/check_cluster_readiness.py --strict-environment \
  > "$PRODOCK_WORKDIR/preflight.json"
"$PRODOCK_PYTHON" -c 'import sys; from pathlib import Path; print(sys.executable); assert Path(sys.executable).is_file()' \
  > "$PRODOCK_WORKDIR/python.txt"
"$PRODOCK_PYTHON" -m pip freeze > "$PRODOCK_WORKDIR/python-packages.txt"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$PRODOCK_WORKDIR/base-commit.txt"
  git diff --binary > "$PRODOCK_WORKDIR/source-working-tree.patch"
else
  printf 'Standalone source archive; see CLUSTER_PACKAGE_MANIFEST.json and per-stage source hashes.\n' \
    > "$PRODOCK_WORKDIR/base-commit.txt"
  : > "$PRODOCK_WORKDIR/source-working-tree.patch"
fi
# Cheap fixtures are separate from actual docking and are not reported as screening data.
"$PRODOCK_PYTHON" Data/validation/run_revision_experiments.py --execute --suite cheap \
  --output "$EVIDENCE_DIR/cheap" --workers "$PRODOCK_WORKERS" "${RESUME[@]}" \
  --only-step full_database_integrity --only-step pinned_native_helper_comparison
# Resume a completed matching quality stage only after verifying its input identity.
QUALITY_REUSE=0
if [[ "${PRODOCK_RESUME:-0}" == "1" && -f "$QUALITY_DIR/report.json" ]]; then
  "$PRODOCK_PYTHON" Data/validation/preparation_sensitivity.py \
    --input-manifest Data/validation/inputs/egfr_structures/manifest.json \
    --output "$QUALITY_DIR" --jobs "$PRODOCK_WORKERS" --dry-run \
    > "$PRODOCK_WORKDIR/current-quality-plan.json"
  if "$PRODOCK_PYTHON" - "$QUALITY_DIR/report.json" "$PRODOCK_WORKDIR/current-quality-plan.json" <<'PY'
import json,sys,hashlib
from pathlib import Path
old=json.load(open(sys.argv[1]));new=json.load(open(sys.argv[2]))
assert old['prespecified_design']['input_manifest_sha256']==new['input_manifest_sha256']
assert old['quality_report']['qualified_for_new_campaign'] is True
assert old['prespecified_design']['parameters']==new['parameters']
assert old['prespecified_design']['default_minimization_max_steps']==new['default_minimization_max_steps']
for name,expected in old['quality_report'].get('source_protocol_sha256',{}).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
PY
  then QUALITY_REUSE=1; fi
fi
if [[ "$QUALITY_REUSE" == "0" ]]; then
  if [[ -f "$QUALITY_DIR/prespecified_design.json" ]]; then
    printf 'Existing quality outcomes did not qualify or no longer match. Inspect them and use a fresh work directory for a changed protocol.\n' >&2
    exit 2
  fi
  QUALITY_EXIT=0
  "$PRODOCK_PYTHON" Data/validation/preparation_sensitivity.py \
    --input-manifest Data/validation/inputs/egfr_structures/manifest.json \
    --output "$QUALITY_DIR" --jobs "$PRODOCK_WORKERS" --timeout 180 --execute || QUALITY_EXIT=$?
  if [[ -f "$QUALITY_DIR/summary.csv" ]]; then
    "$PRODOCK_PYTHON" Data/validation/make_quality_figure.py --quality-directory "$QUALITY_DIR"
  fi
  if [[ "$QUALITY_EXIT" != "0" ]]; then exit "$QUALITY_EXIT"; fi
fi
# No automatic bypass: a failed chemical quality gate stops before the 2,064-task workload.
"$PRODOCK_PYTHON" Data/validation/run_revision_experiments.py --execute "${COMMON[@]}" "${RESUME[@]}"
