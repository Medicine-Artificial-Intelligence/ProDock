#!/usr/bin/env bash
#SBATCH --job-name=prodock-revision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=prodock-revision-%j.log
set -euo pipefail
# Submit from the repository root; supply account/partition options to sbatch.
PRODOCK_REPO="${PRODOCK_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$PRODOCK_REPO/pyproject.toml" ]]; then
  printf 'Set PRODOCK_REPO to the ProDock repository root.\n' >&2; exit 2
fi
if [[ -n "${PRODOCK_CONDA_BASE:-}" ]]; then
  CONDA_BASE_DIR="$PRODOCK_CONDA_BASE"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE_DIR="$(conda info --base)"
else
  printf 'Load your cluster conda module or set PRODOCK_CONDA_BASE before submission.\n' >&2; exit 2
fi
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
conda activate "${PRODOCK_CONDA_ENV:-prodock}"
export PRODOCK_REPO
export PRODOCK_PYTHON="$CONDA_PREFIX/bin/python"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PRODOCK_WORKERS="${PRODOCK_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
exec bash "$PRODOCK_REPO/Data/validation/cluster_revision.sh" --execute
