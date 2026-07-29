"""Shared configuration for the ProDock paper benchmark.

Central place for the receptor panel, engine list, active ligand set, and the
directory layout used by the runner and analysis scripts. Editing values here
keeps ``run_screening.py``, ``run_redocking.py``, and ``analyze_benchmark.py``
consistent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk upward until we find the ProDock repo root (contains ``prodock/``)."""
    for parent in [start, *start.parents]:
        if (parent / "prodock").is_dir() and (parent / "paper").is_dir():
            return parent
    # Fallback: two levels up from Data/benchmark/ -> repo root.
    return start.parents[1]


# Repository root (…/ProDock) inferred robustly from this file's location,
# so the scripts work regardless of where under the repo they are placed.
REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
PAPER_DIR = REPO_ROOT / "paper"

# Ensure the *repo* prodock package is imported, not a stale/older wheel that
# may be installed elsewhere (e.g. site-packages). Import egfr_config BEFORE
# importing prodock in the runner scripts for this to take effect.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Where campaigns, databases, and derived tables/figures are written.
# Kept under Data/benchmark/runs so all benchmark artifacts live together.
BENCH_DIR = REPO_ROOT / "Data" / "benchmark" / "runs"
SCREEN_DIR = BENCH_DIR / "screening"
REDOCK_DIR = BENCH_DIR / "redocking"
REF_DIR = REDOCK_DIR / "reference"  # crystal native-ligand SDFs
TABLES_DIR = PAPER_DIR / "tables"  # LaTeX row files the paper \input's
FIGURE_DIR = PAPER_DIR / "Figure"  # fp_similarity.png lands here

SCREEN_DB = SCREEN_DIR / "screening.db"
REDOCK_DB = REDOCK_DIR / "redocking.db"

# Existing curated ligand library (5 EGFR actives + 20 property-matched decoys).
LIGAND_JSON = REPO_ROOT / "Data" / "case" / "ligand.json"

# EGFR kinase-domain crystal structures and their co-crystallized native
# ligand codes. Each native ligand doubles as the autobox reference and (for
# redocking) the ground-truth pose. 4I23/1C9 is excluded from both campaigns
# because the deposited ligand resolves only 22 of the 33 heavy atoms in the
# ideal definition, so it cannot provide an unambiguous whole-ligand RMSD
# reference.
#
# 2ITY (3.42 A) and 4G5J were dropped from the original four-receptor panel
# and replaced by 4WKQ; see SS_LOG.md for the full investigation:
#   - 2ITY's rank-1 redocking failure tracks its low resolution (3.42 A vs.
#     2.60-2.80 A for the rest of the panel), not the scoring function: 4WKQ
#     is the same ligand (gefitinib/IRE) at 1.85 A and redocks correctly.
#   - 4G5J's reference was built from the wrong ligand copy (`ligand_code
#     "0WM"` selects a free, unreacted afatinib copy 18-25 A from the real,
#     covalently-bound (`0WN`) active-site pose). Two follow-up candidates
#     (1XKK/lapatinib, 3W2S/W2R) were tried as covalent-free replacements and
#     both failed redocking for independent reasons (type-II binding mode;
#     ligand flexibility), so 4G5J was dropped rather than patched.
RECEPTORS = [
    {"pdb_id": "4WKQ", "receptor_name": "EGFR_4WKQ", "ligand_code": "IRE", "chains": ["A"]},
    {"pdb_id": "1M17", "receptor_name": "EGFR_1M17", "ligand_code": "AQ4", "chains": ["A"]},
    {"pdb_id": "4ZAU", "receptor_name": "EGFR_4ZAU", "ligand_code": "YY3", "chains": ["A"]},
]

# Vina-family engines compared in the paper. These are the exact keys
# registered in prodock.dock (note the hyphen in "qvina-w"); the paper's
# column header displays the tidier "qvinaw".
ENGINES = ["smina", "vina", "qvina", "qvina-w"]

# Known EGFR inhibitors; every other ligand_id in the library is a decoy.
ACTIVES = {"gefitinib", "erlotinib", "afatinib", "dacomitinib", "osimertinib"}

# Uniform docking parameters. cpu=1 keeps each individual docking run
# single-threaded and reproducible (comparable to the earlier one-CPU
# panel); n_jobs=6 parallelizes across independent engine/ligand/receptor
# jobs over the 6 CPUs now available, without changing any single job's
# search behavior.
DOCK_PARAMS = dict(
    cpu=1,
    n_jobs=6,
    exhaustiveness=32,
    n_poses=20,
    seed=42,
    box_algorithm="pad",
    box_pad=4.0,
    box_isotropic=True,
    interaction_n_jobs=6,
)


LOG_FILE = BENCH_DIR / "benchmark.log"


def ensure_dirs() -> None:
    for d in (BENCH_DIR, SCREEN_DIR, REDOCK_DIR, REF_DIR, TABLES_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a console+file logger with timestamps for step-level tracing.

    Every script logs to the console and appends to ``Data/benchmark/runs/
    benchmark.log`` so a completed run leaves an auditable step-by-step record.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-9s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:  # logging must never break a run
        pass

    logger.propagate = False
    return logger
