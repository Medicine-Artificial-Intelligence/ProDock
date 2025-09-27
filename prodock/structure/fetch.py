# prodock/structure/fetch.py
from __future__ import annotations
from pathlib import Path
from typing import List

from prodock.io.logging import get_logger
from .constants import ALLOWED_EXT_ORDER
from .utils import decompress_gz, score_path_for_pdb_search  # small utils below

logger = get_logger(__name__)


def _get_pymol_cmd():
    try:
        from pymol import cmd  # type: ignore

        return cmd
    except Exception:
        return None


def fetch_pdb_to_dir(pdb_id: str, fetch_dir: Path) -> Path:
    cmd = _get_pymol_cmd()
    if cmd is None:
        raise RuntimeError("PyMOL 'cmd' is not importable. Install PyMOL for fetching.")
    fetch_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching %s into %s", pdb_id, fetch_dir)
    cmd.fetch(pdb_id, path=str(fetch_dir), type="pdb", async_=0)

    pdb_lower = pdb_id.lower()
    candidates: List[Path] = []
    for p in fetch_dir.iterdir():
        try:
            if pdb_lower in p.name.lower():
                candidates.append(p)
        except Exception:
            continue

    if not candidates:
        for p in fetch_dir.iterdir():
            nl = p.name.lower()
            if (
                nl.startswith(f"pdb{pdb_lower}")
                or nl.endswith(f"{pdb_lower}.pdb")
                or nl.endswith(f"{pdb_lower}.ent")
            ):
                candidates.append(p)

    if not candidates:
        dir_listing = ", ".join(sorted([p.name for p in fetch_dir.iterdir()]))
        raise FileNotFoundError(
            f"No fetched PDB found for {pdb_id} in {fetch_dir}. Contents: [{dir_listing}]"
        )

    chosen = sorted(
        candidates,
        key=lambda p: score_path_for_pdb_search(p, pdb_lower, ALLOWED_EXT_ORDER),
    )[0]

    if chosen.name.lower().endswith(".gz"):
        try:
            chosen = decompress_gz(chosen)
        except Exception as exc:
            logger.warning("Failed to decompress %s: %s", chosen, exc)
    logger.debug("Selected fetched file: %s", chosen)
    return chosen
