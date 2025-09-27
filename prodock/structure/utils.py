# prodock/structure/utils.py
from __future__ import annotations
import gzip
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


@contextmanager
def chdir(path: Path):
    import os

    prev = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


def decompress_gz(gz_path: Path) -> Path:
    if not gz_path.exists():
        raise FileNotFoundError(gz_path)
    if not gz_path.name.lower().endswith(".gz"):
        return gz_path
    out_path = gz_path.with_suffix("")
    if out_path.exists():
        return out_path
    with gzip.open(gz_path, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return out_path


def score_path_for_pdb_search(
    p: Path, pdb_lower: str, allowed_ext_order: Iterable[str]
) -> int:
    nl = p.name.lower()
    for idx, ext in enumerate(allowed_ext_order):
        if nl.endswith(ext):
            return idx
    if nl.startswith(f"pdb{pdb_lower}"):
        return len(list(allowed_ext_order))
    return len(list(allowed_ext_order)) + 10
