# prodock/structure/convert.py
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from prodock.io.logging import get_logger
from .utils import chdir

logger = get_logger(__name__)


def convert_with_obabel(
    src: Path, dst: Path, extra_args: Optional[Sequence[str]] = None
) -> bool:
    """
    Convert src -> dst using Open Babel (obabel). Returns True if a conversion
    product exists after running obabel.

    Behavior:
    - If `obabel` not found in PATH, return False.
    - Run `obabel -i <src_fmt> <src.name> -o <dst_fmt> -O <dst.name> [extra_args]`
      with current working directory set to `src.parent`.
    - After running, check for output in *two* places:
        1. the expected `dst` path (absolute)
        2. `src.parent / dst.name` (some CLI tools write the file into cwd)
      Return True if either exists.
    """
    obabel_bin = shutil.which("obabel")
    if obabel_bin is None:
        logger.debug("Open Babel not found; skipping conversion %s -> %s", src, dst)
        return False

    # ensure destination parent exists (we still create parent dir here)
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_fmt = src.suffix.lstrip(".") or "pdb"
    dst_fmt = dst.suffix.lstrip(".") or "sdf"

    # use only the filename for the -i input (we run from src.parent)
    cmdline = [
        obabel_bin,
        "-i",
        src_fmt,
        str(src.name),
        "-o",
        dst_fmt,
        "-O",
        str(dst.name),
    ]
    if extra_args:
        cmdline += list(extra_args)

    with chdir(src.parent):
        logger.debug("Running obabel: %s", " ".join(cmdline))
        try:
            completed = subprocess.run(cmdline, check=False)
        except Exception as exc:
            logger.warning("obabel call failed for %s -> %s: %s", src, dst, exc)
            completed = None

        if completed is not None and getattr(completed, "returncode", 1) != 0:
            logger.warning(
                "obabel exited with code %s for %s -> %s",
                getattr(completed, "returncode", None),
                src,
                dst,
            )

    # Some CLI tools write to cwd using only the filename.
    fallback_dst = src.parent / dst.name
    exists = dst.exists() or fallback_dst.exists()
    if not exists:
        logger.debug("Conversion produced no output at %s nor %s", dst, fallback_dst)
    else:
        logger.debug(
            "Conversion output found at %s (or %s)",
            dst if dst.exists() else fallback_dst,
            fallback_dst if not dst.exists() else "",
        )
    return exists


def copy_fallback(src: Path, dst: Path) -> bool:
    """
    Fallback copy: copy src -> dst ensuring destination parent exists.
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as exc:
        logger.warning("Copy fallback failed %s -> %s: %s", src, dst, exc)
        return False
