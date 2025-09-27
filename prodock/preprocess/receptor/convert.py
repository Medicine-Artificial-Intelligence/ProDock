"""
Conversion helpers: mekoo and OpenBabel wrappers.

Functions:
- convert_with_mekoo
- convert_with_obabel
"""

from __future__ import annotations

import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def convert_with_mekoo(
    mekoo_cmd: str,
    input_pdb: Path,
    out_basename: Path,
    write_pdbqt: Optional[Path] = None,
    box_center: Optional[tuple] = None,
    box_size: Optional[tuple] = None,
) -> Dict[str, Any]:
    """
    Call mk_prepare_receptor.py (mekoo) to produce PDBQT (or other artifacts).

    :param mekoo_cmd: Path or command name for mk_prepare_receptor.py.
    :type mekoo_cmd: str
    :param input_pdb: path to input PDB.
    :type input_pdb: pathlib.Path
    :param out_basename: base path for outputs (without ext).
    :type out_basename: pathlib.Path
    :param write_pdbqt: optional path to request PDBQT output.
    :type write_pdbqt: pathlib.Path or None
    :param box_center: optional (x,y,z) grid center to pass through to mekoo.
    :type box_center: tuple or None
    :param box_size: optional (sx,sy,sz) grid size to pass through to mekoo.
    :type box_size: tuple or None
    :returns: info dict with keys 'called','rc','stdout','stderr','produced'
    :rtype: dict
    """
    exe = shutil.which(mekoo_cmd) or (mekoo_cmd if Path(mekoo_cmd).exists() else None)
    info = {"called": None, "rc": None, "stdout": None, "stderr": None, "produced": []}
    if not exe:
        info["stderr"] = f"mekoo ({mekoo_cmd}) not found"
        logger.warning(info["stderr"])
        return info

    args: List[str] = [str(exe), "--read_pdb", str(input_pdb), "-o", str(out_basename)]
    if box_center:
        args += [
            "--box_center",
            str(box_center[0]),
            str(box_center[1]),
            str(box_center[2]),
        ]
    if box_size:
        args += ["--box_size", str(box_size[0]), str(box_size[1]), str(box_size[2])]
    if write_pdbqt:
        args += ["--write_pdbqt", str(write_pdbqt)]

    info["called"] = " ".join(args)
    logger.debug("convert_with_mekoo: %s", info["called"])
    try:
        proc = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        info["rc"] = proc.returncode
        info["stdout"] = proc.stdout
        info["stderr"] = proc.stderr
    except Exception as exc:
        info["rc"] = -1
        info["stderr"] = str(exc)
        logger.exception("convert_with_mekoo: invocation failed")
    produced = []
    if write_pdbqt and Path(write_pdbqt).exists():
        produced.append(str(Path(write_pdbqt).resolve()))
    for ext in (".pdbqt", ".json"):
        cand = out_basename.with_suffix(ext)
        if cand.exists():
            produced.append(str(cand.resolve()))
    info["produced"] = produced
    logger.debug("convert_with_mekoo: produced=%s", produced)
    return info


def convert_with_obabel(
    input_path: Path, output_path: Path, extra_args: Optional[List[str]] = None
) -> None:
    """
    Convert files using OpenBabel. Commonly used to convert PDB -> PDBQT when mekoo is not used.

    :param input_path: input file path.
    :type input_path: pathlib.Path
    :param output_path: desired output path (e.g., .pdbqt)
    :type output_path: pathlib.Path
    :param extra_args: list of extra flags to pass to obabel (e.g., ['--partialcharge', 'gasteiger'])
    :type extra_args: List[str] or None
    :raises RuntimeError: when obabel is missing or conversion fails
    """
    exe = shutil.which("obabel") or shutil.which("babel")
    if not exe:
        raise RuntimeError("OpenBabel (obabel/babel) not found in PATH")

    args = [exe, str(input_path), "-O", str(output_path)]
    if extra_args:
        args += extra_args

    logger.debug("convert_with_obabel: %s", " ".join(args))
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    if proc.returncode != 0:
        logger.error(
            "convert_with_obabel: failed rc=%s stderr=%s", proc.returncode, proc.stderr
        )
        raise RuntimeError(f"OpenBabel conversion failed: {proc.stderr.strip()}")

    if not output_path.exists():
        raise RuntimeError("OpenBabel reported success but output file missing")
    logger.info("convert_with_obabel: success -> %s", output_path)
