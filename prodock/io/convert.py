from __future__ import annotations
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Union, Optional, List, Literal


try:
    from rdkit import Chem  # type: ignore

    _RDKIT_AVAILABLE = True
except Exception:
    Chem = None  # type: ignore
    _RDKIT_AVAILABLE = False

from prodock.io.logging import get_logger
from prodock.structure.pdbqt_sanitizer import (
    PDBQTSanitizer,
)

logger = get_logger(__name__)

# ---------------------------
# Types
# ---------------------------

Backend = Literal["meeko", "obabel", "mgltools"]
TmpConv = Literal["rdkit", "obabel"]


# ---------------------------
# Utilities
# ---------------------------


def _ensure_exists(path: Union[str, Path], kind: str) -> Path:
    """
    Ensure path exists and return resolved Path.

    :param path: file path to check
    :param kind: human readable kind for error messages
    :returns: resolved Path
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{kind} not found: {p}")
    return p


def _require_exe(name: str) -> str:
    """
    Ensure an executable exists in PATH; return its absolute path.

    :param name: executable name
    :returns: absolute path to executable
    :raises RuntimeError: if not found
    """
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")
    return exe


def _run(args: List[str]) -> None:
    """
    Run a CLI command and raise on failure.

    :param args: list of executable + arguments
    :returns: None
    :raises RuntimeError: on non-zero return code
    """
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _rdkit_require() -> None:
    """
    Raise if RDKit is not available.

    :returns: None
    :raises RuntimeError: if RDKit not available
    """
    if not _RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for this operation but is not available.")


# ---------------------------
# Fast pre-check for Meeko sanitization
# ---------------------------

# Tokens typically emitted by Meeko/related toolchains that bother smina/qvina
_SUSPECT_TOKENS = {
    "CG0",
    "G0",
    "G",
    "A",
    "OA",
    "OH",
    "OD",
    "HD",
    "HG",
    "HG1",
    "HA",
    "HB",
    "AA",
    "CL1",
}
# Valid elements (two-letter capitalization honored)
_VALID_ELEMENTS = {
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "Mg",
    "Zn",
    "Fe",
    "K",
    "Na",
    "Ca",
}
_FLOAT_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")


def _looks_like_atom_line(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


def _needs_meeko_sanitize_fast(pdbqt: Path) -> bool:
    """
    Very fast scan to decide whether a Meeko-produced PDBQT likely needs sanitization.

    Heuristics:
      - If fixed-column element (cols 77-78) is present and not in _VALID_ELEMENTS (e.g., 'A', 'G') -> sanitize.
      - If the last whitespace token on an ATOM/HETATM line is in _SUSPECT_TOKENS -> sanitize.
      - If the last token is alpha+digits (e.g., 'CG0', 'CL1') -> sanitize.

    :param pdbqt: path to a PDBQT to scan
    :returns: True if sanitize should run, else False
    """
    text = pdbqt.read_text(encoding="utf-8", errors="replace")
    for ln in text.splitlines():
        if not _looks_like_atom_line(ln):
            continue

        # Check fixed-column element
        if len(ln) >= 78:
            elem = ln[76:78].strip()
            if elem and elem not in _VALID_ELEMENTS:
                return True  # clearly wrong like 'A', 'G', etc.

        toks = ln.split()
        if not toks:
            continue
        last = toks[-1]

        # If last token is numeric (often tempFactor/partial charge), also peek prior token
        if _FLOAT_RE.match(last) and len(toks) >= 2:
            last = toks[-2]

        # explicit suspicious tokens
        if last in _SUSPECT_TOKENS:
            return True

        # alpha+digits or uppercase-only aliases (e.g. CG0, NA used ambiguously)
        if re.fullmatch(r"[A-Za-z]{1,3}\d+", last):
            return True

        # single-letter ambiguous alias
        if last in {"A", "G"}:
            return True

    return False


def _sanitize_meeko_if_needed(pdbqt_path: Path) -> None:
    """
    Run the sanitizer with rebuild if the fast scan says it's needed.

    :param pdbqt_path: path to produced .pdbqt
    :returns: None
    """
    try:
        if _needs_meeko_sanitize_fast(pdbqt_path):
            logger.info("Sanitizing Meeko PDBQT (conditional): %s", pdbqt_path)
            # No backup per user request; rebuild fixed columns; conservative (aggressive=False)
            PDBQTSanitizer.sanitize_file(
                pdbqt_path, out_path=None, rebuild=True, aggressive=False, backup=False
            )
        else:
            logger.info("Meeko PDBQT appears clean; skipping sanitize: %s", pdbqt_path)
    except Exception as e:
        # Don't fail the conversion if sanitize encounters something unexpected
        logger.warning("Sanitizer check/operation failed for %s: %s", pdbqt_path, e)


# ---------------------------
# Temporary conversion helpers
# ---------------------------


def _tmp_sdf_to_pdb_with_rdkit(in_sdf: Path, out_pdb: Path) -> None:
    """
    Convert SDF -> PDB via RDKit for temporary intermediate files.

    :param in_sdf: input SDF path
    :param out_pdb: output PDB path
    :returns: None
    """
    _rdkit_require()
    suppl = Chem.SDMolSupplier(str(in_sdf), removeHs=False, sanitize=True)  # type: ignore
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"No valid molecule found in {in_sdf}")
    Chem.MolToPDBFile(mol, str(out_pdb))  # type: ignore


def _tmp_sdf_to_pdb_with_obabel(in_sdf: Path, out_pdb: Path) -> None:
    """
    Convert SDF -> PDB via Open Babel CLI for temporary intermediate files.

    :param in_sdf: input SDF path
    :param out_pdb: output PDB path
    :returns: None
    """
    obabel = _require_exe("obabel") if shutil.which("obabel") else _require_exe("babel")
    _run([obabel, "-isdf", str(in_sdf), "-opdb", "-O", str(out_pdb)])


def _sdf_to_pdb_intermediate(in_sdf: Path, out_pdb: Path, tmp_backend: TmpConv) -> None:
    """
    Create a PDB intermediate from SDF using RDKit or Open Babel.

    :param in_sdf: input SDF path
    :param out_pdb: desired output PDB path
    :param tmp_backend: 'rdkit' or 'obabel'
    :returns: None
    """
    if tmp_backend == "rdkit":
        _tmp_sdf_to_pdb_with_rdkit(in_sdf, out_pdb)
    elif tmp_backend == "obabel":
        _tmp_sdf_to_pdb_with_obabel(in_sdf, out_pdb)
    else:
        raise ValueError("tmp_from_sdf_backend must be 'rdkit' or 'obabel'.")


# ---------------------------
# Core conversions (explicit backends, no fallback)
# ---------------------------


def pdb_to_pdbqt(
    input_pdb: Union[str, Path],
    output_pdbqt: Union[str, Path],
    *,
    mode: Literal["receptor", "ligand"],
    backend: Backend,
    extra_args: Optional[List[str]] = None,
    meeko_cmd: Optional[str] = None,
    mgltools_cmd: Optional[str] = None,
) -> Path:
    """
    Convert PDB -> PDBQT via an explicit backend only (no fallback).

    If ``backend == 'meeko'``, the produced PDBQT is **conditionally sanitized**
    (rebuild fixed columns) **only** if a fast pre-check detects suspicious
    tokens (CG0/G0/A/G/OA/...).

    :param input_pdb: input PDB file
    :param output_pdbqt: output PDBQT path
    :param mode: 'receptor' or 'ligand'
    :param backend: 'meeko' | 'obabel' | 'mgltools'
    :param extra_args: extra CLI args for the chosen backend
    :param meeko_cmd: override Meeko script name (default: mk_prepare_{receptor,ligand}.py)
    :param mgltools_cmd: override MGLTools script name (prepare_{receptor,ligand}4.py)
    :returns: Path to produced PDBQT
    """
    input_pdb = _ensure_exists(input_pdb, "Input PDB")
    output_pdbqt = Path(output_pdbqt).resolve()
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    extra_args = list(extra_args or [])

    if backend == "meeko":
        if mode == "receptor":
            cmd = meeko_cmd or "mk_prepare_receptor.py"
            exe = _require_exe(cmd)
            args = [
                exe,
                "--read_pdb",
                str(input_pdb),
                "--write_pdbqt",
                str(output_pdbqt),
            ]
            args += extra_args
            _run(args)
        else:
            cmd = meeko_cmd or "mk_prepare_ligand.py"
            exe = _require_exe(cmd)
            args = [exe, "-i", str(input_pdb), "-o", str(output_pdbqt)]
            args += extra_args
            _run(args)
    elif backend == "obabel":
        obabel = (
            _require_exe("obabel") if shutil.which("obabel") else _require_exe("babel")
        )
        args = [
            obabel,
            "-ipdb",
            str(input_pdb),
            "-opdbqt",
            "-O",
            str(output_pdbqt),
            "--partialcharge",
            "gasteiger",
        ] + extra_args
        _run(args)
    elif backend == "mgltools":
        if mode == "receptor":
            cmd = mgltools_cmd or "prepare_receptor4.py"
            exe = _require_exe(cmd)
            args = [exe, "-r", str(input_pdb), "-o", str(output_pdbqt)]
            args += extra_args
            _run(args)
        else:
            cmd = mgltools_cmd or "prepare_ligand4.py"
            exe = _require_exe(cmd)
            args = [exe, "-l", str(input_pdb), "-o", str(output_pdbqt)]
            args += extra_args
            _run(args)
    else:
        raise ValueError("backend must be one of: 'meeko', 'obabel', 'mgltools'.")

    if not output_pdbqt.exists():
        raise FileNotFoundError(f"PDBQT not produced: {output_pdbqt}")

    # Must sanitize if using meeko, but only when suspicious pattern is present
    if backend == "meeko":
        _sanitize_meeko_if_needed(output_pdbqt)

    return output_pdbqt


def pdbqt_to_pdb(
    input_pdbqt: Union[str, Path],
    output_pdb: Union[str, Path],
    *,
    backend: Literal["obabel"],
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Convert PDBQT -> PDB via Open Babel.

    :param input_pdbqt: input PDBQT
    :param output_pdb: output PDB
    :param backend: must be 'obabel'
    :param extra_args: extra args for obabel
    :returns: Path to produced PDB
    """
    if backend != "obabel":
        raise NotImplementedError(
            "PDBQT -> PDB is only supported with backend='obabel'."
        )

    input_pdbqt = _ensure_exists(input_pdbqt, "Input PDBQT")
    output_pdb = Path(output_pdb).resolve()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    obabel = _require_exe("obabel") if shutil.which("obabel") else _require_exe("babel")
    args = [obabel, "-ipdbqt", str(input_pdbqt), "-opdb", "-O", str(output_pdb)] + list(
        extra_args or []
    )
    _run(args)

    if not output_pdb.exists():
        raise FileNotFoundError(f"PDB not produced: {output_pdb}")
    return output_pdb


def sdf_to_pdb(
    input_sdf: Union[str, Path],
    output_pdb: Union[str, Path],
    *,
    backend: Literal["rdkit", "obabel"],
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Convert SDF -> PDB via a single explicit backend.

    :param input_sdf: input SDF path
    :param output_pdb: desired PDB path
    :param backend: 'rdkit' or 'obabel'
    :param extra_args: extra CLI args when using Open Babel
    :returns: Path to produced PDB
    """
    input_sdf = _ensure_exists(input_sdf, "SDF")
    output_pdb = Path(output_pdb).resolve()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    if backend == "rdkit":
        _rdkit_require()
        suppl = Chem.SDMolSupplier(str(input_sdf), removeHs=False, sanitize=True)  # type: ignore
        mol = next((m for m in suppl if m is not None), None)
        if mol is None:
            raise ValueError(f"No valid molecule found in {input_sdf}")
        Chem.MolToPDBFile(mol, str(output_pdb))  # type: ignore
    elif backend == "obabel":
        obabel = (
            _require_exe("obabel") if shutil.which("obabel") else _require_exe("babel")
        )
        args = [obabel, "-isdf", str(input_sdf), "-opdb", "-O", str(output_pdb)] + list(
            extra_args or []
        )
        _run(args)
    else:
        raise ValueError("backend must be 'rdkit' or 'obabel'.")

    if not output_pdb.exists():
        raise FileNotFoundError(f"PDB not produced: {output_pdb}")
    return output_pdb


def sdf_to_pdbqt(
    input_sdf: Union[str, Path],
    output_pdbqt: Union[str, Path],
    *,
    backend: Backend,
    tmp_from_sdf_backend: TmpConv = "rdkit",
    extra_args: Optional[List[str]] = None,
    meeko_cmd: Optional[str] = None,
    mgltools_cmd: Optional[str] = None,
) -> Path:
    """
    Convert SDF -> PDBQT using an explicit backend. No fallback.

    If ``backend == 'meeko'``, the produced PDBQT is **conditionally sanitized**
    (rebuild fixed columns) **only** if a fast pre-check detects suspicious
    tokens (CG0/G0/A/G/OA/...).

    :param input_sdf: input SDF path
    :param output_pdbqt: output PDBQT path
    :param backend: 'meeko' | 'obabel' | 'mgltools'
    :param tmp_from_sdf_backend: 'rdkit' or 'obabel' to produce tmp intermediates
    :param extra_args: extra CLI args for the chosen backend
    :param meeko_cmd: override Meeko ligand script name
    :param mgltools_cmd: override MGLTools ligand script name
    :returns: Path to produced PDBQT
    """
    input_sdf = Path(input_sdf).resolve()
    output_pdbqt = Path(output_pdbqt).resolve()
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    extra_args = list(extra_args or [])

    if not input_sdf.exists():
        raise FileNotFoundError(f"SDF not found: {input_sdf}")

    if backend == "obabel":
        obabel = (
            _require_exe("obabel") if shutil.which("obabel") else _require_exe("babel")
        )
        args = [
            obabel,
            "-isdf",
            str(input_sdf),
            "-opdbqt",
            "-O",
            str(output_pdbqt),
            "--partialcharge",
            "gasteiger",
        ] + extra_args
        _run(args)

    elif backend == "meeko":
        meeko_lig = _require_exe(meeko_cmd or "mk_prepare_ligand.py")
        if input_sdf.suffix.lower() == ".sdf":
            args = [
                meeko_lig,
                "-i",
                str(input_sdf),
                "-o",
                str(output_pdbqt),
            ] + extra_args
            _run(args)
        else:
            with tempfile.TemporaryDirectory() as td:
                tmp_sdf = Path(td) / (input_sdf.stem + ".sdf")
                if tmp_from_sdf_backend == "rdkit":
                    _rdkit_require()
                    if input_sdf.suffix.lower() in {".sdf"}:
                        shutil.copyfile(str(input_sdf), str(tmp_sdf))
                    else:
                        if input_sdf.suffix.lower() in {".pdb", ".pdbqt"}:
                            mol = Chem.MolFromPDBFile(str(input_sdf), removeHs=False)  # type: ignore
                        else:
                            if input_sdf.suffix.lower() == ".smi":
                                with open(input_sdf, "r") as fh:
                                    smi = fh.readline().strip().split()[0]
                                mol = Chem.MolFromSmiles(smi)  # type: ignore
                            else:
                                suppl = Chem.SDMolSupplier(
                                    str(input_sdf), removeHs=False, sanitize=True
                                )  # type: ignore
                                mol = next((m for m in suppl if m is not None), None)
                        if mol is None:
                            raise ValueError(
                                f"RDKit could not parse {input_sdf} to create intermediate SDF."
                            )
                        w = Chem.SDWriter(str(tmp_sdf))  # type: ignore
                        w.write(mol)  # type: ignore
                        w.close()  # type: ignore
                else:
                    ob = (
                        _require_exe("obabel")
                        if shutil.which("obabel")
                        else _require_exe("babel")
                    )
                    _run(
                        [
                            ob,
                            f"-i{input_sdf.suffix.lstrip('.')}",
                            str(input_sdf),
                            "-osdf",
                            "-O",
                            str(tmp_sdf),
                        ]
                    )
                args = [
                    meeko_lig,
                    "-i",
                    str(tmp_sdf),
                    "-o",
                    str(output_pdbqt),
                ] + extra_args
                _run(args)

    elif backend == "mgltools":
        mgl_lig = _require_exe(mgltools_cmd or "prepare_ligand4.py")
        if input_sdf.suffix.lower() == ".sdf":
            args = [mgl_lig, "-l", str(input_sdf), "-o", str(output_pdbqt)] + extra_args
            _run(args)
        else:
            with tempfile.TemporaryDirectory() as td:
                tmp_sdf = Path(td) / (input_sdf.stem + ".sdf")
                if tmp_from_sdf_backend == "rdkit":
                    _rdkit_require()
                    suppl = Chem.SDMolSupplier(str(input_sdf), removeHs=False, sanitize=True)  # type: ignore
                    mol = next((m for m in suppl if m is not None), None)
                    if mol is None:
                        raise ValueError(f"No valid molecule found in {input_sdf}")
                    Chem.SDWriter(str(tmp_sdf)).write(mol)  # type: ignore
                else:
                    ob = (
                        _require_exe("obabel")
                        if shutil.which("obabel")
                        else _require_exe("babel")
                    )
                    _run(
                        [
                            ob,
                            f"-i{input_sdf.suffix.lstrip('.')}",
                            str(input_sdf),
                            "-osdf",
                            "-O",
                            str(tmp_sdf),
                        ]
                    )
                args = [
                    mgl_lig,
                    "-l",
                    str(tmp_sdf),
                    "-o",
                    str(output_pdbqt),
                ] + extra_args
                _run(args)
    else:
        raise ValueError("backend must be one of: 'meeko', 'obabel', 'mgltools'.")

    if not output_pdbqt.exists():
        raise FileNotFoundError(f"PDBQT not produced: {output_pdbqt}")

    # Must sanitize if using meeko, but only when suspicious pattern is present
    if backend == "meeko":
        _sanitize_meeko_if_needed(output_pdbqt)

    return output_pdbqt


def ensure_pdbqt(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    backend: Backend,
    mode: Literal["receptor", "ligand"] = "ligand",
    tmp_from_sdf_backend: TmpConv = "rdkit",
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Ensure the given input becomes a PDBQT using the specified backend ONLY (no fallback).
    If input is already .pdbqt, returns it as-is.

    Routes:
    - .pdb -> PDBQT via :py:meth:`pdb_to_pdbqt`
    - .sdf -> PDBQT via :py:meth:`sdf_to_pdbqt`
    - .mol2/.smi: supported only with backend='obabel' (direct to PDBQT). Others: raise.

    For Meeko backend, a **conditional** sanitizer pass runs automatically on the
    produced PDBQT *only when* suspicious tokens are detected by a fast pre-check.

    :param input_path: path to input (pdb/pdbqt/sdf/mol2/smi)
    :param output_dir: directory where PDBQT should be written (if produced)
    :param backend: 'meeko' | 'obabel' | 'mgltools'
    :param mode: 'receptor' or 'ligand' (PDB->PDBQT conversions)
    :param tmp_from_sdf_backend: backend for temporary intermediates ('rdkit'|'obabel')
    :param extra_args: extra args forwarded to underlying backend command
    :returns: Path to PDBQT file (existing or newly created)
    """
    p = Path(input_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = p.suffix.lower()
    out_p = out_dir / (p.stem + ".pdbqt")

    if ext == ".pdbqt":
        # Input already PDBQT; no Meeko conversion here, so we do NOT sanitize by default.
        return p

    if ext == ".pdb":
        return pdb_to_pdbqt(
            p,
            out_p,
            mode=mode,
            backend=backend,
            extra_args=extra_args,
            meeko_cmd=None,
            mgltools_cmd=None,
        )

    if ext == ".sdf":
        return sdf_to_pdbqt(
            p,
            out_p,
            backend=backend,
            tmp_from_sdf_backend=tmp_from_sdf_backend,
            extra_args=extra_args,
            meeko_cmd=None,
            mgltools_cmd=None,
        )

    if ext in {".mol2", ".smi"}:
        if backend != "obabel":
            raise NotImplementedError(
                f"Input extension {ext} to PDBQT is only supported with backend='obabel'."
            )
        obabel = (
            _require_exe("obabel") if shutil.which("obabel") else _require_exe("babel")
        )
        args = [obabel, f"-i{ext.lstrip('.')}", str(p), "-opdbqt", "-O", str(out_p)]
        args += list(extra_args or [])
        _run(args)
        if not out_p.exists():
            raise FileNotFoundError(f"PDBQT not produced: {out_p}")
        # Not Meeko; do not sanitize by default.
        return out_p

    raise NotImplementedError(
        f"Unsupported input extension {ext} for ensure_pdbqt with backend='{backend}'."
    )


def pdb_to_sdf(
    input_pdb: Union[str, Path],
    output_sdf: Union[str, Path],
    *,
    backend: Literal["rdkit", "obabel"],
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Convert PDB -> SDF via a single explicit backend.

    :param input_pdb: input PDB path
    :param output_sdf: output SDF path
    :param backend: 'rdkit' or 'obabel'
    :param extra_args: extra args for obabel
    :returns: Path to produced SDF
    """
    input_pdb = _ensure_exists(input_pdb, "Input PDB")
    output_sdf = Path(output_sdf).resolve()
    output_sdf.parent.mkdir(parents=True, exist_ok=True)

    if backend == "rdkit":
        _rdkit_require()
        mol = Chem.MolFromPDBFile(str(input_pdb), removeHs=False)  # type: ignore
        if mol is None:
            raise ValueError(f"RDKit could not parse PDB: {input_pdb}")
        writer = Chem.SDWriter(str(output_sdf))  # type: ignore
        writer.write(mol)  # type: ignore
        writer.close()  # type: ignore
    elif backend == "obabel":
        obabel = (
            _require_exe("obabel") if shutil.which("obabel") else _require_exe("babel")
        )
        args = [obabel, "-ipdb", str(input_pdb), "-osdf", "-O", str(output_sdf)] + list(
            extra_args or []
        )
        _run(args)
    else:
        raise ValueError("backend must be 'rdkit' or 'obabel'.")

    if not output_sdf.exists():
        raise FileNotFoundError(f"SDF not produced: {output_sdf}")
    return output_sdf


def pdbqt_to_sdf(
    input_pdbqt: Union[str, Path],
    output_sdf: Union[str, Path],
    *,
    backend: Literal["obabel"],
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Convert PDBQT -> SDF via Open Babel.

    :param input_pdbqt: input PDBQT
    :param output_sdf: output SDF
    :param backend: must be 'obabel'
    :param extra_args: extra args for obabel
    :returns: Path to produced SDF
    """
    if backend != "obabel":
        raise NotImplementedError(
            "PDBQT -> SDF is only supported with backend='obabel'."
        )

    input_pdbqt = _ensure_exists(input_pdbqt, "Input PDBQT")
    output_sdf = Path(output_sdf).resolve()
    output_sdf.parent.mkdir(parents=True, exist_ok=True)

    obabel = _require_exe("obabel") if shutil.which("obabel") else _require_exe("babel")
    args = [obabel, "-ipdbqt", str(input_pdbqt), "-osdf", "-O", str(output_sdf)] + list(
        extra_args or []
    )
    _run(args)

    if not output_sdf.exists():
        raise FileNotFoundError(f"SDF not produced: {output_sdf}")
    return output_sdf


# ---------------------------
# Chainable OOP Converter (explicit, no fallback)
# ---------------------------


class Converter:
    """
    Chainable converter helper for ProDock (explicit backend, no fallback).

    Example:
      out = (
          Converter()
          .set_input("lig.sdf")
          .set_output("lig.pdbqt")
          .set_mode("ligand")
          .set_backend("meeko")
          .set_tmp_from_sdf_backend("rdkit")
          .set_extra_args(["--some-flag"])
          .run()
          .output
      )

    Methods are chainable and ``run()`` returns ``self``.
    """

    def __init__(self) -> None:
        """
        Create a Converter.

        :returns: None
        """
        self._input: Optional[Path] = None
        self._output: Optional[Path] = None
        self._mode: Literal["ligand", "receptor"] = "ligand"
        self._backend: Optional[Backend] = None
        self._tmp_from_sdf_backend: TmpConv = "rdkit"
        self._extra_args: Optional[List[str]] = None
        self._meeko_cmd: Optional[str] = None
        self._mgltools_cmd: Optional[str] = None

    def set_input(self, input_path: Union[str, Path]) -> "Converter":
        """
        Set input file path.

        :param input_path: path to input file
        :returns: self
        """
        self._input = Path(input_path)
        return self

    def set_output(self, output_path: Union[str, Path]) -> "Converter":
        """
        Set output file path.

        :param output_path: desired output path
        :returns: self
        """
        self._output = Path(output_path)
        return self

    def set_mode(self, mode: Literal["ligand", "receptor"]) -> "Converter":
        """
        Set conversion mode.

        :param mode: 'ligand' or 'receptor'
        :returns: self
        """
        self._mode = mode
        return self

    def set_backend(self, backend: Backend) -> "Converter":
        """
        Set explicit backend.

        :param backend: 'meeko' | 'obabel' | 'mgltools'
        :returns: self
        """
        self._backend = backend
        return self

    def set_tmp_from_sdf_backend(self, tmp_backend: TmpConv) -> "Converter":
        """
        Set temporary conversion backend for SDF intermediates.

        :param tmp_backend: 'rdkit' | 'obabel'
        :returns: self
        """
        self._tmp_from_sdf_backend = tmp_backend
        return self

    def set_extra_args(self, args: Optional[List[str]]) -> "Converter":
        """
        Set extra CLI args for chosen backend.

        :param args: list of args or None
        :returns: self
        """
        self._extra_args = None if args is None else list(args)
        return self

    def set_meeko_cmd(self, cmd: Optional[str]) -> "Converter":
        """
        Override meeko script name.

        :param cmd: custom command (e.g. full path) or None
        :returns: self
        """
        self._meeko_cmd = cmd
        return self

    def set_mgltools_cmd(self, cmd: Optional[str]) -> "Converter":
        """
        Override mgltools script name.

        :param cmd: custom command (e.g. full path) or None
        :returns: self
        """
        self._mgltools_cmd = cmd
        return self

    def run(self) -> "Converter":
        """
        Execute conversion according to configured inputs/options.

        - If backend is ``'meeko'``, the output PDBQT is **conditionally** sanitized
          (rebuild fixed PDB columns) *only* when a fast pre-check flags suspicious tokens.

        :returns: self
        :raises RuntimeError: if required options are missing
        """
        if self._input is None:
            raise RuntimeError("No input set (call .set_input(...))")
        if self._backend is None:
            raise RuntimeError(
                "No backend set (call .set_backend('meeko'|'obabel'|'mgltools'))"
            )
        if self._output is None:
            self._output = Path.cwd() / (self._input.stem + ".pdbqt")

        inp = self._input.resolve()
        out = self._output.resolve()
        ext = inp.suffix.lower()

        if ext == ".pdb":
            self._output = pdb_to_pdbqt(
                inp,
                out,
                mode=self._mode,
                backend=self._backend,
                extra_args=self._extra_args,
                meeko_cmd=self._meeko_cmd,
                mgltools_cmd=self._mgltools_cmd,
            )
            return self

        if ext == ".sdf":
            self._output = sdf_to_pdbqt(
                inp,
                out,
                backend=self._backend,
                tmp_from_sdf_backend=self._tmp_from_sdf_backend,
                extra_args=self._extra_args,
                meeko_cmd=self._meeko_cmd,
                mgltools_cmd=self._mgltools_cmd,
            )
            return self

        if ext == ".pdbqt":
            # Nothing to do; note: not Meeko conversion here, so no sanitize by default.
            self._output = inp
            return self

        if ext in {".mol2", ".smi"}:
            if self._backend != "obabel":
                raise NotImplementedError(
                    f"Direct {ext}->PDBQT is only supported with backend='obabel' in Converter."
                )
            out_dir = out.parent
            self._output = ensure_pdbqt(
                inp,
                out_dir,
                backend=self._backend,
                mode=self._mode,
                tmp_from_sdf_backend=self._tmp_from_sdf_backend,
                extra_args=self._extra_args,
            )
            return self

        raise NotImplementedError(
            f"Converter: unsupported input extension {ext} for backend '{self._backend}'."
        )

    @property
    def output(self) -> Optional[Path]:
        """
        Return the output Path (if available).

        :returns: Path or None
        """
        return self._output

    def __repr__(self) -> str:
        return (
            f"<Converter input={self._input} output={self._output} "
            f"mode={self._mode} backend={self._backend} tmp_from_sdf={self._tmp_from_sdf_backend}>"
        )

    def help(self) -> None:
        """
        Print short usage help to stdout.

        :returns: None
        """
        print("Converter usage:")
        print("  conv = Converter()")
        print(
            "  (conv.set_input('lig.sdf')"
            ".set_output('lig.pdbqt')"
            ".set_mode('ligand')"
            ".set_backend('meeko')"
            ".set_tmp_from_sdf_backend('rdkit')"
            ".run())"
        )
        print("  print(conv.output)")
