"""
Conversion utilities for ProDock (NO-FALLBACK, explicit backends).

Backends:
- Meeko CLI: mk_prepare_ligand.py / mk_prepare_receptor.py
- Open Babel CLI: obabel (or babel)
- MGLTools (AutoDockTools): prepare_ligand4.py / prepare_receptor4.py
- RDKit (Python API): for SDF <-> PDB where selected

Design:
- Every function takes a required `backend` (and fails if unsupported/unavailable).
- No implicit fallback. If you want a different route, pick the backend explicitly.
- For SDF -> PDBQT with Meeko/MGLTools, you must go via a temp PDB; choose
  the intermediate converter with `tmp_from_sdf_backend="rdkit"|"obabel"`.

Notes:
- Converting receptors reliably for docking typically needs Meeko or MGLTools.
- Open Babel can emit PDBQT but may not set docking-specific atom types.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional, List, Literal

# Optional RDKit (used only when you explicitly choose it)
try:
    from rdkit import Chem  # type: ignore

    _RDKIT_AVAILABLE = True
except Exception:
    Chem = None  # type: ignore
    _RDKIT_AVAILABLE = False

from prodock.io.logging import get_logger

logger = get_logger(__name__)

# ---------------------------
# Utilities
# ---------------------------

Backend = Literal["meeko", "obabel", "mgltools"]
TmpConv = Literal["rdkit", "obabel"]


def _ensure_exists(path: Union[str, Path], kind: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{kind} not found: {p}")
    return p


def _require_exe(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")
    return exe


def _run(args: List[str]) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _rdkit_require() -> None:
    if not _RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for this operation but is not available.")


def _tmp_sdf_to_pdb_with_rdkit(in_sdf: Path, out_pdb: Path) -> None:
    _rdkit_require()
    suppl = Chem.SDMolSupplier(str(in_sdf), removeHs=False, sanitize=True)  # type: ignore
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"No valid molecule found in {in_sdf}")
    Chem.MolToPDBFile(mol, str(out_pdb))  # type: ignore


def _tmp_sdf_to_pdb_with_obabel(in_sdf: Path, out_pdb: Path) -> None:
    obabel = _require_exe("obabel") if shutil.which("obabel") else _require_exe("babel")
    _run([obabel, "-isdf", str(in_sdf), "-opdb", "-O", str(out_pdb)])


def _sdf_to_pdb_intermediate(in_sdf: Path, out_pdb: Path, tmp_backend: TmpConv) -> None:
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

    :param input_pdb: input PDB file
    :param output_pdbqt: output PDBQT path
    :param mode: 'receptor' or 'ligand'
    :param backend: 'meeko' | 'obabel' | 'mgltools'
    :param extra_args: extra CLI args for the chosen backend
    :param meeko_cmd: override Meeko script name (default: mk_prepare_{receptor,ligand}.py)
    :param mgltools_cmd: override MGLTools script name (prepare_{receptor,ligand}4.py)
    """
    input_pdb = _ensure_exists(input_pdb, "Input PDB")
    output_pdbqt = Path(output_pdbqt).resolve()
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    extra_args = list(extra_args or [])

    if backend == "meeko":
        # Choose the appropriate script
        if mode == "receptor":
            cmd = meeko_cmd or "mk_prepare_receptor.py"
            exe = _require_exe(cmd)
            # Meeko receptor typical args
            args = [
                exe,
                "--read_pdb",
                str(input_pdb),
                "--write_pdbqt",
                str(output_pdbqt),
            ]
            args += extra_args
            _run(args)
        else:  # ligand
            cmd = meeko_cmd or "mk_prepare_ligand.py"
            exe = _require_exe(cmd)
            # Modern Meeko ligand CLI supports -i/-o
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
            # Minimal required flags; adjust via extra_args as you like
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
    return output_pdbqt


def pdbqt_to_pdb(
    input_pdbqt: Union[str, Path],
    output_pdb: Union[str, Path],
    *,
    backend: Literal["obabel"],
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Convert PDBQT -> PDB. Only Open Babel provides this direct route.

    :param input_pdbqt: input PDBQT
    :param output_pdb: output PDB
    :param backend: must be 'obabel'
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

    :param backend: 'rdkit' or 'obabel'
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

    - backend='obabel': direct via Open Babel.
    - backend == 'meeko': prefer calling Meeko with the original SDF (Meeko accepts sdf/mol2/mol).
      If input is not an SDF, we fall back to creating a temporary SDF (via RDKit or Open Babel)
      and call Meeko with that temporary SDF.
    - backend == 'mgltools': similar to meeko, try using SDF/MOL2 where possible.

    Note: This avoids creating a temporary PDB and then feeding PDB to Meeko (which can fail).
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
        # If input is already SDF (best case), call Meeko directly with it.
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
            # otherwise create a temporary SDF (rdkit or obabel) and pass that to Meeko
            with tempfile.TemporaryDirectory() as td:
                tmp_sdf = Path(td) / (input_sdf.stem + ".sdf")
                if tmp_from_sdf_backend == "rdkit":
                    _rdkit_require()
                    # try RDKit reading arbitrary formats that RDKit supports
                    # read via RDKit then write SDF
                    if input_sdf.suffix.lower() in {".sdf"}:
                        shutil.copyfile(str(input_sdf), str(tmp_sdf))
                    else:
                        # try to load via RDKit generic readers (MolFromSmiles, MolFromPDBFile, etc.)
                        if input_sdf.suffix.lower() in {".pdb", ".pdbqt"}:
                            # RDKit: read PDB
                            mol = Chem.MolFromPDBFile(str(input_sdf), removeHs=False)  # type: ignore
                        else:
                            # fallback: try reading as SMILES (if .smi), else try SDMolSupplier
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
                    # tmp_from_sdf_backend == "obabel"
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

                # now call Meeko with the tmp SDF
                args = [
                    meeko_lig,
                    "-i",
                    str(tmp_sdf),
                    "-o",
                    str(output_pdbqt),
                ] + extra_args
                _run(args)

    elif backend == "mgltools":
        # For ligands MGLTools' prepare_ligand4.py generally accepts MOL/MOL2/SDF.
        mgl_lig = _require_exe(mgltools_cmd or "prepare_ligand4.py")
        if input_sdf.suffix.lower() == ".sdf":
            args = [mgl_lig, "-l", str(input_sdf), "-o", str(output_pdbqt)] + extra_args
            _run(args)
        else:
            # create tmp sdf as above then call prepare_ligand4.py
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
    return output_pdbqt


def pdb_to_sdf(
    input_pdb: Union[str, Path],
    output_sdf: Union[str, Path],
    *,
    backend: Literal["rdkit", "obabel"],
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Convert PDB -> SDF via a single explicit backend.

    :param backend: 'rdkit' or 'obabel'
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
    Convert PDBQT -> SDF. Only Open Babel provides this direct route.

    :param backend: must be 'obabel'
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
    - .pdb -> PDBQT via `pdb_to_pdbqt` with chosen backend
    - .sdf -> PDBQT via `sdf_to_pdbqt` with chosen backend (and temp converter if needed)
    - .mol2/.smi: supported only with backend='obabel' (direct to PDBQT). Others: raise.

    :param backend: 'meeko' | 'obabel' | 'mgltools'
    """
    p = Path(input_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    if p.suffix.lower() == ".pdbqt":
        return p

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_p = output_dir / (p.stem + ".pdbqt")

    ext = p.suffix.lower()
    if ext == ".pdb":
        return pdb_to_pdbqt(
            p,
            out_p,
            mode=mode,
            backend=backend,
            extra_args=extra_args,
        )

    if ext == ".sdf":
        return sdf_to_pdbqt(
            p,
            out_p,
            backend=backend,
            tmp_from_sdf_backend=tmp_from_sdf_backend,
            extra_args=extra_args,
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
        return out_p

    raise NotImplementedError(
        f"Unsupported input extension {ext} for ensure_pdbqt with backend='{backend}'."
    )


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
          .set_backend("meeko")                 # or "obabel", "mgltools"
          .set_tmp_from_sdf_backend("rdkit")    # or "obabel" (only if needed)
          .set_extra_args(["--some-flag"])
          .run()
          .output
      )
    """

    def __init__(self) -> None:
        self._input: Optional[Path] = None
        self._output: Optional[Path] = None
        self._mode: Literal["ligand", "receptor"] = "ligand"
        self._backend: Optional[Backend] = None
        self._tmp_from_sdf_backend: TmpConv = "rdkit"
        self._extra_args: Optional[List[str]] = None
        self._meeko_cmd: Optional[str] = None
        self._mgltools_cmd: Optional[str] = None

    def set_input(self, input_path: Union[str, Path]) -> "Converter":
        self._input = Path(input_path)
        return self

    def set_output(self, output_path: Union[str, Path]) -> "Converter":
        self._output = Path(output_path)
        return self

    def set_mode(self, mode: Literal["ligand", "receptor"]) -> "Converter":
        self._mode = mode
        return self

    def set_backend(self, backend: Backend) -> "Converter":
        self._backend = backend
        return self

    def set_tmp_from_sdf_backend(self, tmp_backend: TmpConv) -> "Converter":
        self._tmp_from_sdf_backend = tmp_backend
        return self

    def set_extra_args(self, args: Optional[List[str]]) -> "Converter":
        self._extra_args = None if args is None else list(args)
        return self

    def set_meeko_cmd(self, cmd: Optional[str]) -> "Converter":
        self._meeko_cmd = cmd
        return self

    def set_mgltools_cmd(self, cmd: Optional[str]) -> "Converter":
        self._mgltools_cmd = cmd
        return self

    def run(self) -> "Converter":
        if self._input is None:
            raise RuntimeError("No input set (call .set_input(...))")
        if self._backend is None:
            raise RuntimeError(
                "No backend set (call .set_backend('meeko'|'obabel'|'mgltools'))"
            )
        if self._output is None:
            # default: change extension to .pdbqt in CWD if not given
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
            # nothing to do; keep as-is
            self._output = inp
            return self

        # Allow direct OBabel conversions for some other text formats
        if ext in {".mol2", ".smi"}:
            if self._backend != "obabel":
                raise NotImplementedError(
                    f"Direct {ext}->PDBQT is only supported with backend='obabel' in Converter."
                )
            self._output = ensure_pdbqt(
                inp,
                out.parent,
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
        return self._output

    def __repr__(self) -> str:
        return (
            f"<Converter input={self._input} output={self._output} "
            f"mode={self._mode} backend={self._backend} tmp_from_sdf={self._tmp_from_sdf_backend}>"
        )

    def help(self) -> None:
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
