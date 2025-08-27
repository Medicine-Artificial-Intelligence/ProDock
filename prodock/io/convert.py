"""
Conversion utilities for ProDock:
- pdb_to_pdbqt, pdbqt_to_pdb, sdf_to_pdb (improved)
- additional converters: sdf_to_pdbqt, pdb_to_sdf, pdbqt_to_sdf
- Converter class (chainable OOP) for re-usable conversion workflows

Backends tried (in order):
1) Meeko CLI (mk_prepare_ligand.py / mk_prepare_receptor.py) when preparing PDB -> PDBQT
2) Open Babel CLI (`obabel` or `babel`) for many direct format conversions
3) RDKit (python API) for SDF -> PDB conversion (preferred if installed)

Note: converting receptors reliably for docking often requires Meeko or MGLTools;
Open Babel can help but may not set all docking-specific atom types.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional, List

# Try to import RDKit if available (used for sdf -> pdb)
try:
    from rdkit import Chem  # type: ignore

    _RDKit_AVAILABLE = True
except Exception:
    Chem = None  # type: ignore
    _RDKit_AVAILABLE = False

from prodock.io.logging import get_logger

logger = get_logger(__name__)
# ---------------------------
# Original (enhanced) functions
# ---------------------------


def pdb_to_pdbqt(
    input_pdb: Union[str, Path],
    output_pdbqt: Union[str, Path],
    mode: str = "receptor",
    meeko_cmd: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Convert PDB -> PDBQT using Meeko or fallback to Open Babel.

    :param input_pdb: input PDB file.
    :param output_pdbqt: output PDBQT file.
    :param mode: 'receptor' or 'ligand'.
    :param meeko_cmd: explicit command (default: mk_prepare_receptor.py / mk_prepare_ligand.py).
    :param extra_args: extra CLI args to pass.
    :return: Path to PDBQT produced.
    """
    input_pdb = Path(input_pdb).resolve()
    output_pdbqt = Path(output_pdbqt).resolve()
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    if not input_pdb.exists():
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

    # decide Meeko command
    if mode.lower() == "receptor":
        default_meeko = "mk_prepare_receptor.py"
        if meeko_cmd is None:
            meeko_cmd = default_meeko
        # Meeko receptor CLI historically supports: --read_pdb, --write_pdbqt (but CLI variants differ)
        # construct args defensively
        args = [
            meeko_cmd,
            "--read_pdb",
            str(input_pdb),
            "--write_pdbqt",
            str(output_pdbqt),
        ]
        # some Meeko versions expect "-o basename" rather than write_pdbqt;
        # include both possibilities via extra_args if needed
    elif mode.lower() == "ligand":
        default_meeko = "mk_prepare_ligand.py"
        if meeko_cmd is None:
            meeko_cmd = default_meeko
        args = [meeko_cmd, "-i", str(input_pdb), "-o", str(output_pdbqt)]
    else:
        raise ValueError("mode must be 'receptor' or 'ligand'")

    if extra_args:
        args += list(extra_args)

    # Prefer Meeko CLI if present
    exe = shutil.which(meeko_cmd)
    if exe:
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{meeko_cmd} failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_pdbqt.exists():
            # Some Meeko variants produce basename+".pdbqt" — check
            candidate = Path(str(output_pdbqt))
            if not candidate.exists():
                # try basename fallback
                fallback = output_pdbqt.with_suffix(".pdbqt")
                if fallback.exists():
                    return fallback
                raise FileNotFoundError(f"PDBQT not produced: {output_pdbqt}")
        return output_pdbqt

    # fallback: use Open Babel if available (limited for receptor)
    obabel = shutil.which("obabel") or shutil.which("babel")
    if obabel:
        # obabel usage: obabel -ipdb input.pdb -opdbqt -O output.pdbqt --partialcharge gasteiger
        args = [
            obabel,
            "-ipdb",
            str(input_pdb),
            "-opdbqt",
            "-O",
            str(output_pdbqt),
            "--partialcharge",
            "gasteiger",
        ]
        if extra_args:
            args += list(extra_args)
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Open Babel failed producing PDBQT (rc={proc.returncode})\n"
                + f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_pdbqt.exists():
            raise FileNotFoundError(f"PDBQT not produced by Open Babel: {output_pdbqt}")
        return output_pdbqt

    # No tool found
    raise RuntimeError(
        "Neither Meeko nor Open Babel found. Install Meeko (mk_prepare_{ligand,receptor}.py) or Open Babel (obabel)."
    )


def pdbqt_to_pdb(
    input_pdbqt: Union[str, Path],
    output_pdb: Union[str, Path],
    obabel_cmd: str = "obabel",
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Convert PDBQT -> PDB using Open Babel (Meeko has no direct reverse).

    :param input_pdbqt: input PDBQT file.
    :param output_pdb: output PDB file.
    :param obabel_cmd: path to obabel binary.
    :param extra_args: extra CLI args.
    :return: Path to PDB produced.
    """
    input_pdbqt = Path(input_pdbqt).resolve()
    output_pdb = Path(output_pdb).resolve()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    if not input_pdbqt.exists():
        raise FileNotFoundError(f"Input PDBQT not found: {input_pdbqt}")

    exe = shutil.which(obabel_cmd)
    if exe is None:
        raise RuntimeError(f"Open Babel binary not found: {obabel_cmd}")

    args = [exe, "-ipdbqt", str(input_pdbqt), "-opdb", "-O", str(output_pdb)]
    if extra_args:
        args += list(extra_args)

    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"obabel failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if not output_pdb.exists():
        raise FileNotFoundError(f"PDB not produced: {output_pdb}")
    return output_pdb


def sdf_to_pdb(input_sdf: Union[str, Path], output_pdb: Union[str, Path]) -> Path:
    """
    Convert the first molecule in an SDF file to PDB.

    :param input_sdf: path to input .sdf file
    :param output_pdb: path to output .pdb file
    :return: Path to written PDB
    """
    input_sdf = Path(input_sdf).resolve()
    output_pdb = Path(output_pdb).resolve()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    if not input_sdf.exists():
        raise FileNotFoundError(f"SDF not found: {input_sdf}")

    # prefer RDKit if available
    if _RDKit_AVAILABLE:
        suppl = Chem.SDMolSupplier(str(input_sdf), removeHs=False, sanitize=True)
        mol = next((m for m in suppl if m is not None), None)
        if mol is None:
            raise ValueError(f"No valid molecule found in {input_sdf}")
        Chem.MolToPDBFile(mol, str(output_pdb))
        return output_pdb

    # fallback to Open Babel
    obabel = shutil.which("obabel") or shutil.which("babel")
    if obabel:
        args = [obabel, "-isdf", str(input_sdf), "-opdb", "-O", str(output_pdb)]
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"obabel failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_pdb.exists():
            raise FileNotFoundError(f"PDB not produced by Open Babel: {output_pdb}")
        return output_pdb

    raise RuntimeError(
        "Neither RDKit nor Open Babel available for SDF -> PDB conversion."
    )


# ---------------------------
# Additional useful conversions
# ---------------------------


def sdf_to_pdbqt(
    input_sdf: Union[str, Path],
    output_pdbqt: Union[str, Path],
    use_meeko: bool = False,
    obabel_cmd: str = "obabel",
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Convert SDF -> PDBQT. Tries:
      1) RDKit -> temporary PDB -> Meeko ligand prepare (if use_meeko True or Meeko available)
      2) Open Babel direct conversion (obabel -isdf -opdbqt -O out.pdbqt --partialcharge gasteiger)

    :param input_sdf: path to input SDF
    :param output_pdbqt: desired output PDBQT path
    :param use_meeko: if True prefer Meeko pathway (recommended for docking-compatible PDBQT)
    :param obabel_cmd: obabel binary name
    :param extra_args: extra args to pass to underlying tool
    :return: Path to PDBQT produced
    """
    input_sdf = Path(input_sdf).resolve()
    output_pdbqt = Path(output_pdbqt).resolve()
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    if not input_sdf.exists():
        raise FileNotFoundError(f"SDF not found: {input_sdf}")

    # If user requested Meeko pathway and Meeko is available, use RDKit->temp PDB->Meeko
    meeko_lig = shutil.which("mk_prepare_ligand.py")
    if use_meeko or meeko_lig:
        if _RDKit_AVAILABLE:
            with tempfile.TemporaryDirectory() as td:
                tmp_pdb = Path(td) / (input_sdf.stem + ".pdb")
                # create pdb with RDKit
                suppl = Chem.SDMolSupplier(
                    str(input_sdf), removeHs=False, sanitize=True
                )
                mol = next((m for m in suppl if m is not None), None)
                if mol is None:
                    raise ValueError(f"No valid molecule found in {input_sdf}")
                Chem.MolToPDBFile(mol, str(tmp_pdb))
                # call Meeko ligand prepare
                try:
                    return pdb_to_pdbqt(
                        tmp_pdb,
                        output_pdbqt,
                        mode="ligand",
                        meeko_cmd=meeko_lig,
                        extra_args=extra_args,
                    )
                except Exception as e:
                    # fallback to OB if Meeko fails
                    logger.warning(f"Meeko failed: {e}")
                    pass
        # if RDKit missing or Meeko failed, attempt Open Babel fallback below

    # fallback: Open Babel direct conversion
    obabel = shutil.which(obabel_cmd)
    if obabel:
        args = [
            obabel,
            "-isdf",
            str(input_sdf),
            "-opdbqt",
            "-O",
            str(output_pdbqt),
            "--partialcharge",
            "gasteiger",
        ]
        if extra_args:
            args += list(extra_args)
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Open Babel failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_pdbqt.exists():
            raise FileNotFoundError(f"PDBQT not produced: {output_pdbqt}")
        return output_pdbqt

    raise RuntimeError(
        "No available method to convert SDF -> PDBQT (install Meeko + RDKit or Open Babel)."
    )


def pdb_to_sdf(input_pdb: Union[str, Path], output_sdf: Union[str, Path]) -> Path:
    """
    Convert PDB -> SDF (single molecule). Uses RDKit if available, otherwise Open Babel.

    :param input_pdb: input PDB path
    :param output_sdf: output SDF path
    :return: Path to SDF file
    """
    input_pdb = Path(input_pdb).resolve()
    output_sdf = Path(output_sdf).resolve()
    output_sdf.parent.mkdir(parents=True, exist_ok=True)

    if not input_pdb.exists():
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

    if _RDKit_AVAILABLE:
        # RDKit: read PDB as Mol, write as SDF (single molecule)
        mol = Chem.MolFromPDBFile(str(input_pdb), removeHs=False)
        if mol is None:
            raise ValueError(f"RDKit could not parse PDB: {input_pdb}")
        writer = Chem.SDWriter(str(output_sdf))
        writer.write(mol)
        writer.close()
        return output_sdf

    obabel = shutil.which("obabel") or shutil.which("babel")
    if obabel:
        args = [obabel, "-ipdb", str(input_pdb), "-osdf", "-O", str(output_sdf)]
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Open Babel failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_sdf.exists():
            raise FileNotFoundError(f"SDF not produced: {output_sdf}")
        return output_sdf

    raise RuntimeError(
        "No available method to convert PDB -> SDF (install RDKit or Open Babel)."
    )


def pdbqt_to_sdf(
    input_pdbqt: Union[str, Path],
    output_sdf: Union[str, Path],
    obabel_cmd: str = "obabel",
) -> Path:
    """
    Convert PDBQT -> SDF using Open Babel.

    :param input_pdbqt: input PDBQT path
    :param output_sdf: output SDF path
    :param obabel_cmd: obabel binary
    :return: Path to SDF
    """
    input_pdbqt = Path(input_pdbqt).resolve()
    output_sdf = Path(output_sdf).resolve()
    output_sdf.parent.mkdir(parents=True, exist_ok=True)

    if not input_pdbqt.exists():
        raise FileNotFoundError(f"Input PDBQT not found: {input_pdbqt}")

    obabel = shutil.which(obabel_cmd)
    if obabel is None:
        raise RuntimeError("Open Babel not found (required to convert PDBQT -> SDF).")

    args = [obabel, "-ipdbqt", str(input_pdbqt), "-osdf", "-O", str(output_sdf)]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Open Babel failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if not output_sdf.exists():
        raise FileNotFoundError(f"SDF not produced: {output_sdf}")
    return output_sdf


def ensure_pdbqt(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    prefer: str = "meeko",
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Ensure the given input is available as a PDBQT file. If input is already .pdbqt, returns it.
    Otherwise converts to PDBQT into output_dir and returns the Path.

    :param input_path: any input path (.pdb, .sdf, .mol2, .pdbqt)
    :param output_dir: directory where to place converted pdbqt
    :param prefer: 'meeko' or 'obabel' preference for conversion
    :param extra_args: extra args to pass to converter
    :return: Path to PDBQT
    """
    p = Path(input_path)
    if p.suffix.lower() == ".pdbqt":
        return p.resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_p = output_dir / (p.stem + ".pdbqt")

    # route by extension
    ext = p.suffix.lower()
    if ext == ".pdb":
        # prefer meeko if requested
        if prefer == "meeko":
            try:
                return pdb_to_pdbqt(p, out_p, mode="ligand", extra_args=extra_args)
            except Exception:
                # fallback to obabel
                pass
        return pdb_to_pdbqt(p, out_p, mode="ligand", extra_args=extra_args)

    if ext in {".sdf", ".mol2", ".smi", ".smi.gz"}:
        return sdf_to_pdbqt(
            p, out_p, use_meeko=(prefer == "meeko"), extra_args=extra_args
        )

    # final fallback: try obabel conversion with autodetected input format
    obabel = shutil.which("obabel") or shutil.which("babel")
    if obabel:
        args = [obabel, f"-i{ext.lstrip('.')}", str(p), "-opdbqt", "-O", str(out_p)]
        if extra_args:
            args += list(extra_args)
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Open Babel fallback failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not out_p.exists():
            raise FileNotFoundError(f"PDBQT not produced by fallback: {out_p}")
        return out_p

    raise RuntimeError(
        "No available tool to produce PDBQT for input: install Meeko or Open Babel."
    )


# ---------------------------
# Chainable OOP Converter
# ---------------------------


class Converter:
    """
    Chainable converter helper for ProDock.

    Example:
      conv = Converter().set_input("mol.sdf").set_output("mol.pdbqt").set_mode("ligand").use_meeko().run()
      out = conv.output

    :param input_path: initial input (set via set_input)
    """

    def __init__(self):
        self._input: Optional[Path] = None
        self._output: Optional[Path] = None
        self._mode: Optional[str] = None
        self._prefer: str = "meeko"
        self._extra_args: Optional[List[str]] = None
        self._meeko_cmd: Optional[str] = None
        self._obabel_cmd: Optional[str] = None
        self._last_cmd: Optional[List[str]] = None

    def set_input(self, input_path: Union[str, Path]) -> "Converter":
        self._input = Path(input_path)
        return self

    def set_output(self, output_path: Union[str, Path]) -> "Converter":
        self._output = Path(output_path)
        return self

    def set_mode(self, mode: str) -> "Converter":
        if mode.lower() not in {"ligand", "receptor"}:
            raise ValueError("mode must be 'ligand' or 'receptor'")
        self._mode = mode.lower()
        return self

    def use_meeko(self, meeko_cmd: Optional[str] = None) -> "Converter":
        self._prefer = "meeko"
        self._meeko_cmd = meeko_cmd
        return self

    def use_obabel(self, obabel_cmd: Optional[str] = None) -> "Converter":
        self._prefer = "obabel"
        self._obabel_cmd = obabel_cmd
        return self

    def set_extra_args(self, args: Optional[List[str]]) -> "Converter":
        self._extra_args = None if args is None else list(args)
        return self

    def run(self) -> "Converter":
        if self._input is None:
            raise RuntimeError("No input set (call .set_input(...))")
        if self._output is None:
            # infer output from input + desired extension
            self._output = Path.cwd() / (self._input.stem + ".pdbqt")
        # route conversions depending on extension
        inp = self._input.resolve()
        out = self._output.resolve()
        ext = inp.suffix.lower()

        if ext == ".pdb":
            # PDB -> PDBQT
            meeko_cmd = self._meeko_cmd
            if self._prefer != "meeko":
                meeko_cmd = None
            res = pdb_to_pdbqt(
                inp,
                out,
                mode=(self._mode or "ligand"),
                meeko_cmd=meeko_cmd,
                extra_args=self._extra_args,
            )
            self._last_cmd = None
            self._output = res
            return self

        if ext == ".sdf":
            res = sdf_to_pdbqt(
                inp,
                out,
                use_meeko=(self._prefer == "meeko"),
                obabel_cmd=(self._obabel_cmd or "obabel"),
                extra_args=self._extra_args,
            )
            self._output = res
            return self

        if ext == ".pdbqt":
            # nothing to do — just copy
            self._output = inp
            return self

        # fallback: try ensure_pdbqt which will attempt many fallbacks
        res = ensure_pdbqt(
            inp, out.parent, prefer=self._prefer, extra_args=self._extra_args
        )
        self._output = res
        return self

    @property
    def output(self) -> Optional[Path]:
        return self._output

    def __repr__(self) -> str:
        return f"<Converter input={self._input} output={self._output} mode={self._mode} prefer={self._prefer}>"

    def help(self) -> None:
        print("Converter usage:")
        print("  conv = Converter()")
        print(
            "  conv.set_input('lig.sdf').set_output('lig.pdbqt').set_mode('ligand').use_meeko().run()"
        )
        print("  print(conv.output)")
