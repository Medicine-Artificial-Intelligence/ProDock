#!/usr/bin/env python3
"""
prepare_receptor4.py

- Create strict PDBQT for receptor from .pdb or reformat .pdbqt.
- Writes fixed-column ATOM/HETATM and includes AD4 type and Gasteiger charge (if available).
- Validates final file tags minimally (no BRANCH expected for receptor).
"""
from __future__ import annotations
import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Set

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def _fmt_pdbqt_atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resnum: int,
    x: float,
    y: float,
    z: float,
    charge: float,
    atype: str,
    het: bool = False,
) -> str:
    rec = "HETATM" if het else "ATOM  "
    atom_field = f"{atom_name[:4]:>4s}"
    res_field = f"{resname[:3]:>3s}"
    chain_field = (chain or "A")[:1]
    serial = max(1, min(serial, 99999))
    resnum = max(0, min(resnum, 9999))
    return (
        f"{rec:<6s}"
        f"{serial:5d} "
        f"{atom_field}"
        f" {res_field} "
        f"{chain_field}"
        f"{resnum:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{0.00:6.2f}{0.00:6.2f}    "
        f"{charge:7.4f} {atype}\n"
    )


def _reformat_pdbqt_lines(lines: Iterable[str]) -> List[str]:
    out = []
    for L in lines:
        if not L:
            out.append("\n")
            continue
        toks = L.split()
        tag = toks[0] if toks else ""
        if tag in ("ATOM", "HETATM"):
            # find three numeric tokens for coordinates (robust)
            floats = []
            for t in toks:
                try:
                    floats.append(float(t))
                except Exception:
                    pass
            if len(floats) < 3:
                out.append(L if L.endswith("\n") else L + "\n")
                continue
            x, y, z = floats[-3], floats[-2], floats[-1]
            name = toks[2] if len(toks) > 2 else "X"
            resn = toks[3] if len(toks) > 3 else "PRO"
            chain = toks[4] if len(toks) > 4 else "A"
            try:
                resnum = int(toks[5]) if len(toks) > 5 else 1
            except Exception:
                resnum = 1
            # try to find atype as last non-numeric token
            atype = ""
            for tok in reversed(toks):
                try:
                    float(tok)
                except Exception:
                    atype = tok
                    break
            out.append(
                _fmt_pdbqt_atom_line(
                    0 if not toks[1].isdigit() else int(toks[1]),
                    name,
                    resn,
                    chain,
                    resnum,
                    x,
                    y,
                    z,
                    0.0,
                    atype,
                    het=(tag == "HETATM"),
                )
            )
        else:
            out.append(L if L.endswith("\n") else L + "\n")
    return out


def _validate_receptor_text(text: str) -> None:
    # minimal validation: only REMARK, ATOM, HETATM tags allowed (no BRANCH)
    allowed = {"REMARK", "ATOM", "HETATM"}
    for i, L in enumerate(text.splitlines(), start=1):
        if not L.strip():
            continue
        t = L.split()[0]
        if t not in allowed and not L.startswith("REMARK"):
            raise RuntimeError(f"Unknown or inappropriate tag '{t}' on line {i}")


WATERS = {"HOH", "WAT", "H2O"}


def _safe_name(atom: Chem.Atom, serial: int) -> str:
    info = atom.GetPDBResidueInfo()
    raw = (info.GetName().strip() if info else "") or ""
    return (raw[:4] if raw else f"{atom.GetSymbol()}{serial%1000:03d}")[:4].strip()


def assign_ad4_types(mol: Chem.Mol) -> None:
    try:
        aromatic = Chem.MolFromSmarts("a")
        arom = (
            set(idx for (idx,) in mol.GetSubstructMatches(aromatic))
            if aromatic is not None
            else set()
        )
    except Exception:
        arom = set()
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        idx = a.GetIdx()
        try:
            is_arom = a.GetIsAromatic() or (idx in arom)
        except Exception:
            is_arom = idx in arom
        if sym == "H":
            t = "H"
        elif sym == "C":
            t = "A" if is_arom else "C"
        elif sym == "N":
            t = "NA" if is_arom else "N"
        elif sym == "O":
            t = "OA" if is_arom else "O"
        elif sym == "S":
            t = "SA" if is_arom else "S"
        elif sym in {"F", "Cl", "Br", "I", "P"}:
            t = sym
        else:
            t = sym
        a.SetProp("_AD4TYPE", t)


def _gasteiger_val(atom: Chem.Atom) -> float:
    if atom.HasProp("_GasteigerCharge"):
        try:
            q = float(atom.GetProp("_GasteigerCharge"))
            return 0.0 if (math.isnan(q) or math.isinf(q)) else q
        except Exception:
            return 0.0
    return 0.0


class ReceptorPreparer:
    def __init__(self, inp: Path, outp: Path):
        self.inp = Path(inp)
        self.out = Path(outp)

    def _from_pdb(self) -> Path:
        raw = self.inp.read_text(errors="ignore").splitlines(True)
        filtered = []
        for L in raw:
            if L.startswith(("ATOM", "HETATM")):
                alt = L[16:17]
                if alt not in (" ", "A"):
                    continue
                resn = L[17:20].strip().upper()
                if resn in WATERS:
                    continue
            filtered.append(L if L.endswith("\n") else L + "\n")
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tf:
            tf.write("".join(filtered))
            tmp = Path(tf.name)
        try:
            mol = Chem.MolFromPDBFile(str(tmp), sanitize=True, removeHs=False)
        finally:
            try:
                tmp.unlink()
            except Exception:
                pass
        if mol is None:
            raise RuntimeError("RDKit failed to parse cleaned PDB.")
        molH = Chem.AddHs(mol, addCoords=True)
        try:
            AllChem.ComputeGasteigerCharges(molH)
        except Exception:
            pass
        assign_ad4_types(molH)
        conf = molH.GetConformer()
        lines = ["REMARK  Name PROT\n"]
        serial = 1
        for a in molH.GetAtoms():
            info = a.GetPDBResidueInfo()
            name = _safe_name(a, serial)
            resn = (info.GetResidueName().strip() if info else "PRO") or "PRO"
            chain = (info.GetChainId().strip() if info else "A") or "A"
            resnum = (
                int(info.GetResidueNumber())
                if (info and info.GetResidueNumber() is not None)
                else 1
            )
            pos = conf.GetAtomPosition(a.GetIdx())
            charge = _gasteiger_val(a)
            atype = a.GetProp("_AD4TYPE") if a.HasProp("_AD4TYPE") else a.GetSymbol()
            lines.append(
                _fmt_pdbqt_atom_line(
                    serial,
                    name,
                    resn,
                    chain,
                    resnum,
                    pos.x,
                    pos.y,
                    pos.z,
                    charge,
                    atype,
                    het=False,
                )
            )
            serial += 1
        text = "".join(lines)
        _validate_receptor_text(text)
        self.out.write_text(text)
        return self.out

    def _reformat_pdbqt(self) -> Path:
        raw = self.inp.read_text(errors="ignore").splitlines()
        fixed = _reformat_pdbqt_lines(raw)
        text = "".join(fixed)
        _validate_receptor_text(text)
        self.out.write_text(text)
        return self.out

    def run(self) -> Path:
        ext = self.inp.suffix.lower()
        if ext == ".pdb":
            return self._from_pdb()
        if ext == ".pdbqt":
            return self._reformat_pdbqt()
        raise RuntimeError("Receptor input must be .pdb or .pdbqt")


def _cli():
    p = argparse.ArgumentParser(description="Prepare receptor PDBQT (strict format).")
    p.add_argument("-r", "--receptor", required=True)
    p.add_argument("-o", "--output", required=True)
    return p


def main(argv=None):
    ap = _cli()
    args = ap.parse_args(argv)
    try:
        out = ReceptorPreparer(Path(args.receptor), Path(args.output)).run()
        print(f"[OK] wrote {out}")
        return 0
    except Exception as e:
        print("[ERROR] prepare_receptor4 failed:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
