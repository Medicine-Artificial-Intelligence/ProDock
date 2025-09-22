#!/usr/bin/env python3
"""
prepare_ligand4.py

- Writes strict fixed-column PDBQT for flexible ligands (ROOT/BRANCH/ENDBRANCH/ENDROOT/TORSDOF).
- Ensures BRANCH/ENDBRANCH pairs are matched and refer to valid printed (heavy) atom serials.
- Validates the produced file and prints clear diagnostics if invalid (so you can paste them).
- RDKit-only (no OpenBabel), robust to missing RingInfo by using fallbacks.
"""
from __future__ import annotations
import argparse
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# silence RDKit C++ logs
RDLogger.DisableLog("rdApp.*")


# ----------------- Strict PDB atom formatting function -----------------
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
    het: bool = True,
) -> str:
    # Use PDB column positions (1-indexed):
    # record(1-6), serial(7-11), " " (12), name(13-16), " " (17), resname(18-20), " " (21),
    # chain(22), resseq(23-26), coords 31-54 etc. After coords we append charge and type.
    rec = "HETATM" if het else "ATOM  "
    atom_field = f"{atom_name[:4]:>4s}"
    res_field = f"{resname[:3]:>3s}"
    chain_field = (chain or "A")[:1]
    serial = max(1, min(serial, 99999))
    resnum = max(0, min(resnum, 9999))
    # Compose aligned line. Keep occupancy/temp default 0.00.
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


# ----------------- Safe AD4 typing and rotatable detection -----------------
def _safe_aromatic_set(mol: Chem.Mol) -> Set[int]:
    try:
        pat = Chem.MolFromSmarts("a")
        if pat is None:
            return set()
        return set(i for (i,) in mol.GetSubstructMatches(pat))
    except Exception:
        return set()


def assign_ad4_types(mol: Chem.Mol) -> None:
    arom = _safe_aromatic_set(mol)
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        idx = a.GetIdx()
        try:
            is_arom = a.GetIsAromatic() or (idx in arom)
        except Exception:
            is_arom = idx in arom
        if sym == "H":
            t = "H"
            try:
                if a.GetDegree() == 1:
                    neigh = a.GetNeighbors()[0]
                    if neigh.GetSymbol() in {"N", "O", "S"}:
                        t = "HD"
            except Exception:
                pass
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


try:
    _ROT_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
except Exception:
    _ROT_SMARTS = None


def _is_amide(mol: Chem.Mol, a: int, b: int) -> bool:
    try:
        A = mol.GetAtomWithIdx(a)
        B = mol.GetAtomWithIdx(b)
    except Exception:
        return False

    def check(natom, catom):
        if natom.GetSymbol() != "N" or catom.GetSymbol() != "C":
            return False
        for bd in catom.GetBonds():
            if bd.GetBondType() == Chem.BondType.DOUBLE:
                other = bd.GetOtherAtom(catom)
                if other.GetSymbol() == "O":
                    return True
        return False

    try:
        return check(A, B) or check(B, A)
    except Exception:
        return False


def find_rotatables(mol: Chem.Mol) -> List[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    try:
        if _ROT_SMARTS is not None:
            for m in mol.GetSubstructMatches(_ROT_SMARTS):
                a, b = m[0], m[1]
                bd = mol.GetBondBetweenAtoms(a, b)
                if bd is None:
                    continue
                try:
                    if bd.IsInRing():
                        continue
                except Exception:
                    pass
                if bd.GetBondType() != Chem.BondType.SINGLE:
                    continue
                if _is_amide(mol, a, b):
                    continue
                pairs.add(tuple(sorted((a, b))))
            return sorted(pairs)
    except Exception:
        pass
    # fallback
    try:
        for bd in mol.GetBonds():
            if bd.GetBondType() != Chem.BondType.SINGLE:
                continue
            try:
                if bd.IsInRing():
                    continue
            except Exception:
                pass
            a = bd.GetBeginAtomIdx()
            b = bd.GetEndAtomIdx()
            if (
                mol.GetAtomWithIdx(a).GetDegree() <= 1
                or mol.GetAtomWithIdx(b).GetDegree() <= 1
            ):
                continue
            if _is_amide(mol, a, b):
                continue
            pairs.add(tuple(sorted((a, b))))
    except Exception:
        pass
    return sorted(pairs)


# ----------------- Component tree (like ADTool) -----------------
def build_comp_tree(mol: Chem.Mol, rot_pairs: List[Tuple[int, int]]):
    n = mol.GetNumAtoms()
    adj = {i: [] for i in range(n)}
    for bd in mol.GetBonds():
        i = bd.GetBeginAtomIdx()
        j = bd.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)
    rotset = set(tuple(sorted(p)) for p in rot_pairs)
    adj_nr = {i: [] for i in range(n)}
    for i in range(n):
        for j in adj[i]:
            if tuple(sorted((i, j))) in rotset:
                continue
            adj_nr[i].append(j)

    comp_id = {}
    comp_atoms: Dict[int, List[int]] = {}
    seen = set()
    cid = 0
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp_atoms[cid] = []
        while stack:
            u = stack.pop()
            comp_id[u] = cid
            comp_atoms[cid].append(u)
            for v in adj_nr[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        cid += 1

    comp_children_map = {k: [] for k in comp_atoms}
    for a, b in rot_pairs:
        ca, cb = comp_id[a], comp_id[b]
        if ca == cb:
            continue
        comp_children_map[ca].append((cb, (a, b)))
        comp_children_map[cb].append((ca, (b, a)))

    root_atom = next((a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"), 0)
    root_comp = comp_id[root_atom]

    parent = {root_comp: None}
    children = {cid: [] for cid in comp_atoms}
    parent_attach: Dict[int, Tuple[int, int]] = {}
    q = deque([root_comp])
    while q:
        c = q.popleft()
        for nbr, attach in comp_children_map.get(c, []):
            if nbr in parent:
                continue
            parent[nbr] = c
            parent_attach[nbr] = attach
            children[c].append((nbr, attach))
            q.append(nbr)
    return comp_atoms, children, root_comp, parent_attach


# ----------------- Anchor helpers -----------------
def _heavy(mol: Chem.Mol, idx: int) -> bool:
    try:
        return mol.GetAtomWithIdx(idx).GetSymbol() != "H"
    except Exception:
        return True


def _choose_heavy_in_comp(
    mol: Chem.Mol, comp_list: List[int], preferred: int
) -> Optional[int]:
    if _heavy(mol, preferred):
        return preferred
    try:
        for n in mol.GetAtomWithIdx(preferred).GetNeighbors():
            if n.GetIdx() in comp_list and _heavy(mol, n.GetIdx()):
                return n.GetIdx()
    except Exception:
        pass
    for a in comp_list:
        if _heavy(mol, a):
            return a
    return None


# ----------------- Validator to emulate Vina parse checks -----------------
_BRANCH_RE = re.compile(r"^BRANCH\s+(\d+)\s+(\d+)\s*$")
_ENDBR_RE = re.compile(r"^ENDBRANCH\s+(\d+)\s+(\d+)\s*$")
_TORSDOF_RE = re.compile(r"^TORSDOF\s+(\d+)\s*$")


def validate_pdbqt_text(text: str) -> Tuple[bool, List[str]]:
    """
    Validate the generated PDBQT text for the key constraints Vina expects.
    Returns (ok, diagnostics).
    """
    diagnostics: List[str] = []
    lines = text.splitlines()
    # Must have ROOT..ENDROOT enclosing branches and atoms
    in_root = False
    seen_atom_serials: Dict[int, Dict] = {}
    branch_pairs: List[Tuple[int, int, int]] = []  # (line_no, parent, child)
    endbranch_pairs: List[Tuple[int, int, int]] = []
    tor_dof = None
    for i, L in enumerate(lines, start=1):
        s = L.rstrip("\n")
        if not s.strip():
            continue
        if s.startswith("REMARK"):
            continue
        if s == "ROOT":
            if in_root:
                diagnostics.append(f"Line {i}: duplicate ROOT")
            in_root = True
            continue
        if s == "ENDROOT":
            if not in_root:
                diagnostics.append(f"Line {i}: ENDROOT without ROOT")
            in_root = False
            continue
        m = _BRANCH_RE.match(s)
        if m:
            parent = int(m.group(1))
            child = int(m.group(2))
            branch_pairs.append((i, parent, child))
            if not in_root:
                diagnostics.append(f"Line {i}: BRANCH outside ROOT/ENDROOT")
            continue
        m2 = _ENDBR_RE.match(s)
        if m2:
            parent = int(m2.group(1))
            child = int(m2.group(2))
            endbranch_pairs.append((i, parent, child))
            continue
        m3 = _TORSDOF_RE.match(s)
        if m3:
            tor_dof = int(m3.group(1))
            continue
        # ATOM/HETATM: parse columns for serial and element & coords
        if s.startswith("ATOM") or s.startswith("HETATM"):
            # parse serial at cols 7-11 (0-based 6:11)
            try:
                serial = int(s[6:11].strip())
            except Exception:
                diagnostics.append(
                    f"Line {i}: cannot parse atomic serial in ATOM/HETATM line: '{s}'"
                )
                continue
            # record whether heavy (look at atom name or element at end)
            # element/type at end: try to extract last token
            tail = s[66:].strip() if len(s) > 66 else s[54:].strip()
            # tail often like " -0.234 N"
            last_tok = tail.split()[-1] if tail.split() else ""
            # Using atom name to detect heavy element: atom name positions 12-16 (0-based 12:16)
            atom_name = s[12:16].strip()
            # crude heavy detection by first char not H
            is_heavy = not atom_name.startswith("H")
            seen_atom_serials[serial] = {"line": i, "text": s, "heavy": is_heavy}
            continue
        # Unknown tag encountered
        diagnostics.append(f"Line {i}: Unknown or inappropriate tag: '{s.split()[0]}'")
    # Post-checks: every BRANCH must have matching ENDBRANCH
    # We require counts to match and exact pairs present
    branches = [(p, c) for (_, p, c) in branch_pairs]
    endbranches = [(p, c) for (_, p, c) in endbranch_pairs]
    if len(branches) != len(endbranches):
        diagnostics.append(
            f"BRANCH count {len(branches)} != ENDBRANCH count {len(endbranches)}"
        )
    else:
        for idx, (p, c) in enumerate(branches):
            if idx >= len(endbranches) or endbranches[idx] != (p, c):
                diagnostics.append(
                    f"BRANCH pair #{idx+1} (parent {p} child {c}) does not match ENDBRANCH pair at same nesting"
                )
    # BRANCH anchors must refer to actual atom serials and parent must be printed before child
    all_serials = sorted(seen_atom_serials.keys())
    if not all_serials:
        diagnostics.append("No ATOM/HETATM lines found")
    for ln, p, c in branch_pairs:
        if p not in seen_atom_serials:
            diagnostics.append(
                f"Line {ln}: BRANCH parent serial {p} not present in atom lines"
            )
        if c not in seen_atom_serials:
            diagnostics.append(
                f"Line {ln}: BRANCH child serial {c} not present in atom lines"
            )
        # ensure parent line number < child line number
        if p in seen_atom_serials and c in seen_atom_serials:
            if seen_atom_serials[p]["line"] >= seen_atom_serials[c]["line"]:
                diagnostics.append(
                    f"Line {ln}: BRANCH parent serial {p} (line {seen_atom_serials[p]['line']}) is not before child serial {c} (line {seen_atom_serials[c]['line']})"
                )
            # ensure both anchors heavy
            if not seen_atom_serials[p]["heavy"]:
                diagnostics.append(
                    f"Line {ln}: BRANCH parent serial {p} (line {seen_atom_serials[p]['line']}) is not a heavy atom"
                )
            if not seen_atom_serials[c]["heavy"]:
                diagnostics.append(
                    f"Line {ln}: BRANCH child serial {c} (line {seen_atom_serials[c]['line']}) is not a heavy atom"
                )
    # TORSDOF must equal number of branches emitted (conservatively count branches len)
    if tor_dof is not None:
        if tor_dof != len(branches):
            diagnostics.append(
                f"TORSDOF {tor_dof} does not match number of BRANCH entries {len(branches)}"
            )
    else:
        diagnostics.append("TORSDOF not present")
    ok = len(diagnostics) == 0
    return ok, diagnostics


# ----------------- Core writer using comp tree + safe anchors -----------------
def write_strict_pdbqt(mol: Chem.Mol, out_path: Path, title: str = "LIG"):
    if mol.GetNumConformers() == 0:
        raise RuntimeError(
            "No conformer (3D) available; embed first or provide coordinates."
        )
    conf = mol.GetConformer()
    assign_ad4_types(mol)
    rot_pairs = find_rotatables(mol)
    comp_atoms, children, root_comp, parent_attach = build_comp_tree(mol, rot_pairs)

    lines: List[str] = []
    lines.append(f"REMARK  Name {title}\n")
    lines.append("ROOT\n")

    next_serial = 1
    serial_map: Dict[int, int] = {}
    branch_list: List[Tuple[int, int]] = []

    def print_component(cid: int, attach: Optional[Tuple[int, int]]):
        nonlocal next_serial
        local = sorted(comp_atoms[cid])
        if cid in parent_attach:
            anchor_old = parent_attach[cid][1]
            if anchor_old in local:
                local = [anchor_old] + [a for a in local if a != anchor_old]

        branch_emitted = False
        branch_pair: Optional[Tuple[int, int]] = None
        if attach is not None:
            parent_old, child_old = attach
            # find parent component id and choose heavy parent anchor
            parent_cid = None
            for pcid, atoms in comp_atoms.items():
                if parent_old in atoms:
                    parent_cid = pcid
                    break
            if parent_cid is not None:
                parent_anchor = _choose_heavy_in_comp(
                    mol, comp_atoms[parent_cid], parent_old
                )
            else:
                parent_anchor = parent_old
            parent_serial = serial_map.get(parent_anchor)
            child_anchor = _choose_heavy_in_comp(mol, comp_atoms[cid], child_old)
            if (
                child_anchor is not None
                and child_anchor in local
                and parent_serial is not None
                and _heavy(mol, child_anchor)
            ):
                child_serial_expected = (
                    next_serial  # the first atom printed in this component
                )
                lines.append(f"BRANCH {parent_serial} {child_serial_expected}\n")
                branch_emitted = True
                branch_pair = (parent_serial, child_serial_expected)
                branch_list.append(branch_pair)

        # Print atoms:
        for ai in local:
            a = mol.GetAtomWithIdx(ai)
            serial_map[ai] = next_serial
            info = a.GetPDBResidueInfo()
            name = (
                info.GetName().strip()[:4]
                if info
                else f"{a.GetSymbol()}{next_serial%1000:03d}"
            )
            resn = (info.GetResidueName().strip() if info else title) or title
            chain = (info.GetChainId().strip() if info else "A") or "A"
            resnum = (
                int(info.GetResidueNumber())
                if (info and info.GetResidueNumber() is not None)
                else 1
            )
            pos = conf.GetAtomPosition(ai)
            try:
                charge = (
                    float(a.GetProp("_GasteigerCharge"))
                    if a.HasProp("_GasteigerCharge")
                    else 0.0
                )
            except Exception:
                charge = 0.0
            atype = a.GetProp("_AD4TYPE") if a.HasProp("_AD4TYPE") else a.GetSymbol()
            lines.append(
                _fmt_pdbqt_atom_line(
                    next_serial,
                    name,
                    resn,
                    chain,
                    resnum,
                    pos.x,
                    pos.y,
                    pos.z,
                    charge,
                    atype,
                    het=True,
                )
            )
            next_serial += 1

        # Recurse to children and write ENDBRANCH pairs in same order as BRANCH were written
        for child_cid, attach_pair in children.get(cid, []):
            print_component(child_cid, attach_pair)
            if branch_list:
                bp = branch_list.pop(0)
                lines.append(f"ENDBRANCH {bp[0]} {bp[1]}\n")

    print_component(root_comp, None)
    lines.append("ENDROOT\n")
    # torsions = number of BRANCH occurrences we emitted
    torsions_emitted = len(re.findall(r"^BRANCH ", "".join(lines), flags=re.M))
    lines.append(f"TORSDOF {torsions_emitted}\n")

    text = "".join(lines)
    ok, diag = validate_pdbqt_text(text)
    if not ok:
        # Provide diagnostics to the user to debug
        diag_text = "\n".join(diag)
        raise RuntimeError(
            f"PDBQT validation failed:\n{diag_text}\n\nFirst 60 lines of text:\n{''.join(text.splitlines(True)[:60])}"
        )
    out_path.write_text(text)


# ----------------- Load molecule helper (SDF, PDB, PDBQT, SMILES) -----------------
def _pdbqt_to_pdbblock(path: Path) -> str:
    # simplified heuristic: extract coordinates tokens
    lines = path.read_text(errors="ignore").splitlines()
    out_lines = []
    for L in lines:
        if not L.strip():
            continue
        toks = L.split()
        if toks[0] in ("ATOM", "HETATM"):
            # Find three floats that look like coords
            floats = []
            for t in toks:
                try:
                    floats.append(float(t))
                except Exception:
                    pass
            if len(floats) < 3:
                continue
            x, y, z = (
                floats[-6],
                floats[-5],
                floats[-4] if len(floats) >= 6 else floats[-3],
                floats[-2],
                floats[-1],
            )
            # guess fields
            name = toks[2] if len(toks) > 2 else "C"
            resn = toks[3] if len(toks) > 3 else "LIG"
            chain = toks[4] if len(toks) > 4 else "A"
            try:
                resnum = int(toks[5]) if len(toks) > 5 else 1
            except Exception:
                resnum = 1
            out_lines.append(
                f"ATOM  {1:5d} {name:>4s} {resn:>3s} {chain}{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}\n"
            )
    return "".join(out_lines)


def load_any(path: Path) -> Chem.Mol:
    ext = path.suffix.lower()
    mol: Optional[Chem.Mol] = None
    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        mol = next((m for m in suppl if m is not None), None)
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    elif ext == ".mol":
        mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=False)
    elif ext == ".smi":
        smi = path.read_text().strip().split()[0]
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    elif ext == ".pdbqt":
        block = _pdbqt_to_pdbblock(path)
        mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
        if mol is None:
            mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
        if mol is None:
            raise RuntimeError(f"Could not parse coords from {path}")
        if mol.GetNumBonds() == 0:
            # infer bonds by distance
            conf = mol.GetConformer()
            n = mol.GetNumAtoms()
            em = Chem.RWMol()
            for a in mol.GetAtoms():
                em.AddAtom(Chem.Atom(a.GetSymbol()))
            coords = [conf.GetAtomPosition(i) for i in range(n)]
            cov = {
                "H": 0.31,
                "C": 0.76,
                "N": 0.71,
                "O": 0.66,
                "F": 0.57,
                "P": 1.07,
                "S": 1.05,
                "Cl": 1.02,
                "Br": 1.20,
                "I": 1.39,
            }
            for i in range(n):
                ri = cov.get(mol.GetAtomWithIdx(i).GetSymbol(), 0.8)
                for j in range(i + 1, n):
                    rj = cov.get(mol.GetAtomWithIdx(j).GetSymbol(), 0.8)
                    if coords[i].Distance(coords[j]) <= min(ri + rj + 0.45, 1.9):
                        em.AddBond(i, j, Chem.BondType.SINGLE)
            new = em.GetMol()
            new.AddConformer(conf, assignId=True)
            try:
                Chem.SanitizeMol(new)
            except Exception:
                pass
            mol = new
    else:
        raise RuntimeError(f"Unsupported ligand input format: {ext}")
    if mol is None:
        raise RuntimeError(f"Failed to load molecule: {path}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


# ----------------- CLI -----------------
def _cli():
    p = argparse.ArgumentParser(
        description="Prepare flexible ligand PDBQT with strict validation."
    )
    p.add_argument(
        "-l",
        "--ligand",
        required=True,
        help="input ligand (sdf/mol2/mol/pdb/smi/pdbqt)",
    )
    p.add_argument("-o", "--output", required=True, help="output pdbqt")
    return p


def main(argv=None):
    ap = _cli()
    args = ap.parse_args(argv)
    inp = Path(args.ligand)
    outp = Path(args.output)
    try:
        mol = load_any(inp)
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        molH = Chem.AddHs(mol, addCoords=True)
        try:
            AllChem.ComputeGasteigerCharges(molH)
        except Exception:
            pass
        write_strict_pdbqt(molH, outp, title="LIG")
        print(f"[OK] wrote {outp}")
        return 0
    except Exception as e:
        print("[ERROR] prepare_ligand4 failed:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
