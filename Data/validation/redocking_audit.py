"""Audit local redocking in a protein-defined coordinate frame, without ligand fitting.

This reads the original files and database only. It does not regenerate docking
or infer success for missing cases. Historical exclusions precede measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prodock.database.serialization import deserialize_mol  # noqa: E402

AA = dict(
    zip(
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ],
        "ARNDCQEGHILKMFPSTWYV",
    )
)
AA.update(MSE="M", HIE="H", HID="H", HIP="H", CYX="C")
CASES = [
    ("4WKQ", "IRE", True, None),
    ("1M17", "AQ4", True, None),
    ("4ZAU", "YY3", True, None),
    ("2ITY", "IRE", False, None),
    (
        "4I23",
        "1C9",
        False,
        "Incomplete experimental ligand: not an eligible whole-ligand RMSD reference.",
    ),
    (
        "4G5J",
        "0WM",
        False,
        (
            "Selected 0WM is an off-site unreacted copy; true binding-site afatinib is covalent. "
            "Excluded from noncovalent native-pose recovery."
        ),
    ),
]
ENGINES = ["smina", "vina", "qvina", "qvina-w"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cas(path, chain="A"):
    data = []
    seen = set()
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if (
            line[:6].strip() not in {"ATOM", "HETATM"}
            or line[12:16].strip() != "CA"
            or line[21:22] != chain
            or line[16:17] not in {" ", "A"}
        ):
            continue
        name = line[17:20].strip()
        key = (chain, line[22:27].strip())
        if name not in AA or key in seen:
            continue
        seen.add(key)
        data.append(
            dict(
                residue=name,
                residue_id=key[1],
                code=AA[name],
                xyz=[float(line[30:38]), float(line[38:46]), float(line[46:54])],
            )
        )
    return data


def sequence_pairs(left, right):
    """Global sequence alignment: match +2, mismatch -1, linear gap -2."""
    a = "".join(x["code"] for x in left)
    b = "".join(x["code"] for x in right)
    score = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    trace = np.zeros_like(score, dtype=np.int8)
    score[:, 0] = np.arange(len(a) + 1) * -2
    score[0, :] = np.arange(len(b) + 1) * -2
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            candidates = (
                int(score[i - 1, j - 1]) + (2 if a[i - 1] == b[j - 1] else -1),
                int(score[i - 1, j]) - 2,
                int(score[i, j - 1]) - 2,
            )
            choice = max(range(3), key=candidates.__getitem__)
            score[i, j] = candidates[choice]
            trace[i, j] = choice
    i, j = len(a), len(b)
    pairs = []
    mismatches = 0
    while i or j:
        choice = trace[i, j] if i and j else (1 if i else 2)
        if choice == 0:
            if a[i - 1] == b[j - 1]:
                pairs.append((i - 1, j - 1))
            else:
                mismatches += 1
            i -= 1
            j -= 1
        elif choice == 1:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs)), mismatches


def transform(mobile, reference):
    x = np.asarray(mobile, float)
    y = np.asarray(reference, float)
    mx = x.mean(0)
    my = y.mean(0)
    u, _, vh = np.linalg.svd((x - mx).T @ (y - my))
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ vh)
    rotation = u @ sign @ vh
    translation = my - mx @ rotation
    return (
        rotation,
        translation,
        float(np.sqrt(np.mean(np.sum((x @ rotation + translation - y) ** 2, axis=1)))),
    )


def raw_ligand(path, code, chain="A"):
    groups = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if (
            line.startswith("HETATM")
            and line[17:20].strip() == code
            and line[21:22] == chain
            and line[16:17] in {" ", "A"}
        ):
            groups.setdefault(line[22:27].strip(), []).append(line)
    candidates = []
    for resid, lines in groups.items():
        mol = Chem.MolFromPDBBlock(
            "\n".join(lines) + "\nEND\n", sanitize=False, removeHs=False
        )
        if mol is None:
            continue
        mol = Chem.RemoveHs(mol, sanitize=False)
        candidates.append((resid, mol))
    return candidates


def coordinate_distance(mol, other):
    if mol.GetNumAtoms() != other.GetNumAtoms():
        return None
    x = mol.GetConformer().GetPositions()
    y = other.GetConformer().GetPositions()
    cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    for i, atom in enumerate(mol.GetAtoms()):
        for j, b in enumerate(other.GetAtoms()):
            if atom.GetAtomicNum() != b.GetAtomicNum():
                cost[i, j] = 1e6
    left, right = linear_sum_assignment(cost)
    return float(cost[left, right].max())


def pdbqt_molecule(path, rank):
    lines = []
    current = 1
    for line in Path(path).read_text().splitlines():
        if line.startswith("MODEL"):
            current = int(line.split()[1])
            continue
        if line.startswith("ENDMDL") and current == rank:
            break
        if current == rank and line[:6].strip() in {"ATOM", "HETATM"}:
            atom_type = line.split()[-1]
            element = {"A": "C", "NA": "N", "OA": "O", "SA": "S", "HD": "H"}.get(
                atom_type, atom_type
            )
            if element == "H":
                continue
            line = line[:76] + element.rjust(2)
            lines.append(line)
    return Chem.MolFromPDBBlock(
        "\n".join(lines) + "\nEND\n", sanitize=False, removeHs=False
    )


def heavy_atoms(path, chain="A"):
    atoms = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if (
            line[:6].strip() not in {"ATOM", "HETATM"}
            or line[21:22] != chain
            or line[16:17] not in {" ", "A"}
        ):
            continue
        is_pdbqt = Path(path).suffix.lower() == ".pdbqt"
        atom_type = line.split()[-1] if is_pdbqt else None
        element = line[76:78].strip() or line[12:16].strip()[0]
        if is_pdbqt:
            element = {
                "A": "C",
                "NA": "N",
                "OA": "O",
                "SA": "S",
                "HD": "H",
                "HS": "H",
            }.get(atom_type, atom_type)
        if element.upper() == "H":
            continue
        atoms.append(
            dict(
                serial=int(line[6:11]),
                element=element,
                atom_type=atom_type,
                charge=float(line[70:76]) if is_pdbqt else None,
                atom=line[12:16].strip(),
                residue=line[17:20].strip(),
                residue_id=line[22:27].strip(),
                xyz=[float(line[30:38]), float(line[38:46]), float(line[46:54])],
            )
        )
    return atoms


def prepared_diagnostics(
    raw_atoms, prepared_path, prepared_reference_xyz, raw_reference_xyz, case
):
    prepared_atoms = heavy_atoms(prepared_path)
    pdbqt_atoms = heavy_atoms(Path(prepared_path).with_suffix(".pdbqt"))

    def key(atom):
        return (atom["residue_id"], atom["residue"], atom["atom"])

    lookup = {key(atom): atom for atom in prepared_atoms}
    qt_lookup = {key(atom): atom for atom in pdbqt_atoms}
    common = sorted(set(lookup) & set(qt_lookup))
    deltas = [
        np.linalg.norm(np.array(lookup[k]["xyz"]) - qt_lookup[k]["xyz"]) for k in common
    ]
    protein_raw = [a for a in raw_atoms if a["residue"] in AA or a["residue"] == "CSX"]

    def clashes(atoms, ligand_xyz):
        xyz = np.asarray([a["xyz"] for a in atoms])
        distances = np.linalg.norm(xyz[:, None, :] - ligand_xyz[None, :, :], axis=2)
        pairs = []
        for ai, li in np.argwhere(distances < 2.0):
            pairs.append(
                dict(
                    protein_residue=atoms[ai]["residue"],
                    residue_id=atoms[ai]["residue_id"],
                    protein_atom=atoms[ai]["atom"],
                    ligand_atom=int(li),
                    distance_A=float(distances[ai, li]),
                )
            )
        return dict(
            minimum_distance_A=float(distances.min()),
            pairs_below_1p5A=int(np.sum(distances < 1.5)),
            pairs_below_2A=int(np.sum(distances < 2.0)),
            severe_pairs=pairs,
        )

    raw_clashes = clashes(protein_raw, raw_reference_xyz)
    prepared_clashes = clashes(prepared_atoms, prepared_reference_xyz)
    qt_clashes = clashes(pdbqt_atoms, prepared_reference_xyz)
    mapped = {p["raw_residue"]: p["prepared_residue"] for p in case["atom_mapping"]}
    rotation = np.asarray(case["rotation"])
    translation = np.asarray(case["translation"])
    sidechain_changes = []
    for raw_residue, prep_residue in mapped.items():
        old_atoms = {
            a["atom"]: a for a in protein_raw if a["residue_id"] == raw_residue
        }
        new_atoms = {
            a["atom"]: a for a in prepared_atoms if a["residue_id"] == prep_residue
        }
        if not old_atoms:
            continue
        old_xyz = np.asarray([a["xyz"] for a in old_atoms.values()])
        if (
            np.linalg.norm(
                old_xyz[:, None, :] - raw_reference_xyz[None, :, :], axis=2
            ).min()
            > 6.0
        ):
            continue
        matched = sorted(set(old_atoms) & set(new_atoms))
        chain_names = [
            name for name in matched if name not in {"N", "CA", "C", "O", "OXT"}
        ]
        sidechain_rmsd = None
        if chain_names:
            old = np.asarray([old_atoms[name]["xyz"] for name in chain_names])
            new = (
                np.asarray([new_atoms[name]["xyz"] for name in chain_names]) @ rotation
                + translation
            )
            sidechain_rmsd = float(np.sqrt(np.mean(np.sum((new - old) ** 2, axis=1))))
        sidechain_changes.append(
            dict(
                raw_residue=raw_residue,
                prepared_residue=prep_residue,
                residue=next(iter(old_atoms.values()))["residue"],
                sidechain_rmsd_A=sidechain_rmsd,
                raw_only_atom_names=sorted(set(old_atoms) - set(new_atoms)),
                prepared_only_atom_names=sorted(set(new_atoms) - set(old_atoms)),
            )
        )
    hinge_raw = "769" if case["pdb"] == "1M17" else "793"
    hinge_prepared = mapped.get(hinge_raw)
    return dict(
        prepared_pdb_heavy_atoms=len(prepared_atoms),
        prepared_pdbqt_heavy_atoms=len(pdbqt_atoms),
        common_named_heavy_atoms=len(common),
        max_common_coordinate_difference_A=float(max(deltas)) if deltas else None,
        common_coordinate_rmsd_A=(
            float(np.sqrt(np.mean(np.square(deltas)))) if deltas else None
        ),
        pdb_atoms_missing_from_pdbqt=[
            list(k) for k in sorted(set(lookup) - set(qt_lookup))
        ],
        pdbqt_extra_named_atoms=[list(k) for k in sorted(set(qt_lookup) - set(lookup))],
        native_raw_protein_clashes=raw_clashes,
        native_prepared_protein_clashes=prepared_clashes,
        native_prepared_pdbqt_clashes=qt_clashes,
        pocket_sidechain_changes=sidechain_changes,
        hinge_pdbqt_types=[
            a
            for a in pdbqt_atoms
            if a["residue_id"] == hinge_prepared and a["atom"] in {"N", "O", "CA", "C"}
        ],
    )


def validate_frame_controls():
    mobile = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.3, 0.5, 4.0]]
    )
    known_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = mobile @ known_rotation + np.array([10.0, -4.0, 2.0])
    rotation, translation, residual = transform(mobile, target)
    assert residual < 1e-12 and abs(np.linalg.det(rotation) - 1.0) < 1e-12
    assert np.max(np.abs(mobile @ rotation + translation - target)) < 1e-12
    matches, mismatches = sequence_pairs(
        [dict(code=x) for x in "ACD"], [dict(code=x) for x in "AXCD"]
    )
    assert matches == [(0, 0), (1, 2), (2, 3)] and mismatches == 0
    return dict(
        known_rotation_translation=True,
        proper_rotation_no_reflection=True,
        sequence_insertion_mapping=True,
    )


def chemistry_context(cases, output):
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig

    by_id = {c["pdb"]: c for c in cases}
    anchor = cas(ROOT / by_id["1M17"]["raw_path"])
    feature_factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    details = []
    specs = []
    for pdb in ["1M17", "4ZAU", "2ITY", "4WKQ"]:
        case = by_id[pdb]
        raw = ROOT / case["raw_path"]
        prepared = ROOT / case["prepared_path"]
        atoms = heavy_atoms(raw)
        prepared_atoms = heavy_atoms(prepared)
        links = [
            line for line in raw.read_text().splitlines() if line.startswith("LINK")
        ]
        metadata_path = output / f"rcsb_{pdb}_entry.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        polymer_path = output / f"rcsb_{pdb}_polymer_entity1.json"
        polymer_metadata = (
            json.loads(polymer_path.read_text()) if polymer_path.exists() else {}
        )
        reference = Chem.SDMolSupplier(
            str(output / f"{pdb}_native_experimental.sdf"), removeHs=True
        )[0]
        acceptors = {
            i
            for feature in feature_factory.GetFeaturesForMol(reference)
            if feature.GetFamily() == "Acceptor"
            for i in feature.GetAtomIds()
        }
        raw_hinge = "769" if pdb == "1M17" else "793"
        hinge = [
            a for a in atoms if a["residue_id"] == raw_hinge and a["residue"] == "MET"
        ]
        hinge_n = next(a for a in hinge if a["atom"] == "N")
        min_acceptor = min(
            float(
                np.linalg.norm(
                    np.array(hinge_n["xyz"])
                    - reference.GetConformer().GetPositions()[i]
                )
            )
            for i in acceptors
        )
        current = cas(raw)
        pairs, mismatches = sequence_pairs(anchor, current)
        equivalences = [
            dict(
                anchor_residue=anchor[i]["residue_id"],
                candidate_residue=current[j]["residue_id"],
                residue=anchor[i]["residue"],
            )
            for i, j in pairs
        ]
        hinge_map = next(
            x for x in case["atom_mapping"] if x["raw_residue"] == raw_hinge
        )
        modified = [
            a
            for a in atoms
            if a["residue"] not in AA
            and a["residue"] not in {case["ligand_code"], "HOH", "NA", "CL"}
        ]
        unique_mod = {a["residue_id"]: a["residue"] for a in modified}
        modified_loss = []
        for residue_id, residue in unique_mod.items():
            present = any(a["residue"] == residue for a in prepared_atoms)
            modified_loss.append(
                dict(
                    raw_residue_id=residue_id,
                    raw_residue=residue,
                    preserved_same_residue_type=present,
                )
            )
        ligand_serials = {
            a["serial"] for a in atoms if a["residue"] == case["ligand_code"]
        }
        protein_serials = {a["serial"] for a in atoms if a["residue"] in AA}
        cross = []
        for line in raw.read_text().splitlines():
            if not line.startswith("CONECT"):
                continue
            ids = [
                int(line[i : i + 5])
                for i in range(6, len(line), 5)
                if line[i : i + 5].strip().isdigit()
            ]
            if ids and ids[0] in ligand_serials:
                cross.extend((ids[0], x) for x in ids[1:] if x in protein_serials)
        sg_candidates = [
            a
            for a in atoms
            if a["atom"] == "SG"
            and a["residue_id"] in ({"773"} if pdb == "1M17" else {"797"})
        ]
        warhead = reference.GetSubstructMatch(Chem.MolFromSmarts("C=CC(=O)N"))
        sg_beta = (
            float(
                np.linalg.norm(
                    np.array(sg_candidates[0]["xyz"])
                    - reference.GetConformer().GetPositions()[warhead[0]]
                )
            )
            if sg_candidates and warhead
            else None
        )
        detail = dict(
            pdb=pdb,
            rcsb_url=f"https://www.rcsb.org/structure/{pdb}",
            title=metadata.get("struct", {}).get("title"),
            resolution=metadata.get("rcsb_entry_info", {}).get("resolution_combined"),
            ligand_code=case["ligand_code"],
            official_polymer_mutation_count=polymer_metadata.get("entity_poly", {}).get(
                "rcsb_mutation_count"
            ),
            official_nonstandard_monomers=polymer_metadata.get("entity_poly", {}).get(
                "rcsb_non_std_monomers", []
            ),
            official_metadata_sha256=(
                digest(metadata_path) if metadata_path.exists() else None
            ),
            ligand_heavy_atoms=reference.GetNumAtoms(),
            ligand_smiles=Chem.MolToSmiles(reference),
            hinge_raw=raw_hinge,
            hinge_prepared=hinge_map["prepared_residue"],
            hinge_backbone_N_to_nearest_ligand_acceptor_A=min_acceptor,
            ligand_protein_conect_bonds=cross,
            link_records=links,
            modified_residue_treatment=modified_loss,
            acrylamide_warhead=bool(warhead),
            Cys_sulfur_to_acrylamide_beta_carbon_A=sg_beta,
            anchor_sequence_mapping=equivalences,
        )
        if pdb == "4WKQ":
            detail["critical_preparation_issue"] = (
                "Raw CSX797 (oxidized cysteine) is removed by the cached preparation, leaving a missing "
                "pocket residue; do not use cached protein without a declared repair policy."
            )
        if pdb == "4ZAU":
            detail["chemical_scope"] = (
                "RCSB identifies wild-type EGFR/AZD9291 (osimertinib). The ligand has an acrylamide "
                "warhead, but this local deposited model has no protein-ligand LINK/CONECT bond. "
                "Noncovalent docking probes recognition geometry only, not covalent reactivity or mutant "
                "selectivity."
            )
        details.append(detail)
        rotation = np.asarray(case["rotation"])
        translation = np.asarray(case["translation"])
        prepared_reference = Chem.Mol(reference)
        prepared_coordinates = (
            reference.GetConformer().GetPositions() - translation
        ) @ rotation.T
        for atom_id, xyz in enumerate(prepared_coordinates):
            prepared_reference.GetConformer().SetAtomPosition(atom_id, xyz)
        prepared_ref_path = output / f"{pdb}_native_prepared_frame.sdf"
        writer = Chem.SDWriter(str(prepared_ref_path))
        writer.write(prepared_reference)
        writer.close()
        detail["experimental_to_prepared_formula"] = (
            "(experimental_xyz - translation) @ rotation.T"
        )
        detail["prepared_to_experimental_rotation"] = case["rotation"]
        detail["prepared_to_experimental_translation"] = case["translation"]
        detail["protein_fit_residual_A"] = case["protein_ca_rmsd_after"]
        detail["prepared_file_diagnostics"] = prepared_diagnostics(
            atoms,
            prepared,
            prepared_coordinates,
            reference.GetConformer().GetPositions(),
            case,
        )
        coords = prepared_coordinates
        center = coords.mean(0)
        size = float(np.max(coords.max(0) - coords.min(0)) + 8.0)
        if pdb in {"1M17", "4ZAU", "2ITY"}:
            specs.append(
                dict(
                    pdb=pdb,
                    prepared_pdb=str(prepared),
                    prepared_pdbqt=str(prepared.with_suffix(".pdbqt")),
                    experimental_reference=str(
                        output / f"{pdb}_native_experimental.sdf"
                    ),
                    ligand_code=case["ligand_code"],
                    box_reference_prepared_frame=str(prepared_ref_path),
                    prepared_to_experimental_rotation=case["rotation"],
                    prepared_to_experimental_translation=case["translation"],
                    experimental_to_prepared_formula="(experimental_xyz - translation) @ rotation.T",
                    protein_fit_residual_A=case["protein_ca_rmsd_after"],
                    center=center.tolist(),
                    size=[size] * 3,
                    box_definition=(
                        "Reference-coordinate centroid; isotropic maximum coordinate span plus 8 A total padding "
                        "(4A each side)."
                    ),
                    hinge_raw=raw_hinge,
                    hinge_prepared=hinge_map["prepared_residue"],
                    gatekeeper_raw="766" if pdb == "1M17" else "790",
                    sha256=dict(
                        prepared_pdb=digest(prepared),
                        prepared_pdbqt=digest(prepared.with_suffix(".pdbqt")),
                        reference=digest(output / f"{pdb}_native_experimental.sdf"),
                    ),
                )
            )
    result = dict(
        cases=details,
        ready_prepared_specs=specs,
        interpretation=(
            "1M17 and 4ZAU permit a two-conformation WT-EGFR noncovalent-recognition comparison with "
            "distinct ligand chemotypes; does not model covalent reaction. 2ITY is a "
            "first-generation gefitinib alternative with lower 3.42 A resolution. 4WKQ cached preparation "
            "omits CSX797 and must not silently enter a chemically curated campaign."
        ),
    )
    (output / "chemical_context.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    db_path = ROOT / "Data/benchmark/runs/redocking/redocking.db"
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    frame_reports = []
    rows = []
    file_manifest = {str(db_path.relative_to(ROOT)): digest(db_path)}
    for pdb, code, configured, exclusion in CASES:
        project = ROOT / f"Data/benchmark/runs/redocking/projects/{pdb}/{pdb}"
        raw = project / "fetched_protein" / f"{pdb.lower()}.pdb"
        prepared = project / "filtered_protein" / f"{pdb}.pdb"
        if not raw.exists():
            project = ROOT / f"Data/benchmark/runs/redocking/{pdb}"
            raw = project / "fetched_protein" / f"{pdb.lower()}.pdb"
            prepared = project / "filtered_protein" / f"{pdb}.pdb"
        if pdb == "4WKQ":
            project = ROOT / "Data/Demo/4WKQ"
            raw = project / "fetched_protein/4wkq.pdb"
            prepared = project / "filtered_protein/4WKQ.pdb"
        report = dict(
            pdb=pdb,
            ligand_code=code,
            configured_panel=configured,
            predeclared_exclusion=exclusion,
            raw_path=str(raw.relative_to(ROOT)),
            prepared_path=str(prepared.relative_to(ROOT)),
        )
        try:
            original = cas(raw)
            mobile = cas(prepared)
            pairs, mismatch = sequence_pairs(mobile, original)
            x = np.asarray([mobile[a]["xyz"] for a, b in pairs])
            y = np.asarray([original[b]["xyz"] for a, b in pairs])
            rot, trans, residual = transform(x, y)
            report.update(
                raw_ca=len(original),
                prepared_ca=len(mobile),
                matched_ca=len(pairs),
                sequence_mismatches=mismatch,
                prepared_sequence_coverage=len(pairs) / len(mobile),
                raw_sequence_coverage=len(pairs) / len(original),
                protein_ca_rmsd_before=float(
                    np.sqrt(np.mean(np.sum((x - y) ** 2, axis=1)))
                ),
                protein_ca_rmsd_after=residual,
                rotation=rot.tolist(),
                translation=trans.tolist(),
                atom_mapping=[
                    dict(
                        prepared_residue=mobile[a]["residue_id"],
                        raw_residue=original[b]["residue_id"],
                        residue=mobile[a]["residue"],
                    )
                    for a, b in pairs
                ],
            )
            frame_ok = (
                len(pairs) >= 100
                and len(pairs) / len(mobile) >= 0.8
                and mismatch / max(1, len(pairs) + mismatch) < 0.05
            )
            report["sequence_frame_correspondence"] = frame_ok
            template_path = (
                ROOT / f"Data/benchmark/runs/redocking/reference/{code}_ideal.sdf"
            )
            template = Chem.SDMolSupplier(str(template_path), removeHs=True)[0]
            candidates = raw_ligand(raw, code)
            complete = []
            for resid, mol in candidates:
                if mol.GetNumAtoms() != template.GetNumAtoms():
                    continue
                try:
                    complete.append(
                        (resid, AllChem.AssignBondOrdersFromTemplate(template, mol))
                    )
                except Exception:
                    continue
            report["bound_ligand_copies"] = [
                dict(residue_id=r, heavy_atoms=m.GetNumAtoms()) for r, m in candidates
            ]
            report["ccd_heavy_atoms"] = template.GetNumAtoms()
            if not complete:
                raise ValueError(
                    "No complete chemically assignable experimental ligand copy"
                )
            # Choose exact chain/residue copy deterministically before evaluating docking.
            resid, reference = sorted(
                complete, key=lambda a: int(a[0]) if a[0].isdigit() else a[0]
            )[0]
            report["selected_bound_ligand_residue"] = resid
            target = args.output / f"{pdb}_native_experimental.sdf"
            writer = Chem.SDWriter(str(target))
            writer.write(reference)
            writer.close()
            report["recovered_reference"] = str(target)
            native_path = (
                ROOT / f"Data/benchmark/runs/redocking/reference/{pdb}_native.sdf"
            )
            if native_path.exists():
                oldref = Chem.SDMolSupplier(str(native_path), removeHs=True)[0]
                report["old_reference_max_element_matched_coordinate_difference"] = (
                    coordinate_distance(reference, oldref)
                )
                file_manifest[str(native_path.relative_to(ROOT))] = digest(native_path)
            for f in [raw, prepared, template_path]:
                file_manifest[str(f.relative_to(ROOT))] = digest(f)
            for engine in ENGINES:
                records = conn.execute(
                    (
                        "SELECT * FROM poses WHERE receptor_id=? AND ligand_id=? AND engine=? AND pose_rank<=? "
                        "ORDER BY pose_rank"
                    ),
                    (pdb, f"native_{pdb}", engine, args.top_k),
                ).fetchall()
                if not records:
                    rows.append(
                        dict(
                            pdb=pdb,
                            engine=engine,
                            pose_rank=1,
                            configured_panel=configured,
                            status="missing_pose",
                            error="No matching original database pose",
                        )
                    )
                    continue
                if exclusion:
                    rows.append(
                        dict(
                            pdb=pdb,
                            engine=engine,
                            pose_rank=1,
                            configured_panel=configured,
                            status="predeclared_exclusion",
                            error=exclusion,
                        )
                    )
                    continue
                for record in records:
                    row = dict(
                        pdb=pdb,
                        engine=engine,
                        pose_rank=record["pose_rank"],
                        configured_panel=configured,
                        pose_db_id=record["pose_db_id"],
                        run_id=record["run_id"],
                    )
                    try:
                        molecule = deserialize_mol(
                            record["mol_blob"],
                            compressed=bool(record["mol_is_compressed"]),
                        )
                        molecule = Chem.RemoveHs(molecule, sanitize=False)
                        if molecule.GetNumAtoms() != reference.GetNumAtoms():
                            raise ValueError("Pose/reference heavy atom counts differ")
                        molecule = AllChem.AssignBondOrdersFromTemplate(
                            reference, molecule
                        )
                        row["raw_fixed_rmsd"] = float(
                            rdMolAlign.CalcRMS(molecule, reference)
                        )
                        moved = Chem.Mol(molecule)
                        conf = moved.GetConformer()
                        for i, xyz in enumerate(
                            moved.GetConformer().GetPositions() @ rot + trans
                        ):
                            conf.SetAtomPosition(i, xyz)
                        row["protein_aligned_fixed_rmsd"] = float(
                            rdMolAlign.CalcRMS(moved, reference)
                        )
                        row["independently_fitted_rmsd"] = float(
                            rdMolAlign.GetBestRMS(Chem.Mol(molecule), reference)
                        )
                        reference_xyz = reference.GetConformer().GetPositions()
                        pose_xyz = moved.GetConformer().GetPositions()
                        row["experimental_reference_centroid"] = json.dumps(
                            reference_xyz.mean(0).tolist()
                        )
                        row["protein_aligned_pose_centroid"] = json.dumps(
                            pose_xyz.mean(0).tolist()
                        )
                        row["centroid_displacement_A"] = float(
                            np.linalg.norm(reference_xyz.mean(0) - pose_xyz.mean(0))
                        )
                        campaign_path = (
                            ROOT
                            / f"Data/benchmark/runs/redocking/projects/{pdb}/campaign.json"
                        )
                        if campaign_path.exists():
                            file_manifest[str(campaign_path.relative_to(ROOT))] = (
                                digest(campaign_path)
                            )
                            campaign = json.loads(campaign_path.read_text())
                            box = campaign["receptors"][0]["box"]
                            program = next(
                                x
                                for x in campaign["receptors"][0]["softwares"]
                                if x["name"] == engine
                            )
                            row["declared_cpu"] = program.get("cpu")
                            row["declared_exhaustiveness"] = program.get(
                                "exhaustiveness"
                            )
                            row["declared_seed"] = program.get("seed")
                            row["declared_box_center"] = json.dumps(box["center"])
                            row["declared_box_size"] = json.dumps(box["size"])
                            lo = np.array(box["center"]) - np.array(box["size"]) / 2
                            hi = np.array(box["center"]) + np.array(box["size"]) / 2
                            row["reference_atoms_inside_box_fraction"] = float(
                                np.mean(
                                    np.all(
                                        (reference_xyz >= lo) & (reference_xyz <= hi),
                                        axis=1,
                                    )
                                )
                            )
                        from rdkit.Chem import ChemicalFeatures
                        from rdkit import RDConfig

                        Chem.SanitizeMol(moved)
                        factory = ChemicalFeatures.BuildFeatureFactory(
                            str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
                        )
                        acceptor_ids = {
                            i
                            for feat in factory.GetFeaturesForMol(moved)
                            if feat.GetFamily() == "Acceptor"
                            for i in feat.GetAtomIds()
                        }
                        protein_atoms = [
                            a for a in heavy_atoms(raw) if a["residue"] in AA
                        ]
                        hinge_residue = "769" if pdb == "1M17" else "793"
                        hinge_atom = next(
                            a
                            for a in protein_atoms
                            if a["residue_id"] == hinge_residue
                            and a["atom"] == "N"
                            and a["residue"] == "MET"
                        )
                        row["hinge_backbone_N_to_pose_acceptor_A"] = (
                            min(
                                float(
                                    np.linalg.norm(
                                        np.array(hinge_atom["xyz"]) - pose_xyz[i]
                                    )
                                )
                                for i in acceptor_ids
                            )
                            if acceptor_ids
                            else None
                        )
                        protein_xyz = np.asarray([a["xyz"] for a in protein_atoms])
                        ref_d = np.linalg.norm(
                            protein_xyz[:, None, :] - reference_xyz[None, :, :], axis=2
                        ).min(1)
                        pose_d = np.linalg.norm(
                            protein_xyz[:, None, :] - pose_xyz[None, :, :], axis=2
                        ).min(1)
                        ref_contacts = {
                            a["residue_id"]
                            for a, d in zip(protein_atoms, ref_d)
                            if d <= 4.0
                        }
                        pose_contacts = {
                            a["residue_id"]
                            for a, d in zip(protein_atoms, pose_d)
                            if d <= 4.0
                        }
                        row["reference_contact_residues_4A"] = json.dumps(
                            sorted(ref_contacts)
                        )
                        row["pose_contact_residues_4A"] = json.dumps(
                            sorted(pose_contacts)
                        )
                        row["contact_residue_jaccard_4A"] = (
                            len(ref_contacts & pose_contacts)
                            / len(ref_contacts | pose_contacts)
                            if ref_contacts | pose_contacts
                            else 0.0
                        )

                        raw_pose = (
                            project
                            / f"results/docked/{engine}/native_{pdb}_docked.pdbqt"
                        )
                        if raw_pose.exists():
                            parsed = pdbqt_molecule(raw_pose, record["pose_rank"])
                            row[
                                "database_vs_output_max_element_coordinate_difference"
                            ] = coordinate_distance(molecule, parsed)
                            file_manifest[str(raw_pose.relative_to(ROOT))] = digest(
                                raw_pose
                            )
                        row["status"] = (
                            "evaluated"
                            if frame_ok
                            else "unverified_protein_correspondence"
                        )
                    except Exception as e:
                        row.update(status="evaluation_failed", error=str(e))
                    rows.append(row)
        except Exception as exc:
            report["error"] = str(exc)
        frame_reports.append(report)
    conn.close()
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output / "pose_results.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ranks = [r for r in rows if r.get("configured_panel") and r.get("pose_rank") == 1]
    evaluated = [r for r in ranks if r["status"] == "evaluated"]
    report = dict(
        protocol=(
            "Read-only existing-artifact audit; protein CA global sequence alignment (+2/-1/gap-2), "
            "proper Kabsch fit, same transform applied to docked ligand; no independent ligand "
            "fitting for recovery. Existing output/DB equality checked where files available."
        ),
        mathematical_controls=validate_frame_controls(),
        expected_configured_rank1=12,
        evaluated_configured_rank1=len(evaluated),
        configured_rank1_below_2A=sum(
            r["protein_aligned_fixed_rmsd"] <= 2 for r in evaluated
        ),
        cases=frame_reports,
        pose_results=rows,
        source_sha256=file_manifest,
        limitations=[
            "Current artifact audit, not a rerun of the submitted complete campaign.",
            (
                "Ligand symmetry mappings use chemically assigned reference template; protein fits use "
                "matching CA residues and report residual deformation."
            ),
            (
                "4G5J and 4I23 exclusions are predeclared structural eligibility decisions; 2ITY retained "
                "irrespective of RMSD."
            ),
            "Prepared receptor and reference identities must be retained with any new rerun.",
        ],
    )
    report["chemical_context"] = chemistry_context(frame_reports, args.output)
    summaries = []
    for pdb, code, configured, exclusion in CASES:
        for engine in ENGINES:
            group = [
                r
                for r in rows
                if r["pdb"] == pdb
                and r["engine"] == engine
                and r["status"] == "evaluated"
            ]
            if group:
                best = min(group, key=lambda x: x["protein_aligned_fixed_rmsd"])
                summaries.append(
                    dict(
                        pdb=pdb,
                        engine=engine,
                        configured_panel=configured,
                        poses_evaluated=len(group),
                        rank1_fixed_rmsd=next(
                            (
                                r["protein_aligned_fixed_rmsd"]
                                for r in group
                                if r["pose_rank"] == 1
                            ),
                            None,
                        ),
                        best_of_stored_fixed_rmsd=best["protein_aligned_fixed_rmsd"],
                        best_stored_rank=best["pose_rank"],
                        rank1_centroid_displacement_A=next(
                            (
                                r["centroid_displacement_A"]
                                for r in group
                                if r["pose_rank"] == 1
                            ),
                            None,
                        ),
                        rank1_hinge_acceptor_distance_A=next(
                            (
                                r["hinge_backbone_N_to_pose_acceptor_A"]
                                for r in group
                                if r["pose_rank"] == 1
                            ),
                            None,
                        ),
                    )
                )
    report["per_program_summary"] = summaries
    (args.output / "per_program_summary.csv").write_text("")
    if summaries:
        with (args.output / "per_program_summary.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0]))
            w.writeheader()
            w.writerows(summaries)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "expected_configured_rank1",
                    "evaluated_configured_rank1",
                    "configured_rank1_below_2A",
                ]
            },
            indent=2,
        )
    )
    for c in frame_reports:
        print(
            c["pdb"],
            c.get("matched_ca"),
            c.get("protein_ca_rmsd_after"),
            c.get("error"),
        )


if __name__ == "__main__":
    main()
