"""Freeze a chemically annotated ChEMBL EGFR library, without fitting to docking.

Selects deterministic structure-unique compounds from standardized binding-assay
IC50 records (target CHEMBL203). Median pIC50 is an annotation across heterogeneous
assays, not a homogeneous benchmark ground truth. No annotated variants or
explicit mutant-assay descriptions are retained. All filtering decisions and
raw pages remain auditable. Activity strata balance chemical interpretation;
they do not determine docking or contact-analysis parameters.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

BASE = "https://www.ebi.ac.uk/chembl/api/data/"
QUERY = dict(
    target_chembl_id="CHEMBL203",
    standard_type="IC50",
    standard_relation="=",
    standard_units="nM",
    assay_type="B",
    standard_flag=1,
    order_by="activity_id",
    limit=1000,
)
MUTANT = re.compile(r"mutan|T790M|L858R|C797S|exon\s*(?:19|20|21)|\bdel19\b", re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, path):
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                data = r.read()
            obj = json.loads(data)
            path.write_bytes(data)
            return obj
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--size", type=int, default=1000)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    status = fetch(BASE + "status.json", raw / "chembl_status.json")
    first = fetch(
        BASE + "activity.json?" + urllib.parse.urlencode(QUERY),
        raw / "activities_00000.json",
    )
    total = first["page_meta"]["total_count"]
    offsets = list(range(0, total, 1000))

    def page(offset):
        q = dict(QUERY, offset=offset)
        url = BASE + "activity.json?" + urllib.parse.urlencode(q)
        return offset, url, fetch(url, raw / f"activities_{offset:05d}.json")

    pages = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset, url, obj in pool.map(page, offsets):
            pages.append((offset, url, obj))
            print("Fetched", offset, len(obj["activities"]), flush=True)
    by_molecule = defaultdict(list)
    excluded = Counter()
    activities = []
    for _, _, obj in sorted(pages):
        for a in obj["activities"]:
            activities.append(a)
            if a.get("assay_variant_mutation") or MUTANT.search(
                a.get("assay_description") or ""
            ):
                excluded["annotated_or_explicit_variant_activity"] += 1
                continue
            if a.get("data_validity_comment"):
                excluded["activity_validity_flag"] += 1
                continue
            try:
                value = float(a["standard_value"])
            except (TypeError, ValueError):
                excluded["missing_numeric_IC50"] += 1
                continue
            if value <= 0 or not math.isfinite(value):
                excluded["nonpositive_IC50"] += 1
                continue
            if not a.get("canonical_smiles"):
                excluded["missing_smiles"] += 1
                continue
            by_molecule[a["molecule_chembl_id"]].append(a)
    candidates = []
    compound_exclusions = []
    seen = set()
    for mid, records in sorted(by_molecule.items()):
        smiles = records[0]["canonical_smiles"]
        mol = Chem.MolFromSmiles(smiles)
        reason = None
        if mol is None:
            reason = "unparsable_smiles"
        elif len(Chem.GetMolFrags(mol)) != 1:
            reason = "multifragment_input"
        elif any(
            a.GetAtomicNum() not in {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
            for a in mol.GetAtoms()
        ):
            reason = "outside_declared_organic_elements"
        elif not 12 <= mol.GetNumHeavyAtoms() <= 50:
            reason = "outside_12_50_heavy_atoms"
        elif Lipinski.NumRotatableBonds(mol) > 12:
            reason = "above_12_rotatable_bonds"
        if reason:
            compound_exclusions.append(dict(id=mid, reason=reason))
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canonical in seen:
            compound_exclusions.append(
                dict(id=mid, reason="duplicate_exact_canonical_structure")
            )
            continue
        seen.add(canonical)
        pic = [9 - math.log10(float(a["standard_value"])) for a in records]
        median = statistics.median(pic)
        # Conflict is disclosed rather than resolved by choosing a favorable assay.
        stratum = "potent" if median >= 7 else "weak" if median <= 5 else "intermediate"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        candidates.append(
            dict(
                id=mid,
                smiles=canonical,
                median_pic50=median,
                activity_stratum=stratum,
                n_activity_records=len(records),
                activity_pic50_range=max(pic) - min(pic),
                activity_ids=[a["activity_id"] for a in records],
                assay_ids=sorted({a["assay_chembl_id"] for a in records}),
                document_ids=sorted({a["document_chembl_id"] for a in records}),
                heavy_atoms=mol.GetNumHeavyAtoms(),
                rotatable_bonds=Lipinski.NumRotatableBonds(mol),
                molecular_weight=Descriptors.MolWt(mol),
                logp=Descriptors.MolLogP(mol),
                scaffold=scaffold,
                selection_key=sha(("prodock-egfr-v1:" + canonical).encode()),
            )
        )
    pools = {
        k: sorted(
            [x for x in candidates if x["activity_stratum"] == k],
            key=lambda x: x["selection_key"],
        )
        for k in ["potent", "intermediate", "weak"]
    }
    quotas = {
        "potent": args.size * 4 // 10,
        "intermediate": args.size * 4 // 10,
        "weak": args.size * 2 // 10,
    }
    selected = []
    for k, quota in quotas.items():
        selected.extend(pools[k][:quota])
    selected_ids = {x["id"] for x in selected}
    remaining = sorted(
        [x for x in candidates if x["id"] not in selected_ids],
        key=lambda x: x["selection_key"],
    )
    selected.extend(remaining[: max(0, args.size - len(selected))])
    if len(selected) != args.size:
        raise RuntimeError(f"Only {len(selected)} eligible compounds")
    # Interleave strata so pilot prefixes do not accidentally contain one activity class.
    buckets = {k: [x for x in selected if x["activity_stratum"] == k] for k in pools}
    interleaved = []
    while any(buckets.values()):
        for k in ["potent", "intermediate", "weak"]:
            if buckets[k]:
                interleaved.append(buckets[k].pop(0))
    library = dict(
        source="ChEMBL activity API",
        target="CHEMBL203",
        ligands=interleaved,
        interpretation=(
            "No annotated variants retained; absence of annotation is not proof of wild "
            "type. Median IC50 across heterogeneous binding assays is descriptive, not "
            "homogeneous validation ground truth."
        ),
        chemical_question=(
            "How protein-structure choice changes EGFR pose/hinge-contact/compound "
            "priorities, and how these relate to size and measured activity annotations."
        ),
        protocol={
            "size": args.size,
            "seed": "prodock-egfr-v1",
            "sampling_quotas": quotas,
            "filters": (
                "single-fragment supplied chemical state, organic elements, 12–50 heavy "
                "atoms, <=12 rotatable bonds; no docking-based selection"
            ),
            "no_activity_fitted_weights": True,
            "no_prospective_validation": True,
        },
        counts=dict(
            raw_activities=len(activities),
            raw_unique_activity_ids=len({a["activity_id"] for a in activities}),
            eligible_molecules=len(candidates),
            selected=len(interleaved),
            selected_strata=dict(Counter(x["activity_stratum"] for x in interleaved)),
            selected_unique_scaffolds=len({x["scaffold"] for x in interleaved}),
            activity_exclusions=dict(excluded),
        ),
        compound_exclusions=compound_exclusions,
        chembl_status=status,
        rdkit_version=rdBase.rdkitVersion,
        pages=[
            dict(
                offset=o,
                url=u,
                sha256=sha((raw / f"activities_{o:05d}.json").read_bytes()),
                count=len(obj["activities"]),
            )
            for o, u, obj in sorted(pages)
        ],
    )
    data = (json.dumps(library, indent=2) + "\n").encode()
    (out / "library.json").write_bytes(data)
    (out / "library.sha256").write_text(sha(data) + "  library.json\n")
    print(json.dumps(library["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
