"""LoLO Blue: interpret a chemically annotated EGFR campaign without redocking.

Only actual rank-1 scores and exported ProLIF events are analyzed. The fixed
protocol is stored in Data/validation/protocols/chemical_usecase.txt.
ChEMBL IC50 strata are heterogeneous descriptive annotations, not a homogeneous
benchmark; no labels fit weights, choose thresholds, or select shortlist examples.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AA = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
    )
)
RADII = (5.0, 6.0, 8.0)
PRIMARY_ENGINE = "smina"
HBOND_TYPES = {"HBAcceptor", "HBDonor", "HBond", "HydrogenBond"}
STRATA = ("potent", "intermediate", "weak")
COLORS = {"potent": "#2166ac", "intermediate": "#777777", "weak": "#b35806"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
    )


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def correlation(x, y):
    a = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 3 or a.x.nunique() < 2 or a.y.nunique() < 2:
        return {"n": len(a), "spearman_rho": None}
    return {"n": len(a), "spearman_rho": number(spearmanr(a.x, a.y).statistic)}


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def parse_residue(value):
    """Normalize ProLIF identifiers such as MET98.A and PDB insertion codes."""
    if pd.isna(value):
        return None
    match = re.fullmatch(r"([A-Za-z]{3})(-?\d+)([A-Za-z]?)\.?(.*)", str(value).strip())
    if not match:
        return None
    name, rid, insertion, chain = match.groups()
    return name.upper(), int(rid), insertion, chain or "A"


@dataclass
class Residue:
    name: str
    number: int
    insertion: str
    chain: str
    coords: list

    @property
    def key(self):
        return f"{self.name}{self.number}{self.insertion}.{self.chain}"

    @property
    def identity(self):
        return self.name, self.number, self.insertion, self.chain


def protein_residues(path):
    residues = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16:17] not in {" ", "A"}:
            continue
        name = line[17:20].strip()
        element = line[76:78].strip() or line[12:16].strip()[0]
        if name not in AA or element in {"H", "D"}:
            continue
        key = (name, int(line[22:26]), line[26:27].strip(), line[21:22].strip() or "A")
        if key not in residues:
            residues[key] = Residue(*key, coords=[])
        residues[key].coords.append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
    result = list(residues.values())
    if not result:
        raise ValueError(f"No standard protein heavy atoms: {path}")
    # This campaign uses one selected protein chain. Avoid silently merging
    # unrelated chains into the sequence correspondence.
    chains = defaultdict(list)
    for residue in result:
        chains[residue.chain].append(residue)
    chain = sorted(chains, key=lambda k: (-len(chains[k]), k))[0]
    return chains[chain]


def sequence_mapping(anchor, candidate):
    """Global sequence correspondence; only identical residue pairs are mapped."""
    a = "".join(AA[r.name] for r in anchor)
    b = "".join(AA[r.name] for r in candidate)
    scores = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int16)
    trace = np.zeros_like(scores, dtype=np.uint8)
    scores[:, 0] = -2 * np.arange(len(a) + 1)
    scores[0, :] = -2 * np.arange(len(b) + 1)
    trace[1:, 0], trace[0, 1:] = 1, 2
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            values = (
                scores[i - 1, j - 1] + (2 if a[i - 1] == b[j - 1] else -1),
                scores[i - 1, j] - 2,
                scores[i, j - 1] - 2,
            )
            best = int(np.argmax(values))
            scores[i, j], trace[i, j] = values[best], best
    i, j = len(a), len(b)
    mapping, aligned = {}, 0
    while i or j:
        step = trace[i, j]
        if i and j and step == 0:
            aligned += 1
            if a[i - 1] == b[j - 1]:
                mapping[candidate[j - 1].identity] = anchor[i - 1].key
            i, j = i - 1, j - 1
        elif i and (j == 0 or step == 1):
            i -= 1
        else:
            j -= 1
    identity = len(mapping) / aligned if aligned else 0
    coverage = len(mapping) / min(len(a), len(b))
    if identity < 0.90 or coverage < 0.80:
        raise ValueError(
            f"Protein correspondence inadequate: identity={identity}, coverage={coverage}"
        )
    return mapping, {
        "aligned_pairs": aligned,
        "identical_pairs": len(mapping),
        "identity": identity,
        "coverage_of_shorter_sequence": coverage,
    }


def pockets(context, proteins):
    specs = {
        s["pdb"]: s for s in context["ready_prepared_specs"] if s["pdb"] in proteins
    }
    missing = set(proteins) - set(specs)
    if missing:
        raise ValueError(f"Protein context absent: {missing}")
    ordered = sorted(proteins)
    residues = {rid: protein_residues(specs[rid]["prepared_pdb"]) for rid in ordered}
    anchor = ordered[0]
    mapping, stats, native, hinge = {}, {}, {}, {}
    for rid in ordered:
        mapping[rid], stats[rid] = sequence_mapping(residues[anchor], residues[rid])
        if "box_reference_prepared_frame" not in specs[rid]:
            raise ValueError(f"Verified prepared-frame reference required for {rid}")
        ref = Path(specs[rid]["box_reference_prepared_frame"])
        if not ref.is_absolute():
            ref = ROOT / ref
        mols = [m for m in Chem.SDMolSupplier(str(ref), removeHs=True) if m is not None]
        if len(mols) != 1:
            raise ValueError(f"Expected one reference molecule: {ref}")
        tree = cKDTree(mols[0].GetConformer().GetPositions())
        nearest = {
            r.identity: float(np.min(tree.query(np.asarray(r.coords))[0]))
            for r in residues[rid]
        }
        native[rid] = {
            radius: {
                mapping[rid][key]
                for key, distance in nearest.items()
                if distance <= radius and key in mapping[rid]
            }
            for radius in RADII
        }
        candidates = [
            r
            for r in residues[rid]
            if r.name == "MET" and str(r.number) == str(specs[rid]["hinge_prepared"])
        ]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous hinge mapping for {rid}")
        hinge[rid] = candidates[0].identity
        stats[rid]["hinge_prepared"] = candidates[0].key
        stats[rid]["hinge_common"] = mapping[rid].get(candidates[0].identity)
        stats[rid]["nearest_reference_heavy_atom_to_hinge_residue_A"] = nearest[
            candidates[0].identity
        ]
        stats[rid]["reference_sha256"] = digest(ref)
        stats[rid]["prepared_to_experimental_protein_fit_residual_A"] = specs[rid].get(
            "protein_fit_residual_A"
        )
        stats[rid][
            "reference_frame"
        ] = "Experimental ligand transformed by the independently verified protein fit."
        stats[rid]["prepared_protein_sha256"] = digest(specs[rid]["prepared_pdb"])
    if len({stats[r]["hinge_common"] for r in ordered}) != 1:
        raise ValueError("Sequence mapping does not reconcile hinge identities")
    common = {
        radius: set().union(*(native[rid][radius] for rid in ordered))
        for radius in RADII
    }
    report = {
        "anchor_protein": anchor,
        "radii_A": list(RADII),
        "sequence_correspondence": stats,
        "pocket_method": "Union of native-reference heavy-atom neighborhoods in common sequence coordinates.",
        "geometry_caveat": (
            "References use independently verified protein rigid transforms, never ligand-to-pose fits. "
            "Residual nonrigid preparation deformation remains and is reported per protein."
        ),
        "native_pockets": {
            rid: {str(r): sorted(v) for r, v in d.items()} for rid, d in native.items()
        },
        "common_pockets": {str(r): sorted(v) for r, v in common.items()},
        "residue_correspondence": {
            rid: {f"{k[0]}{k[1]}{k[2]}.{k[3]}": v for k, v in m.items()}
            for rid, m in mapping.items()
        },
    }
    return mapping, native, common, hinge, report


def event_payloads(campaign):
    table = read_csv(campaign / "interaction_events.csv")
    errors_path = campaign / "analysis_counts.json"
    counts = json.loads(errors_path.read_text()) if errors_path.exists() else {}
    failed = set()
    for error in counts.get("interaction_errors", []):
        if isinstance(error, dict):
            failed.update(str(x) for x in error.get("batch_pose_ids", []))
    events, present = defaultdict(list), set()
    if "interaction_events_json" in table:
        for row in table.itertuples(index=False):
            pose = str(row.pose_id)
            present.add(pose)
            payload = getattr(row, "interaction_events_json", None)
            if isinstance(payload, str) and payload.strip() not in {"", "null", "None"}:
                decoded = json.loads(payload)
                if decoded:
                    if not isinstance(decoded, list):
                        raise ValueError(f"Unexpected event payload for {pose}")
                    events[pose].extend(decoded)
    elif {"pose_id", "protein_residue", "interaction"}.issubset(table.columns):
        for pose, frame in table.groupby("pose_id"):
            events[str(pose)] = frame.to_dict("records")
            present.add(str(pose))
    elif not table.empty:
        raise ValueError(f"Unknown interaction-event columns: {list(table.columns)}")
    known_complete = bool(counts.get("rank1_only_interactions"))
    success = {p for p in present if p not in failed and (known_complete or events[p])}
    return (
        events,
        success,
        {
            "pose_rows_present": len(present),
            "known_successful_profiled_poses": len(success),
            "failed_pose_ids": sorted(failed),
            "profiling_completion_record_present": known_complete,
        },
    )


def annotate(table, library, mapping, native, common, hinge, events, profiled):
    metadata = {r["id"]: r for r in library["ligands"]}
    pattern = Chem.MolFromSmarts("C=CC(=O)N")
    table = table.copy()
    table["score"] = pd.to_numeric(
        table.get("affinity", table.get("score")), errors="coerce"
    )
    table = table[table.pose_rank.eq(1) & table.score.notna()].copy()
    if table.duplicated(["receptor_id", "ligand_id", "engine"]).any():
        raise ValueError(
            "Duplicate rank-1 channel identity; resolve explicitly upstream"
        )
    extra = []
    for row in table.itertuples(index=False):
        source = metadata.get(row.ligand_id)
        if source is None:
            raise ValueError(f"Ligand not in frozen library: {row.ligand_id}")
        molecule = Chem.MolFromSmiles(source["smiles"])
        if molecule is None:
            raise ValueError(f"Frozen input cannot be parsed: {row.ligand_id}")
        contacts, hinge_contact, hinge_hbond, unmapped = set(), False, False, 0
        for event in events.get(str(row.pose_id), []):
            key = parse_residue(event.get("protein_residue"))
            if key == hinge[row.receptor_id]:
                hinge_contact = True
                hinge_hbond |= event.get("interaction") in HBOND_TYPES
            if key in mapping[row.receptor_id]:
                contacts.add(mapping[row.receptor_id][key])
            else:
                unmapped += 1
        valid = str(row.pose_id) in profiled
        ha = molecule.GetNumHeavyAtoms()
        data = dict(
            smiles=source["smiles"],
            heavy_atoms=ha,
            rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(molecule),
            molecular_weight=Descriptors.MolWt(molecule),
            median_pic50=source.get("median_pic50"),
            activity_stratum=source.get("activity_stratum"),
            n_activity_records=source.get("n_activity_records"),
            activity_pic50_range=source.get("activity_pic50_range"),
            scaffold=source.get("scaffold"),
            acrylamide_substructure=bool(molecule.HasSubstructMatch(pattern)),
            docking_score_per_heavy_atom=-row.score / ha,
            profiled=valid,
            hinge_contact=hinge_contact if valid else np.nan,
            hinge_hbond=hinge_hbond if valid else np.nan,
            mapped_contact_residues=len(contacts) if valid else np.nan,
            unmapped_event_count=unmapped,
            mapped_contacts_json=json.dumps(sorted(contacts)) if valid else None,
        )
        for radius in RADII:
            template = native[row.receptor_id][radius]
            available = common[radius] & set(mapping[row.receptor_id].values())
            union = contacts | template
            prefix = f"{radius:g}A"
            data[f"template_tanimoto_{prefix}"] = (
                len(contacts & template) / len(union) if valid and union else np.nan
            )
            data[f"common_pocket_coverage_{prefix}"] = (
                len(contacts & available) / len(available)
                if valid and available
                else np.nan
            )
            data[f"common_pocket_size_{prefix}"] = len(available)
        extra.append(data)
    additions = pd.DataFrame(extra, index=table.index)
    table = table.drop(
        columns=[c for c in additions if c in table], errors="ignore"
    ).join(additions)
    group = table.groupby(["receptor_id", "engine"])
    table["rank"] = group.score.rank(method="average", ascending=True)
    table["rank_percentile"] = table["rank"] / group.score.transform("size")
    return table


def paired_analysis(table):
    rows, reports = [], []
    proteins = sorted(table.receptor_id.unique())
    for engine in sorted(table.engine.unique()):
        frame = table[table.engine.eq(engine)]
        if len(proteins) != 2:
            continue
        left = frame[frame.receptor_id.eq(proteins[0])].set_index("ligand_id")
        right = frame[frame.receptor_id.eq(proteins[1])].set_index("ligand_id")
        ids = sorted(set(left.index) & set(right.index))
        if not ids:
            continue
        a, b = left.loc[ids], right.loc[ids]
        # Rank complete pairs only, so unequal channel missingness cannot shift
        # the reference population between the two structure summaries.
        ar = a.score.rank(method="average")
        br = b.score.rank(method="average")
        cutoff = math.ceil(len(ids) * 0.10)
        score_cut_a = float(np.sort(a.score.to_numpy())[cutoff - 1])
        score_cut_b = float(np.sort(b.score.to_numpy())[cutoff - 1])
        atop = set(a.index[a.score <= score_cut_a])
        btop = set(b.index[b.score <= score_cut_b])
        for ligand in ids:
            record = a.loc[ligand].to_dict()
            record.update(
                ligand_id=ligand,
                engine=engine,
                protein_a=proteins[0],
                protein_b=proteins[1],
                score_a=float(a.at[ligand, "score"]),
                score_b=float(b.at[ligand, "score"]),
                rank_a=float(ar[ligand]),
                rank_b=float(br[ligand]),
                percentile_a=float(ar[ligand] / len(ids)),
                percentile_b=float(br[ligand] / len(ids)),
                hinge_hbond_a=a.at[ligand, "hinge_hbond"],
                hinge_hbond_b=b.at[ligand, "hinge_hbond"],
                profiled_a=bool(a.at[ligand, "profiled"]),
                profiled_b=bool(b.at[ligand, "profiled"]),
                top_decile_a=ligand in atop,
                top_decile_b=ligand in btop,
            )
            record["mean_percentile"] = (
                record["percentile_a"] + record["percentile_b"]
            ) / 2
            record["worst_percentile"] = max(
                record["percentile_a"], record["percentile_b"]
            )
            record["absolute_percentile_shift"] = abs(
                record["percentile_a"] - record["percentile_b"]
            )
            rows.append(record)
        reports.append(
            dict(
                engine=engine,
                protein_a=proteins[0],
                protein_b=proteins[1],
                paired_compounds=len(ids),
                rank_correlation=correlation(a.score, b.score),
                median_absolute_score_shift=number(
                    np.median(np.abs(a.score - b.score))
                ),
                top_decile_rank_cutoff=cutoff,
                top_a=len(atop),
                top_b=len(btop),
                shared_top=len(atop & btop),
                top_decile_jaccard=(
                    len(atop & btop) / len(atop | btop) if atop | btop else None
                ),
            )
        )
    return pd.DataFrame(rows), reports


def inspection_shortlist(paired):
    if paired.empty:
        return pd.DataFrame(), {}
    primary = paired[paired.engine.eq(PRIMARY_ENGINE)].copy()
    categories = {
        "stable_hinge": (
            primary.top_decile_a
            & primary.top_decile_b
            & primary.hinge_hbond_a.eq(True)
            & primary.hinge_hbond_b.eq(True),
            ["worst_percentile", "mean_percentile", "ligand_id"],
            [True, True, True],
            "Top decile and detected hinge H-bond in both structures; inspect a stable recognition hypothesis.",
        ),
        "good_score_no_hinge": (
            primary.mean_percentile.le(0.10)
            & primary.profiled_a
            & primary.profiled_b
            & primary.hinge_hbond_a.eq(False)
            & primary.hinge_hbond_b.eq(False),
            ["mean_percentile", "ligand_id"],
            [True, True],
            (
                "Good mean score rank without a detected hinge H-bond in either profiled pose; "
                "inspect score/geometry disagreement."
            ),
        ),
        "structure_sensitive": (
            primary.absolute_percentile_shift.ge(0.50),
            ["absolute_percentile_shift", "mean_percentile", "ligand_id"],
            [False, True, True],
            "At least 0.50 rank-percentile shift between structures; inspect conformation dependence.",
        ),
    }
    selected, sizes = [], {}
    for category, (mask, order, ascending, reason) in categories.items():
        candidates = primary.loc[mask].sort_values(order, ascending=ascending)
        sizes[category] = dict(
            eligible=len(candidates), selected=min(3, len(candidates))
        )
        chosen = candidates.head(3).copy()
        chosen["inspection_category"], chosen["inspection_reason"] = category, reason
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True), sizes


def figures(table, paired, shortlist, output):
    primary = table[table.engine.eq(PRIMARY_ENGINE)]
    pair = paired[paired.engine.eq(PRIMARY_ENGINE)] if not paired.empty else paired
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.7), constrained_layout=True)
    ax = axes[0, 0]
    for stratum in STRATA:
        frame = pair[pair.activity_stratum.eq(stratum)] if not pair.empty else pair
        ax.scatter(
            frame.get("percentile_a", []),
            frame.get("percentile_b", []),
            s=15,
            alpha=0.65,
            color=COLORS[stratum],
            label=f"{stratum} annotation",
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    ax.set(
        xlabel="1M17 score-rank percentile",
        ylabel="4ZAU score-rank percentile",
        title="A  Protein-structure rank stability",
        xlim=(0, 1.03),
        ylim=(0, 1.03),
    )
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[0, 1]
    proteins = sorted(primary.receptor_id.unique())
    width = 0.34
    for j, rid in enumerate(proteins):
        heights, labels = [], []
        for stratum in STRATA:
            block = primary[
                primary.receptor_id.eq(rid)
                & primary.activity_stratum.eq(stratum)
                & primary.profiled
            ]
            heights.append(
                float(block.hinge_hbond.astype(float).mean()) if len(block) else 0
            )
            labels.append(f"{int(block.hinge_hbond.astype(float).sum())}/{len(block)}")
        xs = np.arange(len(STRATA)) + (j - 0.5) * width
        bars = ax.bar(
            xs, heights, width=width, label=rid, color=["#4477aa", "#ee9944"][j % 2]
        )
        for bar, text in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                text,
                ha="center",
                fontsize=7,
            )
    ax.set(
        xticks=np.arange(3),
        xticklabels=STRATA,
        ylabel="Fraction with detected MET H-bond",
        ylim=(0, 1.15),
        title="B  MET hinge-residue H-bonds by annotation",
    )
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for rid in proteins:
        block = primary[primary.receptor_id.eq(rid)]
        ax.scatter(block.heavy_atoms, block.score, s=13, alpha=0.5, label=rid)
    ax.set(
        xlabel="Ligand heavy-atom count",
        ylabel="Native docking score",
        title="C  Inspect score–size dependence",
    )
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    block = primary[primary.profiled]
    for name, label_, color in [
        ("template_tanimoto_5A", "Template-set Tanimoto", "#aa3377"),
        ("common_pocket_coverage_5A", "Common-pocket coverage", "#228833"),
    ]:
        ax.scatter(
            block.heavy_atoms, block[name], s=12, alpha=0.45, label=label_, color=color
        )
    ax.set(
        xlabel="Ligand heavy-atom count",
        ylabel="Contact-set summary",
        title="D  Explicit contact-size diagnostic",
    )
    ax.legend(fontsize=8)
    fig.suptitle(
        "Descriptive EGFR noncovalent recognition queries — no fitted weights",
        fontsize=13,
    )
    fig.savefig(output / "chemical_usecase_summary.pdf")
    fig.savefig(output / "chemical_usecase_summary.png", dpi=180)
    plt.close(fig)
    if not shortlist.empty:
        mols = [Chem.MolFromSmiles(s) for s in shortlist.smiles]
        legends = [
            (
                f"{r.ligand_id}\n{r.inspection_category}\n"
                f"mean percentile {r.mean_percentile:.3f}; shift {r.absolute_percentile_shift:.3f}"
            )
            for r in shortlist.itertuples()
        ]
        image = Draw.MolsToGridImage(
            mols, molsPerRow=3, subImgSize=(380, 290), legends=legends
        )
        image.save(output / "rule_selected_structures.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = ROOT / "Data/validation/protocols/chemical_usecase.txt"
    if not protocol.exists():
        raise FileNotFoundError(
            "The prespecified protocol must exist before analyzing results"
        )
    table_path = args.campaign / "annotated_rank1.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Rank-1 campaign export is not ready: {table_path}")
    inputs = [args.library, args.context, table_path]
    inputs += [
        p
        for p in [
            args.campaign / "interaction_events.csv",
            args.campaign / "analysis_counts.json",
        ]
        if p.exists()
    ]
    hashes = {str(p): digest(p) for p in inputs}
    library, context = json.loads(args.library.read_text()), json.loads(
        args.context.read_text()
    )
    table = pd.read_csv(table_path)
    required = {"receptor_id", "ligand_id", "engine", "pose_rank", "pose_id"}
    if not required.issubset(table):
        raise ValueError(f"Missing rank-1 columns: {required - set(table)}")
    mapping, native, common, hinge, pocket_report = pockets(
        context, sorted(table.receptor_id.unique())
    )
    events, profiled, event_report = event_payloads(args.campaign)
    annotated = annotate(
        table, library, mapping, native, common, hinge, events, profiled
    )
    paired, stability = paired_analysis(annotated)
    shortlist, shortlist_counts = inspection_shortlist(paired)
    correlations, hinge_rows = [], []
    for (rid, engine), frame in annotated.groupby(["receptor_id", "engine"]):
        for metric in [
            "score",
            "docking_score_per_heavy_atom",
            "template_tanimoto_5A",
            "common_pocket_coverage_5A",
            "common_pocket_coverage_6A",
            "common_pocket_coverage_8A",
        ]:
            for covariate in ["heavy_atoms", "rotatable_bonds", "median_pic50"]:
                correlations.append(
                    dict(
                        protein=rid,
                        engine=engine,
                        metric=metric,
                        covariate=covariate,
                        **correlation(frame[metric], frame[covariate]),
                    )
                )
        for stratum in STRATA:
            block = frame[frame.activity_stratum.eq(stratum)]
            complete = block[block.profiled]
            hinge_rows.append(
                dict(
                    protein=rid,
                    engine=engine,
                    activity_stratum=stratum,
                    rank1_poses=len(block),
                    profiled_poses=len(complete),
                    hinge_contact_poses=int(complete.hinge_contact.astype(float).sum()),
                    hinge_hbond_poses=int(complete.hinge_hbond.astype(float).sum()),
                    hinge_hbond_fraction=number(
                        complete.hinge_hbond.astype(float).mean()
                    ),
                )
            )
    annotated.to_csv(args.output / "chemical_pose_annotations.csv", index=False)
    paired.to_csv(args.output / "paired_structure_ranks.csv", index=False)
    shortlist.to_csv(args.output / "inspection_shortlist.csv", index=False)
    pd.DataFrame(correlations).to_csv(
        args.output / "descriptive_correlations.csv", index=False
    )
    pd.DataFrame(hinge_rows).to_csv(
        args.output / "hinge_by_activity_annotation.csv", index=False
    )
    write_json(args.output / "pocket_definitions.json", pocket_report)
    figures(annotated, paired, shortlist, args.output)
    report = dict(
        protocol_sha256=digest(protocol),
        input_sha256=hashes,
        rank1_poses=len(annotated),
        compounds=annotated.ligand_id.nunique(),
        proteins=sorted(annotated.receptor_id.unique()),
        engines=sorted(annotated.engine.unique()),
        actual_interaction_events=sum(len(v) for v in events.values()),
        interaction_profile=event_report,
        rank_stability=stability,
        inspection_shortlist=shortlist_counts,
        acrylamide_annotation_compounds=int(
            annotated.groupby("ligand_id").acrylamide_substructure.first().sum()
        ),
        caveats=[
            "Heterogeneous ChEMBL IC50 annotations are descriptive; no homogeneous benchmark or activity prediction.",
            "No fitted weights, activity-selected thresholds, or cherry-picked examples.",
            "Noncovalent recognition only: no covalent-reaction, clinical-potency or mutant-selectivity prediction.",
            "Common pocket is a prespecified native-neighborhood union; coverage can still depend on ligand size.",
            "Only best-scored poses are used for this bounded interaction analysis.",
            (
                "Reference frames use independently validated rigid protein fits; "
                "nonrigid preparation deformation remains."
            ),
        ],
    )
    assert all(
        digest(Path(p)) == h for p, h in hashes.items()
    ), "Read-only inputs changed during analysis"
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
