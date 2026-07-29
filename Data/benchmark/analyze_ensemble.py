#!/usr/bin/env python3
"""Evaluate label-free consensus scoring on the completed EGFR screen.

The docking database contains repeated ligand preparations from interrupted
campaign restarts. By default this script keeps the highest numbered
preparation suffix for every receptor/compound pair, giving every compound one
score per receptor and engine. Active/decoy labels are used only after all
ensemble scores have been constructed.

Outputs:

* ``runs/ensemble/results.csv``
* ``runs/ensemble/results.md``

No docking or database writes are performed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

import egfr_config as cfg  # noqa: E402

from prodock.database import PoseQuery  # noqa: E402
from prodock.postprocess.metrics import ScreenEvaluator  # noqa: E402

_VARIANT_RE = re.compile(r"^(.*?)(?:_(\d+))?$")


def _compound_and_variant(ligand_id: str) -> tuple[str, int]:
    match = _VARIANT_RE.fullmatch(str(ligand_id))
    if match is None:
        return str(ligand_id), 0
    return match.group(1), int(match.group(2) or 0)


def _load_pose_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    query = PoseQuery(str(cfg.SCREEN_DB))
    table = query.poses(include_mol=False, as_dataframe=True)
    receptor_ids = {record["pdb_id"] for record in cfg.RECEPTORS}
    table = table[table["receptor_id"].isin(receptor_ids) & table["engine"].isin(cfg.ENGINES)].copy()
    parsed = table["ligand_id"].map(_compound_and_variant)
    table["compound_id"] = parsed.map(lambda item: item[0])
    table["variant"] = parsed.map(lambda item: item[1])

    # A campaign restart writes the same input compounds with incremented file
    # suffixes. Select one preparation uniformly rather than granting some
    # receptors more independent attempts than others.
    selected_variant = table.groupby(["receptor_id", "compound_id"], sort=False)["variant"].transform("max")
    selected = table[table["variant"] == selected_variant].copy()
    rank1 = selected[selected["pose_rank"] == 1].copy()

    repeat_counts = (
        table.groupby(["receptor_id", "compound_id"])["variant"]
        .nunique()
        .groupby("receptor_id")
        .max()
        .astype(int)
        .to_dict()
    )
    return rank1, selected, repeat_counts


def _score_matrix(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["channel"] = table["receptor_id"] + "/" + table["engine"]
    matrix = table.pivot(index="compound_id", columns="channel", values="affinity").sort_index()
    expected_channels = len(cfg.RECEPTORS) * len(cfg.ENGINES)
    if matrix.shape != (25, expected_channels) or matrix.isna().any().any():
        raise RuntimeError(
            "Expected a complete 25-compound x "
            f"{expected_channels}-channel matrix, got {matrix.shape} "
            f"with {int(matrix.isna().sum().sum())} missing values."
        )
    return matrix.astype(float)


def _ligand_properties(compounds: pd.Index) -> pd.DataFrame:
    records = json.loads(cfg.LIGAND_JSON.read_text())["ligands"]
    rows = []
    for record in records:
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is None:
            raise ValueError(f"Could not parse {record['id']!r}")
        rows.append(
            {
                "compound_id": record["id"],
                "heavy_atoms": float(Descriptors.HeavyAtomCount(mol)),
                "mol_wt": float(Descriptors.MolWt(mol)),
            }
        )
    props = pd.DataFrame(rows).set_index("compound_id").reindex(compounds)
    if props.isna().any().any():
        raise RuntimeError("Ligand properties are incomplete.")
    return props


def _zscore(matrix: pd.DataFrame) -> pd.DataFrame:
    std = matrix.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (matrix - matrix.mean(axis=0)) / std


def _robust_zscore(matrix: pd.DataFrame) -> pd.DataFrame:
    median = matrix.median(axis=0)
    mad = (matrix - median).abs().median(axis=0)
    fallback = matrix.std(axis=0, ddof=0).replace(0.0, 1.0)
    scale = (1.4826 * mad).where(mad > 0.0, fallback)
    return (matrix - median) / scale


def _residualize(matrix: pd.DataFrame, covariate: pd.Series) -> pd.DataFrame:
    design = np.column_stack([np.ones(len(covariate)), covariate.to_numpy()])
    residuals = {}
    for column in matrix:
        values = matrix[column].to_numpy()
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residuals[column] = values - design @ coefficients
    return pd.DataFrame(residuals, index=matrix.index)


def _ensemble_scores(
    matrix: pd.DataFrame,
    props: pd.DataFrame,
    contact_shell: pd.DataFrame,
    contact_core: pd.DataFrame,
    best_pose_contact_shell: pd.DataFrame,
    best_pose_contact_core: pd.DataFrame,
) -> dict[str, tuple[str, pd.Series]]:
    ranks = matrix.rank(axis=0, method="average", ascending=True)
    zscores = _zscore(matrix)
    robust = _robust_zscore(matrix)
    sorted_z = np.sort(zscores.to_numpy(), axis=1)
    contact_shell_ranks = contact_shell.rank(axis=0, method="average", ascending=True)
    contact_core_ranks = contact_core.rank(axis=0, method="average", ascending=True)
    best_pose_shell_ranks = best_pose_contact_shell.rank(axis=0, method="average", ascending=True)
    best_pose_core_ranks = best_pose_contact_core.rank(axis=0, method="average", ascending=True)

    methods: dict[str, tuple[str, pd.Series]] = {}
    for engine in cfg.ENGINES:
        columns = [f"{record['pdb_id']}/{engine}" for record in cfg.RECEPTORS]
        methods[f"raw_best_{engine}"] = (
            "single-engine",
            matrix[columns].min(axis=1),
        )

    # Directly test the intuitive "average the four programs" ensemble.
    # First average the four top-pose affinities within each receptor. The
    # receptor-level matrix then supports either the best receptor (matching
    # the single-engine baseline above) or a mean over all three receptors.
    receptor_engine_means = pd.DataFrame(
        {
            record["pdb_id"]: matrix[[f"{record['pdb_id']}/{engine}" for engine in cfg.ENGINES]].mean(axis=1)
            for record in cfg.RECEPTORS
        }
    )

    methods.update(
        {
            "mean_four_engines_best_receptor": (
                "affinity-ensemble",
                receptor_engine_means.min(axis=1),
            ),
            "mean_four_engines_three_receptors": (
                "affinity-ensemble",
                receptor_engine_means.mean(axis=1),
            ),
            "mean_rank": ("affinity-ensemble", ranks.mean(axis=1)),
            "median_rank": ("affinity-ensemble", ranks.median(axis=1)),
            "geometric_rank": (
                "affinity-ensemble",
                np.exp(np.log(ranks).mean(axis=1)),
            ),
            "rrf_k10": (
                "affinity-ensemble",
                -(1.0 / (10.0 + ranks)).sum(axis=1),
            ),
            "mean_z": ("affinity-ensemble", zscores.mean(axis=1)),
            "median_z": ("affinity-ensemble", zscores.median(axis=1)),
            "trimmed_mean_z": (
                "affinity-ensemble",
                pd.Series(sorted_z[:, 1:-1].mean(axis=1), index=matrix.index),
            ),
            "best_z": ("affinity-ensemble", zscores.min(axis=1)),
            "worst_z": ("affinity-ensemble", zscores.max(axis=1)),
            "mean_robust_z": (
                "affinity-ensemble",
                robust.mean(axis=1),
            ),
            "contact_shell_mean_similarity": (
                "interaction-template",
                contact_shell.mean(axis=1),
            ),
            "contact_shell_mean_rank": (
                "interaction-template",
                contact_shell_ranks.mean(axis=1),
            ),
            "contact_core_mean_similarity": (
                "interaction-template",
                contact_core.mean(axis=1),
            ),
            "contact_core_mean_rank": (
                "interaction-template",
                contact_core_ranks.mean(axis=1),
            ),
            "affinity_contact_shell_equal_rank": (
                "hybrid-template",
                (ranks + contact_shell_ranks).mean(axis=1),
            ),
            "affinity_contact_core_equal_rank": (
                "hybrid-template",
                (ranks + contact_core_ranks).mean(axis=1),
            ),
            "best_pose_contact_shell_mean_similarity": (
                "pose-reranking-template",
                best_pose_contact_shell.mean(axis=1),
            ),
            "best_pose_contact_shell_mean_rank": (
                "pose-reranking-template",
                best_pose_shell_ranks.mean(axis=1),
            ),
            "best_pose_contact_core_mean_similarity": (
                "pose-reranking-template",
                best_pose_contact_core.mean(axis=1),
            ),
            "best_pose_contact_core_mean_rank": (
                "pose-reranking-template",
                best_pose_core_ranks.mean(axis=1),
            ),
        }
    )

    heavy_atom_efficiency = matrix.div(props["heavy_atoms"], axis=0)
    methods["mean_heavy_atom_efficiency_z"] = (
        "size-corrected",
        _zscore(heavy_atom_efficiency).mean(axis=1),
    )
    methods["mean_heavy_atom_residual_z"] = (
        "size-corrected",
        _zscore(_residualize(matrix, props["heavy_atoms"])).mean(axis=1),
    )
    methods["mean_mol_wt_residual_z"] = (
        "size-corrected",
        _zscore(_residualize(matrix, props["mol_wt"])).mean(axis=1),
    )
    return methods


def _reference_contact_fingerprints(
    cutoff: float,
) -> dict[str, set[str]]:
    """Build residue-contact sets directly from cocrystal coordinates."""
    fingerprints = {}
    for record in cfg.RECEPTORS:
        receptor = record["pdb_id"]
        project = cfg.SCREEN_DIR / receptor / receptor
        ligand_path = project / "reference_ligand" / f"{record['ligand_code']}.sdf"
        receptor_path = project / "filtered_protein" / f"{receptor}.pdb"
        supplier = Chem.SDMolSupplier(str(ligand_path), sanitize=False, removeHs=False)
        ligand = supplier[0] if len(supplier) else None
        protein = Chem.MolFromPDBFile(str(receptor_path), sanitize=False, removeHs=False)
        if ligand is None:
            raise RuntimeError(f"Could not read reference ligand {ligand_path}")
        if protein is None:
            raise RuntimeError(f"Could not read receptor {receptor_path}")

        ligand_conf = ligand.GetConformer()
        ligand_coords = np.asarray(
            [list(ligand_conf.GetAtomPosition(atom.GetIdx())) for atom in ligand.GetAtoms() if atom.GetAtomicNum() > 1]
        )
        protein_conf = protein.GetConformer()
        contacts = set()
        for atom in protein.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            info = atom.GetPDBResidueInfo()
            if info is None:
                continue
            coordinate = np.asarray(list(protein_conf.GetAtomPosition(atom.GetIdx())))
            if np.linalg.norm(ligand_coords - coordinate, axis=1).min() <= cutoff:
                residue_name = info.GetResidueName().strip()
                residue_number = info.GetResidueNumber()
                chain = info.GetChainId().strip()
                contacts.add(f"{residue_name}{residue_number}.{chain}")
        fingerprints[receptor] = contacts
        if not fingerprints[receptor]:
            raise RuntimeError(f"No reference contacts were found for {receptor}.")
    return fingerprints


def _contact_score_matrix(
    table: pd.DataFrame,
    references: dict[str, set[str]],
) -> pd.DataFrame:
    """Return negative reference-contact Tanimoto matrix (lower is better)."""
    query = PoseQuery(str(cfg.SCREEN_DB))
    fingerprints = query.fingerprint(
        pose_db_id=table["pose_db_id"].astype(int).tolist(),
        mode="binary",
        index_by="pose_db_id",
    )
    pose_sets = {
        int(pose_db_id): {
            str(feature).split("::", 1)[1]
            for feature in fingerprints.columns[fingerprints.loc[pose_db_id].to_numpy(dtype=bool)]
        }
        for pose_db_id in fingerprints.index
    }

    rows = []
    for row in table.itertuples():
        observed = pose_sets.get(int(row.pose_db_id), set())
        reference = references[str(row.receptor_id)]
        union = observed | reference
        similarity = len(observed & reference) / len(union) if union else 0.0
        rows.append(
            {
                "compound_id": row.compound_id,
                "channel": f"{row.receptor_id}/{row.engine}",
                "similarity": -similarity,
            }
        )

    scores = pd.DataFrame(rows)
    # For a rank-1 table this is a no-op; for an all-pose table it implements
    # template reranking by retaining the most cocrystal-like sampled pose.
    scores = scores.groupby(["compound_id", "channel"], as_index=False)["similarity"].min()
    return scores.pivot(index="compound_id", columns="channel", values="similarity").sort_index()


def _exact_auc_pvalues(
    score_map: dict[str, pd.Series],
    labels: pd.Series,
) -> tuple[dict[str, float], float]:
    """Return exact label-permutation p-values and max-method correction."""
    n_total = len(labels)
    n_active = int(labels.sum())
    combinations = np.asarray(list(itertools.combinations(range(n_total), n_active)), dtype=int)
    auc_null = []
    observed = {}
    denom = n_active * (n_total - n_active)
    offset = n_active * (n_active + 1) / 2.0

    for name, scores in score_map.items():
        # Higher quality rank means a better (lower) ensemble score.
        lower_rank = scores.rank(method="average", ascending=True).to_numpy()
        quality_rank = n_total + 1.0 - lower_rank
        null = (quality_rank[combinations].sum(axis=1) - offset) / denom
        auc_null.append(null)
        active_indices = np.flatnonzero(labels.to_numpy() == 1)
        observed[name] = float((quality_rank[active_indices].sum() - offset) / denom)

    null_matrix = np.column_stack(auc_null)
    pvalues = {
        name: float(np.mean(null_matrix[:, index] >= observed[name] - 1e-12)) for index, name in enumerate(score_map)
    }
    best_observed = max(observed.values())
    max_method_p = float(np.mean(null_matrix.max(axis=1) >= best_observed - 1e-12))
    return pvalues, max_method_p


def _evaluate(
    methods: dict[str, tuple[str, pd.Series]],
) -> tuple[pd.DataFrame, float]:
    evaluator = ScreenEvaluator(higher_is_better=False)
    labels = pd.Series(
        [int(compound in cfg.ACTIVES) for compound in next(iter(methods.values()))[1].index],
        index=next(iter(methods.values()))[1].index,
        dtype=int,
    )
    score_map = {name: values for name, (_, values) in methods.items()}
    pvalues, max_method_p = _exact_auc_pvalues(score_map, labels)

    rows = []
    for name, (kind, scores) in methods.items():
        order = scores.sort_values(kind="mergesort").index
        ordered_labels = labels.loc[order]
        active_ranks = [f"{rank}:{compound}" for rank, compound in enumerate(order, 1) if compound in cfg.ACTIVES]
        top5_hits = int(ordered_labels.iloc[:5].sum())
        rows.append(
            {
                "method": name,
                "kind": kind,
                "roc_auc": evaluator.auc_roc(scores, labels),
                "bedroc_20": evaluator.bedroc(scores, labels, alpha=20.0),
                "top1_hits": int(ordered_labels.iloc[:1].sum()),
                "top5_hits": top5_hits,
                "ef_top5": float(top5_hits),
                "auc_exact_p": pvalues[name],
                "active_ranks": ", ".join(active_ranks),
            }
        )
    results = pd.DataFrame(rows).sort_values(["roc_auc", "bedroc_20"], ascending=False)
    return results, max_method_p


def _write_report(
    results: pd.DataFrame,
    methods: dict[str, tuple[str, pd.Series]],
    repeat_counts: dict[str, int],
    max_method_p: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results.csv", index=False)

    best = results.iloc[0]
    baseline = results[results["kind"] == "single-engine"].iloc[0]
    best_scores = methods[str(best["method"])][1]
    best_order = best_scores.sort_values(kind="mergesort").index
    ranking = pd.DataFrame(
        {
            "compound_id": best_order,
            "is_active": [compound in cfg.ACTIVES for compound in best_order],
            "score": best_scores.loc[best_order].to_numpy(),
            "rank": np.arange(1, len(best_order) + 1),
        }
    )
    ranking.to_csv(output_dir / "best_ranking.csv", index=False)

    template_actives = {"gefitinib", "erlotinib", "osimertinib"}
    non_template_actives = cfg.ACTIVES - template_actives
    diagnostic_ids = [compound for compound in best_scores.index if compound not in template_actives]
    diagnostic_scores = best_scores.loc[diagnostic_ids]
    diagnostic_labels = pd.Series(
        [int(compound in non_template_actives) for compound in diagnostic_ids],
        index=diagnostic_ids,
    )
    evaluator = ScreenEvaluator(higher_is_better=False)
    diagnostic_order = diagnostic_scores.sort_values().index
    diagnostic_top5 = int(diagnostic_labels.loc[diagnostic_order[:5]].sum())
    diagnostic_auc = evaluator.auc_roc(diagnostic_scores, diagnostic_labels)
    diagnostic_p, _ = _exact_auc_pvalues({"diagnostic": diagnostic_scores}, diagnostic_labels)
    lines = [
        "# Affinity ensembles and exploratory contact-template reranking",
        "",
        "Dataset: 25 compounds (5 actives), 3 receptors, 4 engines. "
        "One latest preparation per receptor/compound was retained.",
        "",
        "Interaction-template methods compare stored pose interaction "
        "residues with 4.0/5.0-Angstrom residue-contact fingerprints derived "
        "from each receptor's deposited cocrystal ligand. They therefore "
        "have structural prior information unavailable to affinity-only "
        "methods and should be interpreted separately.",
        "",
        "Restart preparations present in the database: "
        + ", ".join(f"{key}={value}" for key, value in repeat_counts.items())
        + ".",
        "",
        f"Best tested method: `{best['method']}` "
        f"(ROC-AUC {best['roc_auc']:.3f}, BEDROC20 "
        f"{best['bedroc_20']:.3f}, top-5 hits {best['top5_hits']}/5).",
        "",
        f"Best single-engine baseline: `{baseline['method']}` " f"(ROC-AUC {baseline['roc_auc']:.3f}).",
        "",
        "Exact permutation p-value after selecting the best of all listed "
        f"methods: {max_method_p:.4f}. Active labels were not used in the "
        "score calculations, but three active compounds supplied the "
        "cocrystal templates used by the structural methods.",
        "",
        "Non-template-active diagnostic: after removing the three compounds "
        "used as cocrystal templates (gefitinib, erlotinib, osimertinib), "
        f"afatinib and dacomitinib give ROC-AUC {diagnostic_auc:.3f}, "
        f"{diagnostic_top5}/2 hits in the top five, and exact p="
        f"{diagnostic_p['diagnostic']:.4f}. This is a small diagnostic, not "
        "an independent validation set.",
        "",
        "| Method | Type | ROC-AUC | BEDROC20 | Top-1 | Top-5 | " "Exact p | Active ranks |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| `{row['method']}` | {row['kind']} | "
            f"{row['roc_auc']:.3f} | {row['bedroc_20']:.3f} | "
            f"{int(row['top1_hits'])} | {int(row['top5_hits'])} | "
            f"{row['auc_exact_p']:.4f} | {row['active_ranks']} |"
        )
    (output_dir / "results.md").write_text("\n".join(lines) + "\n")


def _write_paper_rows(results: pd.DataFrame) -> Path:
    """Write the simple interaction-similarity table used by the manuscript."""
    rows_by_method = results.set_index("method")
    selected = [
        ("raw_best_vina", "Affinity baseline (Vina)"),
        (
            "contact_shell_mean_similarity",
            "Mean interaction similarity (pose 1)",
        ),
    ]
    lines = []
    for method, label in selected:
        row = rows_by_method.loc[method]
        lines.append(
            f"{label} & {row['roc_auc']:.2f} & {row['bedroc_20']:.3f} & "
            f"{int(row['top5_hits'])}/5 & {row['ef_top5']:.1f} \\\\"
        )
    lines.append("\\bottomrule")
    output = cfg.TABLES_DIR / "ensemble_rows.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


def _write_affinity_paper_rows(results: pd.DataFrame) -> Path:
    """Write the manuscript table comparing top-pose affinity ensembles."""
    rows_by_method = results.set_index("method")
    selected = [
        ("raw_best_vina", "Best single engine (Vina)"),
        (
            "mean_four_engines_best_receptor",
            "Four-engine mean, best receptor",
        ),
        (
            "mean_four_engines_three_receptors",
            "Four-engine mean, three-receptor mean",
        ),
        ("mean_z", "Mean channel-standardized score"),
        ("mean_rank", "Mean channel rank"),
    ]
    lines = []
    for method, label in selected:
        row = rows_by_method.loc[method]
        lines.append(
            f"{label} & {row['roc_auc']:.2f} & {row['bedroc_20']:.3f} & "
            f"{int(row['top5_hits'])}/5 & {row['ef_top5']:.1f} \\\\"
        )
    lines.append("\\bottomrule")
    output = cfg.TABLES_DIR / "affinity_ensemble_rows.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


def _write_ranking_summary_figure(
    results: pd.DataFrame,
    methods: dict[str, tuple[str, pd.Series]],
) -> Path:
    """Compare the affinity baseline and interaction query directly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = results.set_index("method")
    affinity_color = "#566674"
    interaction_color = "#B83B3B"
    random_color = "#B5BBC1"

    def roc_points(scores: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Return empirical ROC coordinates with tied scores grouped."""
        frame = pd.DataFrame(
            {
                "score": scores,
                "active": [int(compound in cfg.ACTIVES) for compound in scores.index],
            }
        )
        grouped = frame.groupby("score", sort=True)["active"].agg(["sum", "count"])
        true_positive = np.r_[0, grouped["sum"].cumsum().to_numpy()]
        false_positive = np.r_[0, (grouped["count"] - grouped["sum"]).cumsum().to_numpy()]
        return (
            false_positive / false_positive[-1],
            true_positive / true_positive[-1],
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 5.0),
        gridspec_kw={"width_ratios": [1.0, 1.08]},
    )

    ax = axes[0]
    roc_curves = [
        (
            "raw_best_vina",
            "Vina affinity",
            affinity_color,
            2.4,
        ),
        (
            "contact_shell_mean_similarity",
            "Mean interaction similarity",
            interaction_color,
            2.8,
        ),
    ]
    for method, label, color, linewidth in roc_curves:
        false_positive, true_positive = roc_points(methods[method][1])
        auc = float(rows.loc[method, "roc_auc"])
        ax.step(
            false_positive,
            true_positive,
            where="post",
            color=color,
            linewidth=linewidth,
            label=f"{label} (AUC = {auc:.2f})",
            zorder=3,
        )
        if method == "contact_shell_mean_similarity":
            ax.fill_between(
                false_positive,
                true_positive,
                step="post",
                color=interaction_color,
                alpha=0.08,
                zorder=1,
            )
    ax.plot(
        [0, 1],
        [0, 1],
        color=random_color,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label="Random-order reference",
        zorder=2,
    )
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.text(
        -0.12,
        1.04,
        "A",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(color="#E7E9EC", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    rank_cutoffs = np.arange(1, 26)
    curves = [
        ("raw_best_vina", "Vina affinity", affinity_color),
        (
            "contact_shell_mean_similarity",
            "Mean interaction similarity",
            interaction_color,
        ),
    ]
    for method, label, color in curves:
        order = methods[method][1].sort_values(kind="mergesort").index
        cumulative = np.cumsum([compound in cfg.ACTIVES for compound in order])
        ax.step(
            rank_cutoffs,
            cumulative,
            where="post",
            color=color,
            linewidth=2.5,
            label=label,
            zorder=3,
        )
        active_ranks = [rank for rank, compound in enumerate(order, start=1) if compound in cfg.ACTIVES]
        ax.scatter(
            active_ranks,
            np.arange(1, len(active_ranks) + 1),
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    ax.plot(
        rank_cutoffs,
        rank_cutoffs * (len(cfg.ACTIVES) / 25.0),
        color=random_color,
        linewidth=1.1,
        linestyle=(0, (3, 3)),
        label="Random-order expectation",
    )
    ax.axvspan(1, 5, color="#EFF3EC", alpha=0.9, zorder=0)
    ax.axvline(5, color="#CAD3C5", linewidth=0.9, zorder=1)
    ax.annotate(
        "2 inhibitors",
        xy=(5, 2),
        xytext=(7.0, 2.65),
        color=interaction_color,
        fontsize=8.5,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": interaction_color,
            "linewidth": 0.9,
        },
    )
    ax.annotate(
        "0 inhibitors",
        xy=(5, 0),
        xytext=(7.0, 0.48),
        color=affinity_color,
        fontsize=8.5,
        arrowprops={
            "arrowstyle": "-",
            "color": affinity_color,
            "linewidth": 0.9,
        },
    )
    ax.set_xlim(1, 25)
    ax.set_ylim(0, 5.2)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_yticks(range(0, 6))
    ax.set_xlabel("Rank cutoff")
    ax.set_ylabel("EGFR inhibitors recovered")
    ax.text(
        -0.12,
        1.04,
        "B",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=affinity_color,
            linewidth=2.5,
            label="Vina affinity",
        ),
        Line2D(
            [0],
            [0],
            color=interaction_color,
            linewidth=2.8,
            label="Mean interaction similarity",
        ),
        Line2D(
            [0],
            [0],
            color=random_color,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            label="Random-order reference",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=9.0,
        handlelength=3.0,
        columnspacing=2.0,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.125,
        top=0.80,
        wspace=0.27,
    )
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = cfg.FIGURE_DIR / "ranking_method_summary.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _write_interaction_ranking_figure(
    methods: dict[str, tuple[str, pd.Series]],
) -> Path:
    """Illustrate paired ranks and the final interaction-score profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    affinity = methods["raw_best_vina"][1].sort_index()
    interaction_loss = methods["contact_shell_mean_similarity"][1].reindex(affinity.index)

    def ordinal_ranks(scores: pd.Series) -> pd.Series:
        order = scores.sort_values(kind="mergesort").index
        return pd.Series(np.arange(1, len(order) + 1, dtype=float), index=order)

    affinity_ranks = ordinal_ranks(affinity)
    interaction_ranks = ordinal_ranks(interaction_loss)
    mean_similarity = -interaction_loss
    interaction_order = interaction_loss.sort_values(kind="mergesort").index

    active_color = "#B33A3A"
    decoy_color = "#B8BDC5"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.6),
        gridspec_kw={"width_ratios": [0.95, 1.45]},
    )

    ax = axes[0]
    for compound in affinity.index:
        is_active = compound in cfg.ACTIVES
        color = active_color if is_active else decoy_color
        ax.plot(
            [0, 1],
            [affinity_ranks[compound], interaction_ranks[compound]],
            color=color,
            linewidth=2.0 if is_active else 0.8,
            alpha=0.95 if is_active else 0.5,
            zorder=3 if is_active else 1,
        )
        ax.scatter(
            [0, 1],
            [affinity_ranks[compound], interaction_ranks[compound]],
            color=color,
            s=28 if is_active else 10,
            zorder=4 if is_active else 2,
        )
        if is_active:
            ax.text(
                1.05,
                interaction_ranks[compound],
                compound,
                va="center",
                fontsize=8,
                color=active_color,
            )
    ax.set_xlim(-0.14, 1.72)
    ax.set_ylim(25.8, 0.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Vina affinity\nranking", "Interaction-similarity\nranking"])
    ax.set_yticks([1, 5, 10, 15, 20, 25])
    ax.set_ylabel("Compound rank (1 = best)")
    ax.axhspan(0.5, 5.5, color="#EEF4EA", alpha=0.8, zorder=0)
    ax.text(
        0.01,
        0.98,
        "A",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    positions = np.arange(1, len(interaction_order) + 1)
    values = mean_similarity.loc[interaction_order].to_numpy()
    colors = [active_color if compound in cfg.ACTIVES else decoy_color for compound in interaction_order]
    ax.vlines(
        positions,
        0,
        values,
        colors=colors,
        linewidth=[2.2 if compound in cfg.ACTIVES else 1.1 for compound in interaction_order],
        alpha=0.9,
    )
    ax.scatter(positions, values, c=colors, s=28, zorder=3)
    for position, compound, value in zip(positions, interaction_order, values):
        if compound in cfg.ACTIVES:
            ax.annotate(
                compound,
                (position, value),
                xytext=(3, -7),
                textcoords="offset points",
                rotation=48,
                ha="left",
                va="top",
                fontsize=7.5,
                color=active_color,
            )
    ax.set_xlim(0.3, 25.7)
    ax.set_ylim(bottom=0)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel("Interaction-similarity rank")
    ax.set_ylabel("Mean residue-set Tanimoto similarity")
    ax.axvspan(0.5, 5.5, color="#EEF4EA", alpha=0.8, zorder=0)
    ax.text(
        0.01,
        0.98,
        "B",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=active_color,
                label="EGFR inhibitor",
                linewidth=1.8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=decoy_color,
                label="Decoy",
                linewidth=1.0,
            ),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(w_pad=2.2)
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = cfg.FIGURE_DIR / "interaction_similarity_ranking.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.BENCH_DIR / "ensemble",
    )
    args = parser.parse_args()

    if not cfg.SCREEN_DB.exists():
        raise FileNotFoundError(cfg.SCREEN_DB)
    rank1_table, all_pose_table, repeat_counts = _load_pose_tables()
    matrix = _score_matrix(rank1_table)
    props = _ligand_properties(matrix.index)
    shell_references = _reference_contact_fingerprints(cutoff=5.0)
    core_references = _reference_contact_fingerprints(cutoff=4.0)
    contact_shell = _contact_score_matrix(rank1_table, shell_references)
    contact_core = _contact_score_matrix(rank1_table, core_references)
    best_pose_contact_shell = _contact_score_matrix(all_pose_table, shell_references)
    best_pose_contact_core = _contact_score_matrix(all_pose_table, core_references)
    methods = _ensemble_scores(
        matrix,
        props,
        contact_shell,
        contact_core,
        best_pose_contact_shell,
        best_pose_contact_core,
    )
    results, max_method_p = _evaluate(methods)
    _write_report(
        results,
        methods,
        repeat_counts,
        max_method_p,
        args.output_dir,
    )
    paper_rows = _write_paper_rows(results)
    affinity_paper_rows = _write_affinity_paper_rows(results)
    summary_figure = _write_ranking_summary_figure(results, methods)
    interaction_figure = _write_interaction_ranking_figure(methods)
    print(results.to_string(index=False))
    print(f"\nMax-method exact permutation p-value: {max_method_p:.4f}")
    print(f"Wrote {args.output_dir / 'results.csv'}")
    print(f"Wrote {args.output_dir / 'best_ranking.csv'}")
    print(f"Wrote {args.output_dir / 'results.md'}")
    print(f"Wrote {paper_rows}")
    print(f"Wrote {affinity_paper_rows}")
    print(f"Wrote {summary_figure}")
    print(f"Wrote {interaction_figure}")


if __name__ == "__main__":
    main()
