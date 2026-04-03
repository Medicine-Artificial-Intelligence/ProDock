from __future__ import annotations

"""
Publication-style visualisation helpers for PoseInteractionTableResult.

This module provides table-first plotting utilities for the batch interaction
result returned by pose-table interaction workflows.

The plotting API is designed for manuscript-quality figures with restrained,
accessible palettes and compact multi-panel layouts suitable for journal
submission or preprint figures.

Supported visualisations include:

- affinity histograms
- best-pose bar charts
- interaction-type frequency charts
- residue-contact frequency charts
- per-pose interaction-count histograms
- affinity versus interaction-count scatter plots
- 2x3 overview summary panels
- similarity heatmaps when bitvectors are available

The implementation uses matplotlib only and intentionally avoids seaborn so
that figure appearance remains predictable and lightweight.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from .exceptions import MissingDependencyError, VisualizationError
from .models import PoseInteractionTableResult
from .similarity import tanimoto_similarity_matrix


@dataclass(frozen=True)
class JournalStyle:
    """
    Visual style configuration for publication-style figures.
    """

    name: str
    palette: tuple[str, ...]
    heatmap_cmap: str = "cividis"
    background: str = "white"
    panel_facecolor: str = "white"
    grid_color: str = "#d9dde3"
    spine_color: str = "#4a4f57"
    text_color: str = "#222222"
    title_size: float = 10.5
    label_size: float = 9.0
    tick_size: float = 8.0
    panel_label_size: float = 11.0
    line_width: float = 0.8
    grid_alpha: float = 0.35
    histogram_alpha: float = 0.85
    scatter_alpha: float = 0.85
    marker_size: float = 20.0


NATURE_STYLE = JournalStyle(
    name="nature",
    palette=(
        "#4e79a7",  # muted blue
        "#59a14f",  # muted green
        "#9c755f",  # warm brown
        "#f28e2b",  # muted orange
        "#b07aa1",  # muted mauve
        "#76b7b2",  # teal
    ),
    heatmap_cmap="cividis",
)

SCIENCE_STYLE = JournalStyle(
    name="science",
    palette=(
        "#3b5b92",  # deeper blue
        "#2f7f5f",  # darker green
        "#c17c3a",  # warm amber
        "#8d5a97",  # purple
        "#5d6d7e",  # slate
        "#bc4b51",  # restrained red
    ),
    heatmap_cmap="viridis",
)

NATURE_BLUE_STYLE = JournalStyle(
    name="nature_blue",
    palette=(
        "#355c7d",
        "#5b8fb9",
        "#7eaed3",
        "#9bbfdc",
        "#4f6d7a",
        "#84a59d",
    ),
    heatmap_cmap="Blues",
)

MONO_STYLE = JournalStyle(
    name="mono",
    palette=(
        "#2b2b2b",
        "#5a5a5a",
        "#7a7a7a",
        "#9a9a9a",
        "#b5b5b5",
        "#d0d0d0",
    ),
    heatmap_cmap="Greys",
)

STYLE_REGISTRY: dict[str, JournalStyle] = {
    "nature": NATURE_STYLE,
    "science": SCIENCE_STYLE,
    "nature_blue": NATURE_BLUE_STYLE,
    "mono": MONO_STYLE,
}


def _import_matplotlib() -> tuple[Any, Any]:
    """
    Import matplotlib modules lazily.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "matplotlib is required for interaction visualisation."
        ) from exc
    return plt, rcParams


def _get_style(style: str | JournalStyle) -> JournalStyle:
    """
    Resolve a visual style name or style object.
    """
    if isinstance(style, JournalStyle):
        return style
    if style not in STYLE_REGISTRY:
        allowed = ", ".join(sorted(STYLE_REGISTRY))
        raise VisualizationError(
            f"Unknown style '{style}'. Allowed styles are: {allowed}."
        )
    return STYLE_REGISTRY[style]


def _apply_journal_rcparams(style: JournalStyle) -> None:
    """
    Apply manuscript-like matplotlib defaults.
    """
    _, rcParams = _import_matplotlib()
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rcParams["axes.linewidth"] = style.line_width
    rcParams["axes.titlesize"] = style.title_size
    rcParams["axes.labelsize"] = style.label_size
    rcParams["xtick.labelsize"] = style.tick_size
    rcParams["ytick.labelsize"] = style.tick_size
    rcParams["figure.facecolor"] = style.background
    rcParams["axes.facecolor"] = style.panel_facecolor
    rcParams["savefig.facecolor"] = style.background
    rcParams["savefig.transparent"] = False


def _style_axes(
    ax: Any,
    style: JournalStyle,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_grid_y: bool = True,
) -> None:
    """
    Apply a restrained publication style to one axes.
    """
    if title is not None:
        ax.set_title(title, color=style.text_color, pad=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=style.text_color)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=style.text_color)

    ax.tick_params(colors=style.text_color, width=style.line_width, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(style.spine_color)
    ax.spines["bottom"].set_color(style.spine_color)

    if show_grid_y:
        ax.yaxis.grid(
            True, color=style.grid_color, alpha=style.grid_alpha, linewidth=0.6
        )
        ax.xaxis.grid(False)
    else:
        ax.grid(False)


def _panel_label(ax: Any, label: str, style: JournalStyle) -> None:
    """
    Draw a bold panel label in the upper-left corner.
    """
    ax.text(
        -0.16,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=style.panel_label_size,
        fontweight="bold",
        va="top",
        ha="left",
        color=style.text_color,
    )


def _col(
    result: PoseInteractionTableResult,
    settings_key: str,
    default: str,
) -> str:
    settings = getattr(result, "settings", None) or {}
    value = settings.get(settings_key, default)
    return str(value)


def _ensure_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _sum_compact_interactions(compact: dict[str, Any]) -> int:
    total = 0
    for residues in compact.values():
        if isinstance(residues, list):
            total += len(residues)
    return total


def _count_detail_events(detail: dict[str, Any]) -> int:
    total = 0
    for residue_map in detail.values():
        if not isinstance(residue_map, dict):
            continue
        for events in residue_map.values():
            if isinstance(events, list):
                total += len(events)
    return total


def _require_column(df: pd.DataFrame, column: str, where: str) -> None:
    if column not in df.columns:
        raise VisualizationError(
            f"Required column '{column}' was not found in {where}."
        )


def _save_figure_from_axes_or_figure(
    obj: Any,
    output_path: str | Path,
    *,
    dpi: int = 300,
    **savefig_kwargs: Any,
) -> Path:
    fig = obj if hasattr(obj, "savefig") else getattr(obj, "figure", None)
    if fig is None:
        raise VisualizationError(
            "Could not recover a matplotlib figure from the plotting object."
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi, **savefig_kwargs)
    return path


def build_pose_visualization_table(
    result: PoseInteractionTableResult,
) -> pd.DataFrame:
    """
    Build a merged pose-level dataframe convenient for plotting.
    """
    pose_id_col = _col(result, "pose_id_col", "pose_id")

    merged_df = result.merged_df.copy()
    summary_df = result.summary_df.copy()

    _require_column(merged_df, pose_id_col, "merged_df")
    _require_column(summary_df, pose_id_col, "summary_df")

    summary = summary_df[[pose_id_col]].copy()

    if "interaction_compact" in summary_df.columns:
        summary["interaction_compact"] = summary_df["interaction_compact"].map(
            _ensure_mapping
        )
    elif "interaction_compact_json" in summary_df.columns:
        summary["interaction_compact"] = summary_df["interaction_compact_json"].map(
            _ensure_mapping
        )
    else:
        summary["interaction_compact"] = [{} for _ in range(len(summary_df))]

    if "interaction_detail" in summary_df.columns:
        summary["interaction_detail"] = summary_df["interaction_detail"].map(
            _ensure_mapping
        )
    elif "interaction_detail_json" in summary_df.columns:
        summary["interaction_detail"] = summary_df["interaction_detail_json"].map(
            _ensure_mapping
        )
    else:
        summary["interaction_detail"] = [{} for _ in range(len(summary_df))]

    if "has_interactions" in summary_df.columns:
        summary["has_interactions"] = summary_df["has_interactions"].astype(bool)
    else:
        summary["has_interactions"] = summary["interaction_compact"].map(bool)

    summary["interaction_count"] = summary["interaction_compact"].map(
        _sum_compact_interactions
    )
    summary["interaction_event_count"] = summary["interaction_detail"].map(
        _count_detail_events
    )
    summary["interaction_type_count"] = summary["interaction_compact"].map(len)

    df = merged_df.merge(summary, on=pose_id_col, how="left")
    return df


def _iter_compact_rows(
    result: PoseInteractionTableResult,
) -> Iterable[tuple[str, str, str]]:
    pose_id_col = _col(result, "pose_id_col", "pose_id")
    summary_df = result.summary_df
    _require_column(summary_df, pose_id_col, "summary_df")

    if "interaction_compact" in summary_df.columns:
        series = summary_df["interaction_compact"].map(_ensure_mapping)
    elif "interaction_compact_json" in summary_df.columns:
        series = summary_df["interaction_compact_json"].map(_ensure_mapping)
    else:
        series = pd.Series([{} for _ in range(len(summary_df))], index=summary_df.index)

    for pose_id, compact in zip(summary_df[pose_id_col], series):
        for interaction_type, residues in compact.items():
            if isinstance(residues, list):
                for residue in residues:
                    yield str(pose_id), str(interaction_type), str(residue)


def _flatten_compact_interactions(
    result: PoseInteractionTableResult,
) -> pd.DataFrame:
    rows = [
        {
            "pose_id": pose_id,
            "interaction_type": interaction_type,
            "residue_id": residue_id,
        }
        for pose_id, interaction_type, residue_id in _iter_compact_rows(result)
    ]
    return pd.DataFrame(rows)


def _group_palette_map(values: Iterable[Any], style: JournalStyle) -> dict[Any, str]:
    unique = list(dict.fromkeys(values))
    colors = {}
    for i, value in enumerate(unique):
        colors[value] = style.palette[i % len(style.palette)]
    return colors


def make_affinity_histogram(
    result: PoseInteractionTableResult,
    *,
    bins: int = 20,
    group_by: str | None = None,
    figsize: tuple[float, float] = (3.35, 2.6),
    title: str = "Affinity distribution",
    xlabel: str = "Affinity",
    ylabel: str = "Count",
    style: str | JournalStyle = "nature",
) -> Any:
    plt, _ = _import_matplotlib()
    style_obj = _get_style(style)
    _apply_journal_rcparams(style_obj)

    affinity_col = _col(result, "affinity_col", "affinity")
    df = result.merged_df.copy()
    _require_column(df, affinity_col, "merged_df")

    fig, ax = plt.subplots(figsize=figsize)

    if group_by is None:
        values = pd.to_numeric(df[affinity_col], errors="coerce").dropna()
        ax.hist(
            values,
            bins=bins,
            color=style_obj.palette[0],
            alpha=style_obj.histogram_alpha,
            edgecolor="white",
            linewidth=0.4,
        )
    else:
        _require_column(df, group_by, "merged_df")
        color_map = _group_palette_map(df[group_by].astype(str), style_obj)
        for name, group in df.groupby(group_by):
            values = pd.to_numeric(group[affinity_col], errors="coerce").dropna()
            if len(values) == 0:
                continue
            ax.hist(
                values,
                bins=bins,
                alpha=0.55,
                label=str(name),
                color=color_map[str(name)],
                edgecolor="white",
                linewidth=0.3,
            )
        ax.legend(frameon=False, fontsize=7)

    _style_axes(
        ax, style_obj, title=title, xlabel=xlabel, ylabel=ylabel, show_grid_y=True
    )
    fig.tight_layout()
    return ax


def make_best_pose_bar(
    result: PoseInteractionTableResult,
    *,
    group_cols: tuple[str, ...] = ("receptor_id", "ligand_id", "engine"),
    figsize: tuple[float, float] = (4.5, 2.8),
    title: str = "Best pose per group",
    xlabel: str = "Group",
    ylabel: str = "Best affinity",
    style: str | JournalStyle = "nature",
) -> Any:
    plt, _ = _import_matplotlib()
    style_obj = _get_style(style)
    _apply_journal_rcparams(style_obj)

    affinity_col = _col(result, "affinity_col", "affinity")
    df = result.merged_df.copy()

    _require_column(df, affinity_col, "merged_df")
    for col in group_cols:
        _require_column(df, col, "merged_df")

    work = df[list(group_cols) + [affinity_col]].copy()
    work[affinity_col] = pd.to_numeric(work[affinity_col], errors="coerce")
    work = work.dropna(subset=[affinity_col])

    grouped = (
        work.groupby(list(group_cols), dropna=False)[affinity_col].min().reset_index()
    )
    labels = grouped[list(group_cols)].astype(str).agg(" | ".join, axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [
        style_obj.palette[i % len(style_obj.palette)] for i in range(len(grouped))
    ]
    ax.bar(
        range(len(grouped)),
        grouped[affinity_col].to_numpy(),
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(labels, rotation=90)

    _style_axes(
        ax, style_obj, title=title, xlabel=xlabel, ylabel=ylabel, show_grid_y=True
    )
    fig.tight_layout()
    return ax


def make_interaction_type_bar(
    result: PoseInteractionTableResult,
    *,
    top_n: int | None = 10,
    normalize: bool = False,
    figsize: tuple[float, float] = (3.35, 2.6),
    title: str = "Interaction type frequency",
    xlabel: str = "Type",
    ylabel: str | None = None,
    style: str | JournalStyle = "nature",
) -> Any:
    plt, _ = _import_matplotlib()
    style_obj = _get_style(style)
    _apply_journal_rcparams(style_obj)

    flat = _flatten_compact_interactions(result)
    if flat.empty:
        raise VisualizationError(
            "No compact interaction data is available for interaction-type plotting."
        )

    counts = flat["interaction_type"].value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    values = counts.astype(float)
    if normalize and values.sum() > 0:
        values = values / values.sum()

    fig, ax = plt.subplots(figsize=figsize)
    colors = [style_obj.palette[i % len(style_obj.palette)] for i in range(len(values))]
    ax.bar(
        range(len(values)),
        values.to_numpy(),
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(list(values.index), rotation=45, ha="right")

    _style_axes(
        ax,
        style_obj,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel or ("Fraction" if normalize else "Count"),
        show_grid_y=True,
    )
    fig.tight_layout()
    return ax


def make_residue_contact_bar(
    result: PoseInteractionTableResult,
    *,
    interaction_type: str | None = None,
    top_n: int = 15,
    normalize: bool = False,
    figsize: tuple[float, float] = (4.0, 2.6),
    title: str = "Residue contact frequency",
    xlabel: str = "Residue",
    ylabel: str | None = None,
    style: str | JournalStyle = "nature",
) -> Any:
    plt, _ = _import_matplotlib()
    style_obj = _get_style(style)
    _apply_journal_rcparams(style_obj)

    flat = _flatten_compact_interactions(result)
    if flat.empty:
        raise VisualizationError(
            "No compact interaction data is available for residue plotting."
        )

    if interaction_type is not None:
        flat = flat[flat["interaction_type"] == interaction_type]

    if flat.empty:
        raise VisualizationError(
            "No residue contacts matched the requested interaction filter."
        )

    counts = flat["residue_id"].value_counts().head(top_n).astype(float)
    if normalize and counts.sum() > 0:
        counts = counts / counts.sum()

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        range(len(counts)),
        counts.to_numpy(),
        color=style_obj.palette[1],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(list(counts.index), rotation=60, ha="right")

    chart_title = title if interaction_type is None else f"{title} ({interaction_type})"
    _style_axes(
        ax,
        style_obj,
        title=chart_title,
        xlabel=xlabel,
        ylabel=ylabel or ("Fraction" if normalize else "Count"),
        show_grid_y=True,
    )
    fig.tight_layout()
    return ax


def make_interaction_count_histogram(
    result: PoseInteractionTableResult,
    *,
    bins: int = 20,
    count_kind: Literal["compact", "event"] = "compact",
    group_by: str | None = None,
    figsize: tuple[float, float] = (3.35, 2.6),
    title: str = "Interaction count distribution",
    xlabel: str | None = None,
    ylabel: str = "Count",
    style: str | JournalStyle = "nature",
) -> Any:
    plt, _ = _import_matplotlib()
    style_obj = _get_style(style)
    _apply_journal_rcparams(style_obj)

    df = build_pose_visualization_table(result)
    count_col = (
        "interaction_count" if count_kind == "compact" else "interaction_event_count"
    )

    fig, ax = plt.subplots(figsize=figsize)

    if group_by is None:
        values = pd.to_numeric(df[count_col], errors="coerce").dropna()
        ax.hist(
            values,
            bins=bins,
            color=style_obj.palette[2],
            alpha=style_obj.histogram_alpha,
            edgecolor="white",
            linewidth=0.35,
        )
    else:
        _require_column(df, group_by, "visualization dataframe")
        color_map = _group_palette_map(df[group_by].astype(str), style_obj)
        for name, group in df.groupby(group_by):
            values = pd.to_numeric(group[count_col], errors="coerce").dropna()
            if len(values) == 0:
                continue
            ax.hist(
                values,
                bins=bins,
                alpha=0.55,
                label=str(name),
                color=color_map[str(name)],
                edgecolor="white",
                linewidth=0.3,
            )
        ax.legend(frameon=False, fontsize=7)

    _style_axes(
        ax,
        style_obj,
        title=title,
        xlabel=xlabel
        or ("Compact count" if count_kind == "compact" else "Event count"),
        ylabel=ylabel,
        show_grid_y=True,
    )
    fig.tight_layout()
    return ax


def make_affinity_vs_interaction_count_scatter(
    result: PoseInteractionTableResult,
    *,
    count_kind: Literal["compact", "event"] = "compact",
    group_by: str | None = None,
    figsize: tuple[float, float] = (3.35, 2.6),
    title: str = "Affinity vs interaction count",
    xlabel: str = "Affinity",
    ylabel: str | None = None,
    style: str | JournalStyle = "nature",
) -> Any:
    plt, _ = _import_matplotlib()
    style_obj = _get_style(style)
    _apply_journal_rcparams(style_obj)

    affinity_col = _col(result, "affinity_col", "affinity")
    df = build_pose_visualization_table(result)

    _require_column(df, affinity_col, "visualization dataframe")
    count_col = (
        "interaction_count" if count_kind == "compact" else "interaction_event_count"
    )

    df[affinity_col] = pd.to_numeric(df[affinity_col], errors="coerce")
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    df = df.dropna(subset=[affinity_col, count_col])

    fig, ax = plt.subplots(figsize=figsize)

    if group_by is None:
        ax.scatter(
            df[affinity_col],
            df[count_col],
            s=style_obj.marker_size,
            alpha=style_obj.scatter_alpha,
            color=style_obj.palette[3],
            linewidths=0,
        )
    else:
        _require_column(df, group_by, "visualization dataframe")
        color_map = _group_palette_map(df[group_by].astype(str), style_obj)
        for name, group in df.groupby(group_by):
            ax.scatter(
                group[affinity_col],
                group[count_col],
                s=style_obj.marker_size,
                alpha=style_obj.scatter_alpha,
                color=color_map[str(name)],
                label=str(name),
                linewidths=0,
            )
        ax.legend(frameon=False, fontsize=7)

    _style_axes(
        ax,
        style_obj,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
        or ("Compact count" if count_kind == "compact" else "Event count"),
        show_grid_y=False,
    )
    fig.tight_layout()
    return ax


def plot_similarity_heatmap(
    result: PoseInteractionTableResult,
    *,
    figsize: tuple[float, float] = (4.2, 3.5),
    annotate: bool = False,
    title: str = "Fingerprint similarity",
    xlabel: str = "Pose",
    ylabel: str = "Pose",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    style: str | JournalStyle = "nature",
    max_labels: int = 30,
) -> Any:
    plt, _ = _import_matplotlib()
    style_obj = _get_style(style)
    _apply_journal_rcparams(style_obj)

    if result.bitvectors is None:
        raise VisualizationError(
            "No bitvectors are available. Re-run interaction extraction with "
            "'include_bitvectors=True' to enable similarity heatmaps."
        )

    sim_df = tanimoto_similarity_matrix(
        result.bitvectors,
        names=result.molecule_names,
    )

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        sim_df.to_numpy(),
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        cmap=style_obj.heatmap_cmap,
    )

    n = len(sim_df.columns)
    show_labels = n <= max_labels

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(list(sim_df.columns) if show_labels else [""] * n, rotation=90)
    ax.set_yticklabels(list(sim_df.index) if show_labels else [""] * n)

    _style_axes(
        ax, style_obj, title=title, xlabel=xlabel, ylabel=ylabel, show_grid_y=False
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if annotate and n <= 20:
        values = sim_df.to_numpy()
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if values[i, j] < 0.5 else "#222222",
                )

    fig.tight_layout()
    return ax


def make_summary_panel_2x3(
    result: PoseInteractionTableResult,
    *,
    style: str | JournalStyle = "nature",
    figsize: tuple[float, float] = (10.5, 6.8),
    top_n_types: int = 8,
    top_n_residues: int = 12,
    residue_interaction_type: str | None = "Hydrophobic",
    scatter_group_by: str | None = "engine",
    hist_group_by: str | None = "engine",
    title: str = "Docking interaction summary",
) -> Any:
    """
    Build a 2x3 publication-style summary panel.

    Panels:
    a) affinity distribution
    b) best affinity per receptor-ligand-engine group
    c) interaction type frequency
    d) residue contact frequency
    e) interaction count distribution
    f) affinity vs interaction count
    """
    plt, _ = _import_matplotlib()
    style_obj = _get_style(style)
    _apply_journal_rcparams(style_obj)

    affinity_col = _col(result, "affinity_col", "affinity")
    df_viz = build_pose_visualization_table(result)
    flat = _flatten_compact_interactions(result)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    # a
    ax = axes[0]
    if hist_group_by is None or hist_group_by not in result.merged_df.columns:
        values = pd.to_numeric(result.merged_df[affinity_col], errors="coerce").dropna()
        ax.hist(
            values,
            bins=20,
            color=style_obj.palette[0],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.85,
        )
    else:
        color_map = _group_palette_map(
            result.merged_df[hist_group_by].astype(str), style_obj
        )
        for name, group in result.merged_df.groupby(hist_group_by):
            values = pd.to_numeric(group[affinity_col], errors="coerce").dropna()
            if len(values) == 0:
                continue
            ax.hist(
                values,
                bins=20,
                alpha=0.5,
                color=color_map[str(name)],
                label=str(name),
                edgecolor="white",
                linewidth=0.25,
            )
        ax.legend(frameon=False, fontsize=7)
    _style_axes(
        ax, style_obj, title="Affinity distribution", xlabel="Affinity", ylabel="Count"
    )
    _panel_label(ax, "a", style_obj)

    # b
    ax = axes[1]
    group_cols = ("receptor_id", "ligand_id", "engine")
    work = result.merged_df[list(group_cols) + [affinity_col]].copy()
    work[affinity_col] = pd.to_numeric(work[affinity_col], errors="coerce")
    work = work.dropna(subset=[affinity_col])
    grouped = (
        work.groupby(list(group_cols), dropna=False)[affinity_col].min().reset_index()
    )
    labels = grouped[list(group_cols)].astype(str).agg(" | ".join, axis=1)
    colors = [
        style_obj.palette[i % len(style_obj.palette)] for i in range(len(grouped))
    ]
    ax.bar(
        range(len(grouped)),
        grouped[affinity_col],
        color=colors,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(labels, rotation=90)
    _style_axes(
        ax,
        style_obj,
        title="Best pose per group",
        xlabel="Group",
        ylabel="Best affinity",
    )
    _panel_label(ax, "b", style_obj)

    # c
    ax = axes[2]
    if flat.empty:
        raise VisualizationError(
            "No compact interaction data is available for summary plotting."
        )
    counts = flat["interaction_type"].value_counts().head(top_n_types)
    colors = [style_obj.palette[i % len(style_obj.palette)] for i in range(len(counts))]
    ax.bar(
        range(len(counts)),
        counts.to_numpy(),
        color=colors,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(list(counts.index), rotation=45, ha="right")
    _style_axes(
        ax, style_obj, title="Interaction type frequency", xlabel="Type", ylabel="Count"
    )
    _panel_label(ax, "c", style_obj)

    # d
    ax = axes[3]
    flat_res = flat
    if residue_interaction_type is not None:
        flat_res = flat_res[flat_res["interaction_type"] == residue_interaction_type]
    counts_res = flat_res["residue_id"].value_counts().head(top_n_residues)
    ax.bar(
        range(len(counts_res)),
        counts_res.to_numpy(),
        color=style_obj.palette[1],
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xticks(range(len(counts_res)))
    ax.set_xticklabels(list(counts_res.index), rotation=60, ha="right")
    title_d = "Residue contact frequency"
    if residue_interaction_type is not None:
        title_d += f" ({residue_interaction_type})"
    _style_axes(ax, style_obj, title=title_d, xlabel="Residue", ylabel="Count")
    _panel_label(ax, "d", style_obj)

    # e
    ax = axes[4]
    ax.hist(
        df_viz["interaction_count"].dropna(),
        bins=20,
        color=style_obj.palette[2],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.85,
    )
    _style_axes(
        ax,
        style_obj,
        title="Interaction count distribution",
        xlabel="Compact count",
        ylabel="Count",
    )
    _panel_label(ax, "e", style_obj)

    # f
    ax = axes[5]
    df_plot = df_viz.copy()
    df_plot[affinity_col] = pd.to_numeric(df_plot[affinity_col], errors="coerce")
    df_plot["interaction_count"] = pd.to_numeric(
        df_plot["interaction_count"], errors="coerce"
    )
    df_plot = df_plot.dropna(subset=[affinity_col, "interaction_count"])

    if scatter_group_by is not None and scatter_group_by in df_plot.columns:
        color_map = _group_palette_map(df_plot[scatter_group_by].astype(str), style_obj)
        for name, group in df_plot.groupby(scatter_group_by):
            ax.scatter(
                group[affinity_col],
                group["interaction_count"],
                s=style_obj.marker_size,
                alpha=style_obj.scatter_alpha,
                color=color_map[str(name)],
                label=str(name),
                linewidths=0,
            )
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.scatter(
            df_plot[affinity_col],
            df_plot["interaction_count"],
            s=style_obj.marker_size,
            alpha=style_obj.scatter_alpha,
            color=style_obj.palette[3],
            linewidths=0,
        )
    _style_axes(
        ax,
        style_obj,
        title="Affinity vs interaction count",
        xlabel="Affinity",
        ylabel="Compact count",
        show_grid_y=False,
    )
    _panel_label(ax, "f", style_obj)

    fig.suptitle(title, fontsize=11.5, y=1.01, color=style_obj.text_color)
    fig.tight_layout()
    return fig


def save_summary_panel_2x3(
    result: PoseInteractionTableResult,
    output_path: str | Path,
    *,
    style: str | JournalStyle = "nature",
    figsize: tuple[float, float] = (10.5, 6.8),
    top_n_types: int = 8,
    top_n_residues: int = 12,
    residue_interaction_type: str | None = "Hydrophobic",
    scatter_group_by: str | None = "engine",
    hist_group_by: str | None = "engine",
    title: str = "Docking interaction summary",
    dpi: int = 300,
) -> Path:
    fig = make_summary_panel_2x3(
        result,
        style=style,
        figsize=figsize,
        top_n_types=top_n_types,
        top_n_residues=top_n_residues,
        residue_interaction_type=residue_interaction_type,
        scatter_group_by=scatter_group_by,
        hist_group_by=hist_group_by,
        title=title,
    )
    return _save_figure_from_axes_or_figure(fig, output_path, dpi=dpi)


def save_affinity_histogram(
    result: PoseInteractionTableResult,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    ax = make_affinity_histogram(result, **kwargs)
    return _save_figure_from_axes_or_figure(ax, output_path)


def save_best_pose_bar(
    result: PoseInteractionTableResult,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    ax = make_best_pose_bar(result, **kwargs)
    return _save_figure_from_axes_or_figure(ax, output_path)


def save_interaction_type_bar(
    result: PoseInteractionTableResult,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    ax = make_interaction_type_bar(result, **kwargs)
    return _save_figure_from_axes_or_figure(ax, output_path)


def save_residue_contact_bar(
    result: PoseInteractionTableResult,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    ax = make_residue_contact_bar(result, **kwargs)
    return _save_figure_from_axes_or_figure(ax, output_path)


def save_interaction_count_histogram(
    result: PoseInteractionTableResult,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    ax = make_interaction_count_histogram(result, **kwargs)
    return _save_figure_from_axes_or_figure(ax, output_path)


def save_affinity_vs_interaction_count_scatter(
    result: PoseInteractionTableResult,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    ax = make_affinity_vs_interaction_count_scatter(result, **kwargs)
    return _save_figure_from_axes_or_figure(ax, output_path)


def save_similarity_heatmap(
    result: PoseInteractionTableResult,
    output_path: str | Path,
    *,
    dpi: int = 300,
    **kwargs: Any,
) -> Path:
    ax = plot_similarity_heatmap(result, **kwargs)
    return _save_figure_from_axes_or_figure(ax, output_path, dpi=dpi)


__all__ = [
    "JournalStyle",
    "NATURE_STYLE",
    "SCIENCE_STYLE",
    "NATURE_BLUE_STYLE",
    "MONO_STYLE",
    "build_pose_visualization_table",
    "make_affinity_histogram",
    "save_affinity_histogram",
    "make_best_pose_bar",
    "save_best_pose_bar",
    "make_interaction_type_bar",
    "save_interaction_type_bar",
    "make_residue_contact_bar",
    "save_residue_contact_bar",
    "make_interaction_count_histogram",
    "save_interaction_count_histogram",
    "make_affinity_vs_interaction_count_scatter",
    "save_affinity_vs_interaction_count_scatter",
    "plot_similarity_heatmap",
    "save_similarity_heatmap",
    "make_summary_panel_2x3",
    "save_summary_panel_2x3",
]
