from __future__ import annotations

"""
Visualization helpers built on top of ProLIF plotting utilities.

This module provides convenience wrappers for rendering and saving common
interaction-analysis visualizations from an
:class:`~prodock.postprocess.interaction.models.InteractionRunResult`.

The helpers are designed to:

- validate that the required fingerprint object is available
- provide a stable ProDock-facing API around ProLIF plotting functions
- raise project-specific exceptions for missing optional dependencies
- support both direct display in interactive environments and saving figures
  to disk

Available functionality includes:

- barcode visualization of interaction fingerprints
- 3D protein-ligand interaction views
- 2D ligand interaction network views
- similarity heatmap plotting with matplotlib

Example
-------
.. code-block:: python

    from prodock.postprocess.interaction.visualization import (
        make_barcode,
        save_similarity_heatmap,
    )

    ax = make_barcode(result, figsize=(10, 8))
    save_similarity_heatmap(similarity_df, "similarity.png", annotate=True)
"""

from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import MissingDependencyError, VisualizationError
from .models import InteractionRunResult


def _import_prolif() -> Any:
    """
    Import :mod:`prolif` lazily.

    This helper centralizes the optional import so that plotting functions can
    fail with a project-specific exception instead of a raw import error.

    :returns:
        Imported :mod:`prolif` module object.
    :rtype: Any

    :raises MissingDependencyError:
        Raised when :mod:`prolif` is not installed or cannot be imported.

    Example
    -------
    .. code-block:: python

        plf = _import_prolif()
    """
    try:
        import prolif as plf
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "ProLIF is required for interaction visualisation."
        ) from exc
    return plf


def _import_matplotlib_pyplot() -> Any:
    """
    Import :mod:`matplotlib.pyplot` lazily.

    This helper centralizes the optional matplotlib import for plotting and
    saving barcode and heatmap figures.

    :returns:
        Imported :mod:`matplotlib.pyplot` module object.
    :rtype: Any

    :raises MissingDependencyError:
        Raised when :mod:`matplotlib` is not installed or cannot be imported.

    Example
    -------
    .. code-block:: python

        plt = _import_matplotlib_pyplot()
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "matplotlib is required for barcode and heatmap visualisation."
        ) from exc
    return plt


def make_barcode(result: InteractionRunResult, **display_kwargs: Any) -> Any:
    """
    Create and display an interaction barcode plot.

    This function renders a barcode-style summary of interaction fingerprints
    for the supplied run result. When the fingerprint object already provides a
    ``plot_barcode`` method, that method is used directly. Otherwise, a ProLIF
    barcode display object is constructed from the fingerprint.

    :param result:
        Interaction extraction result containing the fingerprint and associated
        molecules.
    :type result: InteractionRunResult
    :param display_kwargs:
        Additional keyword arguments forwarded to the underlying barcode display
        function or method.
    :type display_kwargs: Any

    :returns:
        Matplotlib axes object or backend-specific display object returned by
        the plotting implementation.
    :rtype: Any

    :raises VisualizationError:
        Raised when no fingerprint object is available in ``result``.
    :raises MissingDependencyError:
        Raised when ProLIF is required for fallback barcode construction but is
        not installed.

    Example
    -------
    .. code-block:: python

        ax = make_barcode(result, figsize=(10, 8))
    """
    if result.fingerprint is None:
        raise VisualizationError(
            "No fingerprint object is available for barcode plotting."
        )
    if hasattr(result.fingerprint, "plot_barcode"):
        return result.fingerprint.plot_barcode(**display_kwargs)
    plf = _import_prolif()
    barcode = plf.plotting.barcode.Barcode.from_fingerprint(result.fingerprint)
    return barcode.display(**display_kwargs)


def save_barcode(
    result: InteractionRunResult,
    output_path: str | Path,
    **display_kwargs: Any,
) -> Path:
    """
    Render a barcode plot and save it to disk.

    This function first creates the barcode visualization with
    :func:`make_barcode`, then extracts the associated matplotlib figure and
    writes it to ``output_path``.

    Parent directories are created automatically when needed.

    :param result:
        Interaction extraction result containing the fingerprint to visualize.
    :type result: InteractionRunResult
    :param output_path:
        Output image path for the saved barcode figure.
    :type output_path: str | pathlib.Path
    :param display_kwargs:
        Additional keyword arguments forwarded to :func:`make_barcode`.
    :type display_kwargs: Any

    :returns:
        Path to the created output file.
    :rtype: pathlib.Path

    :raises VisualizationError:
        Raised when the barcode plot does not expose a recoverable matplotlib
        figure.
    :raises MissingDependencyError:
        Raised when a required optional visualization dependency is not
        installed.

    Example
    -------
    .. code-block:: python

        path = save_barcode(result, "figures/barcode.png", figsize=(12, 8))
        print(path)
    """
    ax = make_barcode(result, **display_kwargs)
    fig = getattr(ax, "figure", None)
    if fig is None:
        raise VisualizationError(
            "Could not recover a matplotlib figure from the barcode plot."
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def make_complex3d(
    result: InteractionRunResult,
    frame: int = 0,
    **display_kwargs: Any,
) -> Any:
    """
    Create a 3D protein-ligand interaction view for one frame.

    This function visualizes interactions for a selected ligand frame or pose.
    When the fingerprint object provides a native ``plot_3d`` method, that
    method is used directly. Otherwise, the function falls back to constructing
    a ProLIF ``Complex3D`` view.

    :param result:
        Interaction extraction result containing the fingerprint, ligand
        molecules, and protein molecule.
    :type result: InteractionRunResult
    :param frame:
        Zero-based frame or molecule index to display.
    :type frame: int
    :param display_kwargs:
        Additional keyword arguments forwarded to the underlying 3D display
        function or method.
    :type display_kwargs: Any

    :returns:
        3D display object returned by the plotting backend.
    :rtype: Any

    :raises VisualizationError:
        Raised when no fingerprint object is available in ``result``.
    :raises IndexError:
        Raised when ``frame`` is outside the available ligand-molecule range.
    :raises MissingDependencyError:
        Raised when ProLIF is required for fallback 3D view construction but is
        not installed.

    Example
    -------
    .. code-block:: python

        view = make_complex3d(result, frame=0, size=(700, 600))
    """
    if result.fingerprint is None:
        raise VisualizationError("No fingerprint object is available for 3D plotting.")
    ligand_mol = result.prolif_molecules[frame]
    if hasattr(result.fingerprint, "plot_3d"):
        return result.fingerprint.plot_3d(
            ligand_mol,
            result.protein_molecule,
            frame=frame,
            **display_kwargs,
        )
    plf = _import_prolif()
    plot3d = plf.plotting.complex3d.Complex3D.from_fingerprint(
        result.fingerprint,
        ligand_mol,
        result.protein_molecule,
        frame=frame,
    )
    return plot3d.display(**display_kwargs)


def make_ligand_network(
    result: InteractionRunResult,
    *,
    frame: int = 0,
    kind: str = "aggregate",
    **display_kwargs: Any,
) -> Any:
    """
    Create a 2D ligand interaction network view.

    This function renders a ligand-centered interaction network either as an
    aggregate view across frames or as a single-frame view, depending on the
    selected ``kind``.

    When the fingerprint object provides a native ``plot_lignetwork`` method,
    that method is used directly. Otherwise, the function falls back to a
    ProLIF ligand-network object.

    :param result:
        Interaction extraction result containing the fingerprint and ligand
        molecules.
    :type result: InteractionRunResult
    :param frame:
        Zero-based frame index used when the selected network ``kind`` depends
        on a specific frame.
    :type frame: int
    :param kind:
        Network rendering mode. Typical values are ``"aggregate"`` for an
        overall interaction summary or ``"frame"`` for a single-frame view.
    :type kind: str
    :param display_kwargs:
        Additional keyword arguments forwarded to the underlying network
        display function or method.
    :type display_kwargs: Any

    :returns:
        Display object returned by the ligand-network plotting backend.
    :rtype: Any

    :raises VisualizationError:
        Raised when no fingerprint object is available in ``result``.
    :raises IndexError:
        Raised when ``frame`` is outside the available ligand-molecule range.
    :raises MissingDependencyError:
        Raised when ProLIF is required for fallback ligand-network construction
        but is not installed.

    Example
    -------
    .. code-block:: python

        net = make_ligand_network(result, kind="aggregate", threshold=0.2)
    """
    if result.fingerprint is None:
        raise VisualizationError(
            "No fingerprint object is available for ligand-network plotting."
        )
    ligand_mol = result.molecules[frame]
    if hasattr(result.fingerprint, "plot_lignetwork"):
        return result.fingerprint.plot_lignetwork(
            ligand_mol,
            kind=kind,
            frame=frame,
            **display_kwargs,
        )
    plf = _import_prolif()
    network = plf.plotting.network.LigNetwork.from_fingerprint(
        fp=result.fingerprint,
        ligand_mol=ligand_mol,
        kind=kind,
        frame=frame,
    )
    return network.display(**display_kwargs)


def plot_similarity_heatmap(
    similarity_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8.0, 6.0),
    annotate: bool = False,
) -> Any:
    """
    Plot a pairwise interaction-similarity heatmap.

    This helper uses matplotlib only and does not rely on seaborn. The input is
    expected to be a square similarity matrix, typically produced from
    fingerprint similarity calculations.

    :param similarity_df:
        Pairwise similarity matrix whose rows and columns represent the same
        set of molecules, poses, or systems.
    :type similarity_df: pandas.DataFrame
    :param figsize:
        Figure size passed to :func:`matplotlib.pyplot.subplots`.
    :type figsize: tuple[float, float]
    :param annotate:
        Whether to write formatted numeric similarity values inside each cell.
    :type annotate: bool

    :returns:
        Matplotlib axes object containing the heatmap.
    :rtype: Any

    :raises MissingDependencyError:
        Raised when :mod:`matplotlib` is not installed or cannot be imported.

    Example
    -------
    .. code-block:: python

        ax = plot_similarity_heatmap(similarity_df, figsize=(9, 7), annotate=True)
    """
    plt = _import_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(similarity_df.to_numpy())
    ax.set_xticks(range(len(similarity_df.columns)))
    ax.set_xticklabels(list(similarity_df.columns), rotation=90)
    ax.set_yticks(range(len(similarity_df.index)))
    ax.set_yticklabels(list(similarity_df.index))
    ax.set_xlabel("Molecule")
    ax.set_ylabel("Molecule")
    ax.set_title("Interaction fingerprint similarity")
    fig.colorbar(image, ax=ax)

    if annotate:
        values = similarity_df.to_numpy()
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    return ax


def save_similarity_heatmap(
    similarity_df: pd.DataFrame,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (8.0, 6.0),
    annotate: bool = False,
) -> Path:
    """
    Plot and save a similarity heatmap to disk.

    This function creates a heatmap with :func:`plot_similarity_heatmap`,
    extracts the underlying matplotlib figure, and writes it to ``output_path``.
    Parent directories are created automatically when needed.

    :param similarity_df:
        Pairwise similarity matrix to visualize.
    :type similarity_df: pandas.DataFrame
    :param output_path:
        Output image path for the saved heatmap figure.
    :type output_path: str | pathlib.Path
    :param figsize:
        Figure size passed through to :func:`plot_similarity_heatmap`.
    :type figsize: tuple[float, float]
    :param annotate:
        Whether to write formatted numeric similarity values inside each cell.
    :type annotate: bool

    :returns:
        Path to the created output file.
    :rtype: pathlib.Path

    :raises VisualizationError:
        Raised when the heatmap plot does not expose a recoverable matplotlib
        figure.
    :raises MissingDependencyError:
        Raised when :mod:`matplotlib` is not installed or cannot be imported.

    Example
    -------
    .. code-block:: python

        path = save_similarity_heatmap(
            similarity_df,
            "figures/similarity_heatmap.png",
            annotate=True,
        )
    """
    ax = plot_similarity_heatmap(similarity_df, figsize=figsize, annotate=annotate)
    fig = getattr(ax, "figure", None)
    if fig is None:
        raise VisualizationError(
            "Could not recover a matplotlib figure from the heatmap plot."
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path
