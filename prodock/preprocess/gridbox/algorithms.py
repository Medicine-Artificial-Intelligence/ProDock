from __future__ import annotations

import numpy as np
from typing import Tuple, Union, Optional
from rdkit import Chem

from .utils import (
    coords_from_mol,
    gb_coords_from_mol,
    round_tuple,
    snap_tuple,
    ensure_pos_size,
    center_and_span,
)


def expand_by_pad(
    mol: Chem.Mol,
    pad: Union[float, Tuple[float, float, float]] = 4.0,
    isotropic: bool = False,
    min_size: Union[float, Tuple[float, float, float]] = 0.0,
    round_ndigits: int = 3,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    coords = coords_from_mol(mol)
    center, span = center_and_span(coords)
    if isotropic:
        base = float(np.max(span))
        size = np.array([base, base, base], dtype=float)
    else:
        size = span.astype(float)

    pad_vec = (
        np.array([pad, pad, pad], dtype=float)
        if isinstance(pad, (int, float))
        else np.array(pad, dtype=float)
    )
    size = size + 2.0 * pad_vec

    min_vec = (
        np.array([min_size, min_size, min_size], dtype=float)
        if isinstance(min_size, (int, float))
        else np.array(min_size, dtype=float)
    )
    size = np.maximum(size, min_vec)

    return round_tuple(center, round_ndigits), ensure_pos_size(
        round_tuple(size, round_ndigits)
    )


def expand_by_scale(
    mol: Chem.Mol,
    scale: float = 2.0,
    isotropic: bool = False,
    round_ndigits: int = 3,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    size = span * scale, center = (min+max)/2  (LaBOX-style).
    """
    coords = coords_from_mol(mol)
    center, span = center_and_span(coords)
    if isotropic:
        base = float(np.max(span)) * float(scale)
        size = np.array([base, base, base], dtype=float)
    else:
        size = span.astype(float) * float(scale)
    return round_tuple(center, round_ndigits), ensure_pos_size(
        round_tuple(size, round_ndigits)
    )


def expand_by_advanced(
    mol: Chem.Mol,
    pad: Union[float, Tuple[float, float, float]] = 4.0,
    isotropic: bool = False,
    min_size: Union[float, Tuple[float, float, float]] = 0.0,
    heavy_only: bool = False,
    snap_step: Optional[float] = None,
    round_ndigits: int = 3,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Heavy-atom option, snapping, and min_size handling.
    """
    coords = gb_coords_from_mol(mol, heavy_only=heavy_only)
    center, span = center_and_span(coords)
    if isotropic:
        base = float(np.max(span))
        size = np.array([base, base, base], dtype=float)
    else:
        size = span.astype(float)

    pad_vec = (
        np.array([pad, pad, pad], dtype=float)
        if isinstance(pad, (int, float))
        else np.array(pad, dtype=float)
    )
    size = size + 2.0 * pad_vec

    min_vec = (
        np.array([min_size, min_size, min_size], dtype=float)
        if isinstance(min_size, (int, float))
        else np.array(min_size, dtype=float)
    )
    size = np.maximum(size, min_vec)

    center_t = (float(center[0]), float(center[1]), float(center[2]))
    size_t = (float(size[0]), float(size[1]), float(size[2]))

    if snap_step:
        center_t = snap_tuple(center_t, snap_step)
        size_t = snap_tuple(size_t, snap_step)

    return round_tuple(center_t, round_ndigits), ensure_pos_size(
        round_tuple(size_t, round_ndigits)
    )


def expand_by_percentile(
    mol: Chem.Mol,
    low: float = 5.0,
    high: float = 95.0,
    pad: float = 0.0,
    isotropic: bool = False,
    round_ndigits: int = 3,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Robust bounds using coordinate percentiles to reduce influence of outliers.

    size = (q_high - q_low) + 2*pad, center = (q_low + q_high)/2
    """
    coords = coords_from_mol(mol)
    q_low = np.percentile(coords, low, axis=0)
    q_high = np.percentile(coords, high, axis=0)
    center = (q_low + q_high) / 2.0
    span = q_high - q_low

    if isotropic:
        base = float(np.max(span))
        size = np.array([base, base, base], dtype=float)
    else:
        size = span.astype(float)
    size = size + 2.0 * float(pad)

    return round_tuple(center, round_ndigits), ensure_pos_size(
        round_tuple(size, round_ndigits)
    )


def expand_by_pca_aabb(
    mol: Chem.Mol,
    scale: float = 1.0,
    pad: float = 0.0,
    isotropic: bool = False,
    round_ndigits: int = 3,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    PCA-oriented bounding box (OBB) estimated via PCA, then converted to an
    axis-aligned box (AABB) in the original frame by enclosing the rotated box.
    """
    X = coords_from_mol(mol)  # (N,3)
    mean = X.mean(axis=0)
    Y = X - mean
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    R = Vt  # rows are principal axes

    # project
    P = Y @ R.T
    pmin, pmax = P.min(axis=0), P.max(axis=0)
    pcenter = (pmin + pmax) / 2.0
    pspan = pmax - pmin

    # expand in PCA frame
    if isotropic:
        base = float(np.max(pspan)) * float(scale)
        psize = np.array([base, base, base], dtype=float)
    else:
        psize = pspan.astype(float) * float(scale)
    psize = psize + 2.0 * float(pad)

    # form OBB corner points in PCA frame
    hx, hy, hz = psize / 2.0
    corners = np.array(
        [
            [sx * hx, sy * hy, sz * hz]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ],
        dtype=float,
    )
    # translate to pcenter, rotate back, uncenter
    corners_world = (corners + pcenter) @ R + mean

    # final AABB
    wmin, wmax = corners_world.min(axis=0), corners_world.max(axis=0)
    center = (wmin + wmax) / 2.0
    size = wmax - wmin

    return round_tuple(center, round_ndigits), ensure_pos_size(
        round_tuple(size, round_ndigits)
    )


def centroid_fixed(
    mol: Chem.Mol,
    size: Tuple[float, float, float],
    round_ndigits: int = 3,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Center = centroid (mean of atom coords). Size is user-specified.
    """
    X = coords_from_mol(mol)
    center = X.mean(axis=0)
    return round_tuple(center, round_ndigits), ensure_pos_size(size)


def min_cube_from_size(size: tuple[float, float, float]) -> float:
    """
    Return the minimal cube edge length that contains a box with given size.

    :param size: (sx, sy, sz)
    :return: edge length L = max(sx, sy, sz)
    """
    sx, sy, sz = size
    return float(max(sx, sy, sz))


# -----------------------------
# Conversions & helpers
# -----------------------------
def pad_for_scale(span: np.ndarray, scale: float) -> Tuple[float, float, float]:
    """pad = span*(scale-1)/2 (per-axis)."""
    s = np.asarray(span, dtype=float)
    p = s * (float(scale) - 1.0) / 2.0
    return float(p[0]), float(p[1]), float(p[2])


def scale_for_pad(
    span: np.ndarray, pad: float | tuple[float, float, float]
) -> Tuple[float, float, float]:
    """scale = 1 + (2*pad)/span (per-axis); guards degenerate span."""
    s = np.asarray(span, dtype=float)
    p = (
        np.array([pad, pad, pad], dtype=float)
        if isinstance(pad, (int, float))
        else np.asarray(pad, dtype=float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = 1.0 + (2.0 * p) / np.where(s <= 0.0, np.nan, s)
    sc = np.nan_to_num(sc, nan=1.0, posinf=1e6, neginf=1.0)
    return float(sc[0]), float(sc[1]), float(sc[2])


def union_boxes(
    c1: tuple[float, float, float],
    s1: tuple[float, float, float],
    c2: tuple[float, float, float],
    s2: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """AABB union of two center/size boxes."""
    c1x, c1y, c1z = c1
    s1x, s1y, s1z = s1
    c2x, c2y, c2z = c2
    s2x, s2y, s2z = s2

    min1 = np.array([c1x - s1x / 2.0, c1y - s1y / 2.0, c1z - s1z / 2.0])
    max1 = np.array([c1x + s1x / 2.0, c1y + s1y / 2.0, c1z + s1z / 2.0])

    min2 = np.array([c2x - s2x / 2.0, c2y - s2y / 2.0, c2z - s2z / 2.0])
    max2 = np.array([c2x + s2x / 2.0, c2y + s2y / 2.0, c2z + s2z / 2.0])

    mn = np.minimum(min1, min2)
    mx = np.maximum(max1, max2)
    center = (mn + mx) / 2.0
    size = mx - mn
    return (float(center[0]), float(center[1]), float(center[2])), (
        float(size[0]),
        float(size[1]),
        float(size[2]),
    )
