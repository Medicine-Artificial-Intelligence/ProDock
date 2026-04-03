from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from rdkit import Chem

from .utils import (
    center_and_span,
    coords_from_mol,
    ensure_pos_size,
    gb_coords_from_mol,
    round_tuple,
    snap_tuple,
)

Vec3Like = Union[float, Tuple[float, float, float]]
Box3D = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


def _as_vec3(value: Vec3Like, *, name: str) -> np.ndarray:
    """
    Convert a scalar or length-3 tuple into a NumPy float vector.

    :param value:
        Scalar or 3-element tuple to convert.
    :type value: Union[float, Tuple[float, float, float]]

    :param name:
        Parameter name used in error messages.
    :type name: str

    :returns:
        Array of shape ``(3,)`` with ``dtype=float``.
    :rtype: np.ndarray

    :raises ValueError:
        If ``value`` is not a scalar and does not have exactly 3 elements.
    """
    if isinstance(value, (int, float)):
        return np.array([value, value, value], dtype=float)

    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a scalar or a 3-tuple, got shape {arr.shape}")
    return arr


def _to_tuple3(arr: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a length-3 NumPy array into a float tuple.

    :param arr:
        Input array of shape ``(3,)``.
    :type arr: np.ndarray

    :returns:
        Three-element float tuple.
    :rtype: Tuple[float, float, float]

    :raises ValueError:
        If the array shape is not ``(3,)``.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {arr.shape}")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _finalize_box(
    center: np.ndarray,
    size: np.ndarray,
    *,
    round_ndigits: int,
) -> Box3D:
    """
    Round and normalize a center/size box representation.

    :param center:
        Box center vector.
    :type center: np.ndarray

    :param size:
        Box size vector.
    :type size: np.ndarray

    :param round_ndigits:
        Number of decimal places used for rounding.
    :type round_ndigits: int

    :returns:
        ``(center, size)`` as rounded tuples.
    :rtype: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """
    center_t = round_tuple(_to_tuple3(center), round_ndigits)
    size_t = round_tuple(_to_tuple3(size), round_ndigits)
    return center_t, ensure_pos_size(size_t)


def expand_by_pad(
    mol: Chem.Mol,
    pad: Vec3Like = 4.0,
    isotropic: bool = False,
    min_size: Vec3Like = 0.0,
    round_ndigits: int = 3,
) -> Box3D:
    """
    Build a box from the molecular span plus symmetric padding.

    The center is the midpoint of the molecular axis-aligned bounding box.
    The size is computed as:

    - anisotropic: ``span + 2 * pad``
    - isotropic: ``[max(span), max(span), max(span)] + 2 * pad``

    A minimum size can then be enforced per axis.

    :param mol:
        Molecule with 3D coordinates.
    :type mol: Chem.Mol

    :param pad:
        Padding added on both sides of each axis. Can be a scalar or a 3-tuple.
    :type pad: Union[float, Tuple[float, float, float]]

    :param isotropic:
        If ``True``, use the maximum span on all three axes before padding.
    :type isotropic: bool

    :param min_size:
        Minimum allowed size per axis. Can be a scalar or a 3-tuple.
    :type min_size: Union[float, Tuple[float, float, float]]

    :param round_ndigits:
        Number of decimal places used for rounding output.
    :type round_ndigits: int

    :returns:
        ``(center, size)`` as 3-tuples.
    :rtype: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """
    coords = coords_from_mol(mol)
    center, span = center_and_span(coords)

    if isotropic:
        base = float(np.max(span))
        size = np.array([base, base, base], dtype=float)
    else:
        size = span.astype(float)

    size = size + 2.0 * _as_vec3(pad, name="pad")
    size = np.maximum(size, _as_vec3(min_size, name="min_size"))

    return _finalize_box(center, size, round_ndigits=round_ndigits)


def expand_by_scale(
    mol: Chem.Mol,
    scale: float = 2.0,
    isotropic: bool = False,
    round_ndigits: int = 3,
) -> Box3D:
    """
    Build a box by scaling the molecular span around its center.

    This follows a LaBOX-style convention:

    - ``center = (min + max) / 2``
    - ``size = span * scale``

    :param mol:
        Molecule with 3D coordinates.
    :type mol: Chem.Mol

    :param scale:
        Multiplicative factor applied to the span.
    :type scale: float

    :param isotropic:
        If ``True``, use ``max(span) * scale`` on all three axes.
    :type isotropic: bool

    :param round_ndigits:
        Number of decimal places used for rounding output.
    :type round_ndigits: int

    :returns:
        ``(center, size)`` as 3-tuples.
    :rtype: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """
    coords = coords_from_mol(mol)
    center, span = center_and_span(coords)

    if isotropic:
        base = float(np.max(span)) * float(scale)
        size = np.array([base, base, base], dtype=float)
    else:
        size = span.astype(float) * float(scale)

    return _finalize_box(center, size, round_ndigits=round_ndigits)


def expand_by_advanced(
    mol: Chem.Mol,
    pad: Vec3Like = 4.0,
    isotropic: bool = False,
    min_size: Vec3Like = 0.0,
    heavy_only: bool = False,
    snap_step: Optional[float] = None,
    round_ndigits: int = 3,
) -> Box3D:
    """
    Build a padded box with optional heavy-atom filtering and snapping.

    Compared with :func:`expand_by_pad`, this variant supports:

    - using only heavy atoms
    - optional snapping of center and size to a grid
    - minimum size enforcement

    :param mol:
        Molecule with 3D coordinates.
    :type mol: Chem.Mol

    :param pad:
        Padding added on both sides of each axis. Can be a scalar or a 3-tuple.
    :type pad: Union[float, Tuple[float, float, float]]

    :param isotropic:
        If ``True``, use the maximum span on all three axes before padding.
    :type isotropic: bool

    :param min_size:
        Minimum allowed size per axis. Can be a scalar or a 3-tuple.
    :type min_size: Union[float, Tuple[float, float, float]]

    :param heavy_only:
        If ``True``, compute the box from heavy atoms only.
    :type heavy_only: bool

    :param snap_step:
        Optional grid spacing used to snap both center and size.
    :type snap_step: Optional[float]

    :param round_ndigits:
        Number of decimal places used for rounding output.
    :type round_ndigits: int

    :returns:
        ``(center, size)`` as 3-tuples.
    :rtype: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """
    coords = gb_coords_from_mol(mol, heavy_only=heavy_only)
    center, span = center_and_span(coords)

    if isotropic:
        base = float(np.max(span))
        size = np.array([base, base, base], dtype=float)
    else:
        size = span.astype(float)

    size = size + 2.0 * _as_vec3(pad, name="pad")
    size = np.maximum(size, _as_vec3(min_size, name="min_size"))

    center_t = _to_tuple3(center)
    size_t = _to_tuple3(size)

    if snap_step is not None:
        center_t = snap_tuple(center_t, snap_step)
        size_t = snap_tuple(size_t, snap_step)

    return (
        round_tuple(center_t, round_ndigits),
        ensure_pos_size(round_tuple(size_t, round_ndigits)),
    )


def expand_by_percentile(
    mol: Chem.Mol,
    low: float = 5.0,
    high: float = 95.0,
    pad: float = 0.0,
    isotropic: bool = False,
    round_ndigits: int = 3,
) -> Box3D:
    """
    Build a robust box using coordinate percentiles.

    This reduces the influence of outlier atoms by replacing strict min/max
    bounds with percentile bounds:

    - ``q_low = percentile(coords, low)``
    - ``q_high = percentile(coords, high)``
    - ``center = (q_low + q_high) / 2``
    - ``size = (q_high - q_low) + 2 * pad``

    :param mol:
        Molecule with 3D coordinates.
    :type mol: Chem.Mol

    :param low:
        Lower percentile bound.
    :type low: float

    :param high:
        Upper percentile bound.
    :type high: float

    :param pad:
        Scalar padding added on both sides of all axes.
    :type pad: float

    :param isotropic:
        If ``True``, use the maximum robust span on all three axes.
    :type isotropic: bool

    :param round_ndigits:
        Number of decimal places used for rounding output.
    :type round_ndigits: int

    :returns:
        ``(center, size)`` as 3-tuples.
    :rtype: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
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

    return _finalize_box(center, size, round_ndigits=round_ndigits)


def expand_by_pca_aabb(
    mol: Chem.Mol,
    scale: float = 1.0,
    pad: float = 0.0,
    isotropic: bool = False,
    round_ndigits: int = 3,
) -> Box3D:
    """
    Estimate a PCA-oriented bounding box and return its enclosing AABB.

    The method:
    1. centers the coordinates
    2. computes PCA axes using SVD
    3. measures span in PCA space
    4. expands the PCA-space box by ``scale`` and ``pad``
    5. transforms its corners back to world coordinates
    6. returns the enclosing axis-aligned bounding box in the original frame

    :param mol:
        Molecule with 3D coordinates.
    :type mol: Chem.Mol

    :param scale:
        Multiplicative factor applied to the PCA-space span.
    :type scale: float

    :param pad:
        Scalar padding added on both sides of each PCA axis.
    :type pad: float

    :param isotropic:
        If ``True``, use the maximum PCA-space span on all three axes.
    :type isotropic: bool

    :param round_ndigits:
        Number of decimal places used for rounding output.
    :type round_ndigits: int

    :returns:
        ``(center, size)`` as 3-tuples.
    :rtype: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """
    X = coords_from_mol(mol)
    mean = X.mean(axis=0)
    Y = X - mean

    _, _, vt = np.linalg.svd(Y, full_matrices=False)
    rotation = vt

    projected = Y @ rotation.T
    pmin = projected.min(axis=0)
    pmax = projected.max(axis=0)
    pcenter = (pmin + pmax) / 2.0
    pspan = pmax - pmin

    if isotropic:
        base = float(np.max(pspan)) * float(scale)
        psize = np.array([base, base, base], dtype=float)
    else:
        psize = pspan.astype(float) * float(scale)

    psize = psize + 2.0 * float(pad)

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

    corners_world = (corners + pcenter) @ rotation + mean

    wmin = corners_world.min(axis=0)
    wmax = corners_world.max(axis=0)
    center = (wmin + wmax) / 2.0
    size = wmax - wmin

    return _finalize_box(center, size, round_ndigits=round_ndigits)


def centroid_fixed(
    mol: Chem.Mol,
    size: Tuple[float, float, float],
    round_ndigits: int = 3,
) -> Box3D:
    """
    Build a box centered at the molecular centroid with user-specified size.

    :param mol:
        Molecule with 3D coordinates.
    :type mol: Chem.Mol

    :param size:
        Desired fixed box size ``(sx, sy, sz)``.
    :type size: Tuple[float, float, float]

    :param round_ndigits:
        Number of decimal places used for rounding the center.
    :type round_ndigits: int

    :returns:
        ``(center, size)`` as 3-tuples.
    :rtype: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """
    coords = coords_from_mol(mol)
    center = coords.mean(axis=0)
    center_t = round_tuple(_to_tuple3(center), round_ndigits)
    return center_t, ensure_pos_size(size)


def min_cube_from_size(size: Tuple[float, float, float]) -> float:
    """
    Return the minimal cube edge length that contains a given box.

    :param size:
        Box size as ``(sx, sy, sz)``.
    :type size: Tuple[float, float, float]

    :returns:
        Minimal cube edge length, equal to ``max(size)``.
    :rtype: float
    """
    sx, sy, sz = size
    return float(max(sx, sy, sz))


def pad_for_scale(span: np.ndarray, scale: float) -> Tuple[float, float, float]:
    """
    Convert a scale factor into symmetric per-axis padding.

    The formula is:

    ``pad = span * (scale - 1) / 2``

    :param span:
        Original span vector.
    :type span: np.ndarray

    :param scale:
        Multiplicative span scaling factor.
    :type scale: float

    :returns:
        Per-axis padding as ``(px, py, pz)``.
    :rtype: Tuple[float, float, float]
    """
    span_arr = np.asarray(span, dtype=float)
    pad_arr = span_arr * (float(scale) - 1.0) / 2.0
    return _to_tuple3(pad_arr)


def scale_for_pad(
    span: np.ndarray,
    pad: Union[float, Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """
    Convert symmetric padding into per-axis scale factors.

    The formula is:

    ``scale = 1 + (2 * pad) / span``

    Degenerate or non-positive span values are guarded and mapped to a default
    finite result.

    :param span:
        Original span vector.
    :type span: np.ndarray

    :param pad:
        Padding as a scalar or 3-tuple.
    :type pad: Union[float, Tuple[float, float, float]]

    :returns:
        Per-axis scale values as ``(sx, sy, sz)``.
    :rtype: Tuple[float, float, float]
    """
    span_arr = np.asarray(span, dtype=float)
    pad_arr = _as_vec3(pad, name="pad")

    with np.errstate(divide="ignore", invalid="ignore"):
        scale_arr = 1.0 + (2.0 * pad_arr) / np.where(span_arr <= 0.0, np.nan, span_arr)

    scale_arr = np.nan_to_num(scale_arr, nan=1.0, posinf=1e6, neginf=1.0)
    return _to_tuple3(scale_arr)


def union_boxes(
    c1: Tuple[float, float, float],
    s1: Tuple[float, float, float],
    c2: Tuple[float, float, float],
    s2: Tuple[float, float, float],
) -> Box3D:
    """
    Return the AABB union of two boxes represented by center and size.

    :param c1:
        Center of the first box.
    :type c1: Tuple[float, float, float]

    :param s1:
        Size of the first box.
    :type s1: Tuple[float, float, float]

    :param c2:
        Center of the second box.
    :type c2: Tuple[float, float, float]

    :param s2:
        Size of the second box.
    :type s2: Tuple[float, float, float]

    :returns:
        Union box as ``(center, size)``.
    :rtype: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """
    c1_arr = np.asarray(c1, dtype=float)
    s1_arr = np.asarray(s1, dtype=float)
    c2_arr = np.asarray(c2, dtype=float)
    s2_arr = np.asarray(s2, dtype=float)

    min1 = c1_arr - s1_arr / 2.0
    max1 = c1_arr + s1_arr / 2.0

    min2 = c2_arr - s2_arr / 2.0
    max2 = c2_arr + s2_arr / 2.0

    mn = np.minimum(min1, min2)
    mx = np.maximum(max1, max2)

    center = (mn + mx) / 2.0
    size = mx - mn

    return _to_tuple3(center), _to_tuple3(size)
