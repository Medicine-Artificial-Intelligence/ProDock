"""
prodock.preprocess.gridbox.gridbox
=================================

Utilities to compute docking grid boxes from ligand coordinates.

This module provides the :class:`GridBox` class which supports:

- loading ligands (from file-path or raw text),
- building grid boxes using multiple algorithms (scale, pad, advanced,
  percentile, PCA-AABB, centroid-fixed, union),
- post-processing (snap to grid, grow to minimal cube),
- exporting simple Vina-style snippets.

The implementation delegates low-level parsing to
:mod:`prodock.preprocess.gridbox.parsers` and the computational algorithms to
:mod:`prodock.preprocess.gridbox.algorithms`.

Example
-------

Basic usage::

    from prodock.preprocess.gridbox.gridbox import GridBox

    # build from a file (auto-detects format by suffix) and use scale algorithm
    gb = GridBox().load_ligand("ligand.sdf").from_ligand_scale(
        scale=2.0, isotropic=True
    )
    print(gb.center)           # (cx, cy, cz)
    print(gb.size)             # (sx, sy, sz)
    print(gb.to_vina_lines())  # vina-style snippet

Programmatic selection by name (helper)::

    from prodock.preprocess.gridbox.gridbox import compute_with_algo

    gb = compute_with_algo("pad", "ligand.sdf", pad=3.0, isotropic=False)
    print(gb.to_vina_lines())

You can also configure an algorithm on construction so it is applied
automatically after loading a ligand::

    gb = GridBox(algo="advanced", algo_kwargs={"pad": 4.0, "snap": 0.25})
    gb.load_ligand("ligand.sdf")  # configured algorithm runs automatically
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Iterable, Callable, Any

from rdkit import Chem

from .parsers import parse_text_to_mol
from .utils import round_tuple, snap_tuple, ensure_pos_size
from .algorithms import (
    expand_by_pad,
    expand_by_scale,
    expand_by_advanced,
    expand_by_percentile,
    expand_by_pca_aabb,
    centroid_fixed,
    union_boxes,
)

# -------------------------
# Algorithm dispatch map & helper
# -------------------------
# Each mapping receives (gb: GridBox, args: dict) and must return the GridBox.
ALGO_MAP: Dict[str, Callable[["GridBox", Dict[str, Any]], "GridBox"]] = {
    "scale": lambda gb, args: gb.from_ligand_scale(
        scale=args.get("scale", 2.0), isotropic=args.get("isotropic", False)
    ),
    "pad": lambda gb, args: gb.from_ligand_pad(
        pad=args.get("pad", 4.0),
        isotropic=args.get("isotropic", False),
        min_size=args.get("min_size", 0.0),
    ),
    "advanced": lambda gb, args: gb.from_ligand_pad_adv(
        pad=args.get("pad", 4.0),
        isotropic=args.get("isotropic", False),
        min_size=args.get("min_size", 0.0),
        heavy_only=args.get("heavy_only", False),
        snap_step=args.get("snap", None),
        round_ndigits=args.get("round_ndigits", 3),
    ),
    "percentile": lambda gb, args: gb.from_ligand_percentile(
        low=args.get("low", 5.0),
        high=args.get("high", 95.0),
        pad=args.get("pad", 0.0),
        isotropic=args.get("isotropic", False),
        round_ndigits=args.get("round_ndigits", 3),
    ),
    "pca-aabb": lambda gb, args: gb.from_ligand_pca_aabb(
        scale=args.get("pca_scale", 1.0),
        pad=args.get("pca_pad", 0.0),
        isotropic=args.get("isotropic", False),
        round_ndigits=args.get("round_ndigits", 3),
    ),
    "centroid-fixed": lambda gb, args: gb.from_centroid_fixed(tuple(args["size"])),
    "union": lambda gb, args: gb.from_union(
        args["paths"],
        fmt=args.get("fmt", None),
        pad=args.get("pad", 0.0),
        round_ndigits=args.get("round_ndigits", 3),
    ),
}


def compute_with_algo(algoname: str, ligand: Union[str, Path], **kwargs) -> "GridBox":
    """
    Convenience wrapper: load ligand, dispatch builder by name.

    :param algoname: Algorithm key from :data:`ALGO_MAP`.
    :type algoname: str
    :param ligand: Path or raw text for the ligand.
    :type ligand: str or pathlib.Path
    :param kwargs: Algorithm-specific parameters forwarded to the builder.
    :type kwargs: dict
    :return: GridBox instance after the selected algorithm was applied.
    :rtype: GridBox
    :raises ValueError: If the algorithm is unknown or parsing fails.
    """
    gb = GridBox()
    gb.load_ligand(ligand, fmt=kwargs.get("fmt"))
    fn = ALGO_MAP.get(algoname)
    if fn is None:
        raise ValueError(f"Unknown algorithm: {algoname}")
    return fn(gb, kwargs)


# -------------------------
# GridBox
# -------------------------
class GridBox:
    """
    Compute and represent a docking grid box.

    Builders return ``self`` so calls can be chained. You may optionally
    pass ``algo`` and ``algo_kwargs`` when constructing the GridBox. If
    provided, and a ligand is loaded (either via the constructor's ``mol``
    argument or later with :meth:`load_ligand`), the configured algorithm will
    be executed automatically.

    :param mol: Optional RDKit Mol used to initialize.
    :type mol: rdkit.Chem.rdchem.Mol or None
    :param algo: Optional algorithm name to run (key in :data:`ALGO_MAP`).
    :type algo: str or None
    :param algo_kwargs: Optional dict forwarded to the chosen algorithm.
    :type algo_kwargs: dict or None
    :param round_ndigits: Default rounding digits applied by some builders.
    :type round_ndigits: int
    """

    def __init__(
        self,
        mol: Optional[Chem.Mol] = None,
        algo: Optional[str] = None,
        algo_kwargs: Optional[Dict[str, Any]] = None,
        round_ndigits: int = 3,
    ) -> None:
        self._mol: Optional[Chem.Mol] = mol
        self._center: Optional[Tuple[float, float, float]] = None
        self._size: Optional[Tuple[float, float, float]] = None

        # store configured automatic algorithm (may be applied after a load)
        self._init_algo = algo
        self._init_algo_kwargs = dict(algo_kwargs or {})
        # ensure caller-provided rounding preference is present
        if "round_ndigits" not in self._init_algo_kwargs:
            self._init_algo_kwargs["round_ndigits"] = round_ndigits

        # if a mol provided at init and an algo is configured, apply it immediately
        if self._mol is not None and self._init_algo is not None:
            self._apply_init_algo()

    # -------------------------
    # Loading
    # -------------------------
    def load_ligand(
        self, data: Union[str, Path], fmt: Optional[str] = None
    ) -> "GridBox":
        """
        Load ligand from a path or raw text block. If an `algo` was supplied at
        construction time, it will be applied automatically after a successful
        parse.

        :param data: Path-like or raw text containing molecule data.
        :type data: str or pathlib.Path
        :param fmt: Optional format hint (e.g. 'sdf', 'pdb', 'mol2', 'xyz').
        :type fmt: str or None
        :return: self
        :rtype: GridBox
        :raises ValueError: If parsing fails.
        """
        mol = parse_text_to_mol(data, fmt=fmt)
        if mol is None:
            raise ValueError("Failed to parse ligand.")
        self._mol = mol

        # auto-apply configured initial algorithm (if any)
        if self._init_algo is not None:
            self._apply_init_algo()
        return self

    def _apply_init_algo(self) -> None:
        """
        Internal: apply algorithm stored from constructor using :data:`ALGO_MAP`.

        :raises ValueError: If the configured algorithm name is unknown.
        """
        if self._init_algo is None:
            return
        fn = ALGO_MAP.get(self._init_algo)
        if fn is None:
            raise ValueError(f"Unknown algorithm configured: {self._init_algo}")
        # call the mapping which mutates and returns the GridBox
        fn(self, self._init_algo_kwargs)

    # -------------------------
    # Builders (selectable)
    # -------------------------
    def from_ligand_scale(
        self, scale: float = 2.0, isotropic: bool = False, round_ndigits: int = 3
    ) -> "GridBox":
        """
        Build box by scaling the ligand axis-aligned bounding-box (AABB).

        size = span * scale; center = (min + max) / 2

        :param scale: Multiplier applied to ligand bounding span.
        :type scale: float
        :param isotropic: If True, use the maximum span * scale for all axes (cubic box).
        :type isotropic: bool
        :param round_ndigits: Decimal places to round output values to.
        :type round_ndigits: int
        :return: self with computed center and size.
        :rtype: GridBox
        :raises ValueError: If no ligand was loaded.
        """
        self._check_mol()
        self._center, self._size = expand_by_scale(
            self._mol, scale=scale, isotropic=isotropic, round_ndigits=round_ndigits
        )
        return self

    def from_ligand_pad(
        self,
        pad: Union[float, Tuple[float, float, float]] = 4.0,
        isotropic: bool = False,
        min_size: Union[float, Tuple[float, float, float]] = 0.0,
        round_ndigits: int = 3,
    ) -> "GridBox":
        """
        Build box by padding the ligand AABB.

        size = span + 2*pad (per-axis or scalar). Optionally enforce a minimum size per axis.

        :param pad: Padding in Å (scalar or triple).
        :type pad: float or tuple(float, float, float)
        :param isotropic: If True, use the maximum span to create a cube before applying padding.
        :type isotropic: bool
        :param min_size: Minimum edge lengths after expansion (scalar or triple).
        :type min_size: float or tuple(float, float, float)
        :param round_ndigits: Decimal places to round output values to.
        :type round_ndigits: int
        :return: self with computed center and size.
        :rtype: GridBox
        :raises ValueError: If no ligand was loaded.
        """
        self._check_mol()
        self._center, self._size = expand_by_pad(
            self._mol,
            pad=pad,
            isotropic=isotropic,
            min_size=min_size,
            round_ndigits=round_ndigits,
        )
        return self

    def from_ligand_pad_adv(
        self,
        pad: Union[float, Tuple[float, float, float]] = 4.0,
        isotropic: bool = False,
        min_size: Union[float, Tuple[float, float, float]] = 0.0,
        *,
        heavy_only: bool = False,
        snap_step: Optional[float] = None,
        round_ndigits: int = 3,
    ) -> "GridBox":
        """
        Advanced padding builder with heavy-atom filtering and optional snapping.

        :param pad: Padding in Å (scalar or triple).
        :type pad: float or tuple(float, float, float)
        :param isotropic: If True, produce a cubic box using the maximum span.
        :type isotropic: bool
        :param min_size: Minimum edge lengths after expansion (scalar or triple).
        :type min_size: float or tuple(float, float, float)
        :param heavy_only: If True, compute spans using heavy atoms only (exclude hydrogens).
        :type heavy_only: bool
        :param snap_step: If provided, snap center and size to multiples of this step (e.g., 0.25 Å).
        :type snap_step: float or None
        :param round_ndigits: Decimal places to round output values to.
        :type round_ndigits: int
        :return: self with computed center and size.
        :rtype: GridBox
        :raises ValueError: If no ligand was loaded.
        """
        self._check_mol()
        self._center, self._size = expand_by_advanced(
            self._mol,
            pad=pad,
            isotropic=isotropic,
            min_size=min_size,
            heavy_only=heavy_only,
            snap_step=snap_step,
            round_ndigits=round_ndigits,
        )
        return self

    def from_ligand_percentile(
        self,
        low: float = 5.0,
        high: float = 95.0,
        pad: float = 0.0,
        isotropic: bool = False,
        round_ndigits: int = 3,
    ) -> "GridBox":
        """
        Robust bounds using coordinate percentiles to reduce influence of outliers.

        size = (q_high - q_low) + 2*pad, center = (q_low + q_high) / 2

        :param low: Low percentile (0-100).
        :type low: float
        :param high: High percentile (0-100).
        :type high: float
        :param pad: Padding in Å added after percentile span is computed.
        :type pad: float
        :param isotropic: If True, make the box cubic using the maximum span.
        :type isotropic: bool
        :param round_ndigits: Decimal places to round output values to.
        :type round_ndigits: int
        :return: self with computed center and size.
        :rtype: GridBox
        :raises ValueError: If no ligand was loaded.
        """
        self._check_mol()
        self._center, self._size = expand_by_percentile(
            self._mol,
            low=low,
            high=high,
            pad=pad,
            isotropic=isotropic,
            round_ndigits=round_ndigits,
        )
        return self

    def from_ligand_pca_aabb(
        self,
        scale: float = 1.0,
        pad: float = 0.0,
        isotropic: bool = False,
        round_ndigits: int = 3,
    ) -> "GridBox":
        """
        PCA-oriented oriented bounding box (OBB) approximated with PCA, then converted
        to an axis-aligned bounding box (AABB) in the original frame.

        :param scale: Scale factor applied in PCA frame.
        :type scale: float
        :param pad: Padding (Å) applied in PCA frame.
        :type pad: float
        :param isotropic: If True, make the final box cubic using the maximum axis.
        :type isotropic: bool
        :param round_ndigits: Decimal places to round output values to.
        :type round_ndigits: int
        :return: self with computed center and size.
        :rtype: GridBox
        :raises ValueError: If no ligand was loaded.
        """
        self._check_mol()
        self._center, self._size = expand_by_pca_aabb(
            self._mol,
            scale=scale,
            pad=pad,
            isotropic=isotropic,
            round_ndigits=round_ndigits,
        )
        return self

    def from_centroid_fixed(self, size: Tuple[float, float, float]) -> "GridBox":
        """
        Center the box at the centroid (mean of atom coordinates) and use a user-specified size.

        :param size: Explicit box size (sx, sy, sz) in Å.
        :type size: tuple(float, float, float)
        :return: self with computed center and size.
        :rtype: GridBox
        :raises ValueError: If no ligand was loaded or size has non-positive components.
        """
        self._check_mol()
        self._center, self._size = centroid_fixed(self._mol, size=size)
        return self

    def from_union(
        self,
        ligand_paths: Iterable[Union[str, Path]],
        fmt: Optional[str] = None,
        pad: float = 0.0,
        round_ndigits: int = 3,
    ) -> "GridBox":
        """
        Build the axis-aligned union of AABBs computed for multiple ligands.

        :param ligand_paths: Iterable of paths or raw text blocks for ligands.
        :type ligand_paths: iterable
        :param fmt: Optional format hint used when a ligand entry is raw text.
        :type fmt: str or None
        :param pad: Optional padding (Å) applied when computing each ligand's box.
        :type pad: float
        :param round_ndigits: Decimal places to round output values to.
        :type round_ndigits: int
        :return: self with computed center and size representing the union.
        :rtype: GridBox
        :raises ValueError: If parsing of any ligand fails or no ligands provided.
        """
        boxes = []
        for path in ligand_paths:
            m = parse_text_to_mol(path, fmt=fmt)
            if m is None:
                raise ValueError(f"Failed to parse ligand: {path}")
            tmp = GridBox(m).from_ligand_pad(
                pad=pad, isotropic=False, min_size=0.0, round_ndigits=round_ndigits
            )
            boxes.append((tmp.center, tmp.size))

        if not boxes:
            raise ValueError("No ligands provided for union.")
        c, s = boxes[0]
        for c2, s2 in boxes[1:]:
            c, s = union_boxes(c, s, c2, s2)
        self._center, self._size = c, s
        return self

    # -------------------------
    # Post-process / export
    # -------------------------
    def grow_to_min_cube(self) -> "GridBox":
        """
        Expand the current box so all three edges are equal to the maximum edge.

        :return: self with cube-shaped size.
        :rtype: GridBox
        """
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        L = float(max(sx, sy, sz))
        self._center = (cx, cy, cz)
        self._size = (L, L, L)
        return self

    def snap(self, step: float = 0.25, round_ndigits: int = 3) -> "GridBox":
        """
        Snap center and size to multiples of `step` and round to `round_ndigits`.

        :param step: Snap step in Å.
        :type step: float
        :param round_ndigits: Decimal places to round after snapping.
        :type round_ndigits: int
        :return: self with snapped and rounded center/size.
        :rtype: GridBox
        """
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        self._center = round_tuple(snap_tuple((cx, cy, cz), step), round_ndigits)
        self._size = ensure_pos_size(
            round_tuple(snap_tuple((sx, sy, sz), step), round_ndigits)
        )
        return self

    # -------------------------
    # Properties / snippets
    # -------------------------
    @property
    def center(self) -> Tuple[float, float, float]:
        """
        The computed box center as (x, y, z) in Å.

        :return: 3-tuple of floats.
        :rtype: tuple(float, float, float)
        :raises ValueError: If the center has not been computed yet.
        """
        if self._center is None:
            raise ValueError("Center not computed yet.")
        return self._center

    @property
    def size(self) -> Tuple[float, float, float]:
        """
        The computed box size (width, height, depth) in Å.

        :return: 3-tuple of floats.
        :rtype: tuple(float, float, float)
        :raises ValueError: If the size has not been computed yet.
        """
        if self._size is None:
            raise ValueError("Size not computed yet.")
        return self._size

    @property
    def vina_dict(self) -> Dict[str, float]:
        """
        Dict suitable for writing a Vina config with keys:
        center_x/center_y/center_z and size_x/size_y/size_z.

        :return: dictionary of six floats.
        :rtype: dict
        """
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        return {
            "center_x": float(cx),
            "center_y": float(cy),
            "center_z": float(cz),
            "size_x": float(sx),
            "size_y": float(sy),
            "size_z": float(sz),
        }

    def to_vina_lines(self, fmt: str = "{k} = {v:.3f}") -> str:
        """
        Render a Vina-style multiline snippet.

        :param fmt: Per-line format string receiving ``k`` and ``v``.
        :type fmt: str
        :return: Multiline string.
        :rtype: str
        """
        d = self.vina_dict
        return "\n".join(fmt.format(k=k, v=v) for k, v in d.items())

    # -------------------------
    # Utils
    # -------------------------
    def as_tuple(self):
        """
        Return (center, size) as a pair of tuples.

        :return: (center, size)
        :rtype: tuple(tuple(float, float, float), tuple(float, float, float))
        """
        return self.center, self.size

    def __repr__(self) -> str:
        return f"<GridBox center={getattr(self,'_center',None)} size={getattr(self,'_size',None)}>"

    def _check_mol(self) -> None:
        """
        Internal helper to verify a molecule was loaded.

        :raises ValueError: if no molecule is present.
        """
        if self._mol is None:
            raise ValueError("No ligand loaded. Call load_ligand() first.")
