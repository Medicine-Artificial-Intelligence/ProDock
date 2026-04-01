"""
Utilities to compute docking grid boxes from ligand coordinates.

This module provides :class:`GridBox`, a lightweight object-oriented interface
for constructing docking search boxes from ligand coordinates. It supports:

- loading ligands from file paths or raw structure text,
- building boxes with multiple algorithms,
- post-processing boxes by snapping or cubic expansion,
- exporting Vina-style configuration snippets.

Low-level structure parsing is delegated to
:mod:`prodock.preprocess.gridbox.parsers`, while geometric box-construction
algorithms are delegated to :mod:`prodock.preprocess.gridbox.algorithms`.

Supported algorithms
--------------------
The module-level dispatch helper supports the following algorithm names:

- ``"scale"``
- ``"pad"``
- ``"advanced"``
- ``"percentile"``
- ``"pca-aabb"``
- ``"centroid-fixed"``
- ``"union"``

Example
-------
.. code-block:: python

    from prodock.preprocess.gridbox.gridbox import GridBox, compute_with_algo

    gb = GridBox().load_ligand("ligand.sdf").from_ligand_scale(
        scale=2.0,
        isotropic=True,
    )

    print(gb.center)
    print(gb.size)
    print(gb.to_vina_lines())

    gb2 = compute_with_algo("pad", "ligand.sdf", pad=3.0, isotropic=False)
    print(gb2.to_vina_lines())

Configured automatic execution is also supported:

.. code-block:: python

    gb = GridBox(algo="advanced", algo_kwargs={"pad": 4.0, "snap": 0.25})
    gb.load_ligand("ligand.sdf")
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

# Each mapping receives ``(gb, args)`` and must return the mutated GridBox.
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
    Load a ligand and compute a grid box using a named algorithm.

    This convenience wrapper constructs a :class:`GridBox`, loads the ligand,
    looks up the requested algorithm in :data:`ALGO_MAP`, and applies it using
    the provided keyword arguments.

    :param algoname:
        Algorithm key defined in :data:`ALGO_MAP`.
    :type algoname: str
    :param ligand:
        Ligand source, given either as a filesystem path or raw structure text.
    :type ligand: Union[str, Path]
    :param kwargs:
        Algorithm-specific keyword arguments forwarded to the selected builder.
    :type kwargs: dict

    :returns:
        Grid box after algorithm application.
    :rtype: GridBox

    :raises ValueError:
        If the algorithm name is unknown or ligand parsing fails.

    Example
    -------
    .. code-block:: python

        gb = compute_with_algo(
            "pad",
            "ligand.sdf",
            pad=4.0,
            isotropic=False,
        )
    """
    gb = GridBox()
    gb.load_ligand(ligand, fmt=kwargs.get("fmt"))
    fn = ALGO_MAP.get(algoname)
    if fn is None:
        raise ValueError(f"Unknown algorithm: {algoname}")
    return fn(gb, kwargs)


class GridBox:
    """
    Represent and compute a docking grid box.

    Instances store an optional ligand molecule together with a computed box
    center and size. Builder methods mutate the instance and return ``self`` so
    calls can be chained.

    An algorithm may optionally be configured at construction time via
    ``algo`` and ``algo_kwargs``. When present, that algorithm is applied
    automatically after a ligand is loaded, either through the constructor
    ``mol`` argument or later via :meth:`load_ligand`.

    :param mol:
        Optional RDKit molecule used to initialize the object.
    :type mol: Optional[Chem.Mol]
    :param algo:
        Optional algorithm name corresponding to a key in :data:`ALGO_MAP`.
    :type algo: Optional[str]
    :param algo_kwargs:
        Optional keyword arguments forwarded to the configured algorithm.
    :type algo_kwargs: Optional[Dict[str, Any]]
    :param round_ndigits:
        Default number of decimal places inserted into ``algo_kwargs`` when not
        already provided.
    :type round_ndigits: int

    Example
    -------
    .. code-block:: python

        gb = GridBox()
        gb.load_ligand("ligand.sdf")
        gb.from_ligand_pad(pad=4.0, isotropic=False)

        print(gb.center)
        print(gb.size)

    Automatic algorithm execution:

    .. code-block:: python

        gb = GridBox(
            algo="percentile",
            algo_kwargs={"low": 5.0, "high": 95.0, "pad": 2.0},
        )
        gb.load_ligand("ligand.sdf")
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

        self._init_algo = algo
        self._init_algo_kwargs = dict(algo_kwargs or {})
        if "round_ndigits" not in self._init_algo_kwargs:
            self._init_algo_kwargs["round_ndigits"] = round_ndigits

        if self._mol is not None and self._init_algo is not None:
            self._apply_init_algo()

    def load_ligand(
        self, data: Union[str, Path], fmt: Optional[str] = None
    ) -> "GridBox":
        """
        Load a ligand from a path or raw structure text.

        If an initialization algorithm was configured when the object was
        constructed, that algorithm is executed automatically after successful
        parsing.

        :param data:
            Ligand source as a filesystem path or raw molecular text.
        :type data: Union[str, Path]
        :param fmt:
            Optional explicit format hint such as ``"sdf"``, ``"pdb"``,
            ``"mol2"``, or ``"xyz"``.
        :type fmt: Optional[str]

        :returns:
            The current grid box instance.
        :rtype: GridBox

        :raises ValueError:
            If ligand parsing fails.

        Example
        -------
        .. code-block:: python

            gb = GridBox()
            gb.load_ligand("ligand.sdf")
        """
        mol = parse_text_to_mol(data, fmt=fmt)
        if mol is None:
            raise ValueError("Failed to parse ligand.")
        self._mol = mol

        if self._init_algo is not None:
            self._apply_init_algo()
        return self

    def _apply_init_algo(self) -> None:
        """
        Apply the algorithm configured at construction time.

        The configured algorithm is looked up in :data:`ALGO_MAP` and executed
        with the stored ``algo_kwargs``.

        :returns:
            ``None``.
        :rtype: None

        :raises ValueError:
            If the configured algorithm name is unknown.
        """
        if self._init_algo is None:
            return
        fn = ALGO_MAP.get(self._init_algo)
        if fn is None:
            raise ValueError(f"Unknown algorithm configured: {self._init_algo}")
        fn(self, self._init_algo_kwargs)

    def from_ligand_scale(
        self, scale: float = 2.0, isotropic: bool = False, round_ndigits: int = 3
    ) -> "GridBox":
        """
        Build a box by scaling the ligand axis-aligned bounding box.

        The box center is computed from the ligand coordinate bounds, and the
        box size is computed as ligand span multiplied by ``scale``.

        :param scale:
            Multiplicative factor applied to the ligand span.
        :type scale: float
        :param isotropic:
            If ``True``, use the maximum scaled span for all axes to produce a
            cubic box.
        :type isotropic: bool
        :param round_ndigits:
            Number of decimal places used to round output values.
        :type round_ndigits: int

        :returns:
            The current grid box instance.
        :rtype: GridBox

        :raises ValueError:
            If no ligand has been loaded.
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
        Build a box by padding the ligand axis-aligned bounding box.

        The resulting box size is the ligand span plus twice the padding, with
        optional minimum edge lengths enforced afterwards.

        :param pad:
            Padding in Ångström, provided either as a scalar or per-axis triple.
        :type pad: Union[float, Tuple[float, float, float]]
        :param isotropic:
            If ``True``, first convert the ligand span to a cubic span before
            applying padding.
        :type isotropic: bool
        :param min_size:
            Minimum allowed size, provided either as a scalar or per-axis triple.
        :type min_size: Union[float, Tuple[float, float, float]]
        :param round_ndigits:
            Number of decimal places used to round output values.
        :type round_ndigits: int

        :returns:
            The current grid box instance.
        :rtype: GridBox

        :raises ValueError:
            If no ligand has been loaded.
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
        Build a box with advanced padding logic.

        This method extends simple padding with optional heavy-atom-only bounds
        and optional snapping of the resulting center and size.

        :param pad:
            Padding in Ångström, provided either as a scalar or per-axis triple.
        :type pad: Union[float, Tuple[float, float, float]]
        :param isotropic:
            If ``True``, produce a cubic box using the maximum span.
        :type isotropic: bool
        :param min_size:
            Minimum allowed size, provided either as a scalar or per-axis triple.
        :type min_size: Union[float, Tuple[float, float, float]]
        :param heavy_only:
            If ``True``, compute the ligand bounds using heavy atoms only.
        :type heavy_only: bool
        :param snap_step:
            Optional snapping interval in Ångström applied to center and size.
        :type snap_step: Optional[float]
        :param round_ndigits:
            Number of decimal places used to round output values.
        :type round_ndigits: int

        :returns:
            The current grid box instance.
        :rtype: GridBox

        :raises ValueError:
            If no ligand has been loaded.
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
        Build a box from coordinate percentiles.

        This builder reduces the influence of outlier coordinates by using lower
        and upper coordinate percentiles instead of raw extrema.

        :param low:
            Lower percentile in the range ``0`` to ``100``.
        :type low: float
        :param high:
            Upper percentile in the range ``0`` to ``100``.
        :type high: float
        :param pad:
            Padding in Ångström applied after percentile bounds are computed.
        :type pad: float
        :param isotropic:
            If ``True``, make the final box cubic using the maximum span.
        :type isotropic: bool
        :param round_ndigits:
            Number of decimal places used to round output values.
        :type round_ndigits: int

        :returns:
            The current grid box instance.
        :rtype: GridBox

        :raises ValueError:
            If no ligand has been loaded.
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
        Build a box using a PCA-oriented bounding procedure.

        The ligand is analyzed in a PCA frame, expanded there, and the result is
        converted back to an axis-aligned bounding box in the original frame.

        :param scale:
            Scale factor applied in PCA space.
        :type scale: float
        :param pad:
            Padding in Ångström applied in PCA space.
        :type pad: float
        :param isotropic:
            If ``True``, make the final box cubic using the maximum axis.
        :type isotropic: bool
        :param round_ndigits:
            Number of decimal places used to round output values.
        :type round_ndigits: int

        :returns:
            The current grid box instance.
        :rtype: GridBox

        :raises ValueError:
            If no ligand has been loaded.
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
        Center the box at the ligand centroid and use a fixed user-supplied size.

        :param size:
            Explicit box size given as ``(sx, sy, sz)`` in Ångström.
        :type size: Tuple[float, float, float]

        :returns:
            The current grid box instance.
        :rtype: GridBox

        :raises ValueError:
            If no ligand has been loaded or if ``size`` contains non-positive
            values.
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
        Build the axis-aligned union of boxes computed for multiple ligands.

        Each ligand is parsed independently and converted into a padded box
        before all boxes are merged.

        :param ligand_paths:
            Iterable of ligand paths or raw text entries.
        :type ligand_paths: Iterable[Union[str, Path]]
        :param fmt:
            Optional format hint used when entries are raw text.
        :type fmt: Optional[str]
        :param pad:
            Padding in Ångström applied to each ligand before union.
        :type pad: float
        :param round_ndigits:
            Number of decimal places used to round output values.
        :type round_ndigits: int

        :returns:
            The current grid box instance.
        :rtype: GridBox

        :raises ValueError:
            If any ligand fails to parse or if no ligands are provided.

        Example
        -------
        .. code-block:: python

            gb = GridBox().from_union(
                ["lig1.sdf", "lig2.sdf", "lig3.sdf"],
                pad=2.0,
            )
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

    def grow_to_min_cube(self) -> "GridBox":
        """
        Expand the current box into the smallest cube containing it.

        The center is preserved, and all three edge lengths are set to the
        current maximum edge length.

        :returns:
            The current grid box instance.
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
        Snap the current center and size to a regular grid.

        Center and size values are snapped to multiples of ``step`` and then
        rounded to ``round_ndigits`` decimal places.

        :param step:
            Grid step size in Ångström.
        :type step: float
        :param round_ndigits:
            Number of decimal places used after snapping.
        :type round_ndigits: int

        :returns:
            The current grid box instance.
        :rtype: GridBox
        """
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        self._center = round_tuple(snap_tuple((cx, cy, cz), step), round_ndigits)
        self._size = ensure_pos_size(
            round_tuple(snap_tuple((sx, sy, sz), step), round_ndigits)
        )
        return self

    @property
    def center(self) -> Tuple[float, float, float]:
        """
        Return the computed box center.

        :returns:
            Box center as ``(x, y, z)`` in Ångström.
        :rtype: Tuple[float, float, float]

        :raises ValueError:
            If the center has not been computed yet.
        """
        if self._center is None:
            raise ValueError("Center not computed yet.")
        return self._center

    @property
    def size(self) -> Tuple[float, float, float]:
        """
        Return the computed box size.

        :returns:
            Box size as ``(sx, sy, sz)`` in Ångström.
        :rtype: Tuple[float, float, float]

        :raises ValueError:
            If the size has not been computed yet.
        """
        if self._size is None:
            raise ValueError("Size not computed yet.")
        return self._size

    @property
    def vina_dict(self) -> Dict[str, float]:
        """
        Return the box in a Vina-compatible dictionary representation.

        The returned dictionary contains the six standard Vina keys:
        ``center_x``, ``center_y``, ``center_z``, ``size_x``, ``size_y``,
        and ``size_z``.

        :returns:
            Dictionary of Vina-style box parameters.
        :rtype: Dict[str, float]
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
        Render the current box as a Vina-style multiline text block.

        :param fmt:
            Per-line format string receiving ``k`` and ``v`` fields.
        :type fmt: str

        :returns:
            Multiline text snippet containing Vina box parameters.
        :rtype: str

        Example
        -------
        .. code-block:: python

            print(gb.to_vina_lines())
        """
        d = self.vina_dict
        return "\n".join(fmt.format(k=k, v=v) for k, v in d.items())

    def as_tuple(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Return the box as ``(center, size)``.

        :returns:
            Pair of tuples containing center and size.
        :rtype: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        """
        return self.center, self.size

    def __repr__(self) -> str:
        return f"<GridBox center={getattr(self,'_center',None)} size={getattr(self,'_size',None)}>"

    def _check_mol(self) -> None:
        """
        Verify that a ligand molecule is currently loaded.

        :returns:
            ``None``.
        :rtype: None

        :raises ValueError:
            If no ligand is available.
        """
        if self._mol is None:
            raise ValueError("No ligand loaded. Call load_ligand() first.")
