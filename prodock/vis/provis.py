# provis.py
"""
A thin wrapper around py3Dmol to provide convenient receptor/ligand
display and GridBox visualization helpers.

Dependencies
------------
- py3Dmol
- pathlib

This module exposes :class:`ProVis`, a small, chainable convenience wrapper
around py3Dmol.view that captures loaded ligand metadata and provides a few
helpers for drawing GridBox instances computed externally (see prodock.process.gridbox.GridBox).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import py3Dmol

from ..process.gridbox import GridBox


class ProVis:
    """
    ProDock visualization wrapper around py3Dmol.

    The class intentionally uses a small, chainable API so calls can be composed:
    ``ProVis().load_receptor(...).load_ligand(...).style_preset(...).show()``.

    Example
    -------
    >>> viz = ProVis(800, 600)
    >>> viz.load_receptor("protein.pdb").load_ligand("lig.sdf", fmt="sdf")
    >>> viz.set_receptor_style("cartoon", "white").highlight_ligand("stick", "cyan")
    >>> viz.show()

    Parameters
    ----------
    vw : int, optional
        Viewer width in pixels. Defaults to 700.
    vh : int, optional
        Viewer height in pixels. Defaults to 500.
    """

    def __init__(self, vw: int = 700, vh: int = 500) -> None:
        """
        Create a ProVis viewer.

        :param vw: viewer width in pixels.
        :type vw: int
        :param vh: viewer height in pixels.
        :type vh: int
        """
        self._viewer = py3Dmol.view(width=vw, height=vh)
        self._model_count = -1
        self._ligands_meta: List[dict] = []

    # -------------------------
    # IO
    # -------------------------
    @staticmethod
    def _read_file(inpt_file: Union[str, Path]) -> str:
        """Read text from a file path.

        :param inpt_file: Path-like or string path to read.
        :type inpt_file: str | pathlib.Path
        :return: File contents as a string.
        :rtype: str
        """
        with open(inpt_file, "r") as f:
            return f.read()

    def load_receptor(self, inpt_file: Union[str, Path]) -> "ProVis":
        """
        Load receptor PDB (added as model 0).

        :param inpt_file: Path to a receptor PDB file.
        :type inpt_file: str | pathlib.Path
        :return: self (chainable).
        :rtype: ProVis
        """
        data = self._read_file(inpt_file)
        self._viewer.addModel(data, "pdb")
        self._model_count += 1
        return self

    def load_ligand(self, inpt_file: Union[str, Path], fmt: str = "sdf") -> "ProVis":
        """
        Load a ligand from a file path and remember its metadata.

        :param inpt_file: Path to ligand file.
        :type inpt_file: str | pathlib.Path
        :param fmt: Format of the ligand file (e.g., 'sdf','pdb','mol2','xyz').
        :type fmt: str
        :return: self (chainable).
        :rtype: ProVis
        """
        data = self._read_file(inpt_file)
        self._viewer.addModel(data, fmt)
        self._model_count += 1
        self._ligands_meta.append(
            {
                "model": self._model_count,
                "data": data,
                "fmt": fmt.lower(),
                "name": Path(str(inpt_file)).name,
            }
        )
        return self

    def load_ligand_from_text(
        self, text: str, name: str = "ligand", fmt: str = "sdf"
    ) -> "ProVis":
        """
        Load ligand from a raw text block (useful for pasted SDF/PDB data).

        :param text: Ligand content as a string.
        :type text: str
        :param name: Display name for the ligand (used in metadata).
        :type name: str
        :param fmt: Format of the ligand text ('sdf', 'pdb', 'mol2', 'xyz').
        :type fmt: str
        :return: self (chainable).
        :rtype: ProVis
        """
        self._viewer.addModel(text, fmt)
        self._model_count += 1
        self._ligands_meta.append(
            {"model": self._model_count, "data": text, "fmt": fmt.lower(), "name": name}
        )
        return self

    # -------------------------
    # Styles
    # -------------------------
    def set_receptor_style(
        self, style: str = "cartoon", color: str = "spectrum"
    ) -> "ProVis":
        """
        Apply a style to the receptor model (model 0).

        :param style: Representation style ('cartoon','stick', etc.).
        :type style: str
        :param color: Color or color scheme to apply.
        :type color: str
        :return: self (chainable).
        :rtype: ProVis
        """
        self._viewer.setStyle({"model": 0}, {style: {"color": color}})
        return self

    def highlight_ligand(
        self,
        style: str = "stick",
        color: str = "cyan",
        radius: float = 0.25,
        opacity: float = 1.0,
    ) -> "ProVis":
        """
        Apply a visual style to all loaded ligand models.

        :param style: One of 'stick','sphere','line','cartoon'. Falls back to 'stick'.
        :type style: str
        :param color: Color string (e.g., 'cyan' or '0x00ffff').
        :type color: str
        :param radius: Radius for stick/sphere representations.
        :type radius: float
        :param opacity: Opacity used for cartoon representation.
        :type opacity: float
        :return: self (chainable).
        :rtype: ProVis
        """
        for meta in self._ligands_meta:
            lig_model = meta["model"]
            if style == "stick":
                rep = {"stick": {"color": color, "radius": radius}}
            elif style == "sphere":
                rep = {"sphere": {"color": color, "radius": radius}}
            elif style == "line":
                rep = {"line": {"color": color}}
            elif style == "cartoon":
                rep = {"cartoon": {"color": color, "opacity": opacity}}
            else:
                rep = {"stick": {"color": color, "radius": radius}}
            self._viewer.setStyle({"model": lig_model}, rep)
        return self

    def add_surface(self, opacity: float = 0.35, color: str = "lightgray") -> "ProVis":
        """
        Add a solvent-excluded surface (SES) around the receptor model.

        :param opacity: Surface opacity in range [0,1].
        :type opacity: float
        :param color: Surface color string.
        :type color: str
        :return: self (chainable).
        :rtype: ProVis
        """
        self._viewer.addSurface(
            "SES", {"opacity": opacity, "color": color}, {"model": 0}
        )
        return self

    # -------------------------
    # Grid box drawing
    # -------------------------
    def add_gridbox(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        color: str = "skyBlue",
        opacity: float = 0.6,
    ) -> "ProVis":
        """
        Draw a numeric box specified by center and size.

        :param center: Tuple (x, y, z) specifying the box center.
        :type center: tuple[float, float, float]
        :param size: Tuple (w, h, d) specifying box dimensions (Å).
        :type size: tuple[float, float, float]
        :param color: Box color name or hex string.
        :type color: str
        :param opacity: Box face opacity.
        :type opacity: float
        :return: self (chainable).
        :rtype: ProVis
        """
        self._viewer.addBox(
            {
                "center": {"x": center[0], "y": center[1], "z": center[2]},
                "dimensions": {"w": size[0], "h": size[1], "d": size[2]},
                "color": color,
                "opacity": opacity,
            }
        )
        return self

    def add_gridbox_with_labels(
        self,
        grid: "GridBox",
        color: str = "skyBlue",
        opacity: float = 0.6,
        show_center: bool = True,
        show_sizes: bool = True,
        label_fontsize: int = 12,
        label_bg: bool = False,
        label_offset_factor: float = 0.05,
    ) -> "ProVis":
        """
        Draw a GridBox and add textual labels for center coordinates and sizes.

        :param grid: GridBox instance providing `.center` and `.size` attributes.
        :type grid: GridBox
        :param color: Box color.
        :type color: str
        :param opacity: Box opacity.
        :type opacity: float
        :param show_center: Whether to show a center coordinate label.
        :type show_center: bool
        :param show_sizes: Whether to show per-axis size labels.
        :type show_sizes: bool
        :param label_fontsize: Font size for labels (px).
        :type label_fontsize: int
        :param label_bg: Whether to draw a label background rectangle.
        :type label_bg: bool
        :param label_offset_factor: Fraction of axis length used to offset size labels away from the box.
        :type label_offset_factor: float
        :return: self (chainable).
        :rtype: ProVis
        """
        # draw box
        self.add_gridbox(grid.center, grid.size, color=color, opacity=opacity)
        cx, cy, cz = grid.center
        sx, sy, sz = grid.size

        label_opts_base: Dict[str, Any] = {
            "fontSize": label_fontsize,
            "fontColor": "0x000000",
            "showBackground": label_bg,
            "inFront": True,
            "useScreen": False,
        }

        def _add_label(text: str, pos: Tuple[float, float, float]):
            opts = dict(label_opts_base)
            opts["position"] = {"x": pos[0], "y": pos[1], "z": pos[2]}
            self._viewer.addLabel(text, opts)

        if show_center:
            _add_label(
                f"center: {cx:.3f}, {cy:.3f}, {cz:.3f}", (cx, cy + (sz * 0.02), cz)
            )
        if show_sizes:
            off_x = sx * 0.5 + max(sx, 1.0) * label_offset_factor
            off_y = sy * 0.5 + max(sy, 1.0) * label_offset_factor
            off_z = sz * 0.5 + max(sz, 1.0) * label_offset_factor
            _add_label(f"size_x = {sx:.3f} Å", (cx + off_x, cy, cz))
            _add_label(f"size_y = {sy:.3f} Å", (cx, cy + off_y, cz))
            _add_label(f"size_z = {sz:.3f} Å", (cx, cy, cz + off_z))

        return self

    def add_gridbox_from(
        self,
        grid: "GridBox",
        color: str = "skyBlue",
        opacity: float = 0.6,
        labels: bool = False,
    ) -> "ProVis":
        """
        Draw a GridBox; optionally add labels.

        :param grid: GridBox to draw (must have `.center` and `.size`).
        :type grid: GridBox
        :param color: Box color string.
        :type color: str
        :param opacity: Box opacity.
        :type opacity: float
        :param labels: If True, draw labels (center & sizes).
        :type labels: bool
        :return: self (chainable).
        :rtype: ProVis
        """
        return (
            self.add_gridbox_with_labels(grid, color=color, opacity=opacity)
            if labels
            else self.add_gridbox(grid.center, grid.size, color=color, opacity=opacity)
        )

    def add_gridbox_around_ligand(
        self,
        ligand_index: int = -1,
        pad: Union[float, Tuple[float, float, float]] = 4.0,
        isotropic: bool = False,
        min_size: Union[float, Tuple[float, float, float]] = 0.0,
        color: str = "skyBlue",
        opacity: float = 0.6,
        labels: bool = False,
    ) -> "ProVis":
        """
        Compute a GridBox from a previously loaded ligand and draw it.

        This convenience method computes a GridBox by delegating to GridBox
        (computational logic kept in GridBox) and then draws it.

        :param ligand_index: Index of the loaded ligand to base the box on.
                             Negative indices are supported (default -1 -> last).
        :type ligand_index: int
        :param pad: Padding in Å around ligand (scalar or per-axis tuple).
        :type pad: float | tuple[float, float, float]
        :param isotropic: If True, force cubic box sizing.
        :type isotropic: bool
        :param min_size: Minimal edge length (scalar or per-axis tuple).
        :type min_size: float | tuple[float, float, float]
        :param color: Box color.
        :type color: str
        :param opacity: Box opacity.
        :type opacity: float
        :param labels: Whether to draw labels next to the box.
        :type labels: bool
        :return: self (chainable).
        :rtype: ProVis
        :raises ValueError: if no ligand is loaded.
        :raises IndexError: if the ligand_index is out of range.
        """
        if not self._ligands_meta:
            raise ValueError("No ligand loaded. Use load_ligand() first.")
        if ligand_index < 0:
            ligand_index = len(self._ligands_meta) + ligand_index
        if ligand_index < 0 or ligand_index >= len(self._ligands_meta):
            raise IndexError("ligand_index out of range")
        meta = self._ligands_meta[ligand_index]
        # compute box using GridBox locally to avoid importing cycles in single-file use
        # Note: GridBox is also imported at module top; this local import mirrors previous behavior
        from ..process.gridbox import (
            GridBox as _GridBox,
        )  # local alias to emphasize local use

        gb = (
            _GridBox()
            .load_ligand(meta["data"], fmt=meta["fmt"])
            .from_ligand_pad(pad=pad, isotropic=isotropic, min_size=min_size)
        )
        return self.add_gridbox_from(gb, color=color, opacity=opacity, labels=labels)

    # -------------------------
    # Convenience / styling helpers
    # -------------------------
    def load(
        self,
        receptor: Optional[Union[str, Path]] = None,
        ligand: Optional[Union[str, Path]] = None,
        ligand_fmt: Optional[str] = None,
    ) -> "ProVis":
        """
        Convenience loader for receptor and ligand in a single call.

        :param receptor: Path to receptor PDB file (optional).
        :type receptor: str | pathlib.Path | None
        :param ligand: Path to ligand file (optional).
        :type ligand: str | pathlib.Path | None
        :param ligand_fmt: Optional ligand format to pass to load_ligand.
        :type ligand_fmt: str | None
        :return: self (chainable).
        :rtype: ProVis
        """
        if receptor:
            self.load_receptor(receptor)
        if ligand:
            self.load_ligand(ligand, fmt=(ligand_fmt or "sdf"))
        return self

    def style_preset(
        self,
        name: str = "publication",
        *,
        ligand_style: str = "stick",
        background: Optional[str] = None,
        surface: bool = False,
    ) -> "ProVis":
        """
        Apply a quick visual styling preset.

        Presets supported: 'publication', 'dark', 'surface'.

        :param name: Preset name.
        :type name: str
        :param ligand_style: Representation to use for ligands.
        :type ligand_style: str
        :param background: Optional background color (three.js hex-like '0x...' or CSS).
        :type background: str | None
        :param surface: If True, draw a receptor surface as part of the preset.
        :type surface: bool
        :return: self (chainable).
        :rtype: ProVis
        :raises ValueError: if an unknown preset name is given.
        """
        name = name.lower()
        if name == "publication":
            self.set_receptor_style("cartoon", "white")
            if surface:
                self.add_surface(opacity=0.25, color="lightgray")
            self.highlight_ligand(style=ligand_style, color="cyan", radius=0.25)
            self.set_background(background or "0xFFFFFF")
        elif name == "dark":
            self.set_receptor_style("cartoon", "spectrum")
            if surface:
                self.add_surface(opacity=0.25, color="gray")
            self.highlight_ligand(style=ligand_style, color="yellow", radius=0.25)
            self.set_background(background or "0x111111")
        elif name == "surface":
            self.set_receptor_style("cartoon", "lightgray").add_surface(
                opacity=0.35, color="lightgray"
            )
            self.highlight_ligand(style=ligand_style, color="magenta", radius=0.25)
            self.set_background(background or "0xFFFFFF")
        else:
            raise ValueError(f"Unknown style preset: {name}")
        return self

    def focus_ligand(self, index: int = -1) -> "ProVis":
        """
        Zoom the view to a specific loaded ligand.

        :param index: Ligand index to focus on; negative indices are supported (default -1 -> last).
        :type index: int
        :return: self (chainable).
        :rtype: ProVis
        """
        if not self._ligands_meta:
            return self
        if index < 0:
            index = len(self._ligands_meta) + index
        index = max(0, min(index, len(self._ligands_meta) - 1))
        model_id = self._ligands_meta[index]["model"]
        self._viewer.zoomTo({"model": model_id})
        return self

    def hide_waters(self) -> "ProVis":
        """
        Hide water residues (HOH / WAT) from the receptor model.

        :return: self (chainable).
        :rtype: ProVis
        """
        self._viewer.setStyle(
            {"and": [{"model": 0}, {"or": [{"resn": "HOH"}, {"resn": "WAT"}]}]}, {}
        )
        return self

    def dark_mode(self, on: bool = True) -> "ProVis":
        """
        Toggle dark background.

        :param on: If True set a dark background; otherwise set a white background.
        :type on: bool
        :return: self (chainable).
        :rtype: ProVis
        """
        return self.set_background("0x111111" if on else "0xFFFFFF")

    # -------------------------
    # Viewer controls
    # -------------------------
    def set_background(self, color: str = "0xFFFFFF") -> "ProVis":
        """
        Set the viewer background color.

        :param color: Background color (three.js hex like '0xFFFFFF' or CSS color).
        :type color: str
        :return: self (chainable).
        :rtype: ProVis
        """
        self._viewer.setBackgroundColor(color)
        return self

    def show(self, zoom_to: int = -1, orthographic: bool = True) -> "ProVis":
        """
        Render the py3Dmol viewer inline.

        :param zoom_to: Model index to zoom to (-1 zooms to all models).
        :type zoom_to: int
        :param orthographic: If True use orthographic projection; otherwise perspective.
        :type orthographic: bool
        :return: self (chainable).
        :rtype: ProVis
        """
        if orthographic:
            self._viewer.setProjection("orthographic")
        self._viewer.zoomTo({"model": zoom_to})
        self._viewer.show()
        return self

    @property
    def viewer(self) -> py3Dmol.view:
        """
        Return the underlying py3Dmol view object.

        :return: The py3Dmol.view instance used by this wrapper.
        :rtype: py3Dmol.view
        """
        return self._viewer

    @property
    def ligands(self) -> Tuple[str, ...]:
        """
        Names of loaded ligands (in order).

        :return: Tuple of ligand display names.
        :rtype: tuple[str, ...]
        """
        return tuple(m["name"] for m in self._ligands_meta)

    def __repr__(self) -> str:
        return (
            f"<ProVis models={self._model_count+1}, ligands={len(self._ligands_meta)}>"
        )
