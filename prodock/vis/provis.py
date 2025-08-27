"""
provis.py

A thin wrapper around py3Dmol to provide convenient receptor/ligand
display and gridbox visualization helpers.

Dependencies
------------
- py3Dmol
- pathlib
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import py3Dmol

from ..preprocess.gridbox import GridBox


class ProVis:
    """
    ProDock visualization wrapper around py3Dmol.

    Usage example:
    >>> viz = ProVis(800, 600)
    >>> viz.load_receptor("protein.pdb").load_ligand("lig.sdf", fmt="sdf")
    >>> viz.set_receptor_style("cartoon", "white").highlight_ligand("stick", "cyan")
    >>> viz.show()
    """

    def __init__(self, vw: int = 700, vh: int = 500) -> None:
        """
        Create a ProVis viewer.

        :param vw: viewer width in pixels
        :type vw: int
        :param vh: viewer height in pixels
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
        """Read text from a file path."""
        with open(inpt_file, "r") as f:
            return f.read()

    def load_receptor(self, inpt_file: Union[str, Path]) -> "ProVis":
        """
        Load receptor PDB (model 0).

        :param inpt_file: path to PDB file
        :type inpt_file: str or pathlib.Path
        :return: self
        :rtype: ProVis
        """
        data = self._read_file(inpt_file)
        self._viewer.addModel(data, "pdb")
        self._model_count += 1
        return self

    def load_ligand(self, inpt_file: Union[str, Path], fmt: str = "sdf") -> "ProVis":
        """
        Load ligand from a file path (and remember raw data).

        :param inpt_file: path to ligand file
        :type inpt_file: str or pathlib.Path
        :param fmt: ligand format ('sdf','pdb','mol2','xyz')
        :type fmt: str
        :return: self
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
        Load ligand from a raw text block (useful for pasted SDF/PDB).

        :param text: ligand content
        :type text: str
        :param name: display name for ligand
        :type name: str
        :param fmt: format
        :type fmt: str
        :return: self
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
        Apply a style to receptor (model 0).

        :param style: representation style like 'cartoon', 'stick'
        :type style: str
        :param color: color string or scheme
        :type color: str
        :return: self
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
        Apply a style to all loaded ligands.

        :param style: 'stick','sphere','line','cartoon'
        :type style: str
        :param color: color string
        :type color: str
        :param radius: radius for sticks/spheres
        :type radius: float
        :param opacity: opacity for cartoons
        :type opacity: float
        :return: self
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
        Add SES surface around model 0.

        :param opacity: surface opacity
        :type opacity: float
        :param color: surface color
        :type color: str
        :return: self
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
        Draw a numeric box by center and size.

        :param center: (x,y,z)
        :type center: tuple
        :param size: (w,h,d)
        :type size: tuple
        :param color: box color
        :type color: str
        :param opacity: box opacity
        :type opacity: float
        :return: self
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
        Draw a GridBox and add textual labels for center and sizes.

        :param grid: GridBox instance (center/size must be computed)
        :type grid: GridBox
        :param color: box color
        :type color: str
        :param opacity: box opacity
        :type opacity: float
        :param show_center: whether to show the center label
        :type show_center: bool
        :param show_sizes: whether to show size labels
        :type show_sizes: bool
        :param label_fontsize: font size for labels
        :type label_fontsize: int
        :param label_bg: show label background
        :type label_bg: bool
        :param label_offset_factor: fraction of edge to offset labels by
        :type label_offset_factor: float
        :return: self
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

        :param grid: GridBox to draw
        :param color: color
        :param opacity: opacity
        :param labels: whether to draw labels
        :return: self
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
        Convenience: compute a GridBox from a previously loaded ligand and draw it.

        :param ligand_index: index of loaded ligand (supports negative indexing)
        :param pad: padding in Å
        :param isotropic: cubic if True
        :param min_size: minimal edges
        :param color: box color
        :param opacity: box opacity
        :param labels: draw labels if True
        :return: self
        """
        if not self._ligands_meta:
            raise ValueError("No ligand loaded. Use load_ligand() first.")
        if ligand_index < 0:
            ligand_index = len(self._ligands_meta) + ligand_index
        if ligand_index < 0 or ligand_index >= len(self._ligands_meta):
            raise IndexError("ligand_index out of range")
        meta = self._ligands_meta[ligand_index]
        # compute box using GridBox locally to avoid importing cycles in single-file use
        from gridbox import GridBox  # local import to keep files decoupled

        gb = (
            GridBox()
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
        Convenience loader for receptor and ligand in one call.

        :param receptor: path to receptor PDB
        :param ligand: path to ligand
        :param ligand_fmt: ligand format if given
        :return: self
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
        Apply a quick style preset.

        Presets: 'publication', 'dark', 'surface'.

        :param name: preset name
        :param ligand_style: ligand representation
        :param background: optional background color
        :param surface: draw surface if True
        :return: self
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
        Zoom to a specific ligand (default last loaded).

        :param index: ligand index (supports negatives)
        :return: self
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
        Hide water residues in the receptor (HOH/WAT).
        """
        self._viewer.setStyle(
            {"and": [{"model": 0}, {"or": [{"resn": "HOH"}, {"resn": "WAT"}]}]}, {}
        )
        return self

    def dark_mode(self, on: bool = True) -> "ProVis":
        """Switch to dark background when on=True."""
        return self.set_background("0x111111" if on else "0xFFFFFF")

    # -------------------------
    # Viewer controls
    # -------------------------
    def set_background(self, color: str = "0xFFFFFF") -> "ProVis":
        """Set the viewer background color."""
        self._viewer.setBackgroundColor(color)
        return self

    def show(self, zoom_to: int = -1, orthographic: bool = True) -> "ProVis":
        """
        Render the py3Dmol viewer inline.

        :param zoom_to: model index to zoom to (-1 for all)
        :param orthographic: use orthographic projection
        :return: self
        """
        if orthographic:
            self._viewer.setProjection("orthographic")
        self._viewer.zoomTo({"model": zoom_to})
        self._viewer.show()
        return self

    @property
    def viewer(self) -> py3Dmol.view:
        """Return underlying py3Dmol view object."""
        return self._viewer

    @property
    def ligands(self) -> Tuple[str, ...]:
        """Names of loaded ligands (in order)."""
        return tuple(m["name"] for m in self._ligands_meta)

    def __repr__(self) -> str:
        return (
            f"<ProVis models={self._model_count+1}, ligands={len(self._ligands_meta)}>"
        )
