"""
provis_gui.py

Interactive ipywidgets GUI for ProVis + GridBox workflows.

Depends on:
- ipywidgets
- IPython.display
- gridbox.GridBox
- provis.ProVis

The ProVisGUI class encapsulates an interactive widget that allows:
- loading/adding ligands (paste/upload/path)
- selecting ligands
- computing non-isotropic and isotropic boxes with presets
- manually editing center/size and applying/drawing boxes
- importing/exporting Vina config snippets
- retrieving a dict to feed into Vina (current_vina_dict)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import traceback

import ipywidgets as widgets
from IPython.display import display, clear_output

from ..process.gridbox import GridBox, _is_pathlike, _snap_tuple, _round_tuple
from .provis import ProVis


class ProVisGUI:
    """
    GUI to drive ProVis and GridBox interactively.

    Create with :py:meth:`ProVisGUI().build().display()` and then interact.
    Use :pyattr:`current_vina_dict` to retrieve the chosen box dict programmatically.

    Example:
    >>> gui = ProVisGUI().build().display()
    >>> gui.current_vina_dict
    """

    def __init__(self) -> None:
        # Receptor & ligand input
        self._receptor_path = widgets.Text(
            value="", description="Receptor:", placeholder="path/to/protein.pdb"
        )
        self._ligand_path = widgets.Text(
            value="",
            description="Ligand (path or paste):",
            placeholder="path or SDF/PDB block",
        )
        self._ligand_fmt = widgets.Dropdown(
            options=["sdf", "pdb", "mol2", "xyz"],
            value="sdf",
            description="Ligand fmt:",
        )
        self._uploader = widgets.FileUpload(
            accept=".sdf,.pdb,.mol2,.xyz", multiple=False, description="Upload ligand"
        )
        self._add_ligand_btn = widgets.Button(
            description="Add ligand", button_style="success"
        )
        self._ligand_select = widgets.Dropdown(
            options=[], description="Loaded ligands:"
        )

        # Presets / compute controls
        self._preset = widgets.Dropdown(
            options=[("Tight", "tight"), ("Safe", "safe"), ("Vina24", "vina24")],
            value="safe",
            description="Preset:",
        )
        self._pad = widgets.FloatSlider(
            value=4.0, min=0.0, max=12.0, step=0.25, description="pad (Å):"
        )
        self._isotropic = widgets.Checkbox(value=True, description="isotropic (cubic)")
        self._min_size = widgets.FloatText(value=22.5, description="min_size (Å):")
        self._heavy_only = widgets.Checkbox(value=False, description="heavy atoms only")
        self._snap_step = widgets.FloatText(value=0.25, description="snap step (Å):")
        self._round_nd = widgets.IntSlider(
            value=3, min=0, max=4, step=1, description="round digits:"
        )

        # Manual 3×2 fields
        self._center_x = widgets.FloatText(value=0.0, description="center_x:")
        self._center_y = widgets.FloatText(value=0.0, description="center_y:")
        self._center_z = widgets.FloatText(value=0.0, description="center_z:")
        self._size_x = widgets.FloatText(value=20.0, description="size_x:")
        self._size_y = widgets.FloatText(value=20.0, description="size_y:")
        self._size_z = widgets.FloatText(value=20.0, description="size_z:")
        self._use_manual = widgets.Checkbox(value=False, description="Prefer manual")

        # Visibility toggles
        self._show_noniso = widgets.Checkbox(value=True, description="show non-iso")
        self._show_iso = widgets.Checkbox(value=True, description="show iso")
        self._show_manual = widgets.Checkbox(value=True, description="show manual")

        # Vina cfg import
        self._vina_cfg_text = widgets.Textarea(
            value="",
            description="Import cfg:",
            placeholder="Paste center_x/…/size_z here",
        )
        self._vina_import_btn = widgets.Button(
            description="Import into manual", button_style=""
        )

        # Actions
        self._update_btn = widgets.Button(
            description="Update viewer", button_style="primary"
        )
        self._apply_manual_btn = widgets.Button(
            description="Apply manual (draw)", button_style="warning"
        )
        self._fill_noniso_btn = widgets.Button(
            description="Fill manual from non-iso", button_style=""
        )
        self._fill_iso_btn = widgets.Button(
            description="Fill manual from iso", button_style=""
        )
        self._show_vina_btn = widgets.Button(
            description="Show Vina dict(s)", button_style="info"
        )
        self._save_vina_btn = widgets.Button(
            description="Save chosen cfg", button_style=""
        )

        # Status + output
        self._status = widgets.Label(value="No ligands loaded.")
        self._out = widgets.Output(
            layout={"border": "1px solid #ccc", "height": "620px", "overflow": "auto"}
        )

        # State
        self._ligands: List[dict] = []
        self._gb_non: Optional[GridBox] = None
        self._gb_iso: Optional[GridBox] = None
        self._gb_manual: Optional[GridBox] = None
        self._last_viz: Optional[ProVis] = None
        self._ui: Optional[widgets.Widget] = None

        # Wire events
        self._add_ligand_btn.on_click(self._on_add_ligand)
        self._preset.observe(self._on_preset_change, names="value")
        self._update_btn.on_click(self._on_update)
        self._apply_manual_btn.on_click(self._on_apply_manual)
        self._fill_noniso_btn.on_click(self._on_fill_noniso)
        self._fill_iso_btn.on_click(self._on_fill_iso)
        self._show_vina_btn.on_click(self._on_show_vina)
        self._save_vina_btn.on_click(self._on_save_vina)
        self._vina_import_btn.on_click(self._on_vina_import)

    # ---- event helpers ----
    def _on_preset_change(self, change: Dict[str, Any]) -> None:
        """Apply recommended defaults when preset changes."""
        mode = change["new"]
        if mode == "tight":
            self._pad.value = 3.0
            self._isotropic.value = False
            self._min_size.value = 0.0
        elif mode == "safe":
            self._pad.value = 4.0
            self._isotropic.value = True
            self._min_size.value = 22.5
        else:  # vina24
            self._pad.value = 2.0
            self._isotropic.value = True
            self._min_size.value = 24.0
        self._status.value = f"Preset applied: {mode}"

    def _on_add_ligand(self, _btn) -> None:
        """Add ligand either from uploader or from ligand_path (path or pasted text)."""
        with self._out:
            clear_output(wait=True)
            try:
                if self._uploader.value:
                    key = next(iter(self._uploader.value))
                    meta = self._uploader.value[key]
                    blob = meta.get("content", b"")
                    try:
                        text = blob.decode("utf-8")
                    except Exception:
                        text = blob.decode("utf-8", errors="replace")
                    name = meta.get("metadata", {}).get("name", key)
                    fmt = (
                        Path(name).suffix.lstrip(".").lower() or self._ligand_fmt.value
                    )
                    self._ligands.append({"data": text, "fmt": fmt, "name": name})
                    try:
                        self._uploader.value.clear()
                    except Exception:
                        pass
                else:
                    lig = self._ligand_path.value.strip()
                    if not lig:
                        print(
                            "Provide a ligand path or paste ligand content, or upload a file."
                        )
                        return
                    fmt = self._ligand_fmt.value
                    name = (
                        Path(lig).name
                        if _is_pathlike(lig)
                        else f"pasted_{len(self._ligands)+1}.{fmt}"
                    )
                    text = Path(lig).read_text() if _is_pathlike(lig) else lig
                    self._ligands.append({"data": text, "fmt": fmt, "name": name})
                self._ligand_select.options = [
                    (m["name"], i) for i, m in enumerate(self._ligands)
                ]
                self._ligand_select.value = len(self._ligands) - 1
                self._status.value = f"Added ligand: {self._ligands[-1]['name']}"
                print("Added ligand:", self._ligands[-1]["name"])
            except Exception as e:
                print("Add ligand error:", e)
                traceback.print_exc()

    def _compute_for_selected(self) -> None:
        """Compute non-isotropic and isotropic GridBoxes for selected ligand."""
        if not self._ligands:
            raise ValueError("No ligands added.")
        idx = int(self._ligand_select.value)
        meta = self._ligands[idx]
        data, fmt = meta["data"], meta["fmt"]
        # use advanced builder to respect heavy_only, snap and rounding
        self._gb_non = (
            GridBox()
            .load_ligand(data, fmt=fmt)
            .from_ligand_pad_adv(
                pad=self._pad.value,
                isotropic=False,
                min_size=self._min_size.value,
                heavy_only=self._heavy_only.value,
                snap_step=(self._snap_step.value or None),
                round_ndigits=self._round_nd.value,
            )
        )
        self._gb_iso = (
            GridBox()
            .load_ligand(data, fmt=fmt)
            .from_ligand_pad_adv(
                pad=self._pad.value,
                isotropic=True,
                min_size=self._min_size.value,
                heavy_only=self._heavy_only.value,
                snap_step=(self._snap_step.value or None),
                round_ndigits=self._round_nd.value,
            )
        )
        self._status.value = f"Computed boxes for: {meta['name']} (idx {idx})"

    def _draw(self) -> None:
        """Render current scene to the GUI output area."""
        with self._out:
            clear_output(wait=True)
            try:
                viz = ProVis(vw=1100, vh=700)
                if self._receptor_path.value.strip():
                    viz.load_receptor(self._receptor_path.value.strip()).style_preset(
                        "publication", surface=False
                    )
                if self._ligands:
                    idx = int(self._ligand_select.value)
                    meta = self._ligands[idx]
                    viz.load_ligand_from_text(
                        meta["data"], name=meta["name"], fmt=meta["fmt"]
                    )
                viz.highlight_ligand(style="stick", color="cyan", radius=0.25)
                if self._show_noniso.value and self._gb_non is not None:
                    viz.add_gridbox_with_labels(
                        self._gb_non, color="skyBlue", opacity=0.25
                    )
                if self._show_iso.value and self._gb_iso is not None:
                    viz.add_gridbox_with_labels(
                        self._gb_iso, color="orange", opacity=0.25
                    )
                if self._show_manual.value and self._gb_manual is not None:
                    viz.add_gridbox_with_labels(
                        self._gb_manual, color="lime", opacity=0.25
                    )
                viz.set_background("0xFFFFFF").show()
                if self._gb_non:
                    print(
                        "NON-ISOTROPIC:",
                        self._gb_non.vina_dict,
                        "|",
                        self._gb_non.summary(),
                    )
                if self._gb_iso:
                    print(
                        "ISOTROPIC:   ",
                        self._gb_iso.vina_dict,
                        "|",
                        self._gb_iso.summary(),
                    )
                if self._gb_manual:
                    print(
                        "MANUAL:      ",
                        self._gb_manual.vina_dict,
                        "|",
                        self._gb_manual.summary(),
                    )
                self._last_viz = viz
            except Exception as e:
                print("Draw error:", e)
                traceback.print_exc()

    # -------------------------
    # Button handlers
    # -------------------------
    def _on_update(self, _btn) -> None:
        try:
            self._compute_for_selected()
            self._draw()
        except Exception as e:
            with self._out:
                clear_output(wait=True)
                print("Error:", e)
                traceback.print_exc()

    def _on_apply_manual(self, _btn) -> None:
        try:
            cx, cy, cz = (
                float(self._center_x.value),
                float(self._center_y.value),
                float(self._center_z.value),
            )
            sx, sy, sz = (
                float(self._size_x.value),
                float(self._size_y.value),
                float(self._size_z.value),
            )
            if self._snap_step.value:
                cx, cy, cz = _snap_tuple((cx, cy, cz), self._snap_step.value)
                sx, sy, sz = _snap_tuple((sx, sy, sz), self._snap_step.value)
            cx, cy, cz = _round_tuple((cx, cy, cz), self._round_nd.value)
            sx, sy, sz = _round_tuple((sx, sy, sz), self._round_nd.value)
            self._gb_manual = GridBox().from_center_size((cx, cy, cz), (sx, sy, sz))
            self._draw()
        except Exception as e:
            with self._out:
                clear_output(wait=True)
                print("Manual box error:", e)
                traceback.print_exc()

    def _on_fill_noniso(self, _btn) -> None:
        if self._gb_non is None:
            with self._out:
                clear_output(wait=True)
                print("Compute boxes first (Update viewer).")
            return
        cx, cy, cz = self._gb_non.center
        sx, sy, sz = self._gb_non.size
        self._center_x.value, self._center_y.value, self._center_z.value = cx, cy, cz
        self._size_x.value, self._size_y.value, self._size_z.value = sx, sy, sz
        self._status.value = "Manual filled from non-iso"

    def _on_fill_iso(self, _btn) -> None:
        if self._gb_iso is None:
            with self._out:
                clear_output(wait=True)
                print("Compute boxes first (Update viewer).")
            return
        cx, cy, cz = self._gb_iso.center
        sx, sy, sz = self._gb_iso.size
        self._center_x.value, self._center_y.value, self._center_z.value = cx, cy, cz
        self._size_x.value, self._size_y.value, self._size_z.value = sx, sy, sz
        self._status.value = "Manual filled from iso"

    def _on_show_vina(self, _btn) -> None:
        with self._out:
            clear_output(wait=True)
            try:
                if self._gb_non:
                    print("[NON-ISOTROPIC]\n" + self._gb_non.to_vina_lines() + "\n")
                if self._gb_iso:
                    print("[ISOTROPIC]\n" + self._gb_iso.to_vina_lines() + "\n")
                if self._gb_manual:
                    print("[MANUAL]\n" + self._gb_manual.to_vina_lines() + "\n")
            except Exception as e:
                print("Show Vina error:", e)
                traceback.print_exc()

    def _on_save_vina(self, _btn) -> None:
        with self._out:
            clear_output(wait=True)
            try:
                gb = self._choose_box_for_export()
                if gb is None:
                    print("No box to save. Compute or apply manual first.")
                    return
                p = Path.cwd() / "vina_box.cfg"
                gb.to_vina_file(p)
                print("Saved to", p)
                print(gb.to_vina_lines())
            except Exception as e:
                print("Save Vina error:", e)
                traceback.print_exc()

    def _on_vina_import(self, _btn) -> None:
        with self._out:
            try:
                d = GridBox.parse_vina_cfg(self._vina_cfg_text.value)
                self._center_x.value = d["center_x"]
                self._center_y.value = d["center_y"]
                self._center_z.value = d["center_z"]
                self._size_x.value = d["size_x"]
                self._size_y.value = d["size_y"]
                self._size_z.value = d["size_z"]
                self._use_manual.value = True
                print("Imported Vina cfg into manual fields.")
            except Exception as e:
                clear_output(wait=True)
                print("CFG import error:", e)

    # -------------------------
    # Selection / export
    # -------------------------
    def _choose_box_for_export(self) -> Optional[GridBox]:
        if self._use_manual.value:
            return self._gb_manual
        return (
            self._gb_iso
            if (self._isotropic.value and self._gb_iso is not None)
            else self._gb_non
        )

    @property
    def current_vina_dict(self) -> Dict[str, float]:
        """
        Return the currently selected Vina dict according to GUI preferences.

        :return: vina param dict
        """
        gb = self._choose_box_for_export()
        if gb is None:
            raise ValueError("No box available; compute or apply manual first.")
        return gb.vina_dict

    # -------------------------
    # Layout / display
    # -------------------------
    def build(self) -> "ProVisGUI":
        """Compose widget layout and return self (idempotent)."""
        if self._ui is None:
            manual_left = widgets.VBox(
                [self._center_x, self._center_y, self._center_z],
                layout=widgets.Layout(margin="0 6px 0 0"),
            )
            manual_right = widgets.VBox(
                [self._size_x, self._size_y, self._size_z],
                layout=widgets.Layout(margin="0 0 0 6px"),
            )
            manual_grid = widgets.HBox([manual_left, manual_right])

            ligand_controls = widgets.VBox(
                [
                    self._ligand_path,
                    self._ligand_fmt,
                    widgets.HBox([self._uploader, self._add_ligand_btn]),
                    self._ligand_select,
                ],
                layout=widgets.Layout(width="100%"),
            )

            compute_row1 = widgets.HBox([self._preset, self._pad, self._isotropic])
            compute_row2 = widgets.HBox(
                [self._min_size, self._heavy_only, self._snap_step, self._round_nd]
            )

            left = widgets.VBox(
                [
                    self._receptor_path,
                    widgets.HTML("<b>Ligand input / list</b>"),
                    ligand_controls,
                    widgets.HTML("<b>Computed box parameters</b>"),
                    compute_row1,
                    compute_row2,
                    widgets.HBox([self._show_noniso, self._show_iso]),
                    widgets.HTML("<b>Manual box parameters</b>"),
                    manual_grid,
                    widgets.HBox([self._use_manual, self._show_manual]),
                    widgets.HBox([self._update_btn, self._apply_manual_btn]),
                    widgets.HBox([self._fill_noniso_btn, self._fill_iso_btn]),
                    widgets.HTML("<b>Vina cfg import</b>"),
                    self._vina_cfg_text,
                    self._vina_import_btn,
                    widgets.HBox([self._show_vina_btn, self._save_vina_btn]),
                    widgets.HTML("<b>Status</b>"),
                    self._status,
                ],
                layout=widgets.Layout(width="40%", padding="6px"),
            )
            right = widgets.VBox(
                [self._out], layout=widgets.Layout(width="60%", padding="6px")
            )
            self._ui = widgets.HBox([left, right])
        return self

    def display(self) -> "ProVisGUI":
        """Render the GUI and return self."""
        if self._ui is None:
            self.build()
        display(self._ui)
        return self

    def __repr__(self) -> str:
        return "<ProVisGUI (presets + heavy-only + snap + cfg import)>"
