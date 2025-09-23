"""
provis_gui.py

A polished, workflow-oriented ipywidgets GUI for ProVis + GridBox.

Enhancements:
- Keyboard shortcuts: Alt+U (Update viewer), Alt+S (Save cfg)
- Compact single-column layout toggle for small screens
- Reduced console spam; rich Info panel in the right column
- Dark theme fixed via injected CSS (scoped) + code/preview styling
- Screenshot: uses ProVis.screenshot(...) if available; otherwise front-end canvas capture

Usage:
>>> gui = ProVisGUI().build().display()
>>> gui.current_vina_dict

Builder style:
- Mutators return self (e.g., .set_receptor(...).add_ligand(...).compute().draw())
- Retrieval via properties (.current_vina_dict, .vina_lines, .gridboxes, etc.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import json
import traceback

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Javascript

from prodock.process.gridbox import GridBox, _is_pathlike, _snap_tuple, _round_tuple
from prodock.vis.provis import ProVis


class ProVisGUI:
    """
    Visual, workflow-oriented GUI to drive ProVis and GridBox derivation.

    Builder-style API: most mutator methods return ``self`` for chaining.
    Retrieval uses properties like :pyattr:`current_vina_dict`, :pyattr:`vina_lines`,
    :pyattr:`gridboxes`, and :pyattr:`ligand_names`.

    :param vw: Initial viewer width in pixels (used when instantiating ProVis).
    :type vw: int
    :param vh: Initial viewer height in pixels (used when instantiating ProVis).
    :type vh: int

    Notes
    -----
    - The GUI is designed for interactive Jupyter usage. Call :py:meth:`build`
      then :py:meth:`display` to render widgets in the notebook.
    - Most mutator methods are chainable (return ``self``) so you can programmatically
      drive a workflow, e.g.:
        ``ProVisGUI().set_receptor('rec.pdb').add_ligand(...).compute().draw().display()``
    """

    # ---- Presets (extensible) ----------------------------------------------------
    PRESETS: Dict[str, Dict[str, Any]] = {
        "tight": {"pad": 3.0, "isotropic": False, "min_size": 0.0},
        "safe": {"pad": 4.0, "isotropic": True, "min_size": 22.5},
        "vina24": {"pad": 2.0, "isotropic": True, "min_size": 24.0},
    }

    # ---- Themes (used for CSS injection) -----------------------------------------
    THEMES: Dict[str, Dict[str, str]] = {
        "light": {
            "bg": "#ffffff",
            "card_bg": "#fbfbfd",
            "card_border": "#e7e7ef",
            "mute": "#6b7280",
            "accent": "#2563eb",
            "status_ok": "#16a34a",
            "status_warn": "#f59e0b",
            "status_err": "#dc2626",
            "code_bg": "#0b1021",
            "code_fg": "#e5e7eb",
        },
        "dark": {
            "bg": "#0f172a",
            "card_bg": "#0b1222",
            "card_border": "#1f2a44",
            "mute": "#9aa3b2",
            "accent": "#60a5fa",
            "status_ok": "#22c55e",
            "status_warn": "#fbbf24",
            "status_err": "#f87171",
            "code_bg": "#0b1021",
            "code_fg": "#e5e7eb",
        },
    }

    # ------------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------------
    def __init__(self, vw: int = 1100, vh: int = 700) -> None:
        # Viewer dims
        self._vw = int(vw)
        self._vh = int(vh)

        # ---------- Header / Theme / Layout / Actions ----------
        self._theme = widgets.ToggleButtons(
            options=[("🌞 Light", "light"), ("🌙 Dark", "dark")],
            value="light",
            description="",
            button_style="",
        )
        self._compact = widgets.Checkbox(
            value=False, description="Compact layout (1-col)"
        )
        self._btn_screenshot = widgets.Button(
            description="Screenshot",
            icon="camera",
            tooltip="Save PNG (disk if supported, else browser download)",
        )
        self._btn_help = widgets.Button(
            description="", icon="question-circle", tooltip="Open Help"
        )
        self._title = widgets.HTML(
            "<h2 style='margin:8px 0 0 0;'>ProVis GUI</h2>"
            "<div style='color:#6b7280;margin-top:-4px;'>Visual gridbox builder for docking</div>"
        )

        # ---------- Receptor / Ligands ----------
        self._receptor_path = widgets.Text(
            value="", description="Receptor:", placeholder="path/to/protein.pdb/pdbqt"
        )
        self._receptor_browse = widgets.FileUpload(
            accept=".pdb,.pdbqt,.mol2", multiple=False, description="Upload receptor"
        )
        self._ligand_path = widgets.Text(
            value="",
            description="Ligand (path/paste):",
            placeholder="path or SDF/PDB/MOL2/XYZ block",
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
            description="Add ligand", button_style="success", icon="plus"
        )
        self._remove_ligand_btn = widgets.Button(
            description="Remove", button_style="danger", icon="trash"
        )
        self._clear_ligands_btn = widgets.Button(description="Clear", icon="eraser")
        self._ligand_select = widgets.Dropdown(options=[], description="Ligands:")
        self._auto_update = widgets.Checkbox(value=True, description="Auto-update")

        # ---------- Compute / Presets ----------
        self._preset = widgets.ToggleButtons(
            options=[("Tight", "tight"), ("Safe", "safe"), ("Vina24", "vina24")],
            value="safe",
            description="Presets:",
        )
        self._pad = widgets.FloatSlider(
            value=4.0,
            min=0.0,
            max=12.0,
            step=0.25,
            description="pad (Å):",
            continuous_update=False,
        )
        self._isotropic = widgets.Checkbox(value=True, description="isotropic (cubic)")
        self._min_size = widgets.FloatText(value=22.5, description="min_size (Å):")
        self._heavy_only = widgets.Checkbox(value=False, description="heavy atoms only")
        self._snap_step = widgets.FloatText(value=0.25, description="snap step (Å):")
        self._round_nd = widgets.IntSlider(
            value=3,
            min=0,
            max=4,
            step=1,
            description="round digits:",
            continuous_update=False,
        )

        # ---------- Manual fields ----------
        self._center_x = widgets.FloatText(value=0.0, description="center_x:")
        self._center_y = widgets.FloatText(value=0.0, description="center_y:")
        self._center_z = widgets.FloatText(value=0.0, description="center_z:")
        self._size_x = widgets.FloatText(value=20.0, description="size_x:")
        self._size_y = widgets.FloatText(value=20.0, description="size_y:")
        self._size_z = widgets.FloatText(value=20.0, description="size_z:")
        self._use_manual = widgets.Checkbox(value=False, description="Prefer manual")

        # ---------- Visual styling controls ----------
        self._color_non = widgets.ColorPicker(
            value="#7ec8ff", description="Non-iso color:"
        )
        self._color_iso = widgets.ColorPicker(value="#f39c12", description="Iso color:")
        self._color_man = widgets.ColorPicker(
            value="#7bed9f", description="Manual color:"
        )
        self._opacity = widgets.FloatSlider(
            value=0.25,
            min=0.05,
            max=0.6,
            step=0.05,
            description="Opacity:",
            readout_format=".2f",
            continuous_update=False,
        )
        self._lig_color = widgets.ColorPicker(
            value="#00ffff", description="Ligand color:"
        )
        self._lig_radius = widgets.FloatSlider(
            value=0.25, min=0.1, max=0.6, step=0.05, description="Lig radius:"
        )

        # ---------- Visibility toggles ----------
        self._show_noniso = widgets.Checkbox(value=True, description="show non-iso")
        self._show_iso = widgets.Checkbox(value=True, description="show iso")
        self._show_manual = widgets.Checkbox(value=True, description="show manual")

        # ---------- Vina I/O ----------
        self._box_source = widgets.Dropdown(
            options=[
                ("Auto", "auto"),
                ("Isotropic", "iso"),
                ("Non-isotropic", "non"),
                ("Manual", "manual"),
            ],
            value="auto",
            description="Export from:",
        )
        self._vina_cfg_text = widgets.Textarea(
            value="",
            description="Import cfg:",
            placeholder="Paste center_x/…/size_z here",
            layout=widgets.Layout(height="90px"),
        )
        self._vina_import_btn = widgets.Button(
            description="Import → manual", icon="sign-in"
        )
        self._vina_preview = widgets.HTML(
            value=self._code_block(""), placeholder="Preview area"
        )
        self._vina_preview.add_class("provis-vina-preview")  # scoped CSS target
        self._preview_btn = widgets.Button(description="Refresh preview", icon="eye")
        self._save_name = widgets.Text(value="vina_box.cfg", description="Save as:")
        self._save_vina_btn = widgets.Button(description="Save cfg", icon="save")
        # mark these with special tooltips so JS shortcuts can target them robustly
        self._update_btn = widgets.Button(
            description="Update viewer",
            button_style="primary",
            icon="refresh",
            tooltip="update-shortcut",
        )
        self._save_vina_btn.tooltip = "savecfg-shortcut"

        # ---------- Session ----------
        self._save_session_btn = widgets.Button(description="Save session", icon="save")
        self._load_session_up = widgets.FileUpload(
            accept=".json", multiple=False, description="Load session"
        )

        # ---------- Status / Info / Output ----------
        self._status = widgets.HTML(
            self._status_bar("Ready. No ligands loaded.", level="ok")
        )
        self._status.add_class("provis-status-bar")  # scoped CSS target
        self._info_panel = widgets.HTML(self._info_block(""))  # rich info goes here
        self._busy = widgets.HTML(self._spinner_html(False))
        self._out = widgets.Output(
            layout={
                "border": "1px solid transparent",
                "height": "640px",
                "overflow": "auto",
            }
        )

        # ---------- State ----------
        self._ligands: List[dict] = []
        self._gb_non: Optional[GridBox] = None
        self._gb_iso: Optional[GridBox] = None
        self._gb_manual: Optional[GridBox] = None
        self._last_viz: Optional[ProVis] = None
        self._ui: Optional[widgets.Widget] = None
        self._left_col: Optional[widgets.Widget] = None
        self._right_col: Optional[widgets.Widget] = None
        self._root: Optional[widgets.Widget] = None
        self._last_error: Optional[str] = None
        self._theme_css_injected = False
        self._shortcuts_injected = False

        # ---------- Event wiring ----------
        self._theme.observe(self._on_theme, names="value")
        self._compact.observe(self._on_compact_toggle, names="value")
        self._btn_help.on_click(lambda _: setattr(self, "_force_help_tab", True))
        self._btn_screenshot.on_click(self._on_screenshot)

        self._add_ligand_btn.on_click(self._on_add_ligand)
        self._remove_ligand_btn.on_click(self._on_remove_ligand)
        self._clear_ligands_btn.on_click(self._on_clear_ligands)
        self._ligand_select.observe(self._on_selection_change, names="value")
        self._receptor_browse.observe(self._on_receptor_upload, names="value")

        self._preset.observe(self._on_preset_change, names="value")
        for w in (
            self._pad,
            self._isotropic,
            self._min_size,
            self._heavy_only,
            self._snap_step,
            self._round_nd,
            self._color_non,
            self._color_iso,
            self._color_man,
            self._opacity,
            self._lig_color,
            self._lig_radius,
            self._show_noniso,
            self._show_iso,
            self._show_manual,
        ):
            w.observe(self._on_any_param_change, names="value")

        self._update_btn.on_click(self._on_update)
        self._apply_manual_btn = widgets.Button(
            description="Apply manual (draw)", button_style="warning", icon="edit"
        )
        self._apply_manual_btn.on_click(self._on_apply_manual)
        self._fill_noniso_btn = widgets.Button(
            description="Manual ← non-iso", icon="square-o"
        )
        self._fill_iso_btn = widgets.Button(description="Manual ← iso", icon="cube")
        self._clear_boxes_btn = widgets.Button(description="Clear boxes", icon="ban")
        self._fill_noniso_btn.on_click(self._on_fill_noniso)
        self._fill_iso_btn.on_click(self._on_fill_iso)
        self._clear_boxes_btn.on_click(self._on_clear_boxes)

        self._vina_import_btn.on_click(self._on_vina_import)
        self._preview_btn.on_click(self._on_refresh_preview)
        self._save_vina_btn.on_click(self._on_save_vina)
        self._save_session_btn.on_click(self._on_save_session)
        self._load_session_up.observe(self._on_load_session, names="value")

        # for Help button switching after build
        self._force_help_tab = False

    # ==============================================================================
    # Public builder-style helpers
    # ==============================================================================
    def set_receptor(self, path: Union[str, Path]) -> "ProVisGUI":
        """
        Set receptor path.

        :param path: Filesystem path to receptor (PDB/PDBQT/MOL2).
        :type path: str | Path
        :return: self
        """
        self._receptor_path.value = str(Path(str(path)).expanduser())
        return self

    def add_ligand(
        self, data_or_path: str, fmt: Optional[str] = None, name: Optional[str] = None
    ) -> "ProVisGUI":
        """
        Add a ligand from a path or a text block.

        :param data_or_path: Path to file or ligand text block.
        :type data_or_path: str
        :param fmt: Optional explicit format ("sdf", "pdb", "mol2", "xyz").
        :type fmt: str, optional
        :param name: Optional display name; otherwise inferred.
        :type name: str, optional
        :return: self
        """
        try:
            src = data_or_path.strip()
            if _is_pathlike(src):
                text = Path(src).read_text()
                fmt_guess = Path(src).suffix.lstrip(".").lower()
                nm = name or Path(src).name
                fmt_final = (fmt or fmt_guess or "sdf").lower()
            else:
                text = src
                fmt_final = (fmt or self._ligand_fmt.value).lower()
                nm = name or f"pasted_{len(self._ligands)+1}.{fmt_final}"
            self._ligands.append({"data": text, "fmt": fmt_final, "name": nm})
            self._refresh_ligand_dropdown()
            self._ligand_select.value = len(self._ligands) - 1
            self._status.value = self._status_bar(f"Added ligand: {nm}", "ok")
            if self._auto_update.value:
                self._safe_update()
        except Exception as e:
            self._capture_error("Add ligand error", e)
        return self

    def remove_ligand(self, key: Union[int, str]) -> "ProVisGUI":
        """
        Remove a ligand by index or display name.

        :param key: Index or ligand display name.
        :type key: int | str
        :return: self
        """
        try:
            idx = self._index_from_key(key)
            removed = self._ligands.pop(idx)
            self._refresh_ligand_dropdown()
            self._status.value = self._status_bar(
                f"Removed ligand: {removed['name']}", "warn"
            )
            if self._auto_update.value:
                self._safe_update()
        except Exception as e:
            self._capture_error("Remove ligand error", e)
        return self

    def clear_ligands(self) -> "ProVisGUI":
        """
        Clear all ligands.

        :return: self
        """
        self._ligands.clear()
        self._refresh_ligand_dropdown()
        self._status.value = self._status_bar("Ligands cleared.", "warn")
        return self

    def select_ligand(self, key: Union[int, str]) -> "ProVisGUI":
        """
        Select ligand by index or name.

        :param key: Index or ligand display name.
        :type key: int | str
        :return: self
        """
        try:
            self._ligand_select.value = self._index_from_key(key)
            if self._auto_update.value:
                self._safe_update()
        except Exception as e:
            self._capture_error("Select ligand error", e)
        return self

    def set_preset(self, name: str) -> "ProVisGUI":
        """
        Programmatically set the box preset.

        :param name: Preset key ('tight'|'safe'|'vina24' or user-registered).
        :type name: str
        :return: self
        """
        if name in self.PRESETS:
            self._preset.value = name
        else:
            self._status.value = self._status_bar(f"Unknown preset '{name}'.", "warn")
        return self

    def compute(self) -> "ProVisGUI":
        """
        Compute GridBoxes for the selected ligand.

        :return: self
        """
        try:
            self._compute_for_selected()
        except Exception as e:
            self._capture_error("Compute error", e)
        return self

    def draw(self) -> "ProVisGUI":
        """
        Redraw the ProVis scene.

        :return: self
        """
        try:
            self._draw()
        except Exception as e:
            self._capture_error("Draw error", e)
        return self

    def apply_manual(
        self, center: Tuple[float, float, float], size: Tuple[float, float, float]
    ) -> "ProVisGUI":
        """
        Apply a manual box (center/size) and draw.

        :param center: (cx, cy, cz)
        :type center: Tuple[float, float, float]
        :param size: (sx, sy, sz)
        :type size: Tuple[float, float, float]
        :return: self
        """
        try:
            cx, cy, cz = center
            sx, sy, sz = size
            if self._snap_step.value:
                cx, cy, cz = _snap_tuple((cx, cy, cz), float(self._snap_step.value))
                sx, sy, sz = _snap_tuple((sx, sy, sz), float(self._snap_step.value))
            cx, cy, cz = _round_tuple((cx, cy, cz), int(self._round_nd.value))
            sx, sy, sz = _round_tuple((sx, sy, sz), int(self._round_nd.value))
            self._gb_manual = GridBox().from_center_size((cx, cy, cz), (sx, sy, sz))
            # mirror to inputs
            self._center_x.value, self._center_y.value, self._center_z.value = (
                cx,
                cy,
                cz,
            )
            self._size_x.value, self._size_y.value, self._size_z.value = sx, sy, sz
            self._use_manual.value = True
            self._draw()
            self._refresh_preview_area()
        except Exception as e:
            self._capture_error("Apply manual error", e)
        return self

    def choose_box_source(self, source: str = "auto") -> "ProVisGUI":
        """
        Set which box source to use for preview/export.

        :param source: One of {'auto','iso','non','manual'}.
        :type source: str
        :return: self
        """
        if source in {"auto", "iso", "non", "manual"}:
            self._box_source.value = source
            self._refresh_preview_area()
        else:
            self._status.value = self._status_bar(f"Unknown source '{source}'.", "warn")
        return self

    def save_cfg(self, path: Union[str, Path, None] = None) -> "ProVisGUI":
        """
        Save the currently chosen Vina config to disk.

        :param path: Destination file path; if None, uses the 'Save as' field.
        :type path: str | Path | None
        :return: self
        """
        try:
            gb = self._choose_box_for_export(self._box_source.value)
            if gb is None:
                self._status.value = self._status_bar(
                    "No box to save. Compute or apply manual first.", "warn"
                )
                return self
            p = (
                Path(path)
                if path is not None
                else Path(self._save_name.value or "vina_box.cfg")
            )
            gb.to_vina_file(p)
            self._status.value = self._status_bar(f"Saved Vina cfg → {p}", "ok")
            self._refresh_preview_area()
        except Exception as e:
            self._capture_error("Save cfg error", e)
        return self

    def save_session(self, path: Union[str, Path]) -> "ProVisGUI":
        """
        Save ligands + UI state as JSON.

        :param path: Destination JSON file path.
        :type path: str | Path
        :return: self
        """
        try:
            payload = self._session_payload()
            p = Path(path)
            p.write_text(json.dumps(payload, indent=2))
            self._status.value = self._status_bar(f"Session saved → {p}", "ok")
        except Exception as e:
            self._capture_error("Save session error", e)
        return self

    def load_session_text(self, text: str) -> "ProVisGUI":
        """
        Load a session from a JSON text.

        :param text: JSON string produced by :py:meth:`save_session`.
        :type text: str
        :return: self
        """
        try:
            payload = json.loads(text)
            self._restore_session(payload)
            self._status.value = self._status_bar("Session loaded from text.", "ok")
            if self._auto_update.value:
                self._safe_update()
        except Exception as e:
            self._capture_error("Load session text error", e)
        return self

    def screenshot(self, path: Union[str, Path] = "provis_view.png") -> "ProVisGUI":
        """
        Save a screenshot of the current ProVis canvas.

        Strategy:
          1) If ProVis exposes `.screenshot(path)` → save to disk.
          2) Else, capture the largest <canvas> in the page (browser download).
             This is front-end only; file won't appear on the Python filesystem.

        :param path: Output image path (used only if back-end screenshot is available).
        :type path: str | Path
        :return: self
        """
        try:
            if self._last_viz and hasattr(self._last_viz, "screenshot"):
                p = Path(path)
                try:
                    self._last_viz.screenshot(str(p))  # type: ignore[attr-defined]
                    self._status.value = self._status_bar(
                        f"Screenshot saved → {p}", "ok"
                    )
                    return self
                except Exception:
                    # fall through to browser capture
                    pass

            # Front-end capture (largest visible canvas)
            js = """
            (function(){
              try{
                const canvases = Array.from(document.querySelectorAll('canvas')).filter(c=>c.width>0&&c.height>0);
                if(!canvases.length){alert('No canvas found to capture.'); return;}
                let best = canvases[0], bestArea = best.width*best.height;
                for(const c of canvases){
                  const a = c.width*c.height;
                  if(a>bestArea){ best=c; bestArea=a; }
                }
                const data = best.toDataURL('image/png');
                const a = document.createElement('a');
                a.href = data;
                a.download = 'provis_view.png';
                document.body.appendChild(a);
                a.click();
                a.remove();
              }catch(e){ console.error(e); alert('Canvas capture failed: '+e); }
            })();
            """
            display(Javascript(js))
            self._status.value = self._status_bar(
                "Screenshot downloaded by browser (front-end).", "ok"
            )
        except Exception as e:
            self._capture_error("Screenshot error", e)
        return self

    # ==============================================================================
    # Properties (retrieval)
    # ==============================================================================
    @property
    def current_vina_dict(self) -> Dict[str, float]:
        """
        Vina dict for the selected export source.

        :return: Vina parameter dictionary.
        :rtype: Dict[str, float]
        :raises ValueError: if no box is available.
        """
        gb = self._choose_box_for_export(self._box_source.value)
        if gb is None:
            raise ValueError("No box available; compute or apply manual first.")
        return gb.vina_dict

    @property
    def vina_lines(self) -> str:
        """
        Vina config lines for the selected export source.

        :return: Multi-line Vina config.
        :rtype: str
        """
        gb = self._choose_box_for_export(self._box_source.value)
        return gb.to_vina_lines() if gb is not None else ""

    @property
    def gridboxes(self) -> Dict[str, GridBox]:
        """
        Available grid boxes (present keys only).

        :return: {'non': GridBox, 'iso': GridBox, 'manual': GridBox}
        :rtype: Dict[str, GridBox]
        """
        out: Dict[str, GridBox] = {}
        if self._gb_non is not None:
            out["non"] = self._gb_non
        if self._gb_iso is not None:
            out["iso"] = self._gb_iso
        if self._gb_manual is not None:
            out["manual"] = self._gb_manual
        return out

    @property
    def ligand_names(self) -> List[str]:
        """Ligand display names in order."""
        return [m["name"] for m in self._ligands]

    @property
    def selected_index(self) -> int:
        """Index of the selected ligand (or -1 if none)."""
        return int(self._ligand_select.value) if self._ligands else -1

    @property
    def selected_name(self) -> Optional[str]:
        """Name of the selected ligand, if available."""
        idx = self.selected_index
        return self._ligands[idx]["name"] if idx >= 0 else None

    @property
    def has_boxes(self) -> bool:
        """True if any grid box exists."""
        return any((self._gb_non, self._gb_iso, self._gb_manual))

    @property
    def ui(self) -> widgets.Widget:
        """
        Root widget container.

        :return: Root widget.
        :rtype: widgets.Widget
        """
        if self._ui is None:
            self.build()
        return self._ui  # type: ignore[return-value]

    @property
    def last_error(self) -> Optional[str]:
        """Last captured error (message + traceback), if any."""
        return self._last_error

    # ==============================================================================
    # Layout / Display
    # ==============================================================================
    def build(self) -> "ProVisGUI":
        """
        Compose the widget layout and return self (idempotent).

        :return: self
        :rtype: ProVisGUI
        """
        if self._ui is None:
            # Header bar
            header = widgets.HBox(
                [
                    widgets.HBox([self._title]),
                    widgets.HBox(
                        [
                            self._theme,
                            self._compact,
                            self._btn_screenshot,
                            self._btn_help,
                        ]
                    ),
                ],
                layout=widgets.Layout(
                    justify_content="space-between",
                    align_items="center",
                    padding="6px 8px",
                ),
            )

            # Cards
            card_setup = self._card(
                "Receptor & Ligands",
                widgets.VBox(
                    [
                        widgets.VBox([self._receptor_path, self._receptor_browse]),
                        widgets.HTML("<hr/>"),
                        widgets.VBox(
                            [
                                self._ligand_path,
                                self._ligand_fmt,
                                widgets.HBox(
                                    [
                                        self._uploader,
                                        self._add_ligand_btn,
                                        self._remove_ligand_btn,
                                        self._clear_ligands_btn,
                                    ]
                                ),
                                self._ligand_select,
                                self._auto_update,
                            ]
                        ),
                    ]
                ),
            )

            card_boxes = self._card(
                "Computed box parameters",
                widgets.VBox(
                    [
                        self._preset,
                        widgets.HBox([self._pad, self._isotropic, self._min_size]),
                        widgets.HBox(
                            [self._heavy_only, self._snap_step, self._round_nd]
                        ),
                        widgets.HTML("<b>Visibility</b>"),
                        widgets.HBox(
                            [self._show_noniso, self._show_iso, self._show_manual]
                        ),
                        widgets.HTML("<b>Style</b>"),
                        widgets.VBox(
                            [
                                self._color_non,
                                self._color_iso,
                                self._color_man,
                                self._opacity,
                            ]
                        ),
                        widgets.HTML("<b>Ligand style</b>"),
                        widgets.HBox([self._lig_color, self._lig_radius]),
                        widgets.HBox([self._update_btn, self._clear_boxes_btn]),
                    ]
                ),
            )

            card_manual = self._card(
                "Manual box",
                widgets.VBox(
                    [
                        widgets.GridBox(
                            children=[
                                self._center_x,
                                self._center_y,
                                self._center_z,
                                self._size_x,
                                self._size_y,
                                self._size_z,
                            ],
                            layout=widgets.Layout(
                                grid_template_columns="repeat(3, minmax(120px, 1fr))",
                                grid_gap="8px",
                            ),
                        ),
                        widgets.HBox([self._use_manual, self._apply_manual_btn]),
                        widgets.HBox([self._fill_noniso_btn, self._fill_iso_btn]),
                    ]
                ),
            )

            card_vina = self._card(
                "Vina I / O",
                widgets.VBox(
                    [
                        widgets.HBox(
                            [
                                self._box_source,
                                self._preview_btn,
                                self._save_name,
                                self._save_vina_btn,
                            ]
                        ),
                        self._vina_preview,
                        widgets.HTML("<b>Vina cfg import</b>"),
                        self._vina_cfg_text,
                        self._vina_import_btn,
                    ]
                ),
            )

            card_session = self._card(
                "Session",
                widgets.VBox(
                    [
                        widgets.HBox([self._save_session_btn, self._load_session_up]),
                    ]
                ),
            )

            help_text = widgets.HTML(
                "<div style='line-height:1.6'>"
                "<b>Workflow</b>: Add ligand(s) → choose preset → Update → tweak/preview → Save cfg.<br>"
                "<b>Shortcuts</b>: Alt+U Update, Alt+S Save cfg. Compact layout for narrow screens.<br>"
                "<b>Screenshot</b>: Saves to disk if supported; otherwise browser downloads PNG.<br>"
                "</div>"
            )
            card_help = self._card("Help", help_text)

            # Tabs (left column)
            self._tabs = widgets.Tab(
                children=[
                    widgets.VBox([card_setup]),
                    widgets.VBox([card_boxes]),
                    widgets.VBox([card_manual]),
                    widgets.VBox([card_vina]),
                    widgets.VBox([card_session]),
                    widgets.VBox([card_help]),
                ]
            )
            for i, name in enumerate(
                ["Setup", "Boxes", "Manual", "Vina", "Session", "Help"]
            ):
                self._tabs.set_title(i, name)

            # ensure Help button works after build
            def _maybe_open_help(_btn=None):
                if getattr(self, "_force_help_tab", False):
                    self._tabs.selected_index = 5
                    self._force_help_tab = False

            self._btn_help.on_click(_maybe_open_help)

            # Right: viewer + status + info
            right_stack = widgets.VBox(
                [
                    widgets.Box(
                        [self._busy], layout=widgets.Layout(justify_content="flex-end")
                    ),
                    self._out,
                    self._status,
                    self._info_panel,
                ]
            )

            # left/right columns (store refs for layout switching)
            self._left_col = widgets.VBox(
                [self._tabs], layout=widgets.Layout(width="44%", padding="6px")
            )
            self._right_col = widgets.VBox(
                [right_stack], layout=widgets.Layout(width="56%", padding="6px")
            )

            # root container
            self._root = widgets.HBox([self._left_col, self._right_col])
            self._ui = widgets.VBox(
                [header, self._root], layout=widgets.Layout(padding="6px")
            )
            self._ui.add_class("provis-root")  # scope CSS

            # Theme + Shortcuts + Layout
            self._apply_theme(self._theme.value)  # inject CSS
            self._inject_shortcuts()  # Alt+U / Alt+S
            self._apply_layout(self._compact.value)
        return self

    def display(self) -> "ProVisGUI":
        """
        Render the GUI and return self.

        :return: self
        :rtype: ProVisGUI
        """
        if self._ui is None:
            self.build()
        display(self._ui)
        return self

    # ==============================================================================
    # Event helpers / internal logic
    # ==============================================================================
    def _on_theme(self, change: Dict[str, Any]) -> None:
        self._apply_theme(change["new"])

    def _on_compact_toggle(self, change: Dict[str, Any]) -> None:
        self._apply_layout(bool(change["new"]))

    def _on_preset_change(self, change: Dict[str, Any]) -> None:
        mode = change.get("new")
        cfg = self.PRESETS.get(mode, {})
        if cfg:
            try:
                self._pad.value = float(cfg.get("pad", self._pad.value))
                self._isotropic.value = bool(
                    cfg.get("isotropic", self._isotropic.value)
                )
                self._min_size.value = float(cfg.get("min_size", self._min_size.value))
                self._status.value = self._status_bar(f"Preset applied: {mode}", "ok")
            except Exception as e:
                self._capture_error("Preset apply error", e)
        if self._auto_update.value:
            self._safe_update()

    def _on_add_ligand(self, _btn) -> None:
        with self._out:
            try:
                clear_output(wait=True)
                added = False
                if self._uploader.value:
                    key = next(iter(self._uploader.value))
                    meta = self._uploader.value[key]
                    blob = meta.get("content", b"") or b""
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
                        self._uploader._counter = 0
                    added = True
                else:
                    lig = self._ligand_path.value.strip()
                    if lig:
                        fmt = self._ligand_fmt.value
                        name = (
                            Path(lig).name
                            if _is_pathlike(lig)
                            else f"pasted_{len(self._ligands)+1}.{fmt}"
                        )
                        text = Path(lig).read_text() if _is_pathlike(lig) else lig
                        self._ligands.append({"data": text, "fmt": fmt, "name": name})
                        added = True
                if not added:
                    print("Provide a ligand path/paste a ligand, or upload a file.")
                    return
                self._refresh_ligand_dropdown()
                self._ligand_select.value = len(self._ligands) - 1
                self._status.value = self._status_bar(
                    f"Added ligand: {self._ligands[-1]['name']}", "ok"
                )
                if self._auto_update.value:
                    self._safe_update()
            except Exception as e:
                clear_output(wait=True)
                self._capture_error("Add ligand error", e, echo=True)

    def _on_remove_ligand(self, _btn) -> None:
        try:
            if not self._ligands:
                self._status.value = self._status_bar("No ligand to remove.", "warn")
                return
            idx = int(self._ligand_select.value)
            removed = self._ligands.pop(idx)
            self._refresh_ligand_dropdown()
            self._status.value = self._status_bar(
                f"Removed ligand: {removed['name']}", "warn"
            )
            if self._auto_update.value:
                self._safe_update()
        except Exception as e:
            self._capture_error("Remove ligand error", e)

    def _on_clear_ligands(self, _btn) -> None:
        self.clear_ligands()
        if self._auto_update.value:
            self._safe_update()

    def _on_receptor_upload(self, _change: Dict[str, Any]) -> None:
        try:
            if not self._receptor_browse.value:
                return
            key = next(iter(self._receptor_browse.value))
            meta = self._receptor_browse.value[key]
            blob = meta.get("content", b"") or b""
            name = meta.get("metadata", {}).get("name", key)
            p = Path.cwd() / name
            p.write_bytes(blob)
            self._receptor_path.value = str(p)
            self._status.value = self._status_bar(f"Receptor saved → {p}", "ok")
            try:
                self._receptor_browse.value.clear()
            except Exception:
                self._receptor_browse._counter = 0
            if self._auto_update.value:
                self._safe_update()
        except Exception as e:
            self._capture_error("Receptor upload error", e)

    def _on_selection_change(self, _change: Dict[str, Any]) -> None:
        if self._auto_update.value:
            self._safe_update()

    def _on_any_param_change(self, _change: Dict[str, Any]) -> None:
        if self._auto_update.value:
            self._safe_update()

    def _on_update(self, _btn) -> None:
        self._safe_update()

    def _safe_update(self) -> None:
        try:
            self._busy.value = self._spinner_html(True)
            self._compute_for_selected()
            self._draw()
            self._refresh_preview_area()
            self._update_info_panel()  # rich info instead of console spam
            self._status.value = self._status_bar("Viewer updated.", "ok")
        except Exception as e:
            self._capture_error("Update error", e, echo=True)
        finally:
            self._busy.value = self._spinner_html(False)

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
                cx, cy, cz = _snap_tuple((cx, cy, cz), float(self._snap_step.value))
                sx, sy, sz = _snap_tuple((sx, sy, sz), float(self._snap_step.value))
            cx, cy, cz = _round_tuple((cx, cy, cz), int(self._round_nd.value))
            sx, sy, sz = _round_tuple((sx, sy, sz), int(self._round_nd.value))
            self._gb_manual = GridBox().from_center_size((cx, cy, cz), (sx, sy, sz))
            self._use_manual.value = True
            self._draw()
            self._refresh_preview_area()
            self._update_info_panel()
            self._status.value = self._status_bar("Manual box applied.", "ok")
        except Exception as e:
            self._capture_error("Manual box error", e, echo=True)

    def _on_fill_noniso(self, _btn) -> None:
        if self._gb_non is None:
            self._warn_need_compute()
            return
        cx, cy, cz = self._gb_non.center
        sx, sy, sz = self._gb_non.size
        self._center_x.value, self._center_y.value, self._center_z.value = cx, cy, cz
        self._size_x.value, self._size_y.value, self._size_z.value = sx, sy, sz
        self._status.value = self._status_bar("Manual filled from non-iso", "ok")

    def _on_fill_iso(self, _btn) -> None:
        if self._gb_iso is None:
            self._warn_need_compute()
            return
        cx, cy, cz = self._gb_iso.center
        sx, sy, sz = self._gb_iso.size
        self._center_x.value, self._center_y.value, self._center_z.value = cx, cy, cz
        self._size_x.value, self._size_y.value, self._size_z.value = sx, sy, sz
        self._status.value = self._status_bar("Manual filled from iso", "ok")

    def _on_clear_boxes(self, _btn) -> None:
        self._gb_non = None
        self._gb_iso = None
        self._gb_manual = None
        self._status.value = self._status_bar("Boxes cleared.", "warn")
        self._refresh_preview_area()
        self._update_info_panel()

    def _on_refresh_preview(self, _btn) -> None:
        self._refresh_preview_area()

    def _on_save_vina(self, _btn) -> None:
        self.save_cfg(None)

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
                clear_output(wait=True)
                print("Imported Vina cfg into manual fields.")
                self._status.value = self._status_bar("CFG imported → manual.", "ok")
            except Exception as e:
                clear_output(wait=True)
                self._capture_error("CFG import error", e, echo=True)

    def _on_save_session(self, _btn) -> None:
        try:
            p = Path("provis_session.json")
            p.write_text(json.dumps(self._session_payload(), indent=2))
            self._status.value = self._status_bar(f"Session saved → {p}", "ok")
        except Exception as e:
            self._capture_error("Save session error", e)

    def _on_load_session(self, _change: Dict[str, Any]) -> None:
        try:
            if not self._load_session_up.value:
                return
            key = next(iter(self._load_session_up.value))
            meta = self._load_session_up.value[key]
            blob = meta.get("content", b"") or b""
            text = blob.decode("utf-8", errors="replace")
            payload = json.loads(text)
            self._restore_session(payload)
            self._status.value = self._status_bar("Session loaded.", "ok")
            try:
                self._load_session_up.value.clear()
            except Exception:
                self._load_session_up._counter = 0
            if self._auto_update.value:
                self._safe_update()
        except Exception as e:
            self._capture_error("Load session error", e)

    def _on_screenshot(self, _btn) -> None:
        self.screenshot()

    # ---- Core compute/draw --------------------------------------------------------
    def _compute_for_selected(self) -> None:
        """Compute non-isotropic and isotropic GridBoxes for selected ligand."""
        if not self._ligands:
            raise ValueError("No ligands added.")
        idx = int(self._ligand_select.value)
        meta = self._ligands[idx]
        data, fmt = meta["data"], meta["fmt"]
        self._gb_non = (
            GridBox()
            .load_ligand(data, fmt=fmt)
            .from_ligand_pad_adv(
                pad=float(self._pad.value),
                isotropic=False,
                min_size=float(self._min_size.value),
                heavy_only=bool(self._heavy_only.value),
                snap_step=(float(self._snap_step.value) or None),
                round_ndigits=int(self._round_nd.value),
            )
        )
        self._gb_iso = (
            GridBox()
            .load_ligand(data, fmt=fmt)
            .from_ligand_pad_adv(
                pad=float(self._pad.value),
                isotropic=True,
                min_size=float(self._min_size.value),
                heavy_only=bool(self._heavy_only.value),
                snap_step=(float(self._snap_step.value) or None),
                round_ndigits=int(self._round_nd.value),
            )
        )
        self._status.value = self._status_bar(
            f"Computed boxes for: {meta['name']} (idx {idx})", "ok"
        )

    def _draw(self) -> None:
        """Render current scene to the GUI output area (quiet)."""
        with self._out:
            clear_output(wait=True)
            try:
                viz = ProVis(vw=self._vw, vh=self._vh)
                rec = self._receptor_path.value.strip()
                if rec:
                    p = Path(rec).expanduser()
                    if p.exists():
                        viz.load_receptor(str(p)).style_preset(
                            "publication", surface=False
                        )
                    else:
                        print(f"Receptor path does not exist: {p}")
                if self._ligands:
                    idx = int(self._ligand_select.value)
                    meta = self._ligands[idx]
                    viz.load_ligand_from_text(
                        meta["data"], name=meta["name"], fmt=meta["fmt"]
                    )
                    viz.highlight_ligand(
                        style="stick",
                        color=self._css_hex_to_threejs(self._lig_color.value),
                        radius=float(self._lig_radius.value),
                    )

                if self._show_noniso.value and self._gb_non is not None:
                    viz.add_gridbox_with_labels(
                        self._gb_non,
                        color=self._css_hex_to_threejs(self._color_non.value),
                        opacity=float(self._opacity.value),
                    )
                if self._show_iso.value and self._gb_iso is not None:
                    viz.add_gridbox_with_labels(
                        self._gb_iso,
                        color=self._css_hex_to_threejs(self._color_iso.value),
                        opacity=float(self._opacity.value),
                    )
                if self._show_manual.value and self._gb_manual is not None:
                    viz.add_gridbox_with_labels(
                        self._gb_manual,
                        color=self._css_hex_to_threejs(self._color_man.value),
                        opacity=float(self._opacity.value),
                    )

                viz.set_background(self._bg_hex_for_provis()).show()

                # No console prints here — info goes to _info_panel instead
                self._last_viz = viz
            except Exception as e:
                self._capture_error("Draw error", e, echo=True)

    # ---- Support & utility --------------------------------------------------------
    def _choose_box_for_export(self, source: str = "auto") -> Optional[GridBox]:
        if self._use_manual.value and source == "auto":
            source = "manual"
        if source == "manual":
            return self._gb_manual
        if source == "iso":
            return self._gb_iso
        if source == "non":
            return self._gb_non
        if self._isotropic.value and self._gb_iso is not None:
            return self._gb_iso
        return self._gb_non

    def _refresh_ligand_dropdown(self) -> None:
        self._ligand_select.options = [
            (m["name"], i) for i, m in enumerate(self._ligands)
        ]
        self._ligand_select.value = (len(self._ligands) - 1) if self._ligands else None

    def _refresh_preview_area(self) -> None:
        self._vina_preview.value = self._code_block(self.vina_lines)

    def _warn_need_compute(self) -> None:
        self._status.value = self._status_bar(
            "Compute boxes first (Update viewer).", "warn"
        )

    def _index_from_key(self, key: Union[int, str]) -> int:
        if isinstance(key, int):
            if not (0 <= key < len(self._ligands)):
                raise IndexError("Ligand index out of range.")
            return key
        for i, m in enumerate(self._ligands):
            if m["name"] == key:
                return i
        raise KeyError(f"Ligand '{key}' not found.")

    def _session_payload(self) -> Dict[str, Any]:
        return {
            "receptor": self._receptor_path.value,
            "ligands": self._ligands,
            "ui_state": {
                "preset": self._preset.value,
                "pad": self._pad.value,
                "isotropic": self._isotropic.value,
                "min_size": self._min_size.value,
                "heavy_only": self._heavy_only.value,
                "snap_step": self._snap_step.value,
                "round_nd": self._round_nd.value,
                "use_manual": self._use_manual.value,
                "show": {
                    "non": self._show_noniso.value,
                    "iso": self._show_iso.value,
                    "manual": self._show_manual.value,
                },
                "box_source": self._box_source.value,
                "save_name": self._save_name.value,
                "selected_index": self.selected_index,
                "center": [
                    self._center_x.value,
                    self._center_y.value,
                    self._center_z.value,
                ],
                "size": [self._size_x.value, self._size_y.value, self._size_z.value],
                "colors": {
                    "non": self._color_non.value,
                    "iso": self._color_iso.value,
                    "man": self._color_man.value,
                    "lig": self._lig_color.value,
                    "opacity": self._opacity.value,
                    "lig_r": self._lig_radius.value,
                },
                "theme": self._theme.value,
                "compact": self._compact.value,
            },
        }

    def _restore_session(self, payload: Dict[str, Any]) -> None:
        self._receptor_path.value = payload.get("receptor", "")
        self._ligands = list(payload.get("ligands", []))
        self._refresh_ligand_dropdown()
        ui = payload.get("ui_state", {})
        self._preset.value = ui.get("preset", self._preset.value)
        self._pad.value = float(ui.get("pad", self._pad.value))
        self._isotropic.value = bool(ui.get("isotropic", self._isotropic.value))
        self._min_size.value = float(ui.get("min_size", self._min_size.value))
        self._heavy_only.value = bool(ui.get("heavy_only", self._heavy_only.value))
        self._snap_step.value = float(ui.get("snap_step", self._snap_step.value))
        self._round_nd.value = int(ui.get("round_nd", self._round_nd.value))
        self._use_manual.value = bool(ui.get("use_manual", self._use_manual.value))
        show = ui.get("show", {})
        self._show_noniso.value = bool(show.get("non", self._show_noniso.value))
        self._show_iso.value = bool(show.get("iso", self._show_iso.value))
        self._show_manual.value = bool(show.get("manual", self._show_manual.value))
        self._box_source.value = ui.get("box_source", self._box_source.value)
        self._save_name.value = ui.get("save_name", self._save_name.value)
        sel = int(ui.get("selected_index", -1))
        if 0 <= sel < len(self._ligands):
            self._ligand_select.value = sel
        center = ui.get("center", [0.0, 0.0, 0.0])
        size = ui.get("size", [20.0, 20.0, 20.0])
        self._center_x.value, self._center_y.value, self._center_z.value = center
        self._size_x.value, self._size_y.value, self._size_z.value = size
        colors = ui.get("colors", {})
        if colors:
            self._color_non.value = colors.get("non", self._color_non.value)
            self._color_iso.value = colors.get("iso", self._color_iso.value)
            self._color_man.value = colors.get("man", self._color_man.value)
            self._lig_color.value = colors.get("lig", self._lig_color.value)
            self._opacity.value = float(colors.get("opacity", self._opacity.value))
            self._lig_radius.value = float(colors.get("lig_r", self._lig_radius.value))
        self._theme.value = ui.get("theme", self._theme.value)
        self._compact.value = bool(ui.get("compact", self._compact.value))
        # re-apply layout after restoring
        self._apply_layout(self._compact.value)

    def _capture_error(self, msg: str, e: BaseException, echo: bool = False) -> None:
        tb = traceback.format_exc()
        self._last_error = f"{msg}: {e}\n{tb}"
        self._status.value = self._status_bar(f"{msg}: {e}", "err")
        if echo:
            print(self._last_error)

    # ==============================================================================
    # Visual helpers (cards, badges, theme, code, info)
    # ==============================================================================
    def _apply_theme(self, key: str) -> None:
        """
        Inject scoped CSS styles for theme, without fighting ipywidgets internals.
        """
        t = self.THEMES.get(key, self.THEMES["light"])
        # Root background
        if self._ui:
            self._ui.layout = widgets.Layout(
                padding="6px", border="1px solid transparent", background_color=t["bg"]
            )
        # Inject style (overwrites on each call)
        css = f"""
        <style id="provis-gui-theme">
          .provis-root .provis-card {{
            background: {t['card_bg']} !important;
            border: 1px solid {t['card_border']} !important;
            box-shadow: 0 3px 10px rgba(0,0,0,0.06);
            border-radius: 6px;
          }}
          .provis-root .provis-card h3 {{
            margin: 0 0 6px 0; color: {t['accent']};
          }}
          .provis-root .provis-status-bar {{
            color: {t['mute']};
          }}
          .provis-root .provis-vina-preview pre {{
            background: {t['code_bg']} !important;
            color: {t['code_fg']} !important;
          }}
        </style>
        """
        display(HTML(css))
        self._refresh_preview_area()

    def _apply_layout(self, compact: bool) -> None:
        """
        Switch between two-column and compact single-column layout.
        """
        if not (self._root and self._left_col and self._right_col):
            return
        if compact:
            self._left_col.layout = widgets.Layout(width="100%", padding="6px")
            self._right_col.layout = widgets.Layout(width="100%", padding="6px")
            self._root.children = (
                widgets.VBox([self._left_col, self._right_col]),
            )  # stack vertical
        else:
            self._left_col.layout = widgets.Layout(width="44%", padding="6px")
            self._right_col.layout = widgets.Layout(width="56%", padding="6px")
            self._root.children = (self._left_col, self._right_col)

    def _card(self, title: str, body: widgets.Widget) -> widgets.VBox:
        """
        Create a styled card container (scoped with class for CSS).
        """
        head = widgets.HTML(f"<h3>{title}</h3>")
        box = widgets.VBox(
            [head, body], layout=widgets.Layout(padding="10px 12px", margin="8px 0")
        )
        box.add_class("provis-card")  # give the DOM a class we can style
        return box

    def _status_bar(self, text: str, level: str = "ok") -> str:
        """
        Build tinted status HTML.
        """
        t = self.THEMES.get(self._theme.value, self.THEMES["light"])
        color = {
            "ok": t["status_ok"],
            "warn": t["status_warn"],
            "err": t["status_err"],
        }.get(level, t["status_ok"])
        return f"<div class='provis-status-bar' style='padding:8px;border-left:4px solid {color};'>{text}</div>"

    def _code_block(self, txt: str) -> str:
        """
        Render code-like block (used for Vina preview).
        """
        t = self.THEMES.get(self._theme.value, self.THEMES["light"])
        esc = (
            (txt or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return (
            f"<div class='provis-vina-preview'>"
            f"<pre style='background:{t['code_bg']};color:{t['code_fg']};"
            f"padding:10px;border-radius:6px;white-space:pre-wrap;line-height:1.3;margin:0'>{esc}</pre>"
            f"</div>"
        )

    def _info_block(self, html_inner: str) -> str:
        """
        Provide a bordered info panel wrapper.
        """
        return (
            "<div style='padding:10px;border:1px dashed #d1d5db;border-radius:6px;margin-top:8px;"
            "font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
            + "'Liberation Mono', 'Courier New', monospace;'>"
            f"{html_inner}</div>"
        )

    def _update_info_panel(self) -> None:
        """
        Summarize current state (chosen export box + sizes) into the right-side panel.
        """
        gb = self._choose_box_for_export(self._box_source.value)
        if not gb:
            self._info_panel.value = self._info_block("<i>No box selected.</i>")
            return
        cx, cy, cz = gb.center
        sx, sy, sz = gb.size
        pseudo = (
            f"<b>Export source:</b> {self._box_source.value}<br>"
            f"<b>Center:</b> ({cx:.3f}, {cy:.3f}, {cz:.3f})<br>"
            f"<b>Size:</b> ({sx:.3f}, {sy:.3f}, {sz:.3f})<br>"
            f"<b>Isotropic flag:</b> {self._isotropic.value} &nbsp; "
            f"<b>Pad:</b> {self._pad.value} Å &nbsp; "
            f"<b>Min size:</b> {self._min_size.value} Å<br>"
            f"<b>Heavy-only:</b> {self._heavy_only.value} &nbsp; "
            f"<b>Snap:</b> {self._snap_step.value} Å &nbsp; "
            f"<b>Round:</b> {self._round_nd.value} digits"
        )
        self._info_panel.value = self._info_block(pseudo)

    def _spinner_html(self, on: bool) -> str:
        if not on:
            return "<div></div>"
        return "<div style='font-size:12px;color:#8892a6;'><i class='fa fa-refresh fa-spin'></i> working...</div>"

    def _bg_hex_for_provis(self) -> str:
        t = self.THEMES.get(self._theme.value, self.THEMES["light"])
        return self._css_hex_to_threejs(t["bg"])

    @staticmethod
    def _css_hex_to_threejs(hex_color: str) -> str:
        h = hex_color.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return f"0x{h}"

    # ---- Shortcuts ----------------------------------------------------------------
    def _inject_shortcuts(self) -> None:
        """
        Inject Alt+U (Update viewer) and Alt+S (Save cfg) shortcuts.

        We use button tooltips as stable selectors: 'update-shortcut' and 'savecfg-shortcut'.
        """
        if self._shortcuts_injected:
            return
        js = """
        (function(){
          if (window.__provis_shortcuts_loaded__) return;
          window.__provis_shortcuts_loaded__ = true;
          document.addEventListener('keydown', function(e){
            try{
              if (e.altKey && e.key && e.key.toLowerCase() === 'u'){
                const btn = Array.from(document.querySelectorAll('button[title="update-shortcut"]'))[0];
                if (btn){ btn.click(); e.preventDefault(); }
              }
              if (e.altKey && e.key && e.key.toLowerCase() === 's'){
                const btn = Array.from(document.querySelectorAll('button[title="savecfg-shortcut"]'))[0];
                if (btn){ btn.click(); e.preventDefault(); }
              }
            }catch(err){ console.warn('Shortcut error', err); }
          }, true);
        })();
        """
        display(Javascript(js))
        self._shortcuts_injected = True

    # ==============================================================================
    # Class-level utilities
    # ==============================================================================
    @classmethod
    def register_preset(
        cls, name: str, pad: float, isotropic: bool, min_size: float
    ) -> None:
        """
        Register or override a preset at runtime.

        :param name: Preset key name.
        :type name: str
        :param pad: Padding around ligand (Å).
        :type pad: float
        :param isotropic: Whether the box is cubic (isotropic).
        :type isotropic: bool
        :param min_size: Minimal cubic edge length for isotropic boxes (Å).
        :type min_size: float
        """
        cls.PRESETS[name] = {
            "pad": float(pad),
            "isotropic": bool(isotropic),
            "min_size": float(min_size),
        }

    # ------------------------------------------------------------------------------
    # Python niceties
    # ------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<ProVisGUI ligands={len(self._ligands)} preset='{self._preset.value}' "
            f"iso={self._isotropic.value} pad={self._pad.value} compact={self._compact.value}>"
        )

    def __len__(self) -> int:
        return len(self._ligands)

    def __contains__(self, name: str) -> bool:
        return any(m["name"] == name for m in self._ligands)
