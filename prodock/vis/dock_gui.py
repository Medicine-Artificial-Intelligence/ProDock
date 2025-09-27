# dock_gui.py
from __future__ import annotations

import json
import html
import time
import threading
import traceback
import zipfile
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Javascript

from prodock.dock.multiple import MultipleDock
from prodock.io.convert import pdbqt_to_sdf  # PDBQT -> SDF conversion (Open Babel)

# ProVis (see class below in this message)
from prodock.vis.provis import ProVis


class DockGUI:
    """
    Interactive Dock GUI for :class:`prodock.engine.multiple.MultipleDock`.

    Features
    --------
    - Async, **chunked** runs with **Pause/Resume/Stop**
    - Inline **tailing of newest .log** into a right-side log card
    - **ProVis** Top-5 pose preview (if ProVis importable)
      - Loads *first pose* from each PDBQT by converting to **SDF** and then to **PDB** (robust for py3Dmol).
    - **Keyboard shortcuts**: Alt+U (Run), Alt+S (Write CSV), Alt+P (Pause/Resume)
    - **Compact one-column** layout toggle for small screens
    - Themed **Dark/Light** look, reduced console spam (logs go to card)
    - **Export report** (HTML + optional ZIP bundle)

    Typical usage
    -------------
    Construct, build and display the GUI::

        >>> gui = DockGUI(vw=1100, vh=700).build().display()

    Parameters
    ----------
    :param vw: initial viewer width (px), used for ProVis rendering defaults.
    :type vw: int
    :param vh: initial viewer height (px), used for ProVis rendering defaults.
    :type vh: int

    Notes
    -----
    - Autobox is supported for binary backends (smina/qvina), not for Vina API.
    - All fluent mutators return ``self``; retrieve state via properties.
    """

    THEMES = {
        "light": {
            "bg": "#ffffff",
            "card_bg": "#fbfbfd",
            "card_border": "#e7eef6",
            "mute": "#6b7280",
            "accent": "#2563eb",
            "status_ok": "#16a34a",
            "status_warn": "#f59e0b",
            "status_err": "#dc2626",
            "table_head_bg": "#f3f4f6",
            "table_border": "#e5e7eb",
            "link": "#0ea5e9",
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
            "table_head_bg": "#111a2f",
            "table_border": "#1f2a44",
            "link": "#67e8f9",
        },
    }

    def __init__(self, vw: int = 1100, vh: int = 700) -> None:
        # viewer size
        self._vw = int(vw)
        self._vh = int(vh)

        # basic widgets
        self._theme = widgets.ToggleButtons(
            options=[("🌞 Light", "light"), ("🌙 Dark", "dark")],
            value="light",
        )
        self._compact = widgets.Checkbox(
            value=False, description="Compact layout (1-col)"
        )
        self._title = widgets.HTML(
            "<h2 style='margin:6px 0 0 0;'>Dock Console</h2>"
            "<div style='color:#6b7280;margin-top:-4px;'>MultipleDock</div>"
        )
        self._btn_help = widgets.Button(
            description="", icon="question-circle", tooltip="Open help"
        )

        # Sources / ligands
        self._receptor_path = widgets.Text(
            value="",
            description="Receptor:",
            placeholder="path/to/receptor.pdbqt or .pdb",
        )
        self._ligand_dir = widgets.Text(
            value="", description="Ligand dir:", placeholder="path/to/ligands"
        )
        self._ligand_list_json = widgets.Textarea(
            value="",
            description="Ligands (JSON):",
            placeholder='["/path/a.pdbqt","/path/b.pdbqt"]',
            layout=widgets.Layout(height="90px"),
        )
        self._refresh_lig_btn = widgets.Button(
            description="Refresh ligands", icon="refresh"
        )
        self._preview_ligands_btn = widgets.Button(
            description="Preview found", icon="list"
        )
        self._lig_format = widgets.Dropdown(
            options=["pdbqt", "sdf", "mol2", "auto", "any"],
            value="pdbqt",
            description="Ligand fmt:",
        )
        self._filter_pat = widgets.Text(value="*", description="Filter glob:")

        # Backend / runtime params
        self._backend = widgets.Dropdown(
            options=[
                ("Vina (API)", "vina"),
                ("smina (binary)", "smina"),
                ("qvina", "qvina"),
                ("qvina-w", "qvina-w"),
                ("Custom binary", "binary"),
            ],
            value="vina",
            description="Backend:",
        )
        self._custom_backend = widgets.Text(
            value="", description="Custom exe:", placeholder="path/to/exe"
        )
        self._exhaust = widgets.IntSlider(
            value=8,
            min=1,
            max=64,
            step=1,
            description="Exhaustiveness:",
            continuous_update=False,
        )
        self._n_poses = widgets.IntSlider(
            value=9,
            min=1,
            max=50,
            step=1,
            description="Num poses:",
            continuous_update=False,
        )
        self._cpu = widgets.IntText(value=4, description="CPU:")
        self._workers = widgets.IntSlider(
            value=4,
            min=1,
            max=64,
            step=1,
            description="Workers:",
            continuous_update=False,
        )
        self._seed = widgets.IntText(value=0, description="Seed (0 none):")

        # Box / autobox
        self._use_autobox = widgets.Checkbox(
            value=False, description="Use autobox (Binary only)"
        )
        self._autobox_ref = widgets.Text(value="", description="Autobox ref:")
        self._autobox_pad = widgets.FloatText(value=4.0, description="Autobox pad:")
        self._center_x = widgets.FloatText(value=0.0, description="center_x:")
        self._center_y = widgets.FloatText(value=0.0, description="center_y:")
        self._center_z = widgets.FloatText(value=0.0, description="center_z:")
        self._size_x = widgets.FloatText(value=22.5, description="size_x:")
        self._size_y = widgets.FloatText(value=22.5, description="size_y:")
        self._size_z = widgets.FloatText(value=22.5, description="size_z:")
        self._vina_cfg_text = widgets.Textarea(
            value="",
            description="Import cfg:",
            placeholder="Paste Vina center_x/…/size_z here",
            layout=widgets.Layout(height="82px"),
        )
        self._vina_import_btn = widgets.Button(
            description="Import → box", icon="sign-in"
        )

        # IO & runtime
        self._out_dir = widgets.Text(value="./docked", description="Out dir:")
        self._log_dir = widgets.Text(value="", description="Log dir (opt):")
        self._skip_existing = widgets.Checkbox(value=True, description="Skip existing")
        self._max_retries = widgets.IntSlider(
            value=2, min=0, max=10, step=1, description="Max retries:"
        )
        self._backoff = widgets.FloatText(value=1.5, description="Retry backoff:")
        self._timeout = widgets.FloatText(value=0.0, description="Timeout (s, 0=off):")
        self._verbose = widgets.Dropdown(
            options=[(0, 0), (1, 1), (2, 2)], value=1, description="Verbosity:"
        )

        # Actions
        self._run_btn = widgets.Button(
            description="Run docking",
            button_style="primary",
            icon="play",
            tooltip="update-shortcut",
        )
        self._pause_btn = widgets.ToggleButton(
            value=False, description="Pause", icon="pause", tooltip="pause-shortcut"
        )
        self._stop_btn = widgets.Button(description="Stop", icon="stop")
        self._write_csv_btn = widgets.Button(
            description="Write summary CSV", icon="save", tooltip="savecfg-shortcut"
        )
        self._export_report_btn = widgets.Button(
            description="Export report (HTML)", icon="file"
        )
        self._zip_outputs = widgets.Checkbox(
            value=False, description="Zip outputs with report"
        )
        self._chunk_size = widgets.IntSlider(
            value=50,
            min=1,
            max=2000,
            step=1,
            description="Chunk size:",
            continuous_update=False,
        )

        # status/info
        self._status = widgets.HTML(
            self._status_bar("Ready. Provide receptor and ligands.", level="warn")
        )
        self._info_panel = widgets.HTML(self._info_block("<i>No run yet.</i>"))
        self._pbar = widgets.IntProgress(
            value=0, min=0, max=100, description="Progress"
        )
        self._eta = widgets.HTML("<div class='muted'>ETA: —</div>")
        self._ticker = widgets.HTML("<div class='muted'>—</div>")
        self._filter_box = widgets.Text(
            value="", placeholder="Filter results (ligand / status / score)..."
        )
        self._results_html = widgets.HTML(self._results_table_html([]))

        # logs & preview
        self._log_output = widgets.Output(
            layout={"border": "1px solid #ddd", "height": "220px", "overflow": "auto"}
        )
        self._tail_logs = widgets.Checkbox(value=True, description="Tail newest .log")
        self._preview_top_btn = widgets.Button(
            description="Preview Top-5 poses", icon="eye"
        )
        self._viz_screenshot_btn = widgets.Button(
            description="Screenshot viewer", icon="camera"
        )
        self._viz_out = widgets.Output(
            layout={
                "border": "1px solid transparent",
                "height": "420px",
                "overflow": "auto",
            }
        )

        # internal state
        self._ui: Optional[widgets.Widget] = None
        self._left_col: Optional[widgets.Widget] = None
        self._right_col: Optional[widgets.Widget] = None
        self._root: Optional[widgets.Widget] = None
        self._tabs: Optional[widgets.Tab] = None
        self._last_md: Optional[MultipleDock] = None
        self._last_error: Optional[str] = None
        self._agg_lock = threading.Lock()
        self._agg_results: List[Dict[str, Any]] = []
        self._total_ligands = 0
        self._processed_ligands = 0
        self._run_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._log_thread: Optional[threading.Thread] = None
        self._run_start_ts: Optional[float] = None
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._on_finish_callbacks: List[Callable[["DockGUI"], None]] = []

        # wire events
        self._theme.observe(self._on_theme, names="value")
        self._compact.observe(self._on_compact, names="value")
        self._btn_help.on_click(lambda _: self._open_help_tab())

        self._refresh_lig_btn.on_click(self._on_refresh_ligands)
        self._preview_ligands_btn.on_click(self._on_preview_ligands)
        self._vina_import_btn.on_click(self._on_vina_import)

        self._run_btn.on_click(self._on_run)
        self._pause_btn.observe(self._on_pause_toggle, names="value")
        self._stop_btn.on_click(self._on_stop)
        self._write_csv_btn.on_click(self._on_write_csv)
        self._export_report_btn.on_click(self._on_export_report)
        self._filter_box.observe(self._on_filter_change, names="value")
        self._tail_logs.observe(self._on_tail_toggle, names="value")
        self._preview_top_btn.on_click(self._on_preview_top)
        self._viz_screenshot_btn.on_click(self._on_viz_screenshot)

        self._run_btn.tooltip = "update-shortcut"
        self._write_csv_btn.tooltip = "savecfg-shortcut"
        self._pause_btn.tooltip = "pause-shortcut"

    # ---------- Fluent API ----------
    def set_receptor(self, path: Union[str, Path]) -> "DockGUI":
        """
        Set receptor path and return self.

        :param path: path to receptor file (PDBQT or PDB).
        :type path: str | pathlib.Path
        :returns: self for fluent chaining.
        :rtype: DockGUI
        """
        self._receptor_path.value = str(Path(str(path)).expanduser())
        return self

    def set_ligand_dir(self, path: Union[str, Path]) -> "DockGUI":
        """
        Set ligands directory and return self.

        :param path: directory containing ligand files.
        :type path: str | pathlib.Path
        :returns: self for fluent chaining.
        :rtype: DockGUI
        """
        self._ligand_dir.value = str(Path(str(path)).expanduser())
        return self

    def set_ligands(self, ligands: Sequence[Union[str, Path]]) -> "DockGUI":
        """
        Set explicit ligands list (populates the JSON textarea) and return self.

        :param ligands: sequence of ligand paths.
        :type ligands: Sequence[str | pathlib.Path]
        :returns: self for fluent chaining.
        :rtype: DockGUI
        """
        arr = [str(Path(str(p)).expanduser()) for p in ligands]
        self._ligand_list_json.value = json.dumps(arr, indent=2)
        return self

    def set_on_finish(self, callback: Callable[["DockGUI"], None]) -> "DockGUI":
        """
        Register a callback invoked when a run finishes; returns self.

        :param callback: callable that accepts the DockGUI instance.
        :type callback: Callable[[DockGUI], None]
        :returns: self
        :rtype: DockGUI
        """
        if callable(callback):
            self._on_finish_callbacks.append(callback)
        return self

    def set_discovery(
        self, ligand_format: str = "pdbqt", filter_glob: str = "*"
    ) -> "DockGUI":
        """
        Set ligand discovery options; returns self.

        :param ligand_format: one of "pdbqt","sdf","mol2","auto","any".
        :type ligand_format: str
        :param filter_glob: glob pattern (without extension) used to find ligands.
        :type filter_glob: str
        :returns: self
        :rtype: DockGUI
        """
        self._lig_format.value = ligand_format
        self._filter_pat.value = filter_glob
        return self

    def set_output(
        self, out_dir: Union[str, Path], log_dir: Optional[Union[str, Path]] = None
    ) -> "DockGUI":
        """
        Set output and logs directories; returns self.

        :param out_dir: output directory path.
        :type out_dir: str | pathlib.Path
        :param log_dir: optional log directory path; if omitted logs go to out_dir/logs.
        :type log_dir: Optional[str | pathlib.Path]
        :returns: self
        :rtype: DockGUI
        """
        self._out_dir.value = str(Path(str(out_dir)).expanduser())
        self._log_dir.value = (
            "" if log_dir is None else str(Path(str(log_dir)).expanduser())
        )
        return self

    def run(self) -> "DockGUI":
        """
        Start docking asynchronously in chunks and return self.

        :returns: self
        :rtype: DockGUI
        """
        self._start_async_run_chunked()
        return self

    def write_summary(self, path: Optional[Union[str, Path]] = None) -> "DockGUI":
        """
        Write CSV summary immediately; returns self.

        :param path: optional path to write CSV; if None the MultipleDock default is used.
        :type path: Optional[str | pathlib.Path]
        :returns: self
        :rtype: DockGUI
        """
        try:
            if not self._last_md:
                self._status.value = self._status_bar(
                    "Nothing to write — no run yet.", "warn"
                )
                return self
            p = self._last_md.write_summary(path)
            with self._log_output:
                clear_output(wait=True)
                print("Summary written:", p)
            self._status.value = self._status_bar(f"Summary CSV written → {p}", "ok")
        except Exception as e:
            self._capture_error("Write summary error", e, echo=True)
        return self

    def export_report(self, zip_outputs: bool = False) -> "DockGUI":
        """
        Export report (HTML + CSV, optional ZIP) and return self.

        :param zip_outputs: include outputs and logs in a zip bundle with the report.
        :type zip_outputs: bool
        :returns: self
        :rtype: DockGUI
        """
        self._execute_export_report(zip_outputs)
        return self

    # ---------- Build / Display ----------
    def build(self) -> "DockGUI":
        """
        Compose widgets and return self (idempotent).

        :returns: self, ready to display.
        :rtype: DockGUI
        """
        if self._ui is None:
            header = widgets.HBox(
                [
                    widgets.HBox([self._title]),
                    widgets.HBox([self._theme, self._compact, self._btn_help]),
                ],
                layout=widgets.Layout(
                    justify_content="space-between",
                    align_items="center",
                    padding="6px 8px",
                ),
            )

            card_sources = self._card(
                "Sources",
                widgets.VBox(
                    [
                        self._receptor_path,
                        widgets.HTML("<b>Ligands</b>"),
                        self._ligand_dir,
                        self._ligand_list_json,
                        widgets.HBox(
                            [self._refresh_lig_btn, self._preview_ligands_btn]
                        ),
                        widgets.HTML("<b>Discovery filters</b>"),
                        widgets.HBox([self._lig_format, self._filter_pat]),
                    ]
                ),
            )

            card_backend = self._card(
                "Backend & Params",
                widgets.VBox(
                    [
                        widgets.HBox(
                            [self._backend, self._custom_backend, self._verbose]
                        ),
                        widgets.HBox([self._exhaust, self._n_poses]),
                        widgets.HBox([self._cpu, self._workers, self._seed]),
                    ]
                ),
            )

            card_box = self._card(
                "Box / Autobox",
                widgets.VBox(
                    [
                        self._use_autobox,
                        widgets.HBox([self._autobox_ref, self._autobox_pad]),
                        widgets.HTML("<b>Explicit box</b>"),
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
                        widgets.HBox([self._vina_cfg_text, self._vina_import_btn]),
                    ]
                ),
            )

            card_io = self._card(
                "IO & Runtime",
                widgets.VBox(
                    [
                        self._out_dir,
                        self._log_dir,
                        widgets.HBox(
                            [self._skip_existing, self._max_retries, self._backoff]
                        ),
                        widgets.HBox([self._timeout]),
                        widgets.HBox([self._chunk_size]),
                    ]
                ),
            )

            help_text = widgets.HTML(
                "<div style='line-height:1.6'>"
                "<b>Workflow</b>: fill receptor & ligands → choose backend → set box/autobox → Run.<br>"
                "<b>Shortcuts</b>: Alt+U Run, Alt+S Write CSV, Alt+P Pause/Resume."
                "</div>"
            )
            card_help = self._card("Help", help_text)

            self._tabs = widgets.Tab(
                children=[
                    widgets.VBox([card_sources]),
                    widgets.VBox([card_backend]),
                    widgets.VBox([card_box]),
                    widgets.VBox([card_io]),
                    widgets.VBox([card_help]),
                ]
            )
            for i, name in enumerate(["Sources", "Backend", "Box", "IO", "Help"]):
                self._tabs.set_title(i, name)

            actions = widgets.HBox(
                [
                    self._run_btn,
                    self._pause_btn,
                    self._stop_btn,
                    self._write_csv_btn,
                    self._export_report_btn,
                    self._zip_outputs,
                ]
            )
            progress = widgets.HBox([self._pbar, self._eta])
            preview_bar = widgets.HBox(
                [self._preview_top_btn, self._viz_screenshot_btn]
            )

            right_stack = widgets.VBox(
                [
                    actions,
                    progress,
                    self._status,
                    self._info_panel,
                    widgets.HTML("<b>Results</b>"),
                    self._filter_box,
                    self._results_html,
                    widgets.HTML("<b>Pose Preview</b>"),
                    preview_bar,
                    self._viz_out,
                    widgets.HTML("<b>Log tail</b>"),
                    widgets.HBox([self._tail_logs]),
                    self._log_output,
                    widgets.HTML("<b>Status</b>"),
                    self._ticker,
                ]
            )

            self._left_col = widgets.VBox(
                [self._tabs], layout=widgets.Layout(width="42%", padding="6px")
            )
            self._right_col = widgets.VBox(
                [right_stack], layout=widgets.Layout(width="58%", padding="6px")
            )
            self._root = widgets.HBox([self._left_col, self._right_col])
            self._ui = widgets.VBox(
                [header, self._root], layout=widgets.Layout(padding="6px")
            )
            self._ui.add_class("provis-root")

            # apply theme, shortcuts and layout
            self._apply_theme(self._theme.value)
            self._inject_shortcuts()
            self._apply_layout(self._compact.value)
        return self

    def display(self) -> "DockGUI":
        """
        Render the GUI into the current IPython output area and return self.

        :returns: self
        :rtype: DockGUI
        """
        if self._ui is None:
            self.build()
        display(self._ui)
        return self

    def help(self) -> None:
        """Print a brief help message describing the DockGUI usage."""
        print(self.__doc__ or "DockGUI")

    # ---------- Event handlers ----------
    def _open_help_tab(self) -> None:
        if self._tabs:
            self._tabs.selected_index = 4

    def _on_theme(self, change: Dict[str, Any]) -> None:
        self._apply_theme(change["new"])

    def _on_compact(self, change: Dict[str, Any]) -> None:
        self._apply_layout(bool(change["new"]))

    def _on_refresh_ligands(self, _btn) -> None:
        try:
            md = self._build_md(dry=True)
            self._status.value = self._status_bar(
                f"Found {len(md.ligands)} ligands.", "ok"
            )
            self._ticker.value = f"<div>Discovered {len(md.ligands)} ligands</div>"
            with self._log_output:
                clear_output(wait=True)
                print(f"Discovered {len(md.ligands)} ligands. (First 50 listed below)")
                for p in md.ligands[:50]:
                    print(" -", p)
                if len(md.ligands) > 50:
                    print(" ... (+%d more)" % (len(md.ligands) - 50))
        except Exception as e:
            self._capture_error("Refresh ligands error", e, echo=True)

    def _on_preview_ligands(self, _btn) -> None:
        self._on_refresh_ligands(_btn)

    def _on_vina_import(self, _btn) -> None:
        try:
            d = self._parse_vina_cfg(self._vina_cfg_text.value)
            self._center_x.value = d["center_x"]
            self._center_y.value = d["center_y"]
            self._center_z.value = d["center_z"]
            self._size_x.value = d["size_x"]
            self._size_y.value = d["size_y"]
            self._size_z.value = d["size_z"]
            self._status.value = self._status_bar("Vina cfg imported to box.", "OK")
        except Exception as e:
            self._capture_error("CFG import error", e, echo=True)

    def _on_run(self, _btn) -> None:
        self._start_async_run_chunked()

    def _on_pause_toggle(self, change: Dict[str, Any]) -> None:
        val = bool(change.get("value", False))
        if val:
            self._pause_flag.set()
            self._pause_btn.description = "Resume"
            self._pause_btn.icon = "play"
            self._ticker.value = "<div>Paused.</div>"
        else:
            self._pause_flag.clear()
            self._pause_btn.description = "Pause"
            self._pause_btn.icon = "pause"
            self._ticker.value = "<div>Resumed.</div>"

    def _on_stop(self, _btn) -> None:
        self._stop_flag.set()
        self._ticker.value = "<div>Stop requested.</div>"

    def _on_write_csv(self, _btn) -> None:
        self.write_summary(None)

    def _on_export_report(self, _btn) -> None:
        self._execute_export_report(bool(self._zip_outputs.value))

    def _on_filter_change(self, change: Dict[str, Any]) -> None:
        self._results_html.value = self._results_table_html(
            self._agg_results, change.get("value", "")
        )

    def _on_tail_toggle(self, change: Dict[str, Any]) -> None:
        if not change.get("value", True):
            with self._log_output:
                clear_output(wait=True)
                print("Log tail disabled.")

    def _on_preview_top(self, _btn) -> None:
        self._preview_top5_in_provis()

    def _on_viz_screenshot(self, _btn) -> None:
        self._viz_screenshot()

    # ---------- Orchestration ----------
    def _start_async_run_chunked(self) -> None:
        if self._run_thread and self._run_thread.is_alive():
            self._status.value = self._status_bar(
                "A run is already in progress.", "warn"
            )
            return
        try:
            base_md = self._build_md(dry=True)
            ligs = list(base_md.ligands)
            self._total_ligands = len(ligs)
            if self._total_ligands == 0:
                self._status.value = self._status_bar("No ligands discovered.", "warn")
                return

            # reset aggregate state
            with self._agg_lock:
                self._agg_results = []
                self._processed_ligands = 0
            self._results_html.value = self._results_table_html([])
            self._pbar.value = 0
            self._pbar.max = max(1, self._total_ligands)
            self._run_start_ts = time.time()
            self._stop_flag.clear()
            self._pause_flag.clear()
            self._last_md = base_md
            self._status.value = self._status_bar(
                "Running docking (chunked, async)…", "warn"
            )
            self._update_info_panel(base_md, running=True)

            # start log tailer if enabled
            if self._tail_logs.value:
                self._start_log_tailer_thread()

            chunk = max(1, int(self._chunk_size.value))

            def worker():
                try:
                    for start in range(0, self._total_ligands, chunk):
                        if self._stop_flag.is_set():
                            break
                        while (
                            self._pause_flag.is_set() and not self._stop_flag.is_set()
                        ):
                            time.sleep(0.2)
                        # fmt: off
                        batch = ligs[start: start + chunk]
                        # fmt: on
                        md = self._build_md(dry=False)
                        md.set_ligands(batch)
                        md.run(n_workers=None, ligands=None)
                        rows = self._normalize_results(md)
                        with self._agg_lock:
                            self._agg_results.extend(rows)
                            self._processed_ligands = min(
                                self._total_ligands,
                                self._processed_ligands + len(batch),
                            )
                except Exception as e:
                    self._capture_error("Run error", e, echo=True)

            def monitor():
                last_seen = -1
                while True:
                    if self._stop_flag.is_set():
                        break
                    with self._agg_lock:
                        done = self._processed_ligands
                        total = self._total_ligands
                        rows = list(self._agg_results)
                    if done != last_seen:
                        last_seen = done
                        self._results_html.value = self._results_table_html(
                            rows, self._filter_box.value
                        )
                        self._pbar.value = done
                        self._ticker.value = (
                            f"<div>Processed {done}/{total} ligands</div>"
                        )
                        if self._run_start_ts and done > 0:
                            elapsed = time.time() - self._run_start_ts
                            rate = elapsed / max(1, done)
                            remain = int(rate * max(0, total - done))
                            self._eta.value = f"<div class='muted'>ETA: ~{remain}s (elapsed {int(elapsed)}s)</div>"
                    if done >= total:
                        break
                    time.sleep(0.5)

                # finished or stopped
                if self._stop_flag.is_set():
                    self._status.value = self._status_bar("Stopped by user.", "warn")
                    self._pbar.bar_style = "warning"
                else:
                    self._status.value = self._status_bar("Docking finished.", "ok")
                    self._pbar.bar_style = "success"

                self._update_info_panel(self._last_md, running=False)

                # run finish callbacks
                for cb in list(self._on_finish_callbacks):
                    try:
                        cb(self)
                    except Exception as e:
                        self._capture_error("on_finish callback error", e, echo=True)

            self._run_thread = threading.Thread(target=worker, daemon=True)
            self._monitor_thread = threading.Thread(target=monitor, daemon=True)
            self._run_thread.start()
            self._monitor_thread.start()

        except Exception as e:
            self._capture_error("Setup run error", e, echo=True)

    # ---------- Log tailer ----------
    def _start_log_tailer_thread(self) -> None:
        if self._log_thread and self._log_thread.is_alive():
            return

        def tailer():
            last_path: Optional[Path] = None
            while (self._run_thread and self._run_thread.is_alive()) or (
                self._log_thread and self._log_thread is threading.current_thread()
            ):
                try:
                    log_root = (
                        Path(self._log_dir.value).expanduser()
                        if self._log_dir.value.strip()
                        else (Path(self._out_dir.value).expanduser() / "logs")
                    )
                    latest = None
                    latest_mtime = -1.0
                    if log_root.exists():
                        for p in log_root.rglob("*.log"):
                            try:
                                mt = p.stat().st_mtime
                                if mt > latest_mtime:
                                    latest, latest_mtime = p, mt
                            except Exception:
                                pass
                    if latest is None:
                        with self._log_output:
                            clear_output(wait=True)
                            print("No log files in", str(log_root))
                        time.sleep(1.0)
                        continue

                    if last_path is None or latest != last_path:
                        last_path = latest

                    data = latest.read_text(encoding="utf-8", errors="replace")
                    tail = data[-8000:] if len(data) > 8000 else data
                    with self._log_output:
                        clear_output(wait=True)
                        print(tail)
                        print("\n--- file:", str(latest))
                    time.sleep(1.0)
                except Exception:
                    time.sleep(1.0)

        self._log_thread = threading.Thread(target=tailer, daemon=True)
        self._log_thread.start()

    # ---------- ProVis preview (Top-5, robust PDB rendering) ----------
    def _preview_top5_in_provis(self) -> None:
        """
        For up to 5 unique ligands show the *first pose* from each PDBQT file.
        Pipeline: PDBQT -> SDF (obabel via `pdbqt_to_sdf`) -> first SDF record -> PDB (obabel) -> render PDB.
        """
        if ProVis is None:
            self._status.value = self._status_bar(
                "ProVis not available: cannot preview.", "warn"
            )
            return

        rows = [r for r in self.results if r.get("out_path")]
        if not rows:
            self._status.value = self._status_bar(
                "No completed poses to preview.", "warn"
            )
            return

        # unique per ligand filename, sorted by best_score ascending, take up to 5
        def skey(r):
            return 1e9 if r.get("best_score") is None else float(r["best_score"])

        seen = set()
        unique_rows = []
        for r in sorted(rows, key=skey):
            basename = Path(str(r["ligand"])).name
            if basename in seen:
                continue
            seen.add(basename)
            unique_rows.append(r)
            if len(unique_rows) >= 5:
                break

        if not unique_rows:
            self._status.value = self._status_bar("Nothing to preview.", "warn")
            return

        with self._viz_out:
            clear_output(wait=True)
            try:
                viz = ProVis(vw=self._vw, vh=self._vh)

                # Load receptor (convert PDBQT -> PDB if necessary)
                rec_path = self._receptor_path.value.strip()
                if rec_path and Path(rec_path).exists():
                    rec_text, rec_fmt = self._receptor_pdb_text(rec_path)
                    if rec_text:
                        viz.load_receptor_from_text(rec_text, fmt=rec_fmt)
                    else:
                        # fallback: try as-is
                        viz.load_receptor(rec_path)
                # style receptor & background up-front
                viz.set_receptor_style("cartoon", "white")
                viz.set_background(self._bg_hex_for_provis())

                palette = ["#00ffff", "#ff9f1c", "#2ec4b6", "#e71d36", "#9b5de5"]
                tmp_dir = self._ensure_preview_tmp()

                for i, r in enumerate(unique_rows, 1):
                    outp = Path(r["out_path"])
                    if not outp.exists():
                        print(f"[warn] out file missing: {outp}")
                        continue

                    # PDBQT -> SDF
                    try:
                        sdf_tmp = tmp_dir / (outp.stem + "_all.sdf")
                        pdbqt_to_sdf(outp, sdf_tmp, backend="obabel", extra_args=None)
                    except Exception as conv_exc:
                        print(
                            f"[warn] pdbqt->sdf conversion failed for {outp.name}: {conv_exc}"
                        )
                        continue
                    if not sdf_tmp.exists():
                        print(f"[warn] conversion did not produce SDF for {outp.name}")
                        continue

                    sdf_text = sdf_tmp.read_text(encoding="utf-8", errors="replace")
                    sdf_first = self._sdf_first_record(sdf_text)

                    # First SDF record -> PDB (via obabel) and render PDB (most robust for py3Dmol)
                    try:
                        pdb_text = self._sdf_to_pdb_via_obabel(sdf_first)
                        if not pdb_text:
                            print(f"[warn] SDF->PDB produced empty for {outp.name}")
                            continue
                        name = f"{i}. {Path(r['ligand']).name} (best={r.get('best_score')})"
                        viz.load_ligand_from_text(pdb_text, name=name, fmt="pdb")
                        viz.highlight_ligand(
                            style="stick",
                            color=self._css_hex_to_threejs(
                                palette[(i - 1) % len(palette)]
                            ),
                            radius=0.28,
                        )
                    except Exception as e_pdb:
                        print(
                            f"[warn] preview conversion failed for {outp.name}: {e_pdb}"
                        )

                viz.show()
            except Exception as e:
                print("Preview error:", e)
                traceback.print_exc()

    def _sdf_to_pdb_via_obabel(self, sdf_text: str) -> Optional[str]:
        """
        Convert a single SDF record string to PDB text using obabel.
        Returns PDB text on success, None on failure.
        """
        try:
            with tempfile.NamedTemporaryFile("w+", suffix=".sdf", delete=False) as inf:
                inf.write(sdf_text)
                inf.flush()
                in_path = Path(inf.name)
            out_fd, out_path = tempfile.mkstemp(suffix=".pdb")
            Path(out_path).unlink(missing_ok=True)
            cmd = ["obabel", str(in_path), "-O", out_path, "--gen3d"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            in_path.unlink(missing_ok=True)
            if res.returncode != 0:
                raise RuntimeError(
                    f"obabel failed: {res.returncode} stderr={res.stderr.strip()}"
                )
            pdb_text = Path(out_path).read_text(encoding="utf-8", errors="replace")
            try:
                Path(out_path).unlink(missing_ok=True)
            except Exception:
                pass
            return pdb_text
        except Exception as e:
            with self._log_output:
                clear_output(wait=True)
                print("SDF->PDB conversion error (obabel):", e)
            return None

    def _receptor_pdb_text(self, rec_path: str) -> Tuple[Optional[str], str]:
        """
        :returns: (text, fmt) where fmt is 'pdb'. If conversion fails, (None, 'pdb').
        """
        p = Path(rec_path)
        if p.suffix.lower() == ".pdb":
            try:
                return p.read_text(encoding="utf-8", errors="replace"), "pdb"
            except Exception:
                return None, "pdb"
        if p.suffix.lower() == ".pdbqt":
            # Convert receptor PDBQT -> PDB for robust py3Dmol rendering
            try:
                out_fd, out_path = tempfile.mkstemp(suffix=".pdb")
                Path(out_path).unlink(missing_ok=True)
                cmd = ["obabel", str(p), "-O", out_path]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
                if res.returncode != 0:
                    raise RuntimeError(
                        f"obabel receptor conversion failed: {res.stderr.strip()}"
                    )
                text = Path(out_path).read_text(encoding="utf-8", errors="replace")
                try:
                    Path(out_path).unlink(missing_ok=True)
                except Exception:
                    pass
                return text, "pdb"
            except Exception as e:
                with self._log_output:
                    clear_output(wait=True)
                    print("Receptor PDBQT->PDB conversion error (obabel):", e)
                return None, "pdb"
        # try as PDB by default
        try:
            return p.read_text(encoding="utf-8", errors="replace"), "pdb"
        except Exception:
            return None, "pdb"

    def _viz_screenshot(self) -> None:
        # Client-side capture of the largest canvas on the page
        js = """
        (function(){
          try{
            const canvases = Array.from(document.querySelectorAll('canvas')).filter(c=>c.width>0 && c.height>0);
            if(!canvases.length){ alert('No canvas found'); return; }
            let best = canvases[0], bestArea = best.width*best.height;
            for(const c of canvases){ const a = c.width*c.height; if(a>bestArea){ best = c; bestArea=a; } }
            const data = best.toDataURL('image/png');
            const a = document.createElement('a'); a.href = data; a.download = 'dock_preview.png';
            document.body.appendChild(a); a.click(); a.remove();
          }catch(e){ console.error(e); alert('Capture failed: '+e); }
        })();
        """
        display(Javascript(js))
        self._status.value = self._status_bar(
            "Viewer screenshot requested (browser will download).", "ok"
        )

    # ---------- Export / Report ----------
    def _execute_export_report(self, zip_outputs: bool) -> None:
        try:
            if not self._last_md:
                self._status.value = self._status_bar(
                    "Nothing to export — no run yet.", "warn"
                )
                return
            csv_path = self._last_md.write_summary(None)
            rows = self.results
            report_html = self._render_report_html(self._last_md, rows)
            report_path = Path(self._last_md.out_dir) / "report.html"
            report_path.write_text(report_html, encoding="utf-8")

            bundle_msg = ""
            if zip_outputs:
                out_dir = Path(self._last_md.out_dir)
                log_dir = Path(self._last_md.log_dir)
                zip_path = out_dir / "bundle.zip"
                with zipfile.ZipFile(
                    zip_path, "w", compression=zipfile.ZIP_DEFLATED
                ) as zf:
                    zf.write(csv_path, arcname=csv_path.name)
                    zf.write(report_path, arcname=report_path.name)
                    for p in out_dir.rglob("*"):
                        if p.is_file():
                            try:
                                arc = str(p.relative_to(out_dir))
                            except Exception:
                                arc = p.name
                            zf.write(p, arcname=arc)
                    if log_dir.exists():
                        for p in log_dir.rglob("*"):
                            if p.is_file():
                                try:
                                    arc = (
                                        str(Path("logs") / p.name)
                                        if out_dir not in p.parents
                                        else str(p.relative_to(out_dir))
                                    )
                                except Exception:
                                    arc = p.name
                                zf.write(p, arcname=arc)
                bundle_msg = f" & bundle → {zip_path}"

            self._status.value = self._status_bar(
                f"Report → {report_path} & CSV → {csv_path}{bundle_msg}", "ok"
            )
            with self._log_output:
                clear_output(wait=True)
                print("Report written:", report_path)
            self._ticker.value = f"<div>Report: {report_path}</div>"
        except Exception as e:
            self._capture_error("Export report error", e, echo=True)

    # ---------- Build MultipleDock from GUI state ----------
    def _build_md(self, dry: bool) -> MultipleDock:
        receptor = Path(self._receptor_path.value).expanduser()
        lig_dir = (
            Path(self._ligand_dir.value).expanduser()
            if self._ligand_dir.value.strip()
            else None
        )
        lig_json = self._ligand_list_json.value.strip()
        ligs = None
        if lig_json:
            arr = json.loads(lig_json)
            if not isinstance(arr, list):
                raise ValueError("Ligands JSON must be a list")
            ligs = [str(Path(str(p)).expanduser()) for p in arr]

        backend_key = self._backend.value
        backend_spec = backend_key
        if backend_key == "binary":
            exe = self._custom_backend.value.strip()
            if not exe:
                raise ValueError("Provide custom binary path/name")
            backend_spec = exe

        use_autobox = bool(self._use_autobox.value)
        autobox_ref = (
            Path(self._autobox_ref.value).expanduser()
            if self._autobox_ref.value.strip()
            else None
        )
        if use_autobox and backend_key == "vina":
            raise RuntimeError("Autobox unsupported by Vina API")

        center = (
            float(self._center_x.value),
            float(self._center_y.value),
            float(self._center_z.value),
        )
        size = (
            float(self._size_x.value),
            float(self._size_y.value),
            float(self._size_z.value),
        )

        out_dir = Path(self._out_dir.value).expanduser()
        log_dir = (
            Path(self._log_dir.value).expanduser()
            if self._log_dir.value.strip()
            else None
        )
        timeout = (
            float(self._timeout.value) if float(self._timeout.value or 0) > 0 else None
        )
        seed = int(self._seed.value) if int(self._seed.value or 0) > 0 else None

        md = MultipleDock(
            receptor=receptor,
            ligand_dir=lig_dir,
            ligands=ligs,
            backend=backend_spec,
            ligand_format=self._lig_format.value,
            filter_pattern=self._filter_pat.value or "*",
            center=None if use_autobox else center,
            size=None if use_autobox else size,
            autobox=use_autobox,
            autobox_ref=autobox_ref,
            autobox_padding=float(self._autobox_pad.value),
            exhaustiveness=int(self._exhaust.value),
            n_poses=int(self._n_poses.value),
            cpu=int(self._cpu.value) if int(self._cpu.value or 0) > 0 else None,
            seed=seed,
            out_dir=out_dir,
            log_dir=log_dir,
            n_workers=int(self._workers.value),
            skip_existing=bool(self._skip_existing.value),
            max_retries=int(self._max_retries.value),
            retry_backoff=float(self._backoff.value),
            timeout=timeout,
            verbose=int(self._verbose.value),
            cache_per_worker=True,
            autorun=False,
            autowrite=False,
        )
        return md

    # ---------- Visual helpers ----------
    def _apply_theme(self, key: str) -> None:
        t = self.THEMES.get(key, self.THEMES["light"])
        if self._ui:
            self._ui.layout = widgets.Layout(
                padding="6px", border="1px solid transparent", background_color=t["bg"]
            )
        css = (
            '<style id="dock-gui-theme">\n'
            "  .provis-root .provis-card {\n"
            "    background: " + t["card_bg"] + " !important;\n"
            "    border: 1px solid " + t["card_border"] + " !important;\n"
            "    border-radius: 10px;\n"
            "    box-shadow: 0 6px 18px rgba(0,0,0,0.06);\n"
            "  }\n"
            "  .provis-root table.docktbl {\n"
            "    width: 100%;\n"
            "    border-collapse: collapse;\n"
            "    border: 1px solid " + t["table_border"] + ";\n"
            "  }\n"
            "  .provis-root table.docktbl th {\n"
            "    background: " + t["table_head_bg"] + ";\n"
            "    text-align: left;\n"
            "    padding: 6px;\n"
            "    border-bottom: 1px solid " + t["table_border"] + ";\n"
            "  }\n"
            "  .provis-root table.docktbl td {\n"
            "    padding: 6px;\n"
            "    border-bottom: 1px solid " + t["table_border"] + ";\n"
            "    font-family: ui-monospace;\n"
            "    font-size: 12px;\n"
            "  }\n"
            "  .provis-root .muted { color: " + t["mute"] + "; }\n"
            "  .provis-root a { color: " + t["link"] + "; text-decoration: none; }\n"
            "</style>\n"
        )

        display(HTML(css))

    def _apply_layout(self, compact: bool) -> None:
        if not (self._root and self._left_col and self._right_col):
            return
        if compact:
            self._left_col.layout = widgets.Layout(width="100%", padding="6px")
            self._right_col.layout = widgets.Layout(width="100%", padding="6px")
            self._root.children = (widgets.VBox([self._left_col, self._right_col]),)
        else:
            self._left_col.layout = widgets.Layout(width="42%", padding="6px")
            self._right_col.layout = widgets.Layout(width="58%", padding="6px")
            self._root.children = (self._left_col, self._right_col)

    def _card(self, title: str, body: widgets.Widget) -> widgets.VBox:
        head = widgets.HTML(f"<h3>{title}</h3>")
        box = widgets.VBox(
            [head, body], layout=widgets.Layout(padding="10px 12px", margin="8px 0")
        )
        box.add_class("provis-card")
        return box

    def _status_bar(self, text: str, level: str = "ok") -> str:
        col = {
            "ok": self.THEMES[self._theme.value]["status_ok"],
            "warn": self.THEMES[self._theme.value]["status_warn"],
            "err": self.THEMES[self._theme.value]["status_err"],
        }.get(level, "#999")
        return f"<div style='padding:8px;border-left:4px solid {col};'>{html.escape(text)}</div>"

    def _info_block(self, inner: str) -> str:
        return f"<div style='padding:10px;border:1px dashed #d1d5db;border-radius:6px;margin-top:8px;'>{inner}</div>"

    def _update_info_panel(self, md: Optional[MultipleDock], *, running: bool) -> None:
        if not md:
            self._info_panel.value = self._info_block("<i>No run.</i>")
            return
        bdesc = self._describe_backend_for_md()
        out = html.escape(str(md.out_dir))
        log = html.escape(str(md.log_dir))
        bits = [
            f"<b>Backend:</b> {bdesc}",
            f"<b>Workers:</b> {md._n_workers}",
            f"<b>CPU/thread:</b> {md._cpu or '-'}",
            f"<b>Exhaustiveness:</b> {md._exhaustiveness}",
            f"<b>Num poses:</b> {md._num_modes}",
            f"<b>Out:</b> {out}",
            f"<b>Logs:</b> {log}",
        ]
        if md._use_autobox:
            bits.append(
                f"<b>Box:</b> autobox ref={html.escape(str(md._autobox_ref))}, pad={md._autobox_padding}"
            )
        else:
            bits.append(f"<b>Box:</b> center={md._box_center}, size={md._box_size}")
        if running:
            bits.append("<i>Running…</i>")
        elif self._run_start_ts:
            dur = int(time.time() - self._run_start_ts)
            bits.append(f"<b>Duration:</b> {dur}s")
        self._info_panel.value = self._info_block("<br>".join(bits))

    def _results_table_html(self, rows: List[Dict[str, Any]], needle: str = "") -> str:
        """Make the results table HTML."""
        if not rows:
            return "<div class='muted'>No results yet.</div>"
        needle_l = (needle or "").strip().lower()
        filt: List[Dict[str, Any]] = []
        for r in rows:
            if needle_l:
                hay = f"{r.get('ligand','')} {r.get('status','')} {r.get('best_score','')}".lower()
                if needle_l not in hay:
                    continue
            filt.append(r)

        def skey(r):
            bs = r.get("best_score", None)
            return 1e9 if bs is None else float(bs)

        filt = sorted(filt, key=skey)
        head = (
            "<table class='docktbl'><thead><tr>"
            "<th>#</th><th>ligand</th><th>best</th><th>status</th>"
            "<th>tries</th><th>out</th><th>log</th><th>err</th>"
            "</tr></thead><tbody>"
        )
        body = []
        for i, r in enumerate(filt, 1):
            lig = html.escape(Path(str(r.get("ligand", ""))).name)
            best = r.get("best_score")
            best_s = "" if best is None else f"{best:.3f}"
            st = html.escape(str(r.get("status", "")))
            body.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td title='{html.escape(str(r.get('ligand','')))}'>{lig}</td>"
                f"<td>{best_s}</td>"
                f"<td>{st}</td>"
                f"<td>{int(r.get('attempts',0))}</td>"
                f"<td class='muted'>{html.escape(str(r.get('out_path','')))}</td>"
                f"<td class='muted'>{html.escape(str(r.get('log_path','')))}</td>"
                f"<td class='muted'>{html.escape(str(r.get('error','')))}</td>"
                "</tr>"
            )
        return head + "".join(body) + "</tbody></table>"

    # ---------- Utils / parsing ----------
    def _inject_shortcuts(self) -> None:
        js = """
        (function(){
          if (window.__dockgui_shortcuts__) return;
          window.__dockgui_shortcuts__ = true;
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
              if (e.altKey && e.key && e.key.toLowerCase() === 'p'){
                const btn = Array.from(document.querySelectorAll('button[title="pause-shortcut"]'))[0];
                if (btn){ btn.click(); e.preventDefault(); }
              }
            }catch(err){ console.warn('Shortcut error', err); }
          }, true);
        })();
        """
        display(Javascript(js))

    @staticmethod
    def _parse_vina_cfg(text: str) -> Dict[str, float]:
        vals: Dict[str, float] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                k, v = parts[0], parts[1]
            k = k.strip()
            try:
                vals[k] = float(v)
            except Exception:
                pass
        for k in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"):
            if k not in vals:
                raise ValueError(f"Missing '{k}' in cfg.")
        return vals

    def _normalize_results(self, md: MultipleDock) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in md.results:
            out.append(
                {
                    "ligand": str(getattr(r, "ligand_path", "")),
                    "out_path": str(getattr(r, "out_path", "") or ""),
                    "log_path": str(getattr(r, "log_path", "") or ""),
                    "best_score": getattr(r, "best_score", None),
                    "status": getattr(r, "status", ""),
                    "error": getattr(r, "error", None) or "",
                    "attempts": getattr(r, "attempts", 0),
                }
            )
        return out

    def _describe_backend_for_md(self) -> str:
        key = self._backend.value
        if key == "binary":
            exe = self._custom_backend.value.strip()
            return f"binary:{exe}" if exe else "binary:<none>"
        return key

    def _bg_hex_for_provis(self) -> str:
        hexc = self.THEMES[self._theme.value]["bg"].lstrip("#")
        if len(hexc) == 3:
            hexc = "".join(ch * 2 for ch in hexc)
        return "0x" + hexc

    def _css_hex_to_threejs(self, hex_color: str) -> str:
        h = hex_color.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return f"0x{h}"

    def _capture_error(self, msg: str, e: Exception, echo: bool = False) -> None:
        tb = traceback.format_exc()
        self._last_error = f"{msg}: {e}\n{tb}"
        self._status.value = self._status_bar(f"{msg}: {e}", "err")
        with self._log_output:
            clear_output(wait=True)
            print(self._last_error)
        if echo:
            print(self._last_error)

    # --- SDF helpers for preview ---
    def _ensure_preview_tmp(self) -> Path:
        """Ensure a temp subdir under out_dir for preview conversions."""
        base = (
            Path(self._out_dir.value).expanduser()
            if self._out_dir.value.strip()
            else Path.cwd() / "docked"
        )
        tmp = base / "_preview_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

    @staticmethod
    def _sdf_first_record(text: str) -> str:
        """Return only the first SDF record (up to the first '$$$$')."""
        if "$$$$" not in text:
            return text
        idx = text.find("\n$$$$")
        if idx == -1:
            idx = text.find("$$$$")
            if idx == -1:
                return text
            return text[: idx + 4]
        return text[: idx + len("\n$$$$")]

    # --- Report HTML ---
    def _render_report_html(self, md: MultipleDock, rows: List[Dict[str, Any]]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = "Docking Report"
        meta = [
            f"<b>Generated:</b> {now}",
            f"<b>Receptor:</b> {html.escape(str(md.receptor))}",
            f"<b>Backend:</b> {html.escape(self._describe_backend_for_md())}",
            f"<b>Ligands:</b> {len(md.ligands)}",
            f"<b>Out:</b> {html.escape(str(md.out_dir))}",
            f"<b>Logs:</b> {html.escape(str(md.log_dir))}",
        ]
        if md._use_autobox:
            meta.append(
                f"<b>Autobox:</b> ref={html.escape(str(md._autobox_ref))}, pad={md._autobox_padding}"
            )
        else:
            meta.append(f"<b>Box:</b> center={md._box_center}, size={md._box_size}")
        tbl = self._results_table_html(rows)
        css = (
            "<style>\n"
            + "  body { font-family: system-ui; padding: 18px; }\n"
            + "  table { width: 100%; border-collapse: collapse; border: 1px solid #e5e7eb; }\n"
            + "  th { background: #f3f4f6; padding: 6px; }\n"
            + "</style>"
        )
        return (
            f"<!doctype html><html><head><meta charset='utf-8'><title>{title}"
            + f"</title>{css}</head><body><h1>{title}</h1><div>{' • '.join(meta)}"
            + f"</div>{tbl}</body></html>"
        )

    # --- Dunders / Properties ---
    def __repr__(self) -> str:
        running = bool(self._run_thread and self._run_thread.is_alive())
        return f"<DockGUI backend={self._backend.value} workers={self._workers.value} running={running}>"

    def __len__(self) -> int:
        with self._agg_lock:
            return len(self._agg_results)

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Normalized results for external consumption."""
        with self._agg_lock:
            return list(self._agg_results)

    @property
    def is_running(self) -> bool:
        """Whether a run is active."""
        return bool(self._run_thread and self._run_thread.is_alive())

    @property
    def progress(self) -> float:
        """Progress in [0,1]."""
        with self._agg_lock:
            total = self._total_ligands or 0
            done = self._processed_ligands or 0
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, done / float(total)))

    @property
    def summary_path(self) -> Optional[Path]:
        """Last written CSV path (if any)."""
        return self._last_md.summary_path if self._last_md else None

    @property
    def last_error(self) -> Optional[str]:
        """Last captured exception with traceback, if any."""
        return self._last_error
