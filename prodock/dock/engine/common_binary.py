# prodock/dock/engine/common_binary.py
from __future__ import annotations

import shlex
import shutil
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, List

from .base import DockBackend, Vec3, PathLike


def _p(x: PathLike) -> Path:
    """
    Normalize a PathLike to pathlib.Path.
    """
    return x if isinstance(x, Path) else Path(x)


def _ensure_parent(path: Optional[Path]) -> None:
    """
    Ensure the parent directory of `path` exists (no-op if path is None).
    """
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)


class BaseBinaryEngine(DockBackend):
    """
    Shared subprocess runner for Vina-like command-line engines (smina, gnina, qvina, vina CLI, etc).

    This class focuses on building a safe, debuggable command line and running it with
    :func:`subprocess.run`. It intentionally does not parse scores or interpret engine
    outputs — that is left to your postprocessing pipeline.

    Resolution strategy for the executable:
      1. If ``exe_name`` is an existing executable file (absolute or relative), use it.
      2. Try ``shutil.which(exe_name)`` to locate executable on PATH.
      3. Try repository-local candidates such as ``prodock/dock/binary/<exe>`` and fallbacks
         like ``prodock/dock/bin/<exe>``.
      4. Raise ``FileNotFoundError`` with actionable guidance.

    Subclasses should override ``exe_name`` and may tweak ``flag_map`` or ``supports_autobox``.

    Example
    -------
    .. code-block:: python

        eng = SminaEngine().set_receptor("rec.pdbqt").set_ligand("lig.pdbqt")
        eng.set_box((10,10,10), (20,20,20)).set_out("out.pdbqt").set_log("out.log")
        eng.run(exhaustiveness=8, n_poses=9)
    """

    # default binary name; subclasses override (e.g., "smina", "qvina")
    exe_name: str = "smina"
    # whether this engine supports autobox flags (smina/gnina do; qvina/qvina-w do not)
    supports_autobox: bool = False

    # Default CLI flag map. Engines that don't support autobox should remove those keys.
    flag_map: Dict[str, str] = {
        "receptor": "--receptor",
        "ligand": "--ligand",
        "center_x": "--center_x",
        "center_y": "--center_y",
        "center_z": "--center_z",
        "size_x": "--size_x",
        "size_y": "--size_y",
        "size_z": "--size_z",
        "exhaustiveness": "--exhaustiveness",
        "num_modes": "--num_modes",
        "cpu": "--cpu",
        "seed": "--seed",
        "out": "--out",
        "log": "--log",
        # autobox flags (present by default)
        "autobox_ligand": "--autobox_ligand",
        "autobox_add": "--autobox_add",
    }

    def __init__(self) -> None:
        # configuration (chainable setters will populate these)
        self._receptor: Optional[Path] = None
        self._ligand: Optional[Path] = None
        self._center: Optional[Vec3] = None
        self._size: Optional[Vec3] = None
        self._exhaustiveness: Optional[int] = None
        self._num_modes: Optional[int] = None
        self._cpu: Optional[int] = None
        self._seed: Optional[int] = None
        self._out: Optional[Path] = None
        self._log: Optional[Path] = None
        self._autobox_ref: Optional[Path] = None
        self._autobox_pad: Optional[float] = None

        # runtime/debug
        self._last_called: Optional[str] = None

    # ----- chainable setters -----
    def set_receptor(
        self, receptor_path: PathLike, *, validate: bool = False
    ) -> "BaseBinaryEngine":
        """
        Set receptor file.

        :param receptor_path: Path to receptor file (PDBQT).
        :param validate: If True, check that file exists immediately.
        """
        p = _p(receptor_path)
        if validate and not p.is_file():
            raise FileNotFoundError(p)
        self._receptor = p
        return self

    def set_ligand(self, ligand_path: PathLike) -> "BaseBinaryEngine":
        """
        Set ligand file.

        :param ligand_path: Path to ligand file (PDBQT/SDF).
        """
        self._ligand = _p(ligand_path)
        return self

    def set_box(self, center: Vec3, size: Vec3) -> "BaseBinaryEngine":
        """
        Set the docking box.

        :param center: (x, y, z) center coordinates.
        :param size: (sx, sy, sz) box sizes.
        """
        self._center = center
        self._size = size
        return self

    def enable_autobox(
        self, reference_file: PathLike, padding: Optional[float] = None
    ) -> "BaseBinaryEngine":
        """
        Enable autoboxing (only if supported by the engine).

        :param reference_file: Path used by the engine to compute the box (e.g., ligand).
        :param padding: Optional padding to add around inferred box.
        :raises RuntimeError: if the engine does not support autoboxing.
        """
        if not getattr(self, "supports_autobox", False):
            raise RuntimeError(
                f"{self.__class__.__name__} does not support autoboxing."
            )
        self._autobox_ref = _p(reference_file)
        self._autobox_pad = padding
        return self

    def set_exhaustiveness(self, value: Optional[int]) -> "BaseBinaryEngine":
        """Set exhaustiveness (engine dependent)."""
        self._exhaustiveness = value
        return self

    def set_num_modes(self, value: Optional[int]) -> "BaseBinaryEngine":
        """Set number of poses/modes to write."""
        self._num_modes = value
        return self

    def set_cpu(self, value: Optional[int]) -> "BaseBinaryEngine":
        """Set CPU thread count for the engine (if supported)."""
        self._cpu = value
        return self

    def set_seed(self, value: Optional[int]) -> "BaseBinaryEngine":
        """Set RNG seed for reproducibility."""
        self._seed = value
        return self

    def set_out(self, out_path: PathLike) -> "BaseBinaryEngine":
        """Set output pose file path (engine will write to this path)."""
        self._out = _p(out_path)
        return self

    def set_log(self, log_path: PathLike) -> "BaseBinaryEngine":
        """Set engine log file path (engine will write logs here)."""
        self._log = _p(log_path)
        return self

    def set_executable(self, path: PathLike) -> "BaseBinaryEngine":
        """
        Explicitly set the binary executable to use.

        :param path: Full or relative path to the engine binary.
        :returns: self (chainable)

        Example
        -------
        >>> eng = SminaEngine().set_executable("/home/user/bin/smina")
        """
        self.exe_name = str(path)
        return self

    # ----- executable resolution -----
    def _resolve_executable(self) -> str:
        """
        Resolve the real executable path to run.

        Resolution order:
          1. If exe_name is an existing executable file (absolute or relative), use it.
          2. Use shutil.which(exe_name) to find binary on PATH.
          3. Try repo-local bundled candidates:
             - prodock/dock/binary/<exe>
             - prodock/dock/bin/<exe>
             - parent-level bin fallbacks
          4. Raise FileNotFoundError with actionable guidance.

        :returns: absolute path to executable
        :raises FileNotFoundError: if not found.
        """
        exe = str(self.exe_name)
        p = Path(exe)

        # 1) explicit path and executable check
        try:
            if p.exists() and p.is_file() and os.access(str(p), os.X_OK):
                return str(p.resolve())
        except Exception:
            # ignore errors and continue to other resolution methods
            pass

        # 2) PATH lookup
        found = shutil.which(exe)
        if found:
            return found

        # 3) package-local candidates
        base = Path(__file__).resolve()
        # We search reasonable repo-local locations (project-specific)
        candidates: List[Path] = [
            base.parents[1]
            / "binary"
            / exe,  # prodock/dock/binary/<exe> (repo-bundled)
            base.parents[1] / "bin" / exe,  # prodock/dock/bin/<exe>
            base.parents[2] / "bin" / exe,  # higher-level bin/<exe>
            base.parents[3] / "bin" / exe,  # another fallback
        ]
        for cand in candidates:
            try:
                if cand.exists() and cand.is_file() and os.access(str(cand), os.X_OK):
                    return str(cand.resolve())
            except Exception:
                # ignore permission/IO errors and continue
                pass

        # 4) helpful error
        raise FileNotFoundError(
            f"Could not locate executable for '{exe}'. "
            "Please either install the binary in your PATH or call "
            "`set_executable('/full/path/to/<binary>')` on the engine instance, "
            "or place an executable at prodock/dock/binary/<binary>. "
            f"Searched PATH and candidates: {[str(c) for c in candidates]}"
        )

    # ----- build and run -----
    def _build_cmd(
        self, override_exhaustiveness: Optional[int], override_nposes: Optional[int]
    ) -> List[str]:
        """
        Build the CLI argument list for the engine.

        This method consults ``flag_map`` to determine which flags to include and performs
        defensive checks so flags are only added if present in the map.
        """
        f = self.flag_map
        cmd: List[str] = [self.exe_name]

        if self._receptor and "receptor" in f:
            cmd += [f["receptor"], str(self._receptor)]

        if self._ligand and "ligand" in f:
            cmd += [f["ligand"], str(self._ligand)]

        if self._center and all(k in f for k in ("center_x", "center_y", "center_z")):
            cx, cy, cz = self._center
            cmd += [
                f["center_x"],
                str(cx),
                f["center_y"],
                str(cy),
                f["center_z"],
                str(cz),
            ]

        if self._size and all(k in f for k in ("size_x", "size_y", "size_z")):
            sx, sy, sz = self._size
            cmd += [f["size_x"], str(sx), f["size_y"], str(sy), f["size_z"], str(sz)]

        ex = (
            override_exhaustiveness
            if override_exhaustiveness is not None
            else self._exhaustiveness
        )
        if ex is not None and "exhaustiveness" in f:
            cmd += [f["exhaustiveness"], str(ex)]

        nm = override_nposes if override_nposes is not None else self._num_modes
        if nm is not None and "num_modes" in f:
            cmd += [f["num_modes"], str(nm)]

        if self._cpu is not None and "cpu" in f:
            cmd += [f["cpu"], str(self._cpu)]

        if self._seed is not None and "seed" in f:
            cmd += [f["seed"], str(self._seed)]

        if self._out is not None and "out" in f:
            cmd += [f["out"], str(self._out)]

        if self._log is not None and "log" in f:
            cmd += [f["log"], str(self._log)]

        if self._autobox_ref is not None and "autobox_ligand" in f:
            cmd += [f["autobox_ligand"], str(self._autobox_ref)]
            if self._autobox_pad is not None and "autobox_add" in f:
                cmd += [f["autobox_add"], str(self._autobox_pad)]

        return cmd

    def run(
        self, *, exhaustiveness: Optional[int] = None, n_poses: Optional[int] = None
    ) -> "BaseBinaryEngine":
        """
        Execute the docking binary.

        :param exhaustiveness: Optional override for exhaustiveness.
        :param n_poses: Optional override for number of poses.
        :returns: self
        :raises FileNotFoundError: if executable cannot be resolved.
        :raises subprocess.CalledProcessError: if process returns non-zero exit code.
        """
        # Build the command list (first element is placeholder exe_name)
        cmd = self._build_cmd(exhaustiveness, n_poses)

        # Resolve executable path and replace placeholder
        resolved_exe = self._resolve_executable()
        if cmd:
            cmd[0] = resolved_exe
        else:
            cmd = [resolved_exe]

        # ensure parents so the engine can write outputs/logs
        _ensure_parent(self._out)
        _ensure_parent(self._log)

        # record exact command (shell-quoted) for debug/repro
        self._last_called = " ".join(shlex.quote(x) for x in cmd)

        # Run the engine. Let exceptions bubble up to callers (BatchDock or user code).
        subprocess.run(cmd, check=True)
        return self

    @property
    def called(self) -> Optional[str]:
        """
        Last executed command (shell-quoted). Useful for debugging or reproducing a run.
        """
        return self._last_called
