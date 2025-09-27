"""Vina Python-binding engine adapter.

This wrapper uses your existing `VinaDock` class (prodock.dock.vina.VinaDock)
but delays importing it until the engine is actually instantiated/used so
module import of this adapter never fails if the underlying binding or
dependencies are missing.

If the underlying `VinaDock` is unavailable, a clear ImportError is raised
only when an operation requiring it is performed (e.g., run, set_receptor).

:Example:

>>> from prodock.dock.engine.vina_binding import VinaBindingEngine
>>> eng = VinaBindingEngine(cpu=2, seed=42)
>>> eng.set_receptor("rec.pdbqt")  # will raise ImportError if VinaDock can't be imported
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from .base import DockBackend, Vec3, PathLike


class VinaBindingEngine(DockBackend):
    """
    Adapter around the project's VinaDock (Python binding wrapper).

    Import of the upstream wrapper is deferred to runtime to avoid import-time
    crashes when the `vina` package or other dependencies are missing.

    :param sf_name: scoring function name passed to VinaDock (default "vina").
    :param cpu: number of threads passed to VinaDock (default 1).
    :param seed: optional random seed.
    :param verbosity: verbosity level passed to VinaDock (default 1).

    :raises ImportError: only raised when you attempt to use the binding but the
                         upstream `VinaDock` class cannot be imported.
    """

    def __init__(
        self,
        sf_name: str = "vina",
        cpu: int = 1,
        seed: Optional[int] = None,
        verbosity: int = 1,
    ) -> None:
        # store constructor args; instantiate underlying VinaDock lazily
        self._sf_name = sf_name
        self._cpu = cpu
        self._seed = seed
        self._verbosity = verbosity

        # lazy pointers
        self._VinaDock_cls = None
        self._Vina_import_error = None
        self._vina_instance = None

        # staged IO so config calls before instantiation are possible
        self._receptor = None
        self._ligand = None
        self._center = None
        self._size = None
        self._out: Optional[Path] = None
        self._log: Optional[Path] = None

        # import attempt is deferred; do not import at module-import time.

    # ---------- lazy helpers ----------
    def _ensure_vinadock_class(self):
        """Attempt to import the project's VinaDock wrapper if not already loaded."""
        if self._VinaDock_cls is not None:
            return
        try:
            # Import your existing wrapper (the one you uploaded as /mnt/data/vina.py)
            from prodock.dock.vina import VinaDock as _VinaDock  # type: ignore

            self._VinaDock_cls = _VinaDock
        except Exception as e:
            # store error but do not raise at import-time
            self._Vina_import_error = e
            self._VinaDock_cls = None

    def _ensure_instance(self):
        """
        Create an instance of VinaDock if available. If not available raise
        a helpful ImportError referencing the stored import error.
        """
        self._ensure_vinadock_class()
        if self._VinaDock_cls is None:
            raise ImportError(
                "VinaBindingEngine requires prodock.dock.vina.VinaDock but it could not be imported. "
                f"Original import error: {self._Vina_import_error}"
            )
        if self._vina_instance is None:
            # instantiate with stored constructor args
            self._vina_instance = self._VinaDock_cls(
                sf_name=self._sf_name,
                cpu=self._cpu,
                seed=self._seed,
                verbosity=self._verbosity,
            )

            # apply any staged configuration that was provided prior to instantiation
            if self._receptor is not None:
                # VinaDock.set_receptor validates path existence — we keep that behavior
                self._vina_instance.set_receptor(self._receptor)
            if self._center is not None and self._size is not None:
                # VinaDock uses define_box / set_box naming — try both commonly used names
                # prefer define_box if exists
                if hasattr(self._vina_instance, "define_box"):
                    self._vina_instance.define_box(self._center, self._size)
                else:
                    # fallback to set_box if wrapper uses that name
                    self._vina_instance.set_box(self._center, self._size)
            if self._ligand is not None:
                self._vina_instance.set_ligand(self._ligand)

    # ---------- DockBackend API (chainable) ----------
    def set_receptor(
        self, receptor_path: PathLike, *, validate: bool = False
    ) -> "VinaBindingEngine":
        """Set receptor path (chainable)."""
        self._receptor = receptor_path
        if self._vina_instance is not None:
            self._vina_instance.set_receptor(receptor_path, validate=validate)
        return self

    def set_ligand(self, ligand_path: PathLike) -> "VinaBindingEngine":
        """Set ligand path (chainable)."""
        self._ligand = ligand_path
        if self._vina_instance is not None:
            self._vina_instance.set_ligand(ligand_path)
        return self

    def set_box(self, center: Vec3, size: Vec3) -> "VinaBindingEngine":
        """Set docking box (chainable)."""
        self._center = center
        self._size = size
        if self._vina_instance is not None:
            if hasattr(self._vina_instance, "define_box"):
                self._vina_instance.define_box(center, size)
            else:
                self._vina_instance.set_box(center, size)
        return self

    def enable_autobox(
        self, reference_file: PathLike, padding: Optional[float] = None
    ) -> "VinaBindingEngine":
        """Vina python binding does not support autoboxing; raise immediately."""
        raise RuntimeError(
            "VinaBindingEngine (python binding) does not support autobox; provide explicit box."
        )

    def set_exhaustiveness(self, value: Optional[int]) -> "VinaBindingEngine":
        """Set exhaustiveness."""
        # store and forward to instance if present
        self._exhaustiveness = value
        if self._vina_instance is not None:
            self._vina_instance.set_exhaustiveness(value)
        return self

    def set_num_modes(self, value: Optional[int]) -> "VinaBindingEngine":
        """Set number of modes/poses."""
        self._num_modes = value
        if self._vina_instance is not None:
            self._vina_instance.set_num_modes(value)
        return self

    def set_cpu(self, value: Optional[int]) -> "VinaBindingEngine":
        """Set CPU count."""
        self._cpu = value
        # if already instantiated, some wrappers allow set_cpu
        if self._vina_instance is not None and hasattr(self._vina_instance, "set_cpu"):
            self._vina_instance.set_cpu(value)
        return self

    def set_seed(self, value: Optional[int]) -> "VinaBindingEngine":
        """Set RNG seed."""
        self._seed = value
        if self._vina_instance is not None and hasattr(self._vina_instance, "set_seed"):
            self._vina_instance.set_seed(value)
        return self

    def set_out(self, out_path: PathLike) -> "VinaBindingEngine":
        """Set output pose file path (chainable)."""
        self._out = Path(out_path)
        return self

    def set_log(self, log_path: PathLike) -> "VinaBindingEngine":
        """Set log path (chainable)."""
        self._log = Path(log_path)
        return self

    def run(
        self, *, exhaustiveness: Optional[int] = None, n_poses: Optional[int] = None
    ) -> "VinaBindingEngine":
        """
        Execute docking using the underlying VinaDock.

        :raises ImportError: if the underlying VinaDock cannot be imported.
        """
        # ensure class + instance exist (or raise helpful ImportError)
        self._ensure_instance()

        # forward overrides
        if exhaustiveness is not None:
            self._vina_instance.set_exhaustiveness(exhaustiveness)
        if n_poses is not None:
            self._vina_instance.set_num_modes(n_poses)

        # run docking
        self._vina_instance.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

        # write outputs if requested
        if self._out is not None:
            self._vina_instance.write_poses(self._out, n_poses=n_poses)
        if self._log is not None:
            self._vina_instance.write_log(self._log)

        return self
