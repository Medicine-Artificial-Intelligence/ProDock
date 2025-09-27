from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Union

from .base import DockBackend, Vec3, PathLike, RunArtifacts
from .registry import factory as get_factory
from .config import SingleConfig


@dataclass
class SingleResult:
    """
    Normalized single-run result (artifacts only, no scores).

    :param artifacts: :class:`RunArtifacts` with out/log paths and called command (if any).
    """

    artifacts: RunArtifacts


class SingleDock:
    """
    Unified single-ligand docking facade.

    The :class:`SingleDock` class is a thin, chainable façade around engine-specific backends
    (CLI engines like smina/qvina or a Python Vina binding). It focuses on configuring the
    run and invoking the engine — *no score parsing is performed here* (postprocessing only).

    Key features
    - Chainable setters mirroring typical docking options (receptor, ligand, box, exhaustiveness, ...).
    - Support for CLI engines (smina/gnina/qvina/qvina-w/vina CLI) and a Python-binding mode
      (``vina_binding``) that wraps an in-process Vina implementation if available.
    - Convenience helpers to construct from a dataclass or config file: :meth:`from_config`
      and :meth:`run_from_config`.
    - The backend resolves bundled binaries automatically from
      ``prodock/dock/binary/<exe>`` or system PATH; use :meth:`set_executable` to override.

    Notes about autoboxing
    - Only some CLI engines support autobox flags (smina and gnina). qvina / qvina-w and the
      Vina Python binding do **not** support ``--autobox_*``; calling :meth:`enable_autobox`
      on those backends will raise ``RuntimeError`` (the facade will forward that error).
    - We intentionally do not parse or interpret docking scores here; write outputs and parse
      separately in a postprocessing step.

    Examples
    --------
    1) Chainable CLI usage (smina)
    .. code-block:: python

        from prodock.dock.engine import SingleDock

        sd = (SingleDock("smina")
              .set_receptor("rec.pdbqt", validate=True)
              .set_ligand("lig.pdbqt")
              .set_box((32.5, 13.0, 133.75), (22.5, 23.5, 22.5))
              .set_exhaustiveness(8)
              .set_num_modes(9)
              .set_cpu(4)
              .set_seed(42)
              .set_out("out/lig_docked.pdbqt")
              .set_log("out/lig.log"))

        # If your project bundles smina at prodock/dock/binary/smina and it is executable,
        # you don't need to set the executable explicitly; the engine resolver will find it.
        res = sd.run()
        print(res.artifacts.called)  # exact command used (CLI engines)

    2) Using the Vina Python binding (in-process)
    .. code-block:: python

        # Use the registry key "vina_binding" to prefer the Python binding if available.
        sd = SingleDock("vina_binding") \
            .set_receptor("rec.pdbqt") \
            .set_ligand("lig.pdbqt") \
            .set_box((32.5, 13.0, 133.75), (22.5, 23.5, 22.5)) \
            .set_out("out/lig_docked.pdbqt") \
            .set_log("out/lig.log")

        # run via the binding (may raise ImportError if the binding is missing)
        res = sd.run(exhaustiveness=8, n_poses=9)

    3) Load from dataclass or config file (JSON/YAML)
    .. code-block:: python

        # dataclass-based (programmatic)
        from prodock.dock.engine.config_dataclass import SingleConfig
        cfg = SingleConfig(engine="smina", receptor="rec.pdbqt", ligand="lig.pdbqt",
                           box=Box(center=(32.5,13.0,133.75), size=(22.5,23.5,22.5)),
                           exhaustiveness=8, n_poses=9, cpu=4, seed=42,
                           out="out/lig_docked.pdbqt", log="out/lig.log")
        sd = SingleDock.from_config(cfg)
        res = sd.run()

        # file-based (JSON or YAML)
        res = SingleDock.run_from_config("configs/single_smina.json")

    API
    ---
    :param engine: registry key (e.g. "smina", "gnina", "qvina", "qvina-w", "vina", "vina_binding")

    Public methods (chainable):
      - set_receptor(path, validate=False)
      - set_ligand(path)
      - set_box(center, size)
      - enable_autobox(reference_file, padding=None)
      - set_exhaustiveness(int)
      - set_num_modes(int)
      - set_cpu(int)
      - set_seed(int)
      - set_out(path)
      - set_log(path)
      - set_executable(path)  # convenience to override binary resolution

    Config helpers:
      - SingleDock.from_config(config) -> SingleDock (config can be dataclass, dict, or path)
      - SingleDock.run_from_config(config) -> SingleResult (constructs and runs)

    Return:
      - :class:`SingleResult` with :attr:`artifacts` (:class:`RunArtifacts`) containing:
        - out_path (Path or None), log_path (Path or None), called (string or None)
    """

    def __init__(self, engine: str = "vina"):
        self.engine = engine.lower()
        self._backend: DockBackend = get_factory(self.engine)()
        self._out: Optional[Path] = None
        self._log: Optional[Path] = None

    # chainable
    def set_receptor(self, path: PathLike, *, validate: bool = False) -> "SingleDock":
        self._backend.set_receptor(path, validate=validate)
        return self

    def set_ligand(self, path: PathLike) -> "SingleDock":
        self._backend.set_ligand(path)
        return self

    def set_box(self, center: Vec3, size: Vec3) -> "SingleDock":
        self._backend.set_box(center, size)
        return self

    def enable_autobox(
        self, reference_file: PathLike, padding: Optional[float] = None
    ) -> "SingleDock":
        self._backend.enable_autobox(reference_file, padding=padding)
        return self

    def set_exhaustiveness(self, value: Optional[int]) -> "SingleDock":
        self._backend.set_exhaustiveness(value)
        return self

    def set_num_modes(self, value: Optional[int]) -> "SingleDock":
        self._backend.set_num_modes(value)
        return self

    def set_cpu(self, value: Optional[int]) -> "SingleDock":
        self._backend.set_cpu(value)
        return self

    def set_seed(self, value: Optional[int]) -> "SingleDock":
        self._backend.set_seed(value)
        return self

    def set_out(self, out_path: PathLike) -> "SingleDock":
        self._out = Path(out_path)
        self._backend.set_out(self._out)
        return self

    def set_log(self, log_path: PathLike) -> "SingleDock":
        self._log = Path(log_path)
        self._backend.set_log(self._log)
        return self

    def set_executable(self, exe_path: PathLike) -> "SingleDock":
        if hasattr(self._backend, "set_executable"):
            self._backend.set_executable(exe_path)
        else:
            try:
                setattr(self._backend, "exe_name", str(exe_path))
            except Exception:
                pass
        return self

    def run(
        self, *, exhaustiveness: Optional[int] = None, n_poses: Optional[int] = None
    ) -> SingleResult:
        self._backend.run(exhaustiveness=exhaustiveness, n_poses=n_poses)
        called = getattr(self._backend, "called", None)
        arts = RunArtifacts(out_path=self._out, log_path=self._log, called=called)
        return SingleResult(artifacts=arts)

    @classmethod
    def from_config(
        cls, config: Union[str, Dict[str, Any], SingleConfig]
    ) -> "SingleDock":
        if isinstance(config, SingleConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = SingleConfig.from_dict(config)
        else:
            cfg = SingleConfig.from_file(config)
        sd = cls(engine=cfg.engine)
        if cfg.receptor:
            sd.set_receptor(cfg.receptor, validate=cfg.validate_receptor)
        if cfg.ligand:
            sd.set_ligand(cfg.ligand)
        if cfg.box:
            sd.set_box(cfg.box.center, cfg.box.size)
        if cfg.autobox_ref:
            sd.enable_autobox(cfg.autobox_ref, padding=cfg.autobox_pad)
        if cfg.exhaustiveness is not None:
            sd.set_exhaustiveness(cfg.exhaustiveness)
        if cfg.n_poses is not None:
            sd.set_num_modes(cfg.n_poses)
        if cfg.cpu is not None:
            sd.set_cpu(cfg.cpu)
        if cfg.seed is not None:
            sd.set_seed(cfg.seed)
        if cfg.out:
            sd.set_out(cfg.out)
        if cfg.log:
            sd.set_log(cfg.log)
        if cfg.executable:
            sd.set_executable(cfg.executable)
        for k, v in cfg.engine_options.items():
            try:
                setattr(sd._backend, k, v)
            except Exception:
                pass
        return sd

    @classmethod
    def run_from_config(
        cls, config: Union[str, Dict[str, Any], SingleConfig]
    ) -> SingleResult:
        sd = cls.from_config(config)
        if isinstance(config, SingleConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = SingleConfig.from_dict(config)
        else:
            cfg = SingleConfig.from_file(config)
        return sd.run(exhaustiveness=cfg.exhaustiveness, n_poses=cfg.n_poses)

    def run_with_config(
        self,
        config: Union[str, Dict[str, Any], "SingleConfig"],
        *,
        prefer: str = "config",
    ) -> "SingleResult":
        """
        Run using a configuration while allowing choice of engine precedence.

        Parameters
        ----------
        config : str | dict | SingleConfig
            Path to a JSON/YAML config file, a dict, or a SingleConfig dataclass.
        prefer : {"config", "instance"}
            Which `engine` value takes precedence when a different engine is present in
            the config vs the current SingleDock instance:
              - "config": the engine named in the config will be used (default).
              - "instance": the current SingleDock instance's engine will be used
                            and the config's engine (if present) will be ignored.

        Returns
        -------
        SingleResult
            Same as :meth:`run()`.

        Examples
        --------
        # prefer config engine (default, backward-compatible)
        res = SingleDock.run_from_config("configs/single_smina.json")

        # prefer instance engine (use qvina even if config says smina)
        sd = SingleDock("qvina")
        res = sd.run_with_config("configs/single_smina.json", prefer="instance")
        """
        # Normalize config to SingleConfig dataclass
        if isinstance(config, SingleConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = SingleConfig.from_dict(config)
        else:
            cfg = SingleConfig.from_file(config)

        # If config is preferred, build a new SingleDock from config and run (old behavior)
        if prefer == "config":
            # use classmethod to preserve original semantics
            return self.from_config(cfg).run(
                exhaustiveness=cfg.exhaustiveness, n_poses=cfg.n_poses
            )

        # prefer == "instance": apply config to this instance but keep its engine
        # we apply the same options as from_config/_apply_config but do NOT recreate backend.
        if cfg.receptor:
            self.set_receptor(cfg.receptor, validate=cfg.validate_receptor)
        if cfg.ligand:
            self.set_ligand(cfg.ligand)
        if cfg.box:
            self.set_box(cfg.box.center, cfg.box.size)
        if cfg.autobox_ref:
            # may raise for backends that don't support autobox; that's intended
            self.enable_autobox(cfg.autobox_ref, padding=cfg.autobox_pad)
        if cfg.exhaustiveness is not None:
            self.set_exhaustiveness(cfg.exhaustiveness)
        if cfg.n_poses is not None:
            self.set_num_modes(cfg.n_poses)
        if cfg.cpu is not None:
            self.set_cpu(cfg.cpu)
        if cfg.seed is not None:
            self.set_seed(cfg.seed)
        if cfg.out:
            self.set_out(cfg.out)
        if cfg.log:
            self.set_log(cfg.log)
        if cfg.executable:
            # keep this as a backend override even in instance-prefer mode.
            self.set_executable(cfg.executable)
        # best-effort apply engine_options
        for k, v in cfg.engine_options.items():
            try:
                setattr(self._backend, k, v)
            except Exception:
                pass

        # finally run using this instance's engine
        return self.run(exhaustiveness=cfg.exhaustiveness, n_poses=cfg.n_poses)

    def __repr__(self) -> str:
        return f"<SingleDock engine={self.engine}>"
