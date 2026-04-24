from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import DockBackend, PathLike, RunArtifacts, Vec3
from .config import SingleConfig
from .registry import factory as get_factory
from prodock.io.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SingleResult:
    """
    Result wrapper for a single docking execution.

    :param artifacts:
        Collected run artifacts including output path, log path, executed
        command, and backend metadata.
    :type artifacts: RunArtifacts
    """

    artifacts: RunArtifacts


class SingleDock:
    """
    Facade for one receptor-ligand-engine docking run.

    This class provides a fluent interface around a registered docking backend.
    It can be configured manually through chained setter calls or initialized
    from a :class:`SingleConfig` object, mapping, or configuration file.

    :param engine:
        Docking engine key registered in the backend registry.
    :type engine: str

    :raises KeyError:
        If no backend factory is registered for the requested engine.

    Example
    -------
    .. code-block:: python

        result = (
            SingleDock("qvina")
            .set_receptor("protein.pdbqt", validate=True)
            .set_ligand("ligand.pdbqt")
            .set_box((1.0, 2.0, 3.0), (20.0, 20.0, 20.0))
            .set_out("dock_out.pdbqt")
            .set_log("dock.log")
            .run()
        )
    """

    def __init__(self, engine: str = "vina"):
        """
        Initialize a single-run docking facade.

        :param engine:
            Docking engine key registered in the backend registry.
        :type engine: str
        """
        self.engine = engine.lower()
        logger.debug("Initializing SingleDock with engine=%s", self.engine)

        self._backend: DockBackend = get_factory(self.engine)()
        self._out: Optional[Path] = None
        self._log: Optional[Path] = None

        logger.debug(
            "Initialized SingleDock: engine=%s backend=%s",
            self.engine,
            type(self._backend).__name__,
        )

    def set_receptor(self, path: PathLike, *, validate: bool = False) -> "SingleDock":
        """
        Set the receptor structure for the docking backend.

        :param path:
            Receptor input file path.
        :type path: PathLike

        :param validate:
            Whether receptor validation should be performed by the backend.
        :type validate: bool

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock

        Example
        -------
        .. code-block:: python

            dock = SingleDock("qvina").set_receptor("protein.pdbqt", validate=True)
        """
        logger.debug(
            "Setting receptor for engine=%s: path=%s validate=%s",
            self.engine,
            path,
            validate,
        )
        self._backend.set_receptor(path, validate=validate)
        return self

    def set_ligand(self, path: PathLike) -> "SingleDock":
        """
        Set the ligand structure for the docking backend.

        :param path:
            Ligand input file path.
        :type path: PathLike

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock

        Example
        -------
        .. code-block:: python

            dock = SingleDock("qvina").set_ligand("ligand.pdbqt")
        """
        logger.debug("Setting ligand for engine=%s: path=%s", self.engine, path)
        self._backend.set_ligand(path)
        return self

    def set_box(self, center: Vec3, size: Vec3) -> "SingleDock":
        """
        Set an explicit docking box.

        :param center:
            Box center coordinates as ``(x, y, z)``.
        :type center: Vec3

        :param size:
            Box dimensions as ``(sx, sy, sz)``.
        :type size: Vec3

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock

        Example
        -------
        .. code-block:: python

            dock = SingleDock("qvina").set_box(
                (1.0, 2.0, 3.0),
                (20.0, 20.0, 20.0),
            )
        """
        logger.debug(
            "Setting docking box for engine=%s: center=%s size=%s",
            self.engine,
            center,
            size,
        )
        self._backend.set_box(center, size)
        return self

    def enable_autobox(
        self, reference_file: PathLike, padding: Optional[float] = None
    ) -> "SingleDock":
        """
        Enable autoboxing using a reference structure or ligand.

        :param reference_file:
            Path to the structure used as the autobox reference.
        :type reference_file: PathLike

        :param padding:
            Optional padding added around the automatically inferred box.
        :type padding: Optional[float]

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock

        Example
        -------
        .. code-block:: python

            dock = SingleDock("vina").enable_autobox("ref_ligand.pdbqt", padding=4.0)
        """
        logger.debug(
            "Enabling autobox for engine=%s: reference_file=%s padding=%s",
            self.engine,
            reference_file,
            padding,
        )
        self._backend.enable_autobox(reference_file, padding=padding)
        return self

    def set_exhaustiveness(self, value: Optional[int]) -> "SingleDock":
        """
        Set the docking exhaustiveness.

        :param value:
            Exhaustiveness value, or ``None`` to leave backend default behavior.
        :type value: Optional[int]

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock
        """
        logger.debug(
            "Setting exhaustiveness for engine=%s: value=%s",
            self.engine,
            value,
        )
        self._backend.set_exhaustiveness(value)
        return self

    def set_num_modes(self, value: Optional[int]) -> "SingleDock":
        """
        Set the maximum number of output poses.

        :param value:
            Number of docking modes / poses to request.
        :type value: Optional[int]

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock
        """
        logger.debug(
            "Setting num_modes for engine=%s: value=%s",
            self.engine,
            value,
        )
        self._backend.set_num_modes(value)
        return self

    def set_cpu(self, value: Optional[int]) -> "SingleDock":
        """
        Set the number of CPU threads used by the backend.

        :param value:
            CPU count, or ``None`` to defer to backend default behavior.
        :type value: Optional[int]

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock
        """
        logger.debug("Setting cpu for engine=%s: value=%s", self.engine, value)
        self._backend.set_cpu(value)
        return self

    def set_seed(self, value: Optional[int]) -> "SingleDock":
        """
        Set the random seed for the docking backend.

        :param value:
            Random seed, or ``None`` to defer to backend default behavior.
        :type value: Optional[int]

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock
        """
        logger.debug("Setting seed for engine=%s: value=%s", self.engine, value)
        self._backend.set_seed(value)
        return self

    def set_out(self, out_path: PathLike) -> "SingleDock":
        """
        Set the output pose file path.

        :param out_path:
            Output docking pose file path.
        :type out_path: PathLike

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock
        """
        self._out = Path(out_path)
        logger.debug(
            "Setting output path for engine=%s: out=%s", self.engine, self._out
        )
        self._backend.set_out(self._out)
        return self

    def set_log(self, log_path: PathLike) -> "SingleDock":
        """
        Set the backend log file path.

        :param log_path:
            Log file path.
        :type log_path: PathLike

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock
        """
        self._log = Path(log_path)
        logger.debug("Setting log path for engine=%s: log=%s", self.engine, self._log)
        self._backend.set_log(self._log)
        return self

    def set_executable(self, exe_path: PathLike) -> "SingleDock":
        """
        Override the backend executable path or executable name.

        If the backend exposes ``set_executable()``, it is used directly.
        Otherwise the executable name is assigned to ``exe_name``.

        :param exe_path:
            Path to the backend executable.
        :type exe_path: PathLike

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock
        """
        logger.debug(
            "Setting executable for engine=%s: exe_path=%s",
            self.engine,
            exe_path,
        )
        if hasattr(self._backend, "set_executable"):
            self._backend.set_executable(exe_path)  # type: ignore[attr-defined]
        else:
            setattr(self._backend, "exe_name", str(exe_path))
            logger.debug(
                "Backend %s has no set_executable(); assigned exe_name=%s directly",
                type(self._backend).__name__,
                exe_path,
            )
        return self

    def apply_engine_options(self, options: Dict[str, Any]) -> "SingleDock":
        """
        Apply arbitrary engine-specific options.

        Resolution order for each option key is:

        1. call a facade setter named ``set_<key>``, if present,
        2. call a backend setter named ``set_<key>``, if present,
        3. otherwise set the attribute directly on the backend.

        :param options:
            Mapping of engine option names to values.
        :type options: Dict[str, Any]

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock

        Example
        -------
        .. code-block:: python

            dock.apply_engine_options(
                {
                    "cpu": 4,
                    "seed": 42,
                    "exhaustiveness": 8,
                }
            )
        """
        logger.debug(
            "Applying engine options for engine=%s: keys=%s",
            self.engine,
            sorted(options.keys()),
        )
        for key, value in options.items():
            setter = getattr(self, f"set_{key}", None)
            if callable(setter):
                logger.debug(
                    "Applying engine option via facade setter for engine=%s: %s=%r",
                    self.engine,
                    key,
                    value,
                )
                setter(value)
                continue

            backend_setter = getattr(self._backend, f"set_{key}", None)
            if callable(backend_setter):
                logger.debug(
                    "Applying engine option via backend setter for engine=%s: %s=%r",
                    self.engine,
                    key,
                    value,
                )
                backend_setter(value)
                continue

            logger.debug(
                "Applying engine option via backend attribute for engine=%s: %s=%r",
                self.engine,
                key,
                value,
            )
            setattr(self._backend, key, value)
        return self

    @staticmethod
    def _coerce_config(
        config: Union[str, Path, Dict[str, Any], SingleConfig],
    ) -> SingleConfig:
        """
        Normalize supported configuration inputs into a :class:`SingleConfig`.

        :param config:
            Configuration represented as a config object, mapping, or file path.
        :type config: Union[str, Path, Dict[str, Any], SingleConfig]

        :returns:
            Parsed single docking configuration.
        :rtype: SingleConfig
        """
        logger.debug("Coercing config from type=%s", type(config).__name__)
        cfg = (
            config
            if isinstance(config, SingleConfig)
            else (
                SingleConfig.from_dict(config)
                if isinstance(config, dict)
                else SingleConfig.from_file(config)
            )
        )
        logger.debug(
            "Coerced config successfully: engine=%s receptor=%s ligand=%s out=%s log=%s",
            cfg.engine,
            cfg.receptor,
            cfg.ligand,
            cfg.out,
            cfg.log,
        )
        return cfg

    def apply_config(
        self,
        config: Union[str, Path, Dict[str, Any], SingleConfig],
    ) -> "SingleDock":
        """
        Apply a configuration onto the current instance.

        This method mutates the existing instance using configuration values.

        :param config:
            Configuration represented as a config object, mapping, or file path.
        :type config: Union[str, Path, Dict[str, Any], SingleConfig]

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock

        Example
        -------
        .. code-block:: python

            dock = SingleDock("qvina").apply_config("config.json")
        """
        cfg = self._coerce_config(config)

        logger.info(
            "Applying config to SingleDock: current_engine=%s config_engine=%s receptor=%s ligand=%s out=%s log=%s",
            self.engine,
            cfg.engine,
            cfg.receptor,
            cfg.ligand,
            cfg.out,
            cfg.log,
        )

        if cfg.receptor:
            self.set_receptor(cfg.receptor, validate=cfg.validate_receptor)
        if cfg.ligand:
            self.set_ligand(cfg.ligand)
        if cfg.box:
            self.set_box(cfg.box.center, cfg.box.size)
        if cfg.autobox_ref:
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
            self.set_executable(cfg.executable)
        if cfg.engine_options:
            self.apply_engine_options(cfg.engine_options)

        logger.debug("Finished applying config to engine=%s", self.engine)
        return self

    def load_config(
        self,
        config: Union[str, Path, Dict[str, Any], SingleConfig],
    ) -> "SingleDock":
        """
        Alias for :meth:`apply_config` to support fluent configuration loading.

        :param config:
            Configuration represented as a config object, mapping, or file path.
        :type config: Union[str, Path, Dict[str, Any], SingleConfig]

        :returns:
            The current instance for fluent chaining.
        :rtype: SingleDock

        Example
        -------
        .. code-block:: python

            dock = SingleDock("qvina").load_config("config.json")
        """
        logger.debug("Loading config into existing SingleDock: engine=%s", self.engine)
        return self.apply_config(config)

    def run(
        self, *, exhaustiveness: Optional[int] = None, n_poses: Optional[int] = None
    ) -> SingleResult:
        """
        Execute the docking backend and collect run artifacts.

        :param exhaustiveness:
            Optional run-time override for exhaustiveness.
        :type exhaustiveness: Optional[int]

        :param n_poses:
            Optional run-time override for number of poses.
        :type n_poses: Optional[int]

        :returns:
            Result object containing run artifacts and backend metadata.
        :rtype: SingleResult

        Example
        -------
        .. code-block:: python

            result = dock.run(exhaustiveness=8, n_poses=10)
        """
        logger.info(
            "Starting docking run: engine=%s out=%s log=%s exhaustiveness=%s n_poses=%s",
            self.engine,
            self._out,
            self._log,
            exhaustiveness,
            n_poses,
        )

        try:
            self._backend.run(exhaustiveness=exhaustiveness, n_poses=n_poses)
        except Exception:
            logger.exception(
                "Docking run failed: engine=%s out=%s log=%s",
                self.engine,
                self._out,
                self._log,
            )
            raise

        called = getattr(self._backend, "called", None)
        metadata = getattr(self._backend, "metadata", None)
        artifacts = RunArtifacts(
            out_path=self._out,
            log_path=self._log,
            called=called,
            metadata=dict(metadata or {}),
        )

        logger.info(
            "Completed docking run: engine=%s out=%s log=%s command=%s",
            self.engine,
            self._out,
            self._log,
            called,
        )
        logger.debug(
            "Docking metadata for engine=%s: %r", self.engine, artifacts.metadata
        )

        return SingleResult(artifacts=artifacts)

    @classmethod
    def from_config(
        cls, config: Union[str, Path, Dict[str, Any], SingleConfig]
    ) -> "SingleDock":
        """
        Build a new :class:`SingleDock` instance from configuration.

        Unlike :meth:`apply_config`, this method creates a fresh facade instance
        using the engine declared by the configuration.

        :param config:
            Configuration represented as a config object, mapping, or file path.
        :type config: Union[str, Path, Dict[str, Any], SingleConfig]

        :returns:
            Newly configured docking facade.
        :rtype: SingleDock

        Example
        -------
        .. code-block:: python

            dock = SingleDock.from_config("config.json")
        """
        logger.debug("Creating SingleDock from config")
        cfg = cls._coerce_config(config)
        logger.info("Constructing SingleDock from config: engine=%s", cfg.engine)
        inst = cls(engine=cfg.engine)
        return inst.apply_config(cfg)

    @classmethod
    def run_from_config(
        cls, config: Union[str, Path, Dict[str, Any], SingleConfig]
    ) -> SingleResult:
        """
        Construct and execute a docking run directly from configuration.

        :param config:
            Configuration represented as a config object, mapping, or file path.
        :type config: Union[str, Path, Dict[str, Any], SingleConfig]

        :returns:
            Result object for the executed run.
        :rtype: SingleResult

        Example
        -------
        .. code-block:: python

            result = SingleDock.run_from_config("config.json")
        """
        logger.info("Running SingleDock directly from config")
        cfg = cls._coerce_config(config)
        return cls.from_config(cfg).run(
            exhaustiveness=cfg.exhaustiveness,
            n_poses=cfg.n_poses,
        )

    def run_with_config(
        self,
        config: Union[str, Path, Dict[str, Any], SingleConfig],
        *,
        prefer: str = "config",
    ) -> SingleResult:
        """
        Execute a run using a provided configuration and a precedence strategy.

        When ``prefer="config"``, a fresh instance is created from the supplied
        configuration and the current object is not reused. When
        ``prefer="instance"``, the configuration is applied onto the existing
        object before running.

        :param config:
            Configuration represented as a config object, mapping, or file path.
        :type config: Union[str, Path, Dict[str, Any], SingleConfig]

        :param prefer:
            Precedence mode. Must be either ``"config"`` or ``"instance"``.
        :type prefer: str

        :returns:
            Result object for the executed run.
        :rtype: SingleResult

        :raises ValueError:
            If ``prefer`` is not one of ``"config"`` or ``"instance"``.

        Example
        -------
        .. code-block:: python

            result = dock.run_with_config("config.json", prefer="instance")
        """
        logger.info(
            "Running with config: current_engine=%s prefer=%s",
            self.engine,
            prefer,
        )
        cfg = self._coerce_config(config)

        if prefer not in {"config", "instance"}:
            logger.error(
                "Invalid preference mode for run_with_config: prefer=%s", prefer
            )
            raise ValueError("prefer must be 'config' or 'instance'")

        if prefer == "config":
            logger.debug(
                "Using config-preferred execution path: current_engine=%s config_engine=%s",
                self.engine,
                cfg.engine,
            )
            return self.from_config(cfg).run(
                exhaustiveness=cfg.exhaustiveness,
                n_poses=cfg.n_poses,
            )

        logger.debug(
            "Using instance-preferred execution path: current_engine=%s config_engine=%s",
            self.engine,
            cfg.engine,
        )
        self.apply_config(cfg)
        return self.run(exhaustiveness=cfg.exhaustiveness, n_poses=cfg.n_poses)

    def __repr__(self) -> str:
        """
        Return a compact debug representation.

        :returns:
            String representation containing the selected engine.
        :rtype: str
        """
        return f"<SingleDock engine={self.engine}>"
