from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

from .base import DockBackend, PathLike, Vec3


def _as_path(value: PathLike) -> Path:
    """
    Convert a path-like value to a :class:`pathlib.Path`.

    :param value:
        Input path represented either as a string or a :class:`Path` object.
    :type value: PathLike

    :returns:
        Normalized path object.
    :rtype: Path
    """
    return value if isinstance(value, Path) else Path(value)


def _ensure_parent(path: Optional[Path]) -> None:
    """
    Ensure that the parent directory of a path exists.

    If ``path`` is not ``None``, its parent directory is created with
    ``parents=True`` and ``exist_ok=True``.

    :param path:
        Target file path whose parent directory should exist.
    :type path: Optional[Path]

    :returns:
        ``None``.
    :rtype: None
    """
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


class BaseBinaryEngine(DockBackend):
    """
    Base implementation for Vina-like command-line docking engines.

    This class provides shared functionality for docking backends that are
    executed as external binaries. It manages receptor and ligand assignment,
    search-box configuration, optional autoboxing, output and log paths,
    command-line argument construction, executable discovery, JSON config
    loading, and command execution.

    Subclasses typically only need to override class attributes such as
    :attr:`exe_name`, :attr:`supports_autobox`, or :attr:`flag_map`.

    :cvar exe_name:
        Default executable name used to invoke the backend.
    :vartype exe_name: str

    :cvar supports_autobox:
        Whether the backend supports ligand-based automatic box generation.
    :vartype supports_autobox: bool

    :cvar flag_map:
        Mapping from logical option names to backend-specific CLI flags.
    :vartype flag_map: Dict[str, str]

    Example
    -------
    .. code-block:: python

        class VinaEngine(BaseBinaryEngine):
            exe_name = "vina"
            supports_autobox = False

        eng = (
            VinaEngine()
            .set_receptor("rec.pdbqt", validate=True)
            .set_ligand("lig.pdbqt")
            .set_box((10.0, 12.0, 8.0), (20.0, 20.0, 20.0))
            .set_exhaustiveness(16)
            .set_num_modes(10)
            .set_out("out/poses.pdbqt")
            .set_log("out/dock.log")
            .run()
        )

    Notes
    -----
    The :meth:`run` method resolves the executable dynamically. It first checks
    whether :attr:`exe_name` points to an executable file, then searches the
    system ``PATH``, and finally checks several local ``bin`` or ``binary``
    directories relative to this module.
    """

    exe_name: str = "smina"
    supports_autobox: bool = False
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
        "autobox_ligand": "--autobox_ligand",
        "autobox_add": "--autobox_add",
    }

    def __init__(self) -> None:
        """
        Initialize an empty binary-engine configuration.

        All runtime parameters are initially unset. Configuration is accumulated
        through fluent setter methods before calling :meth:`run`.

        :returns:
            ``None``.
        :rtype: None
        """
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
        self._extra_args: List[str] = []
        self._last_called: Optional[str] = None
        self._timeout: Optional[float] = None

    @staticmethod
    def _coerce_vec3(value: Any, *, name: str) -> Vec3:
        """
        Validate and coerce an arbitrary value into a 3-vector of floats.

        :param value:
            Candidate vector value, expected to be a sequence of three numeric
            entries.
        :type value: Any

        :param name:
            Human-readable field name used in error messages.
        :type name: str

        :raises TypeError:
            If *value* is not a sequence of exactly three numeric entries.

        :returns:
            Three-element float tuple.
        :rtype: Vec3
        """
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise TypeError(f"{name} must be a list/tuple of length 3")
        try:
            vec = (float(value[0]), float(value[1]), float(value[2]))
        except Exception as exc:
            raise TypeError(f"{name} must contain numeric values") from exc
        return cast(Vec3, vec)

    def load_config(self, config_path: PathLike) -> "BaseBinaryEngine":
        """
        Load docking options from a JSON configuration file.

        Supported keys are:

        - ``box.center``
        - ``box.size``
        - ``cpu``
        - ``seed``
        - ``exhaustiveness``
        - ``n_poses``

        Unknown keys are ignored.

        :param config_path:
            Path to a JSON file containing docking configuration.
        :type config_path: PathLike

        :raises FileNotFoundError:
            If the configuration file does not exist.

        :raises ValueError:
            If the file content is not a JSON object.

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine

        Example
        -------
        .. code-block:: python

            eng = (
                BaseBinaryEngine()
                .load_config("config.json")
            )
        """
        path = _as_path(config_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config JSON must contain an object at top level")

        return self.load_config_dict(data)

    def load_config_dict(self, config: Mapping[str, Any]) -> "BaseBinaryEngine":
        """
        Load docking options from an in-memory mapping.

        Supported keys are:

        - ``box.center``
        - ``box.size``
        - ``cpu``
        - ``seed``
        - ``exhaustiveness``
        - ``n_poses``

        Unknown keys are ignored.

        :param config:
            Mapping containing docking configuration values.
        :type config: Mapping[str, Any]

        :raises TypeError:
            If provided values have invalid types or malformed vector fields.

        :raises ValueError:
            If the ``box`` block is incomplete.

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine

        Example
        -------
        .. code-block:: python

            eng = BaseBinaryEngine().load_config_dict(
                {
                    "box": {
                        "center": [2.865, 193.257, 21.367],
                        "size": [27.091, 27.091, 27.091],
                    },
                    "cpu": 4,
                    "seed": 42,
                    "exhaustiveness": 16,
                    "n_poses": 20,
                }
            )
        """
        box = config.get("box")
        if box is not None:
            if not isinstance(box, Mapping):
                raise TypeError("'box' must be a mapping")

            has_center = "center" in box
            has_size = "size" in box
            if has_center != has_size:
                raise ValueError(
                    "'box' must define both 'center' and 'size' when provided"
                )

            if has_center and has_size:
                center = self._coerce_vec3(box["center"], name="box.center")
                size = self._coerce_vec3(box["size"], name="box.size")
                self.set_box(center=center, size=size)

        if "cpu" in config and config["cpu"] is not None:
            self.set_cpu(int(config["cpu"]))

        if "seed" in config:
            seed_val = config["seed"]
            self.set_seed(None if seed_val is None else int(seed_val))

        if "exhaustiveness" in config and config["exhaustiveness"] is not None:
            self.set_exhaustiveness(int(config["exhaustiveness"]))

        if "n_poses" in config and config["n_poses"] is not None:
            self.set_num_modes(int(config["n_poses"]))

        return self

    def set_receptor(
        self, receptor_path: PathLike, *, validate: bool = False
    ) -> "BaseBinaryEngine":
        """
        Set the receptor structure file.

        :param receptor_path:
            Path to the receptor input file.
        :type receptor_path: PathLike

        :param validate:
            If ``True``, require the receptor path to already exist as a file.
        :type validate: bool

        :raises FileNotFoundError:
            If ``validate=True`` and the receptor file does not exist.

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        path = _as_path(receptor_path)
        if validate and not path.is_file():
            raise FileNotFoundError(path)
        self._receptor = path
        return self

    def set_ligand(self, ligand_path: PathLike) -> "BaseBinaryEngine":
        """
        Set the ligand structure file.

        :param ligand_path:
            Path to the ligand input file.
        :type ligand_path: PathLike

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self._ligand = _as_path(ligand_path)
        return self

    def set_box(self, center: Vec3, size: Vec3) -> "BaseBinaryEngine":
        """
        Set the docking search box explicitly.

        :param center:
            Box center as ``(x, y, z)`` coordinates.
        :type center: Vec3

        :param size:
            Box size as ``(sx, sy, sz)``.
        :type size: Vec3

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine

        Example
        -------
        .. code-block:: python

            eng.set_box(
                center=(12.0, 8.0, 5.0),
                size=(20.0, 20.0, 20.0),
            )
        """
        self._center = center
        self._size = size
        return self

    def enable_autobox(
        self, reference_file: PathLike, padding: Optional[float] = None
    ) -> "BaseBinaryEngine":
        """
        Enable automatic docking-box generation from a reference ligand.

        :param reference_file:
            Path to the reference ligand or structure used for autoboxing.
        :type reference_file: PathLike

        :param padding:
            Optional padding added to the inferred search box.
        :type padding: Optional[float]

        :raises RuntimeError:
            If the engine does not support autoboxing.

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        if not self.supports_autobox:
            raise RuntimeError(f"{self.__class__.__name__} does not support autoboxing")
        self._autobox_ref = _as_path(reference_file)
        self._autobox_pad = padding
        return self

    def set_exhaustiveness(self, value: Optional[int]) -> "BaseBinaryEngine":
        """
        Set the global exhaustiveness value.

        :param value:
            Search exhaustiveness. If ``None``, backend defaults are used.
        :type value: Optional[int]

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self._exhaustiveness = value
        return self

    def set_num_modes(self, value: Optional[int]) -> "BaseBinaryEngine":
        """
        Set the maximum number of poses to generate.

        :param value:
            Number of output modes. If ``None``, backend defaults are used.
        :type value: Optional[int]

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self._num_modes = value
        return self

    def set_cpu(self, value: Optional[int]) -> "BaseBinaryEngine":
        """
        Set the number of CPU threads.

        :param value:
            Number of threads to use. If ``None``, backend defaults are used.
        :type value: Optional[int]

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self._cpu = value
        return self

    def set_seed(self, value: Optional[int]) -> "BaseBinaryEngine":
        """
        Set the random seed.

        :param value:
            Random seed for reproducible docking. If ``None``, the engine may
            behave non-deterministically.
        :type value: Optional[int]

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self._seed = value
        return self

    def set_out(self, out_path: PathLike) -> "BaseBinaryEngine":
        """
        Set the output path for docked poses.

        :param out_path:
            Output file path for docking poses.
        :type out_path: PathLike

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self._out = _as_path(out_path)
        return self

    def set_log(self, log_path: PathLike) -> "BaseBinaryEngine":
        """
        Set the output path for the docking log.

        :param log_path:
            Output file path for the docking log.
        :type log_path: PathLike

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self._log = _as_path(log_path)
        return self

    def set_executable(self, path: PathLike) -> "BaseBinaryEngine":
        """
        Override the executable used to launch the backend.

        :param path:
            Path or executable name to use instead of :attr:`exe_name`.
        :type path: PathLike

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self.exe_name = str(path)
        return self

    def set_timeout(self, seconds: Optional[float]) -> "BaseBinaryEngine":
        """
        Set a timeout for backend execution.

        :param seconds:
            Maximum wall-clock time in seconds for the docking subprocess. If
            ``None``, no timeout is imposed.
        :type seconds: Optional[float]

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self._timeout = seconds
        return self

    def set_extra_args(self, *args: str) -> "BaseBinaryEngine":
        """
        Set additional raw command-line arguments.

        These arguments are appended to the generated command exactly as given.

        :param args:
            Extra backend-specific CLI arguments.
        :type args: str

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine

        Example
        -------
        .. code-block:: python

            eng.set_extra_args("--scoring", "vina", "--quiet")
        """
        self._extra_args = [str(a) for a in args]
        return self

    def _resolve_executable(self) -> str:
        """
        Resolve the executable path for the backend binary.

        Resolution order is:

        1. explicit executable file path stored in :attr:`exe_name`
        2. executable discoverable on system ``PATH``
        3. local ``binary`` or ``bin`` directories next to this module

        :raises FileNotFoundError:
            If no executable can be located.

        :returns:
            Absolute or discovered path to the executable.
        :rtype: str
        """
        exe = str(self.exe_name)
        explicit = Path(exe)

        try:
            if (
                explicit.exists()
                and explicit.is_file()
                and os.access(str(explicit), os.X_OK)
            ):
                return str(explicit.resolve())
        except Exception:
            pass

        found = shutil.which(exe)
        if found:
            return found

        root = Path(__file__).resolve().parent
        # print(root)
        candidates = [
            root / "binary" / exe,
            root / "bin" / exe,
        ]

        for candidate in candidates:
            try:
                if (
                    candidate.exists()
                    and candidate.is_file()
                    and os.access(str(candidate), os.X_OK)
                ):
                    return str(candidate.resolve())
            except Exception:
                pass

        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Could not locate executable for {exe!r}. Install it on PATH, "
            f"call set_executable('/path/to/{exe}'), or bundle it under a local "
            f"bin/binary directory. Searched: {searched}"
        )

    def _build_cmd(
        self,
        override_exhaustiveness: Optional[int] = None,
        override_nposes: Optional[int] = None,
    ) -> List[str]:
        """
        Build the command-line argument vector for the docking run.

        :param override_exhaustiveness:
            Optional per-run override for exhaustiveness.
        :type override_exhaustiveness: Optional[int]

        :param override_nposes:
            Optional per-run override for number of poses.
        :type override_nposes: Optional[int]

        :returns:
            Command vector suitable for :func:`subprocess.run`.
        :rtype: List[str]
        """
        f = self.flag_map
        cmd: List[str] = [self.exe_name]

        if self._receptor is not None and "receptor" in f:
            cmd += [f["receptor"], str(self._receptor)]
        if self._ligand is not None and "ligand" in f:
            cmd += [f["ligand"], str(self._ligand)]

        if self._center is not None:
            cx, cy, cz = self._center
            cmd += [
                f["center_x"],
                str(cx),
                f["center_y"],
                str(cy),
                f["center_z"],
                str(cz),
            ]
        if self._size is not None:
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

        if self._extra_args:
            cmd.extend(self._extra_args)
        return cmd

    def _validate_ready(self) -> None:
        """
        Validate that the engine has enough configuration to run.

        The backend requires both receptor and ligand to be set. In addition,
        either an explicit docking box must be configured or autoboxing must be
        enabled.

        :raises ValueError:
            If mandatory inputs are missing.

        :returns:
            ``None``.
        :rtype: None
        """
        if self._receptor is None:
            raise ValueError("Receptor was not set")
        if self._ligand is None:
            raise ValueError("Ligand was not set")
        if self._center is None or self._size is None:
            if self._autobox_ref is None:
                raise ValueError("Docking box was not set and autobox was not enabled")

    def run(
        self, *, exhaustiveness: Optional[int] = None, n_poses: Optional[int] = None
    ) -> "BaseBinaryEngine":
        """
        Execute the docking job as a subprocess.

        This method validates the current configuration, builds the command,
        resolves the backend executable, creates output directories if needed,
        records the quoted command string, and launches the subprocess.

        :param exhaustiveness:
            Optional per-run override for exhaustiveness.
        :type exhaustiveness: Optional[int]

        :param n_poses:
            Optional per-run override for output pose count.
        :type n_poses: Optional[int]

        :raises ValueError:
            If required docking inputs have not been configured.
        :raises FileNotFoundError:
            If the backend executable cannot be resolved.
        :raises subprocess.CalledProcessError:
            If the docking subprocess exits with a non-zero status.
        :raises subprocess.TimeoutExpired:
            If the subprocess exceeds the configured timeout.

        :returns:
            The engine instance itself.
        :rtype: BaseBinaryEngine
        """
        self._validate_ready()
        cmd = self._build_cmd(exhaustiveness, n_poses)
        cmd[0] = self._resolve_executable()

        _ensure_parent(self._out)
        _ensure_parent(self._log)

        self._last_called = " ".join(shlex.quote(x) for x in cmd)
        subprocess.run(cmd, check=True, timeout=self._timeout)
        return self

    @property
    def called(self) -> Optional[str]:
        """
        Return the most recent executed command string.

        The command is stored as a shell-quoted string after a successful call
        to :meth:`run` reaches subprocess invocation.

        :returns:
            Last executed command string, or ``None`` if the engine has not run.
        :rtype: Optional[str]
        """
        return self._last_called
