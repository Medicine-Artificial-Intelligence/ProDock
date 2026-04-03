from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, cast

from .base import DockBackend, PathLike, Vec3


class VinaEngine(DockBackend):
    """
    AutoDock Vina Python-package backend.

    This backend uses ``from vina import Vina`` from the pip package rather than a
    project-local wrapper. Import is deferred until runtime so module import stays
    lightweight even when the package is not installed.

    The engine supports a chainable configuration style. Receptor, ligand, box,
    output, and docking options may be set manually, or loaded from a JSON file
    using :meth:`load_config`.

    The expected JSON structure is:

    .. code-block:: json

        {
          "box": {
            "center": [2.865, 193.257, 21.367],
            "size": [27.091, 27.091, 27.091]
          },
          "cpu": 4,
          "seed": 42,
          "exhaustiveness": 16,
          "n_poses": 20
        }

    Example
    -------
    .. code-block:: python

        engine = (
            VinaEngine()
            .set_receptor("protein.pdbqt", validate=True)
            .set_ligand("ligand.pdbqt")
            .load_config("config.json")
            .set_out("poses.pdbqt")
            .set_log("dock.log")
            .run()
        )

    Example
    -------
    .. code-block:: python

        engine = VinaEngine()
        engine.load_config_dict(
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

    def __init__(
        self,
        sf_name: str = "vina",
        cpu: int = 0,
        seed: Optional[int] = None,
        verbosity: int = 1,
        no_refine: bool = False,
    ) -> None:
        """
        Initialize the Vina backend.

        :param sf_name:
            Scoring function name passed to the Python ``vina.Vina`` constructor.
            Common values include ``"vina"``.
        :type sf_name: str

        :param cpu:
            Number of CPU threads to request. A value of ``0`` lets Vina decide.
        :type cpu: int

        :param seed:
            Optional random seed for reproducible docking.
        :type seed: Optional[int]

        :param verbosity:
            Verbosity level forwarded to the Vina Python API.
        :type verbosity: int

        :param no_refine:
            Whether to disable post-docking refinement in the Vina backend.
        :type no_refine: bool
        """
        self.sf_name = sf_name
        self._cpu = cpu
        self._seed = seed
        self.verbosity = verbosity
        self.no_refine = no_refine

        self._receptor: Optional[Path] = None
        self._ligand: Optional[Path] = None
        self._center: Optional[Vec3] = None
        self._size: Optional[Vec3] = None
        self._exhaustiveness: Optional[int] = None
        self._num_modes: Optional[int] = None
        self._out: Optional[Path] = None
        self._log: Optional[Path] = None
        self._last_called: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Return a shallow copy of the last recorded runtime metadata.

        :returns:
            Metadata dictionary describing the most recent docking run.
        :rtype: Dict[str, Any]
        """
        return dict(self._metadata)

    @property
    def called(self) -> Optional[str]:
        """
        Return a textual representation of the last Vina API call chain.

        :returns:
            String describing the last executed docking call, or ``None`` if the
            engine has not yet been run.
        :rtype: Optional[str]
        """
        return self._last_called

    def _import_vina(self):
        """
        Import the Python Vina binding lazily.

        :raises ImportError:
            If the ``vina`` pip package is not installed.

        :returns:
            The imported ``vina.Vina`` class.
        :rtype: Any
        """
        try:
            from vina import Vina  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on external install
            raise ImportError(
                "VinaEngine requires the pip package 'vina'. "
                "Install it with `pip install vina`."
            ) from exc
        return Vina

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

    def set_receptor(
        self, receptor_path: PathLike, *, validate: bool = False
    ) -> "VinaEngine":
        """
        Set the receptor structure file.

        :param receptor_path:
            Path to the receptor file, typically a ``.pdbqt`` file.
        :type receptor_path: PathLike

        :param validate:
            If ``True``, require that the receptor path already exists on disk.
        :type validate: bool

        :raises FileNotFoundError:
            If *validate* is ``True`` and the file does not exist.

        :returns:
            The current engine instance for method chaining.
        :rtype: VinaEngine
        """
        path = Path(receptor_path)
        if validate and not path.is_file():
            raise FileNotFoundError(path)
        self._receptor = path
        return self

    def set_ligand(self, ligand_path: PathLike) -> "VinaEngine":
        """
        Set the ligand structure file.

        :param ligand_path:
            Path to the ligand file, typically a ``.pdbqt`` file.
        :type ligand_path: PathLike

        :returns:
            The current engine instance for method chaining.
        :rtype: VinaEngine
        """
        self._ligand = Path(ligand_path)
        return self

    def set_box(self, center: Vec3, size: Vec3) -> "VinaEngine":
        """
        Set the docking search box.

        :param center:
            Docking box center as ``(x, y, z)``.
        :type center: Vec3

        :param size:
            Docking box size as ``(sx, sy, sz)``.
        :type size: Vec3

        :returns:
            The current engine instance for method chaining.
        :rtype: VinaEngine
        """
        self._center = center
        self._size = size
        return self

    def load_config(self, config_path: PathLike) -> "VinaEngine":
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
            The current engine instance for method chaining.
        :rtype: VinaEngine

        Example
        -------
        .. code-block:: python

            engine = (
                VinaEngine()
                .set_receptor("protein.pdbqt")
                .set_ligand("ligand.pdbqt")
                .load_config("config.json")
            )
        """
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config JSON must contain an object at top level")

        return self.load_config_dict(data)

    def load_config_dict(self, config: Mapping[str, Any]) -> "VinaEngine":
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
            The current engine instance for method chaining.
        :rtype: VinaEngine

        Example
        -------
        .. code-block:: python

            engine = VinaEngine().load_config_dict(
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

    def enable_autobox(
        self, reference_file: PathLike, padding: Optional[float] = None
    ) -> "VinaEngine":
        """
        Reject autobox requests for the Python Vina backend.

        :param reference_file:
            Reference structure path that would otherwise be used for autoboxing.
        :type reference_file: PathLike

        :param padding:
            Optional padding distance that would otherwise enlarge the inferred box.
        :type padding: Optional[float]

        :raises RuntimeError:
            Always raised, because the Python Vina backend requires an explicit box.

        :returns:
            This method never returns normally.
        :rtype: VinaEngine
        """
        raise RuntimeError(
            "VinaEngine uses the Python Vina API and requires an explicit box. "
            "Use SminaEngine or GninaEngine when you need autoboxing."
        )

    def set_exhaustiveness(self, value: Optional[int]) -> "VinaEngine":
        """
        Set the default docking exhaustiveness.

        :param value:
            Exhaustiveness value, or ``None`` to leave unset.
        :type value: Optional[int]

        :returns:
            The current engine instance for method chaining.
        :rtype: VinaEngine
        """
        self._exhaustiveness = value
        return self

    def set_num_modes(self, value: Optional[int]) -> "VinaEngine":
        """
        Set the default number of output poses.

        :param value:
            Number of output modes, or ``None`` to leave unset.
        :type value: Optional[int]

        :returns:
            The current engine instance for method chaining.
        :rtype: VinaEngine
        """
        self._num_modes = value
        return self

    def set_cpu(self, value: Optional[int]) -> "VinaEngine":
        """
        Set the number of CPU threads.

        :param value:
            CPU thread count. If ``None``, the current value is preserved.
        :type value: Optional[int]

        :returns:
            The current engine instance for method chaining.
        :rtype: VinaEngine
        """
        if value is not None:
            self._cpu = int(value)
        return self

    def set_seed(self, value: Optional[int]) -> "VinaEngine":
        """
        Set the docking random seed.

        :param value:
            Random seed, or ``None`` to unset the seed.
        :type value: Optional[int]

        :returns:
            The current engine instance for method chaining.
        :rtype: VinaEngine
        """
        self._seed = value
        return self

    def set_out(self, out_path: PathLike) -> "VinaEngine":
        """
        Set the output PDBQT path for docked poses.

        :param out_path:
            Output file path for docked poses.
        :type out_path: PathLike

        :returns:
            The current engine instance for method chaining.
        :rtype: VinaEngine
        """
        self._out = Path(out_path)
        return self

    def set_log(self, log_path: PathLike) -> "VinaEngine":
        """
        Set the log file path.

        :param log_path:
            Output file path for the docking log.
        :type log_path: PathLike

        :returns:
            The current engine instance for method chaining.
        :rtype: VinaEngine
        """
        self._log = Path(log_path)
        return self

    def _validate_ready(self) -> None:
        """
        Validate that all mandatory runtime inputs are available.

        :raises ValueError:
            If receptor, ligand, or docking box has not been defined.
        """
        if self._receptor is None:
            raise ValueError("Receptor was not set")
        if self._ligand is None:
            raise ValueError("Ligand was not set")
        if self._center is None or self._size is None:
            raise ValueError("Docking box was not set")

    def _capture_native_output(self, func, *args, **kwargs) -> Tuple[Any, str]:
        """
        Run a callable while capturing native stdout and stderr.

        This helper captures output written by compiled extensions directly to the
        process file descriptors, which is often necessary for Vina progress and
        score-table output.

        :param func:
            Callable to execute.
        :type func: Any

        :param args:
            Positional arguments forwarded to *func*.
        :type args: tuple

        :param kwargs:
            Keyword arguments forwarded to *func*.
        :type kwargs: dict

        :returns:
            Two-element tuple containing the function return value and the captured
            output text.
        :rtype: Tuple[Any, str]
        """
        sys.stdout.flush()
        sys.stderr.flush()

        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        with tempfile.TemporaryFile(mode="w+b") as tmp:
            try:
                os.dup2(tmp.fileno(), 1)
                os.dup2(tmp.fileno(), 2)

                result = func(*args, **kwargs)

                sys.stdout.flush()
                sys.stderr.flush()
                os.fsync(tmp.fileno())
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)

            tmp.seek(0)
            captured = tmp.read().decode("utf-8", errors="replace")

        return result, captured

    def _build_log_text(
        self,
        *,
        exhaustiveness: Optional[int],
        n_poses: Optional[int],
        captured_output: str,
    ) -> str:
        """
        Build the final log text.

        :param exhaustiveness:
            Effective exhaustiveness used for the run.
        :type exhaustiveness: Optional[int]

        :param n_poses:
            Effective number of poses requested for the run.
        :type n_poses: Optional[int]

        :param captured_output:
            Native Vina console output captured during docking.
        :type captured_output: str

        :returns:
            Final log text to be written to disk.
        :rtype: str
        """
        header = [
            "AutoDock Vina Python backend",
            "",
            f"Scoring function : {self.sf_name}",
            f"Receptor         : {self._receptor}",
            f"Ligand           : {self._ligand}",
            f"Center           : {list(self._center) if self._center else None}",
            f"Box size         : {list(self._size) if self._size else None}",
            f"Exhaustiveness   : {exhaustiveness}",
            f"CPU              : {self._cpu}",
            f"Seed             : {self._seed}",
            f"Verbosity        : {self.verbosity}",
            f"No refine        : {self.no_refine}",
            f"Requested poses  : {n_poses}",
            "",
        ]

        if captured_output.strip():
            body = captured_output.strip()
        else:
            body = (
                "No docking console output was captured from the Vina Python API.\n"
                "This can happen on some platforms/builds."
            )

        return "\n".join(header) + body + "\n"

    def _write_log(
        self,
        *,
        exhaustiveness: Optional[int],
        n_poses: Optional[int],
        captured_output: str = "",
    ) -> None:
        """
        Write the docking log file if a log path was configured.

        :param exhaustiveness:
            Effective exhaustiveness used for the run.
        :type exhaustiveness: Optional[int]

        :param n_poses:
            Effective number of poses requested for the run.
        :type n_poses: Optional[int]

        :param captured_output:
            Captured native Vina console output.
        :type captured_output: str
        """
        if self._log is None:
            return
        self._log.parent.mkdir(parents=True, exist_ok=True)
        self._log.write_text(
            self._build_log_text(
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                captured_output=captured_output,
            ),
            encoding="utf-8",
        )

    def run(
        self, *, exhaustiveness: Optional[int] = None, n_poses: Optional[int] = None
    ) -> "VinaEngine":
        """
        Execute docking with the configured receptor, ligand, and search box.

        Runtime arguments override values previously loaded through
        :meth:`load_config`, :meth:`load_config_dict`, :meth:`set_exhaustiveness`,
        and :meth:`set_num_modes`.

        :param exhaustiveness:
            Optional per-run exhaustiveness override. If omitted, the stored engine
            value is used. If still unset, a default of ``8`` is used.
        :type exhaustiveness: Optional[int]

        :param n_poses:
            Optional per-run pose-count override. If omitted, the stored engine
            value is used. If still unset, a default of ``9`` is used.
        :type n_poses: Optional[int]

        :raises ValueError:
            If mandatory receptor, ligand, or box inputs are missing.

        :raises ImportError:
            If the Python ``vina`` package is not installed.

        :returns:
            The current engine instance after docking.
        :rtype: VinaEngine

        Example
        -------
        .. code-block:: python

            engine = (
                VinaEngine()
                .set_receptor("protein.pdbqt", validate=True)
                .set_ligand("ligand.pdbqt")
                .load_config("config.json")
                .set_out("poses.pdbqt")
                .set_log("dock.log")
                .run()
            )
        """
        self._validate_ready()
        ex = exhaustiveness if exhaustiveness is not None else self._exhaustiveness
        nm = n_poses if n_poses is not None else self._num_modes
        if ex is None:
            ex = 8
        if nm is None:
            nm = 9

        Vina = self._import_vina()
        vina = Vina(
            sf_name=self.sf_name,
            cpu=self._cpu,
            seed=self._seed,
            verbosity=self.verbosity,
            no_refine=self.no_refine,
        )
        vina.set_receptor(str(self._receptor))
        vina.set_ligand_from_file(str(self._ligand))
        vina.compute_vina_maps(center=list(self._center), box_size=list(self._size))

        _, captured_output = self._capture_native_output(
            vina.dock,
            exhaustiveness=int(ex),
            n_poses=int(nm),
        )

        if self._out is not None:
            self._out.parent.mkdir(parents=True, exist_ok=True)
            vina.write_poses(str(self._out), n_poses=int(nm), overwrite=True)

        self._last_called = (
            "vina.Vina(sf_name={!r}, cpu={!r}, seed={!r}, verbosity={!r}, no_refine={!r})"
            ".set_receptor({!r}).set_ligand_from_file({!r}).compute_vina_maps(center={!r}, box_size={!r})"
            ".dock(exhaustiveness={!r}, n_poses={!r})"
        ).format(
            self.sf_name,
            self._cpu,
            self._seed,
            self.verbosity,
            self.no_refine,
            str(self._receptor),
            str(self._ligand),
            list(self._center),
            list(self._size),
            int(ex),
            int(nm),
        )
        self._metadata = {
            "receptor": str(self._receptor),
            "ligand": str(self._ligand),
            "center": list(self._center),
            "size": list(self._size),
            "exhaustiveness": int(ex),
            "n_poses": int(nm),
            "cpu": self._cpu,
            "seed": self._seed,
        }
        self._write_log(
            exhaustiveness=int(ex),
            n_poses=int(nm),
            captured_output=captured_output,
        )
        return self
