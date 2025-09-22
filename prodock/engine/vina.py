from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, List, Tuple, Iterable, Dict, Any
import json
import logging
import numpy as np

try:
    from vina import Vina
except Exception as e:
    raise ImportError(
        "AutoDock Vina Python bindings are required. Install with `pip install vina`.\n"
        f"Original error: {e}"
    )


class VinaDock:
    """
    Object-oriented wrapper for AutoDock Vina (Python API).

    See inline examples below for one-shot and staged usage.

    Parameters
    ----------
    sf_name : str, optional
        Scoring function name (``"vina"``, ``"vinardo"``, ``"ad4"``). Default ``"vina"``.
    cpu : int, optional
        Number of CPU cores to use. Default ``1``.
    seed : int or None, optional
        Random seed for reproducibility. Default ``None``.
    no_refine : bool, optional
        If ``True``, skip the final refinement step. Default ``False``.
    verbosity : int, optional
        Logger + Vina verbosity: ``0`` -> ERROR, ``1`` -> INFO, ``2+`` -> DEBUG. Default ``1``.
    config : str, Path or dict, optional
        Path to a JSON config or a dict with config keys. Explicit kwargs override this config.
    receptor : str or Path, optional
        Path to receptor PDBQT. If provided together with ``center`` and ``size`` maps are computed.
    center : tuple, optional
        Box center ``(x, y, z)``. Required with ``size`` to compute maps.
    size : tuple, optional
        Box size ``(sx, sy, sz)``. Required with ``center`` to compute maps.
    ligand : str or Path, optional
        Path to ligand PDBQT.
    ligand_from_string : str, optional
        Ligand PDBQT string (in-memory).
    ligand_rdkit : object, optional
        RDKit ``Chem.Mol`` instance (if Vina supports RDKit binding).
    exhaustiveness : int, optional
        Docking exhaustiveness. Default ``8``.
    n_poses : int, optional
        Number of poses to request. Default ``9``.
    out_poses : str or Path, optional
        Destination path for writing poses (if autowrite enabled).
    log_path : str or Path, optional
        Destination path for writing human-readable log (if autowrite enabled).
    overwrite : bool, optional
        Overwrite outputs if they exist. Default ``True``.
    validate_pdbqt : bool, optional
        If ``True``, run a quick PDBQT sanity check on provided files. Default ``False``.
    autorun : bool, optional
        If True and receptor/box/ligand are present, compute maps and run docking in ``__init__``.
    autowrite : bool, optional
        If True and output paths are provided, write poses and log after autorun.

    Examples
    --------
    One-shot configuration (autorun + autowrite):
    >>> vd = VinaDock(
    ...     sf_name="vina",
    ...     cpu=4,
    ...     seed=42,
    ...     receptor="Data/testcase/filtered_protein/5N2F.pdbqt",
    ...     center=(32.5, 13.0, 133.75),
    ...     size=(22.5, 23.5, 22.5),
    ...     ligand="Data/testcase/reference_ligand/8HW.pdbqt",
    ...     exhaustiveness=8,
    ...     n_poses=9,
    ...     out_poses="./Data/testcase/vina/vina_out.pdbqt",
    ...     log_path="./Data/testcase/vina/log.txt",
    ...     autorun=True,
    ...     autowrite=True,
    ... )
    >>> print(vd.scores)         # doctest: +SKIP
    >>> print(vd.best_score)     # doctest: +SKIP

    Staged chaining (configure then run):
    >>> vd = VinaDock(sf_name="vina", cpu=4, seed=42)
    >>> (vd.set_receptor("Data/testcase/filtered_protein/5N2F.pdbqt")
    ...    .define_box(center=(32.5, 13.0, 133.75), size=(22.5, 23.5, 22.5))
    ...    .set_ligand("Data/testcase/reference_ligand/8HW.pdbqt")
    ...    .dock(exhaustiveness=8, n_poses=9)
    ...    .write_poses("./Data/testcase/vina/vina_out.pdbqt")
    ...    .write_log("./Data/testcase/vina/log.txt"))
    >>> print(vd.scores)         # doctest: +SKIP
    """

    DEFAULTS: Dict[str, Any] = {
        "sf_name": "vina",
        "cpu": 1,
        "seed": None,
        "no_refine": False,
        "verbosity": 1,
        "exhaustiveness": 8,
        "n_poses": 9,
        "overwrite": True,
        "validate_pdbqt": False,
    }

    # keep __init__ short — helpers do the heavy lifting to reduce cyclomatic complexity
    def __init__(
        self,
        *,
        sf_name: str = DEFAULTS["sf_name"],
        cpu: int = DEFAULTS["cpu"],
        seed: Optional[int] = DEFAULTS["seed"],
        no_refine: bool = DEFAULTS["no_refine"],
        verbosity: int = DEFAULTS["verbosity"],
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        receptor: Optional[Union[str, Path]] = None,
        center: Optional[Tuple[float, float, float]] = None,
        size: Optional[Tuple[float, float, float]] = None,
        ligand: Optional[Union[str, Path]] = None,
        ligand_from_string: Optional[str] = None,
        ligand_rdkit: Optional[Any] = None,
        exhaustiveness: int = DEFAULTS["exhaustiveness"],
        n_poses: int = DEFAULTS["n_poses"],
        out_poses: Optional[Union[str, Path]] = None,
        log_path: Optional[Union[str, Path]] = None,
        overwrite: bool = DEFAULTS["overwrite"],
        validate_pdbqt: bool = DEFAULTS["validate_pdbqt"],
        autorun: bool = False,
        autowrite: bool = False,
    ):
        # small initializer: delegate to private helpers to avoid complexity
        self._merge_config_and_args(
            config=config,
            sf_name=sf_name,
            cpu=cpu,
            seed=seed,
            no_refine=no_refine,
            verbosity=verbosity,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            overwrite=overwrite,
            validate_pdbqt=validate_pdbqt,
        )

        self._setup_logger()
        self._init_vina()

        # runtime state
        self.receptor = None
        self.ligand = None
        self.center = None
        self.size = None
        self._scores = None
        self._last_poses = None
        self._last_score = None
        self._last_optimized_score = None
        self._maps_ready = False

        # staged outputs
        self._out_poses_path = Path(out_poses) if out_poses else None
        self._log_path = Path(log_path) if log_path else None

        # apply provided initial inputs (delegated)
        self._apply_initial_inputs(
            receptor=receptor,
            center=center,
            size=size,
            ligand=ligand,
            ligand_from_string=ligand_from_string,
            ligand_rdkit=ligand_rdkit,
        )

        # autorun/autowrite if requested
        if autorun:
            self._autorun_pipeline(autowrite=autowrite)

    # -------------------- small helper methods (reduce complexity in __init__) -------------------- #
    def _merge_config_and_args(self, **kwargs) -> None:
        """Merge DEFAULTS, optional config file/dict and explicit args into self._config."""
        base_cfg = dict(self.DEFAULTS)
        config = kwargs.pop("config", None)
        if config is not None:
            if isinstance(config, (str, Path)):
                base_cfg.update(self._load_config_file(Path(config)))
            elif isinstance(config, dict):
                base_cfg.update(config)
            else:
                raise TypeError("config must be a dict, or a path to a JSON file")

        # explicit args override config
        base_cfg.update(
            {
                "sf_name": kwargs.pop("sf_name"),
                "cpu": int(kwargs.pop("cpu")),
                "seed": kwargs.pop("seed"),
                "no_refine": bool(kwargs.pop("no_refine")),
                "verbosity": int(kwargs.pop("verbosity")),
                "exhaustiveness": int(kwargs.pop("exhaustiveness")),
                "n_poses": int(kwargs.pop("n_poses")),
                "overwrite": bool(kwargs.pop("overwrite")),
                "validate_pdbqt": bool(kwargs.pop("validate_pdbqt")),
            }
        )
        self._config = base_cfg

    def _setup_logger(self) -> None:
        """Create and configure instance logger."""
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
            )
            self._logger.addHandler(handler)
        self.set_verbosity(self._config["verbosity"])

    def _init_vina(self) -> None:
        """Instantiate the Vina binding with sanitized kwargs."""
        vina_kwargs = {
            "sf_name": self._config["sf_name"],
            "cpu": int(self._config["cpu"]),
            "no_refine": bool(self._config["no_refine"]),
            "verbosity": int(self._config["verbosity"]),
        }
        if self._config.get("seed") is not None:
            vina_kwargs["seed"] = int(self._config["seed"])
        self._logger.debug("Initializing Vina: %s", vina_kwargs)
        self._vina = Vina(**vina_kwargs)

    def _apply_initial_inputs(
        self,
        *,
        receptor: Optional[Union[str, Path]],
        center: Optional[Tuple[float, float, float]],
        size: Optional[Tuple[float, float, float]],
        ligand: Optional[Union[str, Path]],
        ligand_from_string: Optional[str],
        ligand_rdkit: Optional[Any],
    ) -> None:
        """Apply initial receptor/box/ligand inputs passed to __init__."""
        if receptor is not None:
            self.set_receptor(receptor, validate=self._config["validate_pdbqt"])
        if center is not None and size is not None:
            self.define_box(center, size)
        if sum(x is not None for x in (ligand, ligand_from_string, ligand_rdkit)) > 1:
            raise ValueError(
                "Provide only one of ligand | ligand_from_string | ligand_rdkit"
            )
        if ligand is not None:
            self.set_ligand(ligand, validate=self._config["validate_pdbqt"])
        elif ligand_from_string is not None:
            self.set_ligand_from_string(ligand_from_string)
        elif ligand_rdkit is not None:
            self.set_ligand_rdkit(ligand_rdkit)

    def _autorun_pipeline(self, *, autowrite: bool) -> None:
        """If inputs are ready, compute maps and run docking (called from __init__)."""
        self.build()
        if self._ready_to_dock():
            self.dock(
                exhaustiveness=self._config["exhaustiveness"],
                n_poses=self._config["n_poses"],
            )
            if autowrite:
                if self._out_poses_path:
                    self.write_poses(
                        self._out_poses_path,
                        n_poses=self._config["n_poses"],
                        overwrite=self._config["overwrite"],
                    )
                if self._log_path:
                    self.write_log(self._log_path)

    # -------------------- config helpers -------------------- #
    @staticmethod
    def _load_config_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as fh:
            return json.load(fh)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._config)

    def save_config(self, out_path: Union[str, Path]) -> "VinaDock":
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        self._logger.info("Saved config to %s", p)
        return self

    @classmethod
    def from_config(cls, cfg: Union[str, Path, Dict[str, Any]]) -> "VinaDock":
        if isinstance(cfg, (str, Path)):
            cfg = cls._load_config_file(Path(cfg))
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict or a path to JSON")
        return cls(**cfg)

    def update_config(self, **kwargs) -> "VinaDock":
        self._config.update(kwargs)
        if "verbosity" in kwargs:
            self.set_verbosity(self._config["verbosity"])
        return self

    # -------------------- logging -------------------- #
    def set_verbosity(self, verbosity: int) -> "VinaDock":
        level = (
            logging.ERROR
            if verbosity <= 0
            else logging.INFO if verbosity == 1 else logging.DEBUG
        )
        self._logger.setLevel(level)
        self._config["verbosity"] = int(verbosity)
        try:
            if hasattr(self, "_vina"):
                # Vina may or may not have set_verbosity — guard it
                _ = getattr(self._vina, "set_verbosity", None)
                if callable(_):
                    self._vina.set_verbosity(int(verbosity))
        except Exception:
            pass
        return self

    # -------------------- validators & small helpers -------------------- #
    @staticmethod
    def _pdbqt_quick_check(path: Path) -> None:
        with open(path, "r", errors="ignore") as fh:
            head = [next(fh, "") for _ in range(50)]
        text = "".join(head)
        tokens = ("ROOT", "TORSDOF", "ATOM", "HETATM", "BRANCH", "ENDROOT")
        if not any(tok in text for tok in tokens):
            raise ValueError(
                f"{path} does not appear to be a valid PDBQT (missing basic tokens)"
            )

    # -------------------- setup methods (public, chainable) -------------------- #
    def set_receptor(
        self, receptor_path: Union[str, Path], *, validate: bool = False
    ) -> "VinaDock":
        p = Path(receptor_path)
        if not p.exists():
            raise FileNotFoundError(f"Receptor file not found: {p}")
        if validate:
            self._pdbqt_quick_check(p)
        self._vina.set_receptor(str(p))
        self.receptor = str(p)
        self._maps_ready = False
        self._logger.debug("Receptor set: %s", p)
        return self

    def set_receptor_from_string(self, pdbqt_str: str) -> "VinaDock":
        self._vina.set_receptor_from_string(pdbqt_str)
        self.receptor = "<pdbqt-string>"
        self._maps_ready = False
        self._logger.debug("Receptor loaded from string")
        return self

    def set_ligand(
        self, ligand_path: Union[str, Path], *, validate: bool = False
    ) -> "VinaDock":
        p = Path(ligand_path)
        if not p.exists():
            raise FileNotFoundError(f"Ligand file not found: {p}")
        if validate:
            self._pdbqt_quick_check(p)
        self._vina.set_ligand_from_file(str(p))
        self.ligand = str(p)
        self._logger.debug("Ligand set: %s", p)
        return self

    def set_ligand_from_string(self, pdbqt_str: str) -> "VinaDock":
        self._vina.set_ligand_from_string(pdbqt_str)
        self.ligand = "<pdbqt-string>"
        self._logger.debug("Ligand loaded from string")
        return self

    def set_ligand_rdkit(self, rdkit_mol: Any) -> "VinaDock":
        self._vina.set_ligand_from_rdkit(rdkit_mol)
        self.ligand = "<rdkit-mol>"
        self._logger.debug("Ligand set from RDKit object")
        return self

    def define_box(
        self, center: Tuple[float, float, float], size: Tuple[float, float, float]
    ) -> "VinaDock":
        if len(center) != 3 or len(size) != 3:
            raise ValueError("center and size must be 3-tuples")
        centerf = tuple(map(float, center))
        sizef = tuple(map(float, size))
        self.center, self.size = centerf, sizef
        if self.receptor is not None:
            self._vina.compute_vina_maps(center=centerf, box_size=sizef)
            self._maps_ready = True
            self._logger.info("Maps computed: center=%s size=%s", centerf, sizef)
        else:
            self._maps_ready = False
            self._logger.debug(
                "Box staged; maps will be computed once receptor is set."
            )
        self._config.update({"center": self.center, "size": self.size})
        return self

    def build(self) -> "VinaDock":
        if not self._maps_ready and self.receptor and self.center and self.size:
            self._vina.compute_vina_maps(center=self.center, box_size=self.size)
            self._maps_ready = True
            self._logger.info("Maps computed during build()")
        return self

    # -------------------- scoring / dock / IO -------------------- #
    @staticmethod
    def _normalize_scores(raw_scores: Iterable) -> List[Tuple[float, float, float]]:
        try:
            arr = np.asarray(list(raw_scores))
        except Exception:
            arr = np.asarray(raw_scores)

        if arr.ndim == 2 and arr.shape[1] >= 3:
            return [tuple(map(float, arr[i, :3])) for i in range(arr.shape[0])]
        if arr.ndim == 1 and arr.size % 3 == 0 and arr.size > 0:
            n = arr.size // 3
            reshaped = arr.reshape((n, 3))
            return [tuple(map(float, reshaped[i])) for i in range(n)]
        out: List[Tuple[float, float, float]] = []
        for item in raw_scores:
            it = np.asarray(item).flatten()
            if it.size >= 3:
                out.append((float(it[0]), float(it[1]), float(it[2])))
            elif it.size == 2:
                out.append((float(it[0]), float(it[1]), 0.0))
            elif it.size == 1:
                out.append((float(it[0]), 0.0, 0.0))
            else:
                raise ValueError(f"Unable to parse docking score item: {item!r}")
        return out

    def dock(
        self, *, exhaustiveness: Optional[int] = None, n_poses: Optional[int] = None
    ) -> "VinaDock":
        self.build()
        if not (self.receptor and self.center and self.size):
            raise RuntimeError(
                "Receptor or box not defined. Set receptor and define_box first."
            )
        if not self.ligand:
            self._logger.warning(
                "Ligand appears unset (path or in-memory). Ensure a ligand was provided."
            )

        ex = int(
            exhaustiveness
            if exhaustiveness is not None
            else self._config["exhaustiveness"]
        )
        npz = int(n_poses if n_poses is not None else self._config["n_poses"])
        self._logger.info("Docking (exhaustiveness=%d, n_poses=%d)", ex, npz)

        self._vina.dock(exhaustiveness=ex, n_poses=npz)
        raw = self._vina.energies(n_poses=npz)
        self._scores = self._normalize_scores(raw)

        try:
            self._last_poses = self._vina.poses(n_poses=npz)
        except Exception:
            self._last_poses = None

        self._logger.debug(
            "Docking finished with %d modes", len(self._scores) if self._scores else 0
        )
        return self

    def score(self) -> "VinaDock":
        s = self._vina.score()
        self._last_score = (
            float(np.asarray(s).flat[0])
            if isinstance(s, (list, tuple, np.ndarray))
            else float(s)
        )
        self._logger.debug("score() = %.3f", self._last_score)
        return self

    def optimize(self) -> "VinaDock":
        s = self._vina.optimize()
        self._last_optimized_score = (
            float(np.asarray(s).flat[0])
            if isinstance(s, (list, tuple, np.ndarray))
            else float(s)
        )
        self._logger.debug("optimize() = %.3f", self._last_optimized_score)
        return self

    def write_poses(
        self,
        out_path: Union[str, Path],
        *,
        n_poses: Optional[int] = None,
        overwrite: Optional[bool] = None,
    ) -> "VinaDock":
        npz = int(n_poses if n_poses is not None else self._config["n_poses"])
        ovw = bool(self._config["overwrite"] if overwrite is None else overwrite)
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._vina.write_poses(str(p), n_poses=npz, overwrite=ovw)
        self._logger.info("Wrote poses to %s", p)
        return self

    def write_log(
        self, log_path: Union[str, Path], *, human_readable: bool = True
    ) -> "VinaDock":
        if self._scores is None:
            raise RuntimeError("No docking results to log. Run .dock() first.")
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:

            def log_print(msg: str):
                if human_readable:
                    print(msg, flush=True)
                f.write(msg + "\n")

            seed_txt = self._config.get("seed", "auto")
            log_print("VinaDock log")
            log_print(f"Scoring function: {self._config['sf_name']}")
            log_print(f"CPU: {self._config['cpu']} | seed: {seed_txt}")
            if self.center and self.size:
                log_print(f"Box center: {self.center} | size: {self.size}")
            if self.receptor:
                log_print(f"Receptor: {self.receptor}")
            if self.ligand:
                log_print(f"Ligand: {self.ligand}")
            log_print("mode |   affinity | rmsd l.b.| rmsd u.b.")
            log_print("-----+------------+----------+----------")
            for i, (e, rmsd_lb, rmsd_ub) in enumerate(self._scores, start=1):
                log_print(f"{i:4d} {e:12.3f} {rmsd_lb:10.3f} {rmsd_ub:10.3f}")

        self._logger.info("Wrote log to %s", p)
        return self

    # -------------------- properties -------------------- #
    @property
    def scores(self) -> Optional[List[Tuple[float, float, float]]]:
        return None if self._scores is None else list(self._scores)

    @property
    def best_score(self) -> Optional[Tuple[float, float, float]]:
        return None if not self._scores else self._scores[0]

    @property
    def n_modes(self) -> int:
        return 0 if not self._scores else len(self._scores)

    @property
    def last_poses(self) -> Optional[str]:
        return self._last_poses

    @property
    def last_score(self) -> Optional[float]:
        return self._last_score

    @property
    def last_optimized_score(self) -> Optional[float]:
        return self._last_optimized_score

    # -------------------- misc -------------------- #
    def _ready_to_dock(self) -> bool:
        return bool(
            self.receptor
            and self.center
            and self.size
            and (self.ligand or self.ligand in ("<pdbqt-string>", "<rdkit-mol>"))
        )

    def help(self) -> None:
        print(
            "Example — fully configured in __init__ with autorun & autowrite:\n"
            "  vd = VinaDock(\n"
            "        sf_name='vina', cpu=4, seed=42,\n"
            "        receptor='rec.pdbqt', center=(x,y,z), size=(sx,sy,sz),\n"
            "        ligand='lig.pdbqt', exhaustiveness=8, n_poses=9,\n"
            "        out_poses='out/poses.pdbqt', log_path='out/log.txt',\n"
            "        autorun=True, autowrite=True,\n"
            "  )\n"
            "Afterwards, access results via properties: vd.scores, vd.best_score, vd.n_modes."
        )

    def __repr__(self) -> str:
        return (
            f"<VinaDock sf={self._config['sf_name']} cpu={self._config['cpu']} "
            f"receptor={self.receptor} ligand={self.ligand} "
            f"center={self.center} size={self.size} modes={self.n_modes}>"
        )

    def __enter__(self) -> "VinaDock":
        self._logger.debug("Entering context")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._logger.debug("Exiting context")
        cleanup = getattr(self._vina, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass
