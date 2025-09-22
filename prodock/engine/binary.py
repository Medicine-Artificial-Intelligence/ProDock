from __future__ import annotations

import os
import shutil
import subprocess
import re
import json
from pathlib import Path
from typing import Union, Optional, Sequence, Dict, Any, Iterable, Tuple, List
import logging


class BinaryDock:
    """
    OOP wrapper for external docking binaries (smina, qvina, qvina-w, ...).

    Two usage patterns (both supported):
    1) One-shot `__init__` configuration (autorun + autowrite):
       bd = BinaryDock(
           binary_name="smina",
           cpu=8,
           seed=42,
           receptor="rec.pdbqt",
           center=(32.5, 13.0, 133.75),
           size=(22.5, 23.5, 22.5),
           ligand="lig.pdbqt",
           exhaustiveness=8,
           n_poses=9,
           out_poses="out/lig_docked.pdbqt",
           log_path="out/lig.log",
           autorun=True,
           autowrite=True,
       )

    2) Staged chaining (configure then run):
       bd = (BinaryDock("smina", cpu=8, seed=42)
             .set_receptor("rec.pdbqt")
             .set_box((32.5,13,133.75),(22.5,23.5,22.5))
             .set_ligand("lig.pdbqt")
             .set_out("out/lig_docked.pdbqt")
             .set_log("out/lig.log")
             .set_exhaustiveness(8)
             .set_num_modes(9)
             .set_cpu(8)
             .run())

    Parameters
    ----------
    binary_name : str, optional
        Binary name or path (default ``"smina"``).
    binary_dir : str or Path, optional
        Directory to search for the binary (default ``"prodock/binary"``).
    config : dict or str/Path (JSON), optional
        Config dict or path to JSON (explicit kwargs override it).
    cpu : int, optional
        Number of CPU threads to request (will be added if binary supports `--cpu`).
    seed : int, optional
        RNG seed passed to binary if supported.
    receptor : str or Path, optional
        Receptor PDBQT path.
    center : tuple, optional
        Box center (x,y,z).
    size : tuple, optional
        Box size (sx,sy,sz).
    ligand : str or Path, optional
        Ligand path (PDBQT recommended).
    exhaustiveness : int, optional
        Search exhaustiveness (if supported).
    n_poses : int, optional
        Number of poses to request (mapped to `--num_modes` where supported).
    out_poses : str or Path, optional
        Path where output PDBQT should be written.
    log_path : str or Path, optional
        Path for human-readable log.
    overwrite : bool, optional
        Overwrite outputs when writing (default True).
    validate_pdbqt : bool, optional
        Do light PDBQT sanity checks on provided files (default False).
    autorun : bool, optional
        If True and inputs provided, run pipeline on construction.
    autowrite : bool, optional
        If True and outputs provided, write poses/log after autorun.
    verbosity : int, optional
        Internal logger level (0 ERROR, 1 INFO, 2+ DEBUG). Default 1.

    Notes
    -----
    - The wrapper probes `<exe> --help` to decide which options to pass.
    - After `run()` inspect `.result` for {"rc","stdout","stderr","out","log","called"}.
    - Parsing helpers: :meth:`parse_scores_from_log`, :meth:`scores_to_csv`, :meth:`scores_as_dataframe`.
    """

    DEFAULTS: Dict[str, Any] = {
        "binary_name": "smina",
        "binary_dir": "prodock/binary",
        "cpu": None,
        "seed": None,
        "exhaustiveness": 8,
        "n_poses": 9,
        "overwrite": True,
        "validate_pdbqt": False,
        "verbosity": 1,
    }

    def __init__(
        self,
        binary_name: str = DEFAULTS["binary_name"],
        binary_dir: Union[str, Path] = DEFAULTS["binary_dir"],
        *,
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        cpu: Optional[int] = DEFAULTS["cpu"],
        seed: Optional[int] = DEFAULTS["seed"],
        receptor: Optional[Union[str, Path]] = None,
        center: Optional[Tuple[float, float, float]] = None,
        size: Optional[Tuple[float, float, float]] = None,
        ligand: Optional[Union[str, Path]] = None,
        exhaustiveness: Optional[int] = DEFAULTS["exhaustiveness"],
        n_poses: Optional[int] = DEFAULTS["n_poses"],
        out_poses: Optional[Union[str, Path]] = None,
        log_path: Optional[Union[str, Path]] = None,
        overwrite: bool = DEFAULTS["overwrite"],
        validate_pdbqt: bool = DEFAULTS["validate_pdbqt"],
        autorun: bool = False,
        autowrite: bool = False,
        verbosity: int = DEFAULTS["verbosity"],
    ):
        # keep __init__ small: merge config, init state, resolve binary, and apply initial inputs
        self.binary_name = str(binary_name)
        self.binary_dir = Path(binary_dir)
        self._merge_config_and_args(
            config=config,
            cpu=cpu,
            seed=seed,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            out_poses=out_poses,
            log_path=log_path,
            overwrite=overwrite,
            validate_pdbqt=validate_pdbqt,
            verbosity=verbosity,
        )
        self._init_state()
        self._setup_logger()
        self._resolve_executable()
        self._probe_capabilities()
        self._apply_initial_inputs(
            receptor=receptor, center=center, size=size, ligand=ligand
        )

        if autorun:
            self._autorun(autowrite=autowrite)

    # -------------------- small helpers to keep __init__ simple -------------------- #
    def _merge_config_and_args(self, **kwargs) -> None:
        cfg = dict(self.DEFAULTS)
        config = kwargs.pop("config", None)
        if config is not None:
            if isinstance(config, (str, Path)):
                cfg.update(self._load_config_file(Path(config)))
            elif isinstance(config, dict):
                cfg.update(config)
            else:
                raise TypeError("config must be dict or path to JSON")
        # explicit overrides
        cfg.update(
            {
                "cpu": kwargs.pop("cpu"),
                "seed": kwargs.pop("seed"),
                "exhaustiveness": kwargs.pop("exhaustiveness"),
                "n_poses": kwargs.pop("n_poses"),
                "out_poses": kwargs.pop("out_poses"),
                "log_path": kwargs.pop("log_path"),
                "overwrite": kwargs.pop("overwrite"),
                "validate_pdbqt": kwargs.pop("validate_pdbqt"),
                "verbosity": kwargs.pop("verbosity"),
            }
        )
        self._config: Dict[str, Any] = cfg

    @staticmethod
    def _load_config_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as fh:
            return json.load(fh)

    def _init_state(self) -> None:
        # IO and runtime
        self.ligand_path: Optional[Path] = None
        self.receptor_path: Optional[Path] = None
        self.flex_path: Optional[Path] = None
        self.out_path: Optional[Path] = (
            Path(self._config["out_poses"]) if self._config.get("out_poses") else None
        )
        self.log_path: Optional[Path] = (
            Path(self._config["log_path"]) if self._config.get("log_path") else None
        )
        self.config_path: Optional[Path] = None

        # box
        self.autobox: bool = False
        self.autobox_reference: Optional[Path] = None
        self.autobox_add: Optional[float] = None
        self.center: Optional[Tuple[float, float, float]] = None
        self.size: Optional[Tuple[float, float, float]] = None

        # options
        self.exhaustiveness: Optional[int] = (
            int(self._config["exhaustiveness"])
            if self._config.get("exhaustiveness") is not None
            else None
        )
        self.num_modes: Optional[int] = (
            int(self._config["n_poses"])
            if self._config.get("n_poses") is not None
            else None
        )
        self.energy_range: Optional[float] = None
        self.spacing: Optional[float] = None
        self.cpu: Optional[int] = (
            int(self._config["cpu"]) if self._config.get("cpu") is not None else None
        )
        self.seed: Optional[int] = (
            int(self._config["seed"]) if self._config.get("seed") is not None else None
        )

        # flags/options passthrough
        self._flags: List[str] = []
        self._options: List[str] = []

        # subprocess
        self.timeout: Optional[float] = None
        self.env: Optional[dict] = None
        self.cwd: Optional[Union[str, Path]] = None
        self.dry_run: bool = False

        # capability probe
        self._exe: Optional[str] = None
        self._help_text: Optional[str] = None

        # last run
        self._last_result: Optional[Dict[str, Any]] = None

    def _setup_logger(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self._logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
            )
            self._logger.addHandler(h)
        level = (
            logging.ERROR
            if self._config.get("verbosity", 1) <= 0
            else (
                logging.INFO if self._config.get("verbosity", 1) == 1 else logging.DEBUG
            )
        )
        self._logger.setLevel(level)

    def _apply_initial_inputs(
        self,
        *,
        receptor: Optional[Union[str, Path]],
        center: Optional[Tuple[float, float, float]],
        size: Optional[Tuple[float, float, float]],
        ligand: Optional[Union[str, Path]],
    ) -> None:
        if receptor is not None:
            self.set_receptor(
                receptor, validate=self._config.get("validate_pdbqt", False)
            )
        if center is not None and size is not None:
            self.set_box(center, size)
        if ligand is not None:
            self.set_ligand(ligand)

    def _autorun(self, *, autowrite: bool) -> None:
        # Only run when basic inputs are present
        try:
            self.run()
            if autowrite:
                if self.out_path:
                    # some binaries write output themselves; we still ensure a path was set
                    pass
                if self.log_path:
                    # ensure log contains driver info
                    pass
        except Exception as exc:
            self._logger.error(exc)
            # autorun shouldn't crash silently; re-raise for explicit failure
            raise

    # -------------------- executable detection & probing -------------------- #
    def _resolve_executable(self) -> Optional[str]:
        maybe = Path(self.binary_name)
        if maybe.exists() and os.access(str(maybe), os.X_OK):
            self._exe = str(maybe.resolve())
            return self._exe
        exe = shutil.which(self.binary_name)
        if exe:
            self._exe = exe
            return exe
        candidate = Path(self.binary_dir) / self.binary_name
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            self._exe = str(candidate.resolve())
            return self._exe
        alt = shutil.which(self.binary_name + "02") or shutil.which(
            self.binary_name + "-w"
        )
        if alt:
            self._exe = alt
            return alt
        self._exe = None
        return None

    def _probe_capabilities(self) -> None:
        self._help_text = None
        if not self._exe:
            return
        try:
            proc = subprocess.run(
                [self._exe, "--help"], capture_output=True, text=True, timeout=5
            )
            txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
            self._help_text = txt.lower()
        except Exception:
            self._help_text = ""

    def _supports(self, opt: str) -> bool:
        if self._help_text is None:
            safe = {
                "--receptor",
                "--ligand",
                "--out",
                "--center_x",
                "--center_y",
                "--center_z",
                "--size_x",
                "--size_y",
                "--size_z",
                "--exhaustiveness",
                "--num_modes",
                "--seed",
                "--cpu",
                "--config",
            }
            return opt in safe
        return opt.lower() in self._help_text

    # -------------------- validators -------------------- #
    def _validate_inputs(self) -> None:
        if self._exe is None:
            raise RuntimeError(f"Docking binary not found (tried '{self.binary_name}')")
        if self.receptor_path is None:
            raise ValueError("receptor_path not set (call .set_receptor(...))")
        if self.ligand_path is None:
            raise ValueError("ligand_path not set (call .set_ligand(...))")
        if self.out_path is None:
            raise ValueError("out_path not set (call .set_out(...))")
        if self.log_path is None:
            raise ValueError("log_path not set (call .set_log(...))")
        if self.autobox:
            if not self._supports("--autobox_ligand"):
                raise ValueError(
                    "This binary does not support --autobox_ligand; disable autobox or use smina."
                )
            if self.autobox_reference is None:
                raise ValueError("autobox=True requires enable_autobox(reference_file)")
        else:
            if self.center is None or self.size is None:
                raise ValueError(
                    "When autobox is False, set the box via .set_box(center,size)."
                )
        for p in (self.receptor_path, self.ligand_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing file: {p}")
        if self.flex_path and not self.flex_path.exists():
            raise FileNotFoundError(f"Flexible file not found: {self.flex_path}")
        if self.config_path and not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    # -------------------- chainable setters -------------------- #
    def set_binary(
        self, binary_name: str, binary_dir: Union[str, Path] = None
    ) -> "BinaryDock":
        self.binary_name = str(binary_name)
        if binary_dir is not None:
            self.binary_dir = Path(binary_dir)
        self._resolve_executable()
        self._probe_capabilities()
        return self

    def set_receptor(
        self, receptor_path: Union[str, Path], *, validate: bool = False
    ) -> "BinaryDock":
        p = Path(receptor_path)
        if not p.exists():
            raise FileNotFoundError(f"Receptor not found: {p}")
        if validate:
            self._pdbqt_quick_check(p)
        self.receptor_path = p
        return self

    def set_ligand(self, ligand_path: Union[str, Path]) -> "BinaryDock":
        p = Path(ligand_path)
        if not p.exists():
            raise FileNotFoundError(f"Ligand not found: {p}")
        self.ligand_path = p
        return self

    def set_flex(self, flex_path: Union[str, Path]) -> "BinaryDock":
        self.flex_path = Path(flex_path)
        return self

    def set_out(self, out_path: Union[str, Path]) -> "BinaryDock":
        self.out_path = Path(out_path)
        return self

    def set_log(self, log_path: Union[str, Path]) -> "BinaryDock":
        self.log_path = Path(log_path)
        return self

    def set_config(self, config_path: Union[str, Path]) -> "BinaryDock":
        self.config_path = Path(config_path)
        return self

    def set_box(
        self, center: Tuple[float, float, float], size: Tuple[float, float, float]
    ) -> "BinaryDock":
        self.center = (float(center[0]), float(center[1]), float(center[2]))
        self.size = (float(size[0]), float(size[1]), float(size[2]))
        return self

    def enable_autobox(
        self, reference_file: Union[str, Path], padding: Optional[float] = None
    ) -> "BinaryDock":
        self.autobox = True
        self.autobox_reference = Path(reference_file)
        self.autobox_add = None if padding is None else float(padding)
        return self

    def disable_autobox(self) -> "BinaryDock":
        self.autobox = False
        self.autobox_reference = None
        self.autobox_add = None
        return self

    def set_exhaustiveness(self, value: Optional[int]) -> "BinaryDock":
        self.exhaustiveness = None if value is None else int(value)
        return self

    def set_num_modes(self, value: Optional[int]) -> "BinaryDock":
        self.num_modes = None if value is None else int(value)
        return self

    def set_spacing(self, value: Optional[float]) -> "BinaryDock":
        self.spacing = None if value is None else float(value)
        return self

    def set_cpu(self, value: Optional[int]) -> "BinaryDock":
        self.cpu = None if value is None else int(value)
        return self

    def set_seed(self, value: Optional[int]) -> "BinaryDock":
        self.seed = None if value is None else int(value)
        return self

    def add_flags(self, flags: Sequence[str]) -> "BinaryDock":
        self._flags.extend(map(str, flags))
        return self

    def add_options(self, kv_pairs: Sequence[Union[str, int, float]]) -> "BinaryDock":
        self._options.extend(map(lambda x: str(x), kv_pairs))
        return self

    def set_timeout(self, seconds: Optional[float]) -> "BinaryDock":
        self.timeout = None if seconds is None else float(seconds)
        return self

    def set_env(self, env: Optional[dict]) -> "BinaryDock":
        self.env = None if env is None else dict(env)
        return self

    def set_cwd(self, cwd: Optional[Union[str, Path]]) -> "BinaryDock":
        self.cwd = None if cwd is None else str(cwd)
        return self

    def set_dry_run(self, dry: bool = True) -> "BinaryDock":
        self.dry_run = bool(dry)
        return self

    # -------------------- command assembly -------------------- #
    def _build_args(self) -> List[str]:
        if not self._exe:
            raise RuntimeError("Executable not resolved")
        args: List[str] = [str(self._exe)]

        # IO
        if self._supports("--receptor"):
            args += ["--receptor", str(self.receptor_path)]
        else:
            args += ["-r", str(self.receptor_path)]

        if self._supports("--ligand"):
            args += ["--ligand", str(self.ligand_path)]
        else:
            args += ["-l", str(self.ligand_path)]

        if self.flex_path and self._supports("--flex"):
            args += ["--flex", str(self.flex_path)]

        # Output
        if self._supports("--out"):
            args += ["--out", str(self.out_path)]
        else:
            args += ["-o", str(self.out_path)]

        # Log
        if self.log_path and self._supports("--log"):
            args += ["--log", str(self.log_path)]

        # Config
        if self.config_path and self._supports("--config"):
            args += ["--config", str(self.config_path)]

        # CPU/seed
        if self.cpu is not None and self._supports("--cpu"):
            args += ["--cpu", str(self.cpu)]
        if self.seed is not None and self._supports("--seed"):
            args += ["--seed", str(self.seed)]

        # Box / autobox
        if self.autobox:
            args += ["--autobox_ligand", str(self.autobox_reference)]
            if self.autobox_add is not None and self._supports("--autobox_add"):
                args += ["--autobox_add", str(self.autobox_add)]
        else:
            if self.center is not None:
                if self._supports("--center_x"):
                    args += [
                        "--center_x",
                        str(self.center[0]),
                        "--center_y",
                        str(self.center[1]),
                        "--center_z",
                        str(self.center[2]),
                    ]
                else:
                    args += [
                        "-c",
                        str(self.center[0]),
                        str(self.center[1]),
                        str(self.center[2]),
                    ]
            if self.size is not None:
                if self._supports("--size_x"):
                    args += [
                        "--size_x",
                        str(self.size[0]),
                        "--size_y",
                        str(self.size[1]),
                        "--size_z",
                        str(self.size[2]),
                    ]
                else:
                    args += [
                        "-s",
                        str(self.size[0]),
                        str(self.size[1]),
                        str(self.size[2]),
                    ]

        # Search / outputs controls
        if self.exhaustiveness is not None and self._supports("--exhaustiveness"):
            args += ["--exhaustiveness", str(self.exhaustiveness)]
        if self.num_modes is not None and self._supports("--num_modes"):
            args += ["--num_modes", str(self.num_modes)]
        if self.energy_range is not None and self._supports("--energy_range"):
            args += ["--energy_range", str(self.energy_range)]
        if self.spacing is not None and self._supports("--spacing"):
            args += ["--spacing", str(self.spacing)]

        # Behavioral flags
        if self._supports("--local_only") and getattr(self, "local_only", False):
            args += ["--local_only"]
        if self._supports("--randomize_only") and getattr(
            self, "randomize_only", False
        ):
            args += ["--randomize_only"]
        if self._supports("--minimize") and getattr(self, "minimize", False):
            args += ["--minimize"]

        # user raw flags/options
        for flg in self._flags:
            if flg.startswith("-") and not self._supports(flg):
                continue
            args.append(flg)

        i = 0
        while i < len(self._options):
            opt = self._options[i]
            if opt.startswith("-") and not self._supports(opt):
                i += 2
                continue
            args.append(opt)
            i += 1

        return args

    # -------------------- runtime -------------------- #
    def run(self) -> "BinaryDock":
        """
        Execute the assembled command. After run, inspect `.result`.
        """
        self._resolve_executable()
        self._probe_capabilities()
        self._validate_inputs()

        # ensure parent dirs
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        args = self._build_args()
        called = " ".join(args)

        if self.dry_run:
            self._last_result = {
                "rc": None,
                "stdout": None,
                "stderr": None,
                "out": str(self.out_path),
                "log": str(self.log_path),
                "called": called,
                "dry_run": True,
            }
            return self

        env = os.environ.copy()
        if self.env:
            env.update(self.env)

        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            env=env,
            cwd=self.cwd,
            timeout=self.timeout,
        )

        # append driver-captured stdout/stderr to the log file (do NOT clobber binary's own log)
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write("\n\n--- DRIVER STDOUT ---\n")
                fh.write(proc.stdout or "")
                fh.write("\n--- DRIVER STDERR ---\n")
                fh.write(proc.stderr or "")
        except Exception:
            pass

        self._last_result = {
            "rc": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "out": str(self.out_path),
            "log": str(self.log_path),
            "called": called,
        }
        return self

    def run_many(
        self,
        ligands: Iterable[Union[str, Path]],
        out_dir: Union[str, Path],
        log_dir: Union[str, Path],
        *,
        autobox_refs: Optional[Iterable[Union[str, Path]]] = None,
        overwrite: bool = True,
    ) -> "BinaryDock":
        out_dir = Path(out_dir)
        log_dir = Path(log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        refs_iter = iter(autobox_refs) if autobox_refs is not None else None

        for lig in ligands:
            lig = Path(lig)
            out_p = out_dir / f"{lig.stem}_docked.pdbqt"
            log_p = log_dir / f"{lig.stem}.log"
            if (not overwrite) and out_p.exists() and log_p.exists():
                continue
            self.set_ligand(lig).set_out(out_p).set_log(log_p)
            if self.autobox and refs_iter is not None:
                try:
                    ref = next(refs_iter)
                    self.enable_autobox(ref, padding=self.autobox_add)
                except StopIteration:
                    pass
            self.run()
        return self

    # -------------------- accessors & parsing -------------------- #
    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return self._last_result

    def __repr__(self) -> str:
        return (
            f"<BinaryDock exe={self._exe or self.binary_name}"
            + f" ligand={self.ligand_path} receptor={self.receptor_path}"
            + f" autobox={self.autobox} cpu={self.cpu} seed={self.seed}>"
        )

    def help(self) -> None:
        print("See class docstring for examples.")

    # -------------------- score parsing/export (unchanged, robust) -------------------- #
    def _infer_id(self, path: Optional[Union[str, Path]]) -> str:
        return "" if path is None else Path(path).stem

    def parse_scores_from_log(
        self,
        log_path: Union[str, Path],
        ligand_path: Optional[Union[str, Path]] = None,
        receptor_path: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        lp = Path(log_path)
        if not lp.exists():
            raise FileNotFoundError(f"log file not found: {log_path}")
        text_lines = lp.read_text(errors="ignore").splitlines()

        header_idx = None
        for i, line in enumerate(text_lines):
            if re.search(r"\bmode\b", line, re.I) and re.search(
                r"\baffinity\b", line, re.I
            ):
                header_idx = i
                break
        if header_idx is None:
            for i, line in enumerate(text_lines):
                if re.search(r"^-{3,}\+?-{3,}", line.strip()):
                    header_idx = i
                    break
        if header_idx is None:
            return []

        sep_idx = None
        for j in range(header_idx, min(header_idx + 6, len(text_lines))):
            L = text_lines[j].strip()
            if re.match(r"^[-\s\+]{5,}$", L) or re.search(r"-{3,}\+?-{3,}", L):
                sep_idx = j
                break

        start = (sep_idx + 1) if sep_idx is not None else (header_idx + 1)
        while start < len(text_lines) and (
            text_lines[start].strip() == ""
            or re.match(r"[-=+\s]+$", text_lines[start].strip())
        ):
            start += 1

        row_re = re.compile(
            r"^\s*(\d+)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*$"
        )
        rows: List[Dict[str, Any]] = []
        ligand_id = self._infer_id(ligand_path or self.ligand_path)
        receptor_id = self._infer_id(receptor_path or self.receptor_path)
        footer_markers = re.compile(
            r"(refine time|loop time|--- driver stderr ---|--- driver stdout ---)", re.I
        )
        for idx in range(start, len(text_lines)):
            line = text_lines[idx].rstrip()
            if line.strip() == "":
                look_ahead = False
                for k in range(idx + 1, min(idx + 4, len(text_lines))):
                    if row_re.match(text_lines[k].strip()):
                        look_ahead = True
                        break
                if not look_ahead:
                    break
                else:
                    continue
            if footer_markers.search(line):
                break
            m = row_re.match(line)
            if m:
                rows.append(
                    {
                        "ligand_id": ligand_id,
                        "receptor_id": receptor_id,
                        "mode": int(m.group(1)),
                        "affinity": float(m.group(2)),
                        "rmsd_lb": float(m.group(3)),
                        "rmsd_ub": float(m.group(4)),
                    }
                )
                continue
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 4:
                try:
                    rows.append(
                        {
                            "ligand_id": ligand_id,
                            "receptor_id": receptor_id,
                            "mode": int(parts[0]),
                            "affinity": float(parts[1]),
                            "rmsd_lb": float(parts[2]),
                            "rmsd_ub": float(parts[3]),
                        }
                    )
                    continue
                except Exception:
                    continue
            continue
        return rows

    def scores_to_csv(
        self,
        log_path: Union[str, Path],
        csv_path: Union[str, Path],
        ligand_path: Optional[Union[str, Path]] = None,
        receptor_path: Optional[Union[str, Path]] = None,
        append: bool = False,
    ) -> None:
        rows = self.parse_scores_from_log(
            log_path, ligand_path=ligand_path, receptor_path=receptor_path
        )
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(csv_path, mode, encoding="utf-8") as fh:
            if not append:
                fh.write("ligand_id,receptor_id,mode,affinity,rmsd_lb,rmsd_ub\n")
            for r in rows:
                fh.write(
                    f"{r['ligand_id']},{r['receptor_id']},{r['mode']},{r['affinity']},{r['rmsd_lb']},{r['rmsd_ub']}\n"
                )

    def scores_as_dataframe(
        self,
        log_path: Union[str, Path],
        ligand_path: Optional[Union[str, Path]] = None,
        receptor_path: Optional[Union[str, Path]] = None,
    ):
        rows = self.parse_scores_from_log(
            log_path, ligand_path=ligand_path, receptor_path=receptor_path
        )
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise ImportError(
                "pandas is required for DataFrame output. Install with `pip install pandas`."
            ) from e
        if not rows:
            return pd.DataFrame(
                columns=[
                    "ligand_id",
                    "receptor_id",
                    "mode",
                    "affinity",
                    "rmsd_lb",
                    "rmsd_ub",
                ]
            )
        return pd.DataFrame(
            rows,
            columns=[
                "ligand_id",
                "receptor_id",
                "mode",
                "affinity",
                "rmsd_lb",
                "rmsd_ub",
            ],
        )
