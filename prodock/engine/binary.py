import os
import shutil
import subprocess
import re
from pathlib import Path
from typing import Union, Optional, Sequence, Dict, Any, Iterable, Tuple, List


class BinaryDock:
    """
    Run external docking binaries (smina, qvina, qvina-w, etc.) with a modern OOP wrapper.

    Chainable API (all mutators return self). Execute via `.run()` or `.run_many()`,
    then read results from the `.result` property.

    Auto-detects which CLI flags the chosen binary supports by parsing `<exe> --help`,
    so only valid options are passed (prevents spurious failures across binaries).

    :param binary_name: Name or path to the docking binary (e.g., "smina", "qvina", "qvina-w").
    :param binary_dir: Directory to search for the binary if not on PATH (default: "prodock/binary").
    """

    # -------------------- lifecycle -------------------- #
    def __init__(
        self,
        binary_name: str = "smina",
        binary_dir: Union[str, Path] = "prodock/binary",
    ):
        self.binary_name = str(binary_name)
        self.binary_dir = Path(binary_dir)

        # core IO
        self.ligand_path: Optional[Path] = None
        self.receptor_path: Optional[Path] = None
        self.flex_path: Optional[Path] = None  # flexible sidechains file (if supported)
        self.out_path: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self.config_path: Optional[Path] = None

        # box
        self.autobox: bool = False
        self.autobox_reference: Optional[Path] = (
            None  # file for --autobox_ligand (smina)
        )
        self.autobox_add: Optional[float] = None  # padding for autobox (smina)
        self.center: Optional[Tuple[float, float, float]] = None
        self.size: Optional[Tuple[float, float, float]] = None

        # core numeric options
        self.exhaustiveness: Optional[int] = 8
        self.num_modes: Optional[int] = 9
        self.energy_range: Optional[float] = None
        self.spacing: Optional[float] = None
        self.cpu: Optional[int] = None
        self.seed: Optional[int] = None

        # scoring / behavior flags (only added if supported)
        self.scoring: Optional[str] = None  # e.g., "vinardo" (smina)
        self.local_only: bool = False
        self.randomize_only: bool = False
        self.minimize: bool = False

        # raw passthroughs
        self._flags: list[str] = []  # e.g., ["--noxyz"]
        self._options: list[str] = (
            []
        )  # e.g., ["--weight_gauss1", "0.5", "--weight_repulsion", "0.8"]

        # subprocess config
        self.timeout: Optional[float] = None
        self.env: Optional[dict] = None
        self.cwd: Optional[Union[str, Path]] = None
        self.dry_run: bool = False

        # resolution & capability probe
        self._exe: Optional[str] = None
        self._help_text: Optional[str] = None
        self._resolve_executable()
        self._probe_capabilities()

        # last result
        self._last_result: Optional[Dict[str, Any]] = None

    # -------------------- internal helpers -------------------- #
    def _resolve_executable(self) -> Optional[str]:
        """
        Resolve the executable path: exact path → PATH → binary_dir.
        Stores the resolved path in self._exe (or leaves None).
        """
        maybe = Path(self.binary_name)
        if maybe.exists() and os.access(str(maybe), os.X_OK):
            self._exe = str(maybe.resolve())
            return self._exe

        exe = shutil.which(self.binary_name)
        if exe:
            self._exe = exe
            return exe

        candidate = self.binary_dir / self.binary_name
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            self._exe = str(candidate.resolve())
            return self._exe

        # also try common alternative suffixes for qvina
        alt = shutil.which(self.binary_name + "02") or shutil.which(
            self.binary_name + "-w"
        )
        if alt:
            self._exe = alt
            return alt

        self._exe = None
        return None

    def _probe_capabilities(self) -> None:
        """
        Run `<exe> --help` once and cache the text. Used to decide which options
        we should pass to the binary safely.
        """
        self._help_text = None
        if not self._exe:
            return
        try:
            # small timeout so help probing cannot hang indefinitely
            proc = subprocess.run(
                [self._exe, "--help"], capture_output=True, text=True, timeout=5
            )
            txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
            self._help_text = txt.lower()
        except Exception:
            self._help_text = ""

    def _supports(self, opt: str) -> bool:
        """
        Return True if `opt` appears in the probed help text.
        If probing failed, be conservative: only pass very common options.
        """
        if self._help_text is None:
            # not probed: allow only the safest common options
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

    def _binary_is(self, name_substring: str) -> bool:
        if not self._exe:
            return False
        return name_substring.lower() in Path(self._exe).name.lower()

    # -------------------- validation -------------------- #
    def _validate_inputs(self) -> None:
        if self._exe is None:
            raise RuntimeError(
                f"Docking binary not found: tried '{self.binary_name}' and '{self.binary_dir / self.binary_name}'"
            )
        if self.receptor_path is None:
            raise ValueError("receptor_path not set (call .set_receptor(...))")
        if self.ligand_path is None:
            raise ValueError("ligand_path not set (call .set_ligand(...))")
        if self.out_path is None:
            raise ValueError("out_path not set (call .set_out(...))")
        if self.log_path is None:
            raise ValueError("log_path not set (call .set_log(...))")

        # autobox vs box
        if self.autobox:
            if not self._supports("--autobox_ligand"):
                raise ValueError(
                    "This binary does not support --autobox_ligand; disable autobox or use smina."
                )
            if self.autobox_reference is None:
                raise ValueError(
                    "autobox=True requires .enable_autobox(<reference_file>)"
                )
        else:
            if self.center is None or self.size is None:
                raise ValueError(
                    "When autobox is False, you must set the box via .set_box(center, size)."
                )

        # existence
        for p in [self.receptor_path, self.ligand_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing input file: {p}")
        if self.flex_path and not self.flex_path.exists():
            raise FileNotFoundError(
                f"Flexible residue file not found: {self.flex_path}"
            )
        if self.config_path and not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    # -------------------- setters (chainable) -------------------- #
    def set_binary(
        self, binary_name: str, binary_dir: Union[str, Path] = None
    ) -> "BinaryDock":
        """
        Set a new binary and (optionally) search directory; re-probes capabilities.

        :param binary_name: Binary name or path.
        :param binary_dir: Optional directory to search (defaults to previous).
        :return: self
        """
        self.binary_name = str(binary_name)
        if binary_dir is not None:
            self.binary_dir = Path(binary_dir)
        self._resolve_executable()
        self._probe_capabilities()
        return self

    def set_receptor(self, receptor_path: Union[str, Path]) -> "BinaryDock":
        """:param receptor_path: Receptor PDBQT path. :return: self"""
        self.receptor_path = Path(receptor_path)
        return self

    def set_ligand(self, ligand_path: Union[str, Path]) -> "BinaryDock":
        """:param ligand_path: Ligand file path (PDBQT recommended). :return: self"""
        self.ligand_path = Path(ligand_path)
        return self

    def set_flex(self, flex_path: Union[str, Path]) -> "BinaryDock":
        """
        :param flex_path: Flexible sidechains file (if supported by the binary).
        :return: self
        """
        self.flex_path = Path(flex_path)
        return self

    def set_out(self, out_path: Union[str, Path]) -> "BinaryDock":
        """:param out_path: Output PDBQT path. :return: self"""
        self.out_path = Path(out_path)
        return self

    def set_log(self, log_path: Union[str, Path]) -> "BinaryDock":
        """:param log_path: Log file path (binary log if supported; driver appends stdout/stderr). :return: self"""
        self.log_path = Path(log_path)
        return self

    def set_config(self, config_path: Union[str, Path]) -> "BinaryDock":
        """:param config_path: Vina/Smina config file path. :return: self"""
        self.config_path = Path(config_path)
        return self

    def set_box(
        self, center: Tuple[float, float, float], size: Tuple[float, float, float]
    ) -> "BinaryDock":
        """
        :param center: Box center (x, y, z).
        :param size: Box size (sx, sy, sz) in Å.
        :return: self
        """
        self.center = (float(center[0]), float(center[1]), float(center[2]))
        self.size = (float(size[0]), float(size[1]), float(size[2]))
        return self

    def enable_autobox(
        self, reference_file: Union[str, Path], padding: Optional[float] = None
    ) -> "BinaryDock":
        """
        Enable autobox mode (if binary supports --autobox_ligand).

        :param reference_file: File passed to --autobox_ligand.
        :param padding: Optional padding passed to --autobox_add (Å) if supported.
        :return: self
        """
        self.autobox = True
        self.autobox_reference = Path(reference_file)
        self.autobox_add = None if padding is None else float(padding)
        return self

    def disable_autobox(self) -> "BinaryDock":
        """:return: self"""
        self.autobox = False
        self.autobox_reference = None
        self.autobox_add = None
        return self

    def set_exhaustiveness(self, value: Optional[int]) -> "BinaryDock":
        """:param value: Exhaustiveness (None to omit). :return: self"""
        self.exhaustiveness = None if value is None else int(value)
        return self

    def set_num_modes(self, value: Optional[int]) -> "BinaryDock":
        """:param value: Number of output poses (None to omit). :return: self"""
        self.num_modes = None if value is None else int(value)
        return self

    def set_energy_range(self, value: Optional[float]) -> "BinaryDock":
        """:param value: Energy range in kcal/mol (None to omit). :return: self"""
        self.energy_range = None if value is None else float(value)
        return self

    def set_spacing(self, value: Optional[float]) -> "BinaryDock":
        """:param value: Grid spacing in Å (None to omit). :return: self"""
        self.spacing = None if value is None else float(value)
        return self

    def set_cpu(self, value: Optional[int]) -> "BinaryDock":
        """:param value: Number of CPU threads (None to omit). :return: self"""
        self.cpu = None if value is None else int(value)
        return self

    def set_seed(self, value: Optional[int]) -> "BinaryDock":
        """:param value: RNG seed (None to omit). :return: self"""
        self.seed = None if value is None else int(value)
        return self

    # def set_scoring(self, name: Optional[str]) -> "BinaryDock":
    #     """
    #     :param name: Scoring function name (e.g., "vinardo" for smina). None to omit.
    #     :return: self
    #     """
    #     self.scoring = None if name is None else str(name)
    #     return self

    def set_local_only(self, enabled: bool = True) -> "BinaryDock":
        """:param enabled: True → pass --local_only if supported. :return: self"""
        self.local_only = bool(enabled)
        return self

    def set_randomize_only(self, enabled: bool = True) -> "BinaryDock":
        """:param enabled: True → pass --randomize_only if supported. :return: self"""
        self.randomize_only = bool(enabled)
        return self

    # def set_minimize(self, enabled: bool = True) -> "BinaryDock":
    #     """:param enabled: True → pass --minimize if supported. :return: self"""
    #     self.minimize = bool(enabled)
    #     return self

    def add_flags(self, flags: Sequence[str]) -> "BinaryDock":
        """
        Add raw boolean flags (no values), e.g., ["--cpu_affinity"].

        :param flags: Sequence of flag strings (each should start with "-").
        :return: self
        """
        self._flags.extend(map(str, flags))
        return self

    def add_options(self, kv_pairs: Sequence[Union[str, int, float]]) -> "BinaryDock":
        """
        Add raw key-value options, e.g., ["--weight_gauss1", 0.5, "--weight_repulsion", 0.8].

        :param kv_pairs: Flat sequence of alternating option and value(s).
        :return: self
        """
        self._options.extend(map(lambda x: str(x), kv_pairs))
        return self

    def set_timeout(self, seconds: Optional[float]) -> "BinaryDock":
        """:param seconds: Subprocess timeout in seconds (None to disable). :return: self"""
        self.timeout = None if seconds is None else float(seconds)
        return self

    def set_env(self, env: Optional[dict]) -> "BinaryDock":
        """:param env: Extra environment variables dict (merged with os.environ). :return: self"""
        self.env = None if env is None else dict(env)
        return self

    def set_cwd(self, cwd: Optional[Union[str, Path]]) -> "BinaryDock":
        """:param cwd: Working directory for subprocess. :return: self"""
        self.cwd = None if cwd is None else str(cwd)
        return self

    def set_dry_run(self, dry: bool = True) -> "BinaryDock":
        """:param dry: If True, do not execute; just fill `.result` with the planned command. :return: self"""
        self.dry_run = bool(dry)
        return self

    # -------------------- command assembly -------------------- #
    def _build_args(self) -> list[str]:
        """
        Build a safe argument vector based on supported options of the selected binary.
        """
        exe = str(self._exe)
        args: list[str] = [exe]

        # Standard IO
        if self._supports("--receptor"):
            args += ["--receptor", str(self.receptor_path)]
        else:
            # Some binaries also accept "-r"
            args += ["-r", str(self.receptor_path)]

        if self._supports("--ligand"):
            args += ["--ligand", str(self.ligand_path)]
        else:
            # Some binaries accept "-l"
            args += ["-l", str(self.ligand_path)]

        # Flexible residues file
        if self.flex_path and self._supports("--flex"):
            args += ["--flex", str(self.flex_path)]

        # Output
        if self._supports("--out"):
            args += ["--out", str(self.out_path)]
        else:
            args += ["-o", str(self.out_path)]

        # Log (only if supported; smina supports --log; qvina often doesn't)
        if self.log_path and self._supports("--log"):
            args += ["--log", str(self.log_path)]

        # Config
        if self.config_path and self._supports("--config"):
            args += ["--config", str(self.config_path)]

        # CPU / seed
        if self.cpu is not None and self._supports("--cpu"):
            args += ["--cpu", str(self.cpu)]
        if self.seed is not None and self._supports("--seed"):
            args += ["--seed", str(self.seed)]

        # Box vs autobox
        if self.autobox:
            # only added if _supports("--autobox_ligand") passed in validation
            args += ["--autobox_ligand", str(self.autobox_reference)]
            if self.autobox_add is not None and self._supports("--autobox_add"):
                args += ["--autobox_add", str(self.autobox_add)]
        else:
            # explicit box
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
                    # some variants accept "-c x y z"
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
                    # some variants accept "-s sx sy sz"
                    args += [
                        "-s",
                        str(self.size[0]),
                        str(self.size[1]),
                        str(self.size[2]),
                    ]

        # Search / output controls
        if self.exhaustiveness is not None and self._supports("--exhaustiveness"):
            args += ["--exhaustiveness", str(self.exhaustiveness)]
        if self.num_modes is not None and self._supports("--num_modes"):
            args += ["--num_modes", str(self.num_modes)]
        if self.energy_range is not None and self._supports("--energy_range"):
            args += ["--energy_range", str(self.energy_range)]
        if self.spacing is not None and self._supports("--spacing"):
            args += ["--spacing", str(self.spacing)]

        # Scoring + behavioral flags (only when supported)
        if self.scoring and self._supports("--scoring"):
            args += ["--scoring", str(self.scoring)]
        if self.local_only and self._supports("--local_only"):
            args += ["--local_only"]
        if self.randomize_only and self._supports("--randomize_only"):
            args += ["--randomize_only"]
        if self.minimize and self._supports("--minimize"):
            args += ["--minimize"]

        # User-provided raw flags/options (we filter by support when they look like options)
        for flg in self._flags:
            # if it looks like an option and not supported, skip; otherwise include
            if flg.startswith("-") and not self._supports(flg):
                continue
            args.append(flg)
        i = 0
        while i < len(self._options):
            opt = self._options[i]
            if opt.startswith("-") and not self._supports(opt):
                # skip this option and its immediate value if present (best-effort)
                i += 2
                continue
            args.append(opt)
            i += 1

        return args

    # -------------------- execution -------------------- #
    def run(self) -> "BinaryDock":
        """
        Execute the built command once. Creates parent directories, appends driver logs.

        :return: self
        """
        # refresh exe/help in case binary changed after construction
        self._resolve_executable()
        self._probe_capabilities()
        self._validate_inputs()

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

        # prepare environment
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

        # Append driver-captured stdout/stderr to the log file (do NOT clobber binary's own log)
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
        """
        Batch docking: iterate ligands, reusing receptor/box settings.

        :param ligands: Iterable of ligand file paths.
        :param out_dir: Directory for output PDBQT files.
        :param log_dir: Directory for log files.
        :param autobox_refs: Optional iterable of autobox reference files
                             (one per ligand) used when autobox=True.
        :param overwrite: If False, skip ligands whose outputs already exist.
        :return: self
        """
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

            # if autobox enabled and per-ligand refs provided, update the reference
            if self.autobox and refs_iter is not None:
                try:
                    ref = next(refs_iter)
                    self.enable_autobox(ref, padding=self.autobox_add)
                except StopIteration:
                    pass

            self.run()

        return self

    # -------------------- accessors -------------------- #
    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """
        :return: Last run result dict:
                 { rc, stdout, stderr, out, log, called, dry_run? }
        """
        return self._last_result

    def __repr__(self) -> str:
        return (
            f"<BinaryDock exe={self._exe or self.binary_name} "
            f"ligand={self.ligand_path} receptor={self.receptor_path} "
            f"autobox={self.autobox} cpu={self.cpu} seed={self.seed}>"
        )

    # small helper
    def help(self) -> None:
        """Print a minimal usage example."""
        print(
            "Example:\n"
            "  bd = BinaryDock('smina')\n"
            "  (bd.set_receptor('rec.pdbqt')\n"
            "     .set_ligand('lig.pdbqt')\n"
            "     .set_out('out/lig_docked.pdbqt')\n"
            "     .set_log('out/lig.log')\n"
            "     .set_box((x,y,z),(sx,sy,sz))\n"
            "     .set_exhaustiveness(16)\n"
            "     .set_num_modes(9)\n"
            "     .set_cpu(8)\n"
            "     .run())\n"
        )

    # ---------------- score parsing & export ---------------- #
    def _infer_id(self, path: Optional[Union[str, Path]]) -> str:
        """Infer identifier (stem) from path or return empty string."""
        if path is None:
            return ""
        return Path(path).stem

    def parse_scores_from_log(
        self,
        log_path: Union[str, Path],
        ligand_path: Optional[Union[str, Path]] = None,
        receptor_path: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse smina/Vina-style log file and return list of dicts:
        {ligand_id, receptor_id, mode, affinity, rmsd_lb, rmsd_ub}.

        :param log_path: path to log file (text)
        :param ligand_path: optional ligand path to infer ligand_id (overrides self.ligand_path)
        :param receptor_path: optional receptor path to infer receptor_id (overrides self.receptor_path)
        :return: list of parsed rows
        """
        lp = Path(log_path)
        if not lp.exists():
            raise FileNotFoundError(f"log file not found: {log_path}")
        text_lines = lp.read_text(errors="ignore").splitlines()

        # 1) find header line containing 'mode' and 'affinity' (robust, case-insensitive)
        header_idx = None
        for i, line in enumerate(text_lines):
            if re.search(r"\bmode\b", line, re.I) and re.search(
                r"\baffinity\b", line, re.I
            ):
                header_idx = i
                break

        # 2) fallback: look for a dashed separator that usually appears before numeric rows
        if header_idx is None:
            for i, line in enumerate(text_lines):
                if re.search(r"^-{3,}\+?-{3,}", line.strip()):
                    header_idx = i
                    break

        # 3) if still not found, return empty — nothing to parse
        if header_idx is None:
            return []

        # 4) find the separator line (e.g. '-----+------------+----------+----------') after header
        sep_idx = None
        for j in range(header_idx, min(header_idx + 6, len(text_lines))):
            L = text_lines[j].strip()
            # typical separators contain '-' and '+' (or many '-' only)
            if re.match(r"^[-\s\+]{5,}$", L) or re.search(r"-{3,}\+?-{3,}", L):
                sep_idx = j
                break

        # if separator found, data starts after it. Otherwise start after header and skip a couple incidental lines
        start = (sep_idx + 1) if sep_idx is not None else (header_idx + 1)
        # skip blank or purely separator/comment lines
        while start < len(text_lines) and (
            text_lines[start].strip() == ""
            or re.match(r"[-=+\s]+$", text_lines[start].strip())
        ):
            start += 1

        # 5) regex to match numeric rows
        # matches: mode  affinity  rmsd_lb  rmsd_ub  (whitespace separated)
        row_re = re.compile(
            r"^\s*(\d+)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*$"
        )

        rows: List[Dict[str, Any]] = []
        ligand_id = self._infer_id(ligand_path or self.ligand_path)
        receptor_id = self._infer_id(receptor_path or self.receptor_path)

        # 6) iterate lines and parse until we hit footer markers
        footer_markers = re.compile(
            r"(refine time|loop time|--- driver stderr ---|--- driver stdout ---)", re.I
        )
        for idx in range(start, len(text_lines)):
            line = text_lines[idx].rstrip()
            if line.strip() == "":
                # blank line — could be end of table; break conservatively if next non-blank is non-numeric
                # peek next few lines to see if more numeric rows exist; if not, break.
                look_ahead = False
                for k in range(idx + 1, min(idx + 4, len(text_lines))):
                    if row_re.match(text_lines[k].strip()):
                        look_ahead = True
                        break
                if not look_ahead:
                    break
                else:
                    continue

            # stop if we hit known footer text
            if footer_markers.search(line):
                break

            m = row_re.match(line)
            if m:
                mode = int(m.group(1))
                affinity = float(m.group(2))
                rmsd_lb = float(m.group(3))
                rmsd_ub = float(m.group(4))
                rows.append(
                    {
                        "ligand_id": ligand_id,
                        "receptor_id": receptor_id,
                        "mode": mode,
                        "affinity": affinity,
                        "rmsd_lb": rmsd_lb,
                        "rmsd_ub": rmsd_ub,
                    }
                )
                continue

            # Some logs omit decimals for rmsd columns or have extra spaces — try a more permissive parse:
            parts = re.split(r"\s{2,}", line.strip())  # split on 2+ spaces
            if len(parts) >= 4:
                try:
                    mode = int(parts[0])
                    affinity = float(parts[1])
                    rmsd_lb = float(parts[2])
                    rmsd_ub = float(parts[3])
                    rows.append(
                        {
                            "ligand_id": ligand_id,
                            "receptor_id": receptor_id,
                            "mode": mode,
                            "affinity": affinity,
                            "rmsd_lb": rmsd_lb,
                            "rmsd_ub": rmsd_ub,
                        }
                    )
                    continue
                except Exception:
                    # couldn't parse this line — skip it
                    continue

            # otherwise skip non-matching lines and keep scanning
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
        """
        Parse scores from log and write to CSV.

        :param log_path: path to smina log (one per ligand)
        :param csv_path: output CSV path
        :param ligand_path: optional ligand path to infer ligand_id
        :param receptor_path: optional receptor path to infer receptor_id
        :param append: if True, append to CSV instead of overwriting
        """
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
        """
        Return parsed scores as a pandas.DataFrame.

        :param log_path: path to smina log
        :param ligand_path: optional ligand path to infer ligand_id
        :param receptor_path: optional receptor path to infer receptor_id
        :return: pandas.DataFrame with columns [ligand_id,receptor_id,mode,affinity,rmsd_lb,rmsd_ub]
        :raises ImportError: if pandas is not installed
        """
        rows = self.parse_scores_from_log(
            log_path, ligand_path=ligand_path, receptor_path=receptor_path
        )
        try:
            import pandas as pd
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
