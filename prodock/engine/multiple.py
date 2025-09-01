# prodock/dock/multiple.py
"""
MultipleDock (enhanced)
-----------------------

Batch-run docking for a receptor and a folder of ligand PDBQT files.

Features added:
- verbose levels: 0 (silent), 1 (tqdm only), 2+ (detailed prints)
- backend mapping: "vina"/"smina" -> VinaDock; "qvina"/"qvina-w"/"binary" -> BinaryDock
- skip_existing: skip ligands whose output file already exists
- max_retries with exponential backoff
- threaded parallelism (n_workers)
- filter_pattern to select ligands by glob
- convenience setters, better default behavior
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore

from prodock.engine.vina import VinaDock
from prodock.engine.binary import BinaryDock

logger = logging.getLogger("prodock.dock.multiple")
logger.addHandler(logging.NullHandler())


@dataclass
class DockResult:
    ligand_path: Path
    out_path: Optional[Path] = None
    log_path: Optional[Path] = None
    scores: Optional[List[float]] = field(default_factory=list)
    best_score: Optional[float] = None
    status: str = "pending"  # pending | ok | failed | skipped
    error: Optional[str] = None
    attempts: int = 0


class MultipleDock:
    def __init__(
        self,
        receptor: Union[str, Path],
        ligand_dir: Union[str, Path],
        backend: str = "vina",
        filter_pattern: str = "*.pdbqt",
    ) -> None:
        """
        :param receptor: receptor PDBQT path
        :param ligand_dir: folder containing ligand PDBQT files
        :param backend: "vina" | "smina" | "qvina" | "qvina-w" | "binary"
        :param filter_pattern: glob pattern to select ligand files inside ligand_dir
        """
        self.receptor = Path(receptor)
        if not self.receptor.exists():
            raise FileNotFoundError(f"Receptor not found: {self.receptor}")

        self.ligand_dir = Path(ligand_dir)
        if not self.ligand_dir.exists():
            raise FileNotFoundError(f"Ligand directory not found: {self.ligand_dir}")

        self.backend = (backend or "vina").lower()
        valid_backends = {"vina", "smina", "qvina", "qvina-w", "binary"}
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")

        # ensure wrapper classes are available if required
        if self.backend in {"vina"} and VinaDock is None:
            raise RuntimeError(
                "VinaDock wrapper not importable - ensure prodock.dock.VinaDock is available"
            )
        if self.backend in {"qvina", "qvina-w", "smina"} and BinaryDock is None:
            raise RuntimeError(
                "BinaryDock wrapper not importable - ensure prodock.dock.BinaryDock is available"
            )

        # docking params
        self._box_center: Optional[Tuple[float, float, float]] = None
        self._box_size: Optional[Tuple[float, float, float]] = None
        self._use_autobox: bool = False
        self._autobox_ligand: Optional[Path] = None
        self._autobox_padding: float = 4.0

        self._exhaustiveness: int = 8
        self._num_modes: int = 9
        self._cpu: int = 1
        self._seed: Optional[int] = None
        self._extra_args: Dict = {}

        # IO defaults
        self.out_dir: Path = Path("./docked")
        self.log_dir: Path = self.out_dir / "logs"
        self.pose_suffix: str = "_docked.pdbqt"
        self.log_suffix: str = ".log"

        # parallel & retry config
        self._n_workers: int = 1
        self._max_retries: int = 2
        self._retry_backoff: float = 1.5  # multiplier
        self._timeout: Optional[float] = None  # not used directly (wrappers may accept)

        # operation flags
        self._skip_existing: bool = True
        self._verbose: int = 1  # default: show tqdm
        self._filter_pattern = filter_pattern

        # internal state
        self._ligands: List[Path] = sorted(self.ligand_dir.glob(self._filter_pattern))
        self.results: List[DockResult] = []

    # -------------------------
    # configuration setters
    # -------------------------
    def set_box(
        self, center: Tuple[float, float, float], size: Tuple[float, float, float]
    ) -> "MultipleDock":
        self._box_center = tuple(float(x) for x in center)
        self._box_size = tuple(float(x) for x in size)
        self._use_autobox = False
        return self

    def enable_autobox(
        self, reference_ligand: Union[str, Path], padding: float = 4.0
    ) -> "MultipleDock":
        self._use_autobox = True
        self._autobox_ligand = Path(reference_ligand)
        self._autobox_padding = float(padding)
        return self

    def set_exhaustiveness(self, ex: int) -> "MultipleDock":
        self._exhaustiveness = int(ex)
        return self

    def set_num_modes(self, n: int) -> "MultipleDock":
        self._num_modes = int(n)
        return self

    def set_cpu(self, cpu: int) -> "MultipleDock":
        self._cpu = int(cpu)
        return self

    def set_seed(self, seed: Optional[int]) -> "MultipleDock":
        self._seed = int(seed) if seed is not None else None
        return self

    def set_extra_args(self, **kwargs) -> "MultipleDock":
        self._extra_args.update(kwargs)
        return self

    def set_out_dirs(
        self, out_dir: Union[str, Path], log_dir: Optional[Union[str, Path]] = None
    ) -> "MultipleDock":
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if log_dir is None:
            self.log_dir = self.out_dir / "logs"
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self

    def set_workers(self, n_workers: int) -> "MultipleDock":
        self._n_workers = max(1, int(n_workers))
        return self

    def set_skip_existing(self, skip: bool) -> "MultipleDock":
        self._skip_existing = bool(skip)
        return self

    def set_max_retries(self, max_retries: int, backoff: float = 1.5) -> "MultipleDock":
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff = float(backoff)
        return self

    def set_timeout(self, seconds: Optional[float]) -> "MultipleDock":
        self._timeout = None if seconds is None else float(seconds)
        return self

    def set_verbose(self, verbose: int) -> "MultipleDock":
        """0: silent, 1: tqdm only, 2+: detailed prints"""
        self._verbose = max(0, int(verbose))
        return self

    def set_filter_pattern(self, glob_pat: str) -> "MultipleDock":
        self._filter_pattern = str(glob_pat)
        self._ligands = sorted(self.ligand_dir.glob(self._filter_pattern))
        return self

    # -------------------------
    # utilities
    # -------------------------
    def _validate_run(self) -> None:
        if not self._use_autobox and (
            self._box_center is None or self._box_size is None
        ):
            raise RuntimeError(
                "Docking box not defined. Call set_box(...) or enable_autobox(...)."
            )
        # refresh ligands list in case pattern changed
        self._ligands = sorted(self.ligand_dir.glob(self._filter_pattern))
        if not self._ligands:
            raise RuntimeError(
                f"No ligand files found in {self.ligand_dir} using pattern {self._filter_pattern}."
            )

    def _make_out_paths(self, ligand_path: Path) -> Tuple[Path, Path]:
        name = ligand_path.stem
        out_path = self.out_dir / f"{name}{self.pose_suffix}"
        log_path = self.log_dir / f"{name}{self.log_suffix}"
        return out_path, log_path

    def _should_skip(self, out_path: Path) -> bool:
        if not self._skip_existing:
            return False
        return out_path.exists()

    def _log(self, *args, **kwargs) -> None:
        # central print helper for verbose>=2
        if self._verbose >= 2:
            print(*args, **kwargs)

    # -------------------------
    # core docking single-ligand
    # -------------------------
    def _dock_single(self, ligand_path: Path) -> DockResult:
        res = DockResult(ligand_path=ligand_path)
        out_path, log_path = self._make_out_paths(ligand_path)
        res.attempts = 0

        # skip pre-check
        if self._should_skip(out_path):
            res.status = "skipped"
            res.out_path = out_path
            res.log_path = log_path
            return res

        # perform up to max_retries+1 attempts
        attempt = 0
        while attempt <= self._max_retries:
            attempt += 1
            res.attempts = attempt
            try:
                if self.backend in {"vina"}:
                    sf = "vina" if self.backend == "vina" else "vinardo"
                    docker = VinaDock(sf_name=sf, cpu=self._cpu, seed=self._seed or 0)
                    docker.set_receptor(str(self.receptor))
                    if self._use_autobox:
                        # prefer wrapper API; if not present, raise early
                        if hasattr(docker, "enable_autobox"):
                            docker.enable_autobox(
                                str(self._autobox_ligand), padding=self._autobox_padding
                            )
                        else:
                            raise RuntimeError(
                                "VinaDock wrapper lacks enable_autobox method; compute box manually."
                            )
                    else:
                        docker.define_box(center=self._box_center, size=self._box_size)
                    docker.set_ligand(str(ligand_path))
                    # call docking; wrapper expected to provide dock() compatible with earlier snippet
                    docker.dock(
                        exhaustiveness=self._exhaustiveness, n_poses=self._num_modes
                    )
                    # write outputs
                    if hasattr(docker, "write_poses"):
                        docker.write_poses(str(out_path))
                    else:
                        # some wrappers expose poses as attribute
                        poses = getattr(docker, "poses", None)
                        if poses:
                            with out_path.open("wb") as fh:
                                fh.write(poses)
                    if hasattr(docker, "write_log"):
                        docker.write_log(str(log_path))
                    else:
                        log_content = getattr(docker, "log", None)
                        if log_content:
                            with log_path.open("w", encoding="utf-8") as fh:
                                fh.write(str(log_content))
                    scores = getattr(docker, "scores", None) or []
                    best = getattr(docker, "get_best", lambda: None)()
                    best_score = None
                    if best is not None:
                        best_score = (
                            best[0] if isinstance(best, (list, tuple)) else best
                        )
                    res.scores = list(scores)
                    res.best_score = (
                        float(best_score) if best_score is not None else None
                    )

                else:
                    # BinaryDock path
                    # backend could be "qvina" or "qvina-w" or "binary"
                    docker = BinaryDock(self.backend)
                    docker.set_receptor(str(self.receptor))
                    if self._use_autobox:
                        if hasattr(docker, "enable_autobox"):
                            docker.enable_autobox(
                                str(self._autobox_ligand), padding=self._autobox_padding
                            )
                        else:
                            raise RuntimeError(
                                "BinaryDock wrapper lacks enable_autobox API; compute box manually."
                            )
                    else:
                        docker.set_box(center=self._box_center, size=self._box_size)
                    docker.set_ligand(str(ligand_path))
                    docker.set_out(str(out_path))
                    docker.set_log(str(log_path))
                    docker.set_exhaustiveness(self._exhaustiveness)
                    docker.set_num_modes(self._num_modes)
                    docker.set_cpu(self._cpu)
                    if self._seed is not None:
                        docker.set_seed(self._seed)
                    # run
                    docker.run()
                    scores = getattr(docker, "scores", None) or []
                    best = (
                        getattr(docker, "best", None)
                        or getattr(docker, "get_best", lambda: None)()
                    )
                    res.scores = list(scores)
                    res.best_score = float(best) if best is not None else None

                res.out_path = out_path
                res.log_path = log_path
                res.status = "ok"
                res.error = None
                return res

            except Exception as exc:
                # On failure, either retry or fail after max attempts
                res.status = "failed"
                res.error = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "Dock attempt %d failed for %s: %s", attempt, ligand_path, exc
                )
                if attempt > self._max_retries:
                    return res
                # backoff before retrying
                wait = self._retry_backoff ** (attempt - 1)
                # cap wait to a sensible max (e.g., 30s * attempts)
                wait = min(wait, 30.0 * attempt)
                if self._verbose >= 2:
                    print(
                        f"[retry] ligand={ligand_path.name} attempt={attempt}/{self._max_retries} sleeping {wait:.1f}s"
                    )
                time.sleep(wait)

        return res  # fallback

    # -------------------------
    # run method (parallel / sequential)
    # -------------------------
    def run(
        self,
        n_workers: Optional[int] = None,
        ligands: Optional[Sequence[Union[str, Path]]] = None,
    ) -> List[DockResult]:
        """
        Run docking.

        :param n_workers: number of parallel workers (threads). None uses configured self._n_workers.
        :param ligands: optional explicit ligand file list (overrides ligand_dir glob).
        :returns: list of DockResult objects (order is best-effort; results appended as completed)
        """
        if ligands is not None:
            self._ligands = [Path(x) for x in ligands]
        else:
            self._ligands = sorted(self.ligand_dir.glob(self._filter_pattern))

        self._validate_run()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        n_workers = self._n_workers if n_workers is None else max(1, int(n_workers))
        self._n_workers = n_workers

        total = len(self._ligands)
        self.results = []

        # Helper to show tqdm only when requested and tqdm available
        use_tqdm = (self._verbose >= 1) and (tqdm is not None)

        if n_workers <= 1:
            # sequential
            iterator = self._ligands
            if use_tqdm:
                iterator = tqdm(iterator, desc="Docking", unit="ligand", ncols=80)
            for lig in iterator:
                if self._verbose >= 2:
                    print(f"[dock] {lig.name}")
                res = self._dock_single(lig)
                # print summary line if verbose>=2
                if self._verbose >= 2:
                    if res.status == "ok":
                        print(
                            f"[ok] {lig.name} best={res.best_score} out={res.out_path}"
                        )
                    elif res.status == "skipped":
                        print(f"[skipped] {lig.name} (exists: {res.out_path})")
                    else:
                        print(f"[fail] {lig.name} err={res.error}")
                self.results.append(res)
            return self.results

        # parallel
        # We'll submit tasks and optionally show a tqdm bar that updates as futures complete
        futures = []
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            for lig in self._ligands:
                futures.append(exe.submit(self._dock_single, lig))

            if use_tqdm:
                pbar = tqdm(total=total, desc="Docking", unit="ligand", ncols=80)
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as exc:
                        logger.exception("Unhandled exception during docking: %s", exc)
                        # create a failure result wrapper
                        res = DockResult(
                            ligand_path=Path("<unknown>"),
                            status="failed",
                            error=str(exc),
                        )
                    self.results.append(res)
                    pbar.update(1)
                    if self._verbose >= 2:
                        if res.status == "ok":
                            print(
                                f"[ok] {res.ligand_path.name} best={res.best_score} out={res.out_path}"
                            )
                        elif res.status == "skipped":
                            print(f"[skipped] {res.ligand_path.name}")
                        else:
                            print(f"[fail] {res.ligand_path.name} err={res.error}")
                pbar.close()
            else:
                # no tqdm: just collect results as they finish
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as exc:
                        logger.exception("Unhandled exception during docking: %s", exc)
                        res = DockResult(
                            ligand_path=Path("<unknown>"),
                            status="failed",
                            error=str(exc),
                        )
                    self.results.append(res)

        return self.results

    # -------------------------
    # output helpers
    # -------------------------
    def write_summary(self, path: Union[str, Path] = None) -> Path:
        import csv

        path = (
            Path(path) if path is not None else (self.out_dir / "docking_summary.csv")
        )
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "ligand",
                    "out_path",
                    "log_path",
                    "best_score",
                    "status",
                    "error",
                    "attempts",
                ]
            )
            for r in self.results:
                writer.writerow(
                    [
                        str(r.ligand_path),
                        str(r.out_path) if r.out_path else "",
                        str(r.log_path) if r.log_path else "",
                        r.best_score if r.best_score is not None else "",
                        r.status,
                        r.error or "",
                        r.attempts,
                    ]
                )
        logger.info("Wrote docking summary to %s", path)
        return path

    def get_best_per_ligand(self) -> Dict[str, Optional[float]]:
        return {r.ligand_path.name: r.best_score for r in self.results}

    def get_ok_results(self) -> List[DockResult]:
        return [r for r in self.results if r.status == "ok"]
