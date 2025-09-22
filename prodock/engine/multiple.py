# prodock/engine/multiple.py
from __future__ import annotations

import csv
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from prodock.engine.vina import VinaDock
from prodock.engine.binary import BinaryDock

logger = logging.getLogger("prodock.engine.multiple")
logger.addHandler(logging.NullHandler())


@dataclass
class DockResult:
    """Container for one ligand docking outcome."""

    ligand_path: Path
    out_path: Optional[Path] = None
    log_path: Optional[Path] = None
    # Unified scores: list of (affinity, rmsd_lb, rmsd_ub)
    scores: List[Tuple[float, float, float]] = field(default_factory=list)
    best_score: Optional[float] = None
    status: str = "pending"  # pending | ok | failed | skipped
    error: Optional[str] = None
    attempts: int = 0


class MultipleDock:
    """
    Batch docking over a directory or explicit list of ligands using either
    :class:`prodock.engine.vina.VinaDock` (Vina API) or
    :class:`prodock.engine.binary.BinaryDock` (smina/qvina).

    Performance optimization
    ------------------------
    - When `cache_per_worker=True` (default), each worker/thread **caches one backend**:
      - **Vina**: the worker builds a single `VinaDock`, calls `set_receptor()` and
        `define_box()` **once**, then **reuses the precomputed maps** for all ligands
        handled by that worker. This eliminates repeated "Computing Vina grid ... done."
        lines and saves significant time.
      - **Binary** (smina/qvina): binaries still compute their own internal grids per
        subprocess run, but per-worker caching avoids repeated executable resolution,
        `--help` probing, and repeated option wiring.

    Usage styles
    ------------
    1) **One-shot** (mirrors your VinaDock compact init + autorun):
       >>> md = MultipleDock(
       ...     receptor="Data/testcase/dock/receptor/5N2F.pdbqt",
       ...     ligand_dir="Data/testcase/dock/ligand",
       ...     backend="vina",
       ...     ligand_format="pdbqt",
       ...     center=(32.5, 13.0, 133.75),
       ...     size=(22.5, 23.5, 22.5),
       ...     exhaustiveness=8,
       ...     n_poses=9,
       ...     cpu=4,
       ...     out_dir="./Data/testcase/dock/out",
       ...     log_dir="./Data/testcase/dock/logs",
       ...     n_workers=4,
       ...     skip_existing=True,
       ...     verbose=1,
       ...     autorun=True,
       ...     autowrite=True,
       ... )
       >>> print("best per ligand:", md.best_per_ligand)  # doctest: +SKIP

    2) **Staged chaining** (fluent API, like your VinaDock chain):
       >>> md = (MultipleDock(
       ...         receptor="Data/testcase/dock/receptor/5N2F.pdbqt",
       ...         ligand_dir="Data/testcase/dock/ligand",
       ...         backend="vina",
       ...         ligand_format="pdbqt")
       ...       .set_box((32.5,13.0,133.75), (22.5,23.5,22.5))
       ...       .set_exhaustiveness(8)
       ...       .set_num_modes(9)
       ...       .set_cpu(4)
       ...       .set_out_dirs("./Data/testcase/dock/out", "./Data/testcase/dock/logs")
       ...       .set_workers(4)
       ...       .set_skip_existing(True)
       ...       .set_verbose(1)
       ...       .run()
       ... )
       >>> print(len(md.ok_results))  # doctest: +SKIP

    Parameters
    ----------
    receptor : str | Path
        Receptor **PDBQT** path (required).
    ligand_dir : str | Path, optional
        Directory with ligands (discovery per `ligand_format` + `filter_pattern`).
    ligands : sequence of paths, optional
        Explicit ligands list (overrides folder discovery).
    backend : str | Callable[[], Any] | VinaDock | BinaryDock, optional
        "vina" | "smina" | "qvina" | "qvina-w" | "binary" | "<custom exe>" |
        factory callable returning a backend instance | prebuilt backend instance.
    ligand_format : {"pdbqt","sdf","mol2","auto","any"}, optional
        Default "pdbqt". Use "sdf"/"mol2" with smina.
    filter_pattern : str, optional
        Glob (e.g., "LIG_*") without extension, default "*".
    center, size : tuple, optional
        Docking box; required if `autobox=False`.
    autobox : bool, optional
        If True, use autobox with `autobox_ref` and `autobox_padding` (BinaryDock/smina).
        VinaDock does **not** support autobox.
    autobox_ref : str | Path, optional
        Reference file for autobox.
    autobox_padding : float, optional
        Padding Å for autobox (default 4.0).
    exhaustiveness : int, optional
        Search exhaustiveness (default 8).
    n_poses : int, optional
        Number of poses (default 9).
    cpu : int, optional
        Worker CPU threads to request (if supported).
    seed : int, optional
        RNG seed (if supported).
    out_dir, log_dir : str | Path, optional
        Output root and logs directory (defaults: "./docked" and "<out_dir>/logs").
    pose_suffix, log_suffix : str, optional
        Filenames suffixes (defaults: "_docked.pdbqt", ".log").
    n_workers : int, optional
        Thread workers (default 1). For Vina, each worker builds one maps cache.
    skip_existing : bool, optional
        Skip ligands whose output already exists (default True).
    max_retries : int, optional
        Retries on failure (default 2).
    retry_backoff : float, optional
        Exponential backoff base (default 1.5).
    timeout : float, optional
        Per-run timeout for BinaryDock subprocesses.
    verbose : int, optional
        0 silent, 1 tqdm, 2+ per-ligand prints.
    cache_per_worker : bool, optional
        Cache a prepared backend per worker (default True).
    autorun, autowrite : bool, optional
        Convenience flags: run on construction and write summary CSV.

    Notes
    -----
    - For Vina + `n_workers=1`, maps are computed **once total**; for `n_workers>1`,
      maps computed **once per worker**.
    """

    def __init__(
        self,
        receptor: Union[str, Path],
        ligand_dir: Optional[Union[str, Path]] = None,
        *,
        ligands: Optional[Sequence[Union[str, Path]]] = None,
        backend: Union[str, Callable[[], Any], VinaDock, BinaryDock] = "vina",
        ligand_format: str = "pdbqt",
        filter_pattern: str = "*",
        center: Optional[Tuple[float, float, float]] = None,
        size: Optional[Tuple[float, float, float]] = None,
        autobox: bool = False,
        autobox_ref: Optional[Union[str, Path]] = None,
        autobox_padding: float = 4.0,
        exhaustiveness: int = 8,
        n_poses: int = 9,
        cpu: Optional[int] = None,
        seed: Optional[int] = None,
        out_dir: Union[str, Path] = "./docked",
        log_dir: Optional[Union[str, Path]] = None,
        pose_suffix: str = "_docked.pdbqt",
        log_suffix: str = ".log",
        n_workers: int = 1,
        skip_existing: bool = True,
        max_retries: int = 2,
        retry_backoff: float = 1.5,
        timeout: Optional[float] = None,
        verbose: int = 1,
        cache_per_worker: bool = True,
        autorun: bool = False,
        autowrite: bool = False,
    ):
        # Receptor
        self.receptor = Path(receptor)
        if not self.receptor.exists():
            raise FileNotFoundError(f"Receptor not found: {self.receptor}")
        if self.receptor.suffix.lower() != ".pdbqt":
            raise ValueError("Receptor must be a PDBQT file.")

        # Ligands source
        self.ligand_dir: Optional[Path] = Path(ligand_dir) if ligand_dir else None
        if self.ligand_dir and not self.ligand_dir.exists():
            raise FileNotFoundError(f"Ligand directory not found: {self.ligand_dir}")
        self._explicit_ligands: Optional[List[Path]] = (
            [Path(p) for p in ligands] if ligands else None
        )

        # Backend spec
        self._backend_spec = backend

        # Discovery
        self._ligand_format = ligand_format.lower().strip()
        self._filter_pattern = filter_pattern

        # Box / autobox
        self._box_center = tuple(map(float, center)) if center is not None else None
        self._box_size = tuple(map(float, size)) if size is not None else None
        self._use_autobox = bool(autobox)
        self._autobox_ref = Path(autobox_ref) if autobox_ref is not None else None
        self._autobox_padding = float(autobox_padding)

        # Params
        self._exhaustiveness = int(exhaustiveness)
        self._num_modes = int(n_poses)
        self._cpu = None if cpu is None else int(cpu)
        self._seed = None if seed is None else int(seed)

        # IO
        self.out_dir = Path(out_dir)
        self.log_dir = Path(log_dir) if log_dir is not None else (self.out_dir / "logs")
        self.pose_suffix = str(pose_suffix)
        self.log_suffix = str(log_suffix)

        # Runtime
        self._n_workers = max(1, int(n_workers))
        self._skip_existing = bool(skip_existing)
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff = float(retry_backoff)
        self._timeout = None if timeout is None else float(timeout)
        self._verbose = max(0, int(verbose))
        self._cache_per_worker = bool(cache_per_worker)

        # State
        self._ligands: List[Path] = []
        self.results: List[DockResult] = []
        self._summary_path: Optional[Path] = None

        # Thread-local cache for per-worker backends
        self._tls = threading.local()

        # Discovery
        if self._explicit_ligands is not None:
            self._ligands = [p for p in self._explicit_ligands if p.exists()]
        else:
            self._refresh_ligands()

        # Autorun
        if autorun:
            self.run(n_workers=self._n_workers)
            if autowrite:
                self.write_summary()

    # -------------------- Discovery -------------------- #
    def _join_glob(self, ext_wo_dot: str) -> str:
        base = self._filter_pattern
        if base.endswith(f".{ext_wo_dot}"):
            return base
        return f"{base}.{ext_wo_dot}"

    def _refresh_ligands(self) -> None:
        if self._explicit_ligands is not None:
            self._ligands = [p for p in self._explicit_ligands if p.exists()]
            return
        if not self.ligand_dir:
            self._ligands = []
            return
        ext_sets: Dict[str, List[Path]] = {
            "pdbqt": sorted(self.ligand_dir.glob(self._join_glob("pdbqt"))),
            "sdf": sorted(self.ligand_dir.glob(self._join_glob("sdf"))),
            "mol2": sorted(self.ligand_dir.glob(self._join_glob("mol2"))),
        }
        fmt = self._ligand_format
        if fmt == "pdbqt":
            self._ligands = ext_sets["pdbqt"]
        elif fmt == "sdf":
            self._ligands = ext_sets["sdf"]
        elif fmt == "mol2":
            self._ligands = ext_sets["mol2"]
        elif fmt == "auto":
            self._ligands = ext_sets["pdbqt"] or ext_sets["sdf"] or ext_sets["mol2"]
        elif fmt == "any":
            self._ligands = ext_sets["pdbqt"] + ext_sets["sdf"] + ext_sets["mol2"]
        else:
            raise ValueError(
                "ligand_format must be one of: 'pdbqt', 'sdf', 'mol2', 'auto', 'any'"
            )

    def _validate_ready(self) -> None:
        if not self._use_autobox and (
            self._box_center is None or self._box_size is None
        ):
            raise RuntimeError(
                "Docking box not defined. Call set_box(...) or enable_autobox(...)."
            )
        if not self._ligands:
            raise RuntimeError(
                "No ligands discovered. Check ligand_dir/ligands and ligand_format/filter_pattern."
            )
        if isinstance(self._backend_spec, str) and self._backend_spec.lower() in {
            "vina",
            "qvina",
            "qvina-w",
        }:
            wrong = [p for p in self._ligands if p.suffix.lower() != ".pdbqt"]
            if wrong:
                raise ValueError(
                    "Selected backend requires PDBQT ligands. Offending: "
                    + ", ".join(x.name for x in wrong)
                )

    # -------------------- Backend factory & caching -------------------- #
    def _backend_key(self) -> str:
        """A stable key describing the prepared backend configuration."""
        spec = self._backend_spec
        if isinstance(spec, str):
            name = spec.lower()
        elif isinstance(spec, (VinaDock, BinaryDock)):
            name = type(spec).__name__.lower()
        else:
            name = "factory"
        # We include receptor + box/autobox in the signature to ensure correctness
        return "|".join(
            [
                name,
                str(self.receptor.resolve()),
                f"autobox={int(self._use_autobox)}",
                f"ref={str(self._autobox_ref) if self._autobox_ref else ''}",
                f"pad={self._autobox_padding}",
                f"center={self._box_center}",
                f"size={self._box_size}",
                f"cpu={self._cpu}",
                f"seed={self._seed}",
                f"exh={self._exhaustiveness}",
                f"nposes={self._num_modes}",
            ]
        )

    def _get_cached_backend(self) -> Any:
        """Return a cached backend for this worker (thread), if available & compatible."""
        if not self._cache_per_worker:
            return None
        if not hasattr(self._tls, "cached"):
            self._tls.cached = None  # type: ignore[attr-defined]
            self._tls.sig = None  # type: ignore[attr-defined]
        if self._tls.sig == self._backend_key():  # type: ignore[attr-defined]
            return self._tls.cached  # type: ignore[attr-defined]
        return None

    def _set_cached_backend(self, backend: Any) -> None:
        if not self._cache_per_worker:
            return
        self._tls.cached = backend  # type: ignore[attr-defined]
        self._tls.sig = self._backend_key()  # type: ignore[attr-defined]

    def _create_backend_fresh(self) -> Any:
        """Create and fully prepare a fresh backend according to the current settings."""
        spec = self._backend_spec

        # If a callable factory is provided, call it (recommended for parallel)
        if callable(spec) and not isinstance(spec, str):
            backend = spec()
        elif isinstance(spec, (VinaDock, BinaryDock)):
            backend = spec  # reuse (only safe sequentially)
        elif isinstance(spec, str):
            key = spec.lower()
            if key == "vina":
                backend = VinaDock(
                    sf_name="vina",
                    cpu=(self._cpu or 1),
                    seed=self._seed,
                    verbosity=(
                        1 if self._verbose == 1 else (2 if self._verbose >= 2 else 0)
                    ),
                )
            else:
                backend = BinaryDock(binary_name=key)
        else:
            # Unknown type → assume BinaryDock with custom executable name
            backend = BinaryDock(str(spec))

        # Prepare common state
        if isinstance(backend, VinaDock) or (type(backend).__name__ == "VinaDock"):
            backend.set_receptor(str(self.receptor))
            if self._use_autobox:
                # Not supported in Vina Python API — enforce explicit box
                raise RuntimeError(
                    "Autobox is not supported by VinaDock; provide center/size."
                )
            backend.define_box(center=self._box_center, size=self._box_size)
        else:
            # BinaryDock path
            backend.set_receptor(str(self.receptor))
            if self._use_autobox:
                backend.enable_autobox(
                    str(self._autobox_ref), padding=self._autobox_padding
                )
            else:
                backend.set_box(self._box_center, self._box_size)
            if self._cpu is not None:
                backend.set_cpu(self._cpu)
            if self._seed is not None:
                backend.set_seed(self._seed)
            backend.set_exhaustiveness(self._exhaustiveness)
            backend.set_num_modes(self._num_modes)
            if self._timeout is not None and hasattr(backend, "set_timeout"):
                backend.set_timeout(self._timeout)

        return backend

    def _get_backend(self) -> Any:
        """
        Retrieve a backend for the current worker. If caching is enabled,
        return the cached backend or create+cache a new one with maps already
        computed (Vina) or options set (Binary).
        """
        cached = self._get_cached_backend()
        if cached is not None:
            return cached
        backend = self._create_backend_fresh()
        self._set_cached_backend(backend)
        return backend

    # -------------------- Fluent setters -------------------- #
    def set_backend(
        self, backend: Union[str, Callable[[], Any], VinaDock, BinaryDock]
    ) -> "MultipleDock":
        self._backend_spec = backend
        # invalidate cache if backend changed
        if hasattr(self._tls, "sig"):
            self._tls.sig = None  # type: ignore[attr-defined]
        return self

    def set_box(
        self, center: Tuple[float, float, float], size: Tuple[float, float, float]
    ) -> "MultipleDock":
        self._box_center = tuple(float(x) for x in center)
        self._box_size = tuple(float(x) for x in size)
        self._use_autobox = False
        if hasattr(self._tls, "sig"):
            self._tls.sig = None  # invalidate cache
        return self

    def enable_autobox(
        self, reference_ligand: Union[str, Path], padding: float = 4.0
    ) -> "MultipleDock":
        self._use_autobox = True
        self._autobox_ref = Path(reference_ligand)
        self._autobox_padding = float(padding)
        if hasattr(self._tls, "sig"):
            self._tls.sig = None
        return self

    def set_exhaustiveness(self, ex: int) -> "MultipleDock":
        self._exhaustiveness = int(ex)
        if hasattr(self._tls, "sig"):
            self._tls.sig = None
        return self

    def set_num_modes(self, n: int) -> "MultipleDock":
        self._num_modes = int(n)
        if hasattr(self._tls, "sig"):
            self._tls.sig = None
        return self

    def set_cpu(self, cpu: int) -> "MultipleDock":
        self._cpu = int(cpu)
        if hasattr(self._tls, "sig"):
            self._tls.sig = None
        return self

    def set_seed(self, seed: Optional[int]) -> "MultipleDock":
        self._seed = int(seed) if seed is not None else None
        if hasattr(self._tls, "sig"):
            self._tls.sig = None
        return self

    def set_out_dirs(
        self, out_dir: Union[str, Path], log_dir: Optional[Union[str, Path]] = None
    ) -> "MultipleDock":
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir) if log_dir else (self.out_dir / "logs")
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
        if hasattr(self._tls, "sig"):
            self._tls.sig = None
        return self

    def set_verbose(self, verbose: int) -> "MultipleDock":
        self._verbose = max(0, int(verbose))
        return self

    def set_filter_pattern(self, glob_pat: str) -> "MultipleDock":
        self._filter_pattern = str(glob_pat)
        self._refresh_ligands()
        return self

    def set_ligand_format(self, ligand_format: str) -> "MultipleDock":
        self._ligand_format = ligand_format.lower().strip()
        self._refresh_ligands()
        return self

    def set_ligand_dir(self, ligand_dir: Union[str, Path]) -> "MultipleDock":
        self.ligand_dir = Path(ligand_dir)
        if not self.ligand_dir.exists():
            raise FileNotFoundError(f"Ligand directory not found: {self.ligand_dir}")
        self._refresh_ligands()
        return self

    def set_ligands(self, ligands: Sequence[Union[str, Path]]) -> "MultipleDock":
        self._explicit_ligands = [Path(p) for p in ligands]
        self._refresh_ligands()
        return self

    # -------------------- Helpers -------------------- #
    def _make_out_paths(self, ligand_path: Path) -> Tuple[Path, Path]:
        name = ligand_path.stem
        return (
            self.out_dir / f"{name}{self.pose_suffix}",
            self.log_dir / f"{name}{self.log_suffix}",
        )

    def _should_skip(self, out_path: Path) -> bool:
        return self._skip_existing and out_path.exists()

    # -------------------- Single-ligand execution -------------------- #
    def _dock_single(self, ligand_path: Path) -> DockResult:
        res = DockResult(ligand_path=ligand_path)
        out_path, log_path = self._make_out_paths(ligand_path)
        res.out_path, res.log_path = out_path, log_path

        if self._should_skip(out_path):
            res.status = "skipped"
            return res

        attempt = 0
        while attempt <= self._max_retries:
            attempt += 1
            res.attempts = attempt
            try:
                backend = self._get_backend()

                # VinaDock-like flow
                if isinstance(backend, VinaDock) or (
                    type(backend).__name__ == "VinaDock"
                ):
                    backend.set_ligand(str(ligand_path))
                    backend.dock(
                        exhaustiveness=self._exhaustiveness, n_poses=self._num_modes
                    )
                    if hasattr(backend, "write_poses"):
                        backend.write_poses(str(out_path))
                    if hasattr(backend, "write_log"):
                        backend.write_log(str(log_path))
                    # gather scores
                    sc = getattr(backend, "scores", None) or []
                    res.scores = list(sc)
                    # best may be a tuple from get_best or a property best_score
                    best = getattr(backend, "best_score", None) or (
                        getattr(backend, "get_best", lambda: None)()
                    )
                    if best is None:
                        res.best_score = None
                    else:
                        res.best_score = (
                            float(best[0])
                            if isinstance(best, (list, tuple))
                            else float(best)
                        )

                else:
                    # BinaryDock-like flow
                    backend.set_ligand(str(ligand_path))
                    backend.set_out(str(out_path))
                    backend.set_log(str(log_path))
                    backend.run()
                    # parse scores from log (unified)
                    if hasattr(backend, "parse_scores_from_log"):
                        rows = backend.parse_scores_from_log(log_path)
                        res.scores = [
                            (
                                float(r["affinity"]),
                                float(r["rmsd_lb"]),
                                float(r["rmsd_ub"]),
                            )
                            for r in rows
                        ]
                        res.best_score = res.scores[0][0] if res.scores else None
                    else:
                        sc = getattr(backend, "scores", None) or []
                        res.scores = list(sc)
                        best = getattr(backend, "best_score", None) or (
                            getattr(backend, "get_best", lambda: None)()
                        )
                        res.best_score = (
                            (
                                float(best[0])
                                if isinstance(best, (list, tuple))
                                else float(best)
                            )
                            if best is not None
                            else None
                        )

                res.status = "ok"
                res.error = None
                return res

            except Exception as exc:
                res.status = "failed"
                res.error = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "Dock attempt %d failed for %s: %s", attempt, ligand_path, exc
                )
                if attempt > self._max_retries:
                    return res
                wait = min(30.0 * attempt, (self._retry_backoff ** (attempt - 1)))
                if self._verbose >= 2:
                    print(
                        f"[retry] {ligand_path.name} attempt={attempt}/{self._max_retries} sleep {wait:.1f}s"
                    )
                time.sleep(wait)

        return res  # fallback

    # -------------------- Public run -------------------- #
    def run(
        self,
        *,
        n_workers: Optional[int] = None,
        ligands: Optional[Sequence[Union[str, Path]]] = None,
    ) -> "MultipleDock":
        """
        Execute docking for all discovered/explicit ligands.

        Parameters
        ----------
        n_workers : int, optional
            Thread count (defaults to configured value).
        ligands : sequence of paths, optional
            Override the ligands set for this run.

        Returns
        -------
        MultipleDock
            self (inspect :pyattr:`results`, :pyattr:`best_per_ligand`).
        """
        if ligands is not None:
            self.set_ligands(ligands)

        self._refresh_ligands()
        self._validate_ready()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        n_workers = self._n_workers if n_workers is None else max(1, int(n_workers))
        self._n_workers = n_workers  # affects backend cache key (per-worker maps)
        total = len(self._ligands)
        self.results = []

        use_tqdm = (self._verbose >= 1) and (tqdm is not None)

        if n_workers <= 1:
            iterator: Iterable[Path] = self._ligands
            if use_tqdm:
                iterator = tqdm(iterator, desc="Docking", unit="ligand", ncols=80)
            for lig in iterator:
                if self._verbose >= 2:
                    print(f"[dock] {lig.name}")
                res = self._dock_single(lig)
                self.results.append(res)
                if self._verbose >= 2:
                    self._print_one_line(res)
            return self

        # parallel
        futures = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for lig in self._ligands:
                futures.append(pool.submit(self._dock_single, lig))

            if use_tqdm:
                pbar = tqdm(total=total, desc="Docking", unit="ligand", ncols=80)
                for fut in as_completed(futures):
                    res = self._resolve_future(fut)
                    self.results.append(res)
                    pbar.update(1)
                    if self._verbose >= 2:
                        self._print_one_line(res)
                pbar.close()
            else:
                for fut in as_completed(futures):
                    res = self._resolve_future(fut)
                    self.results.append(res)

        return self

    def _resolve_future(self, fut) -> DockResult:
        try:
            return fut.result()
        except Exception as exc:  # pragma: no cover
            logger.exception("Unhandled exception during docking: %s", exc)
            return DockResult(
                ligand_path=Path("<unknown>"), status="failed", error=str(exc)
            )

    def _print_one_line(self, r: DockResult) -> None:  # pragma: no cover
        if r.status == "ok":
            print(f"[ok] {r.ligand_path.name} best={r.best_score} out={r.out_path}")
        elif r.status == "skipped":
            print(f"[skipped] {r.ligand_path.name} (exists)")
        else:
            print(f"[fail] {r.ligand_path.name} err={r.error}")

    # -------------------- Outputs -------------------- #
    def write_summary(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Write a CSV summary of results. Returns the written path.
        """
        path = (
            Path(path) if path is not None else (self.out_dir / "docking_summary.csv")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
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
                        f"{r.best_score:.3f}" if r.best_score is not None else "",
                        r.status,
                        r.error or "",
                        r.attempts,
                    ]
                )
        self._summary_path = path
        logger.info("Wrote docking summary to %s", path)
        return path

    # -------------------- Convenience accessors -------------------- #
    @property
    def ligands(self) -> List[Path]:
        return list(self._ligands)

    @property
    def best_per_ligand(self) -> Dict[str, Optional[float]]:
        return {r.ligand_path.name: r.best_score for r in self.results}

    @property
    def ok_results(self) -> List[DockResult]:
        return [r for r in self.results if r.status == "ok"]

    @property
    def failed_results(self) -> List[DockResult]:
        return [r for r in self.results if r.status == "failed"]

    @property
    def summary_path(self) -> Optional[Path]:
        return self._summary_path

    def help(self) -> None:  # pragma: no cover
        print(self.__doc__ or "MultipleDock: batch docking wrapper.")
