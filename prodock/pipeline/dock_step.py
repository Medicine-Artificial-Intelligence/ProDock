from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import logging
from joblib import Parallel, delayed

# Project-level imports at top to satisfy flake8/E402
from prodock.dock.engine import SingleDock
from prodock.pipeline.utils import iter_progress

# Optional tqdm helpers (progress for parallel execution)
try:
    from tqdm.contrib.concurrent import process_map, thread_map  # type: ignore
except Exception:  # pragma: no cover - tqdm optional
    process_map = None  # type: ignore
    thread_map = None  # type: ignore

logger = logging.getLogger("prodock.pipeline.dock_step")


def _set_if_present(obj: Any, method_name: str, *args, **kwargs) -> None:
    """
    Try to call ``obj.method_name(*args, **kwargs)`` if the attribute exists.

    This is used so the worker can use chaining APIs on different dock classes
    without asserting exact method presence.
    """
    method = getattr(obj, method_name, None)
    if callable(method):
        method(*args, **kwargs)


def _module_level_dock_worker(
    dock_cls: Type,
    backend: str,
    receptor_path: str,
    lig_path: str,
    out_path: str,
    log_path: str,
    cfg_box: Dict[str, float],
    exhaustiveness: int,
    num_modes: int,
    cpu: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """
    Module-level worker that constructs a ``dock_cls`` and runs docking.

    This function is intentionally module-level so it can be pickled when used
    by joblib or tqdm.contrib.concurrent in parallel execution.

    :param dock_cls: Dock class constructor (e.g., prodock.dock.engine.SingleDock).
    :param backend: Backend engine name passed to the constructor.
    :param receptor_path: Path to prepared receptor.
    :param lig_path: Path to ligand file.
    :param out_path: Path to write docked poses.
    :param log_path: Path to write textual log of docking.
    :param cfg_box: Vina-style box dictionary with keys center_x/y/z and size_x/y/z.
    :param exhaustiveness: Docking exhaustiveness.
    :param num_modes: Number of output poses.
    :param cpu: CPU threads / cores to request.
    :param seed: Optional seed for deterministic runs.
    :returns: Dict with keys ``ligand``, ``out``, ``log``, ``success``.
    """
    lig_path_p = Path(lig_path)
    out_path_p = Path(out_path)
    log_path_p = Path(log_path)

    if not cfg_box:
        raise RuntimeError("cfg_box not provided to worker.")

    # Instantiate dock object (SingleDock-like)
    dock_obj = dock_cls(backend)

    # Best-effort setter calls (some SingleDock implementations chain)
    _set_if_present(dock_obj, "set_receptor", str(receptor_path))
    _set_if_present(dock_obj, "set_ligand", str(lig_path_p))
    _set_if_present(dock_obj, "set_out", str(out_path_p))
    _set_if_present(dock_obj, "set_log", str(log_path_p))

    center = (cfg_box["center_x"], cfg_box["center_y"], cfg_box["center_z"])
    size = (cfg_box["size_x"], cfg_box["size_y"], cfg_box["size_z"])
    _set_if_present(dock_obj, "set_box", center, size)

    _set_if_present(dock_obj, "set_exhaustiveness", exhaustiveness)
    _set_if_present(dock_obj, "set_num_modes", num_modes)
    _set_if_present(dock_obj, "set_cpu", cpu)
    if seed is not None:
        _set_if_present(dock_obj, "set_seed", seed)

    # Some SingleDock implementations chain (.set_... returning self).
    # Attempt to call '.run' or '.execute' using whichever exists.
    success = False
    try:
        if hasattr(dock_obj, "run") and callable(getattr(dock_obj, "run")):
            dock_obj.run()
        elif hasattr(dock_obj, "execute") and callable(getattr(dock_obj, "execute")):
            dock_obj.execute()
        else:
            # As a last resort, attempt to call a "run_from_config" if present.
            runcfg = getattr(dock_obj, "run_from_config", None)
            if callable(runcfg):
                # We do not expect a config file here, but some implementations
                # accept an inline dict. Try best-effort.
                try:
                    dock_obj.run_from_config(
                        {
                            "receptor": str(receptor_path),
                            "ligand": str(lig_path_p),
                            "out": str(out_path_p),
                            "log": str(log_path_p),
                        }
                    )
                except Exception:
                    raise RuntimeError(
                        "Dock class missing runnable entrypoint (run/execute/run_from_config)."
                    )
        logger.info("Docked %s -> %s", lig_path_p.name, out_path_p)
        success = True
    except Exception:
        logger.exception("Dock failed for %s", lig_path_p.name)
        success = False

    return {
        "ligand": str(lig_path_p),
        "out": str(out_path_p),
        "log": str(log_path_p),
        "success": success,
    }


def _unpack_for_map(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Small adapter to allow mapping functions to accept tuple input."""
    return _module_level_dock_worker(*args)


class DockStep:
    """
    Docking step wrapper around a SingleDock-like engine.

    This class prefers per-ligand docking (recommended). It can use
    ``tqdm.contrib.concurrent`` mapping helpers to show progress bars during
    parallel execution; when those helpers are not available it falls back to
    joblib or sequential execution.

    :param dock_cls: Class to instantiate for docking; defaults to
                     :class:`prodock.dock.engine.SingleDock`.
                     Inject a test-friendly class in unit tests (see example).
    :param prefer_tqdm_for_parallel: If True and tqdm.contrib.concurrent is
                     available, prefer using it for progress bars during
                     parallel execution (still obeys ``n_jobs``).
    """

    def __init__(
        self, dock_cls: Type = SingleDock, prefer_tqdm_for_parallel: bool = True
    ) -> None:
        self._dock_cls = dock_cls
        self._prefer_tqdm = bool(prefer_tqdm_for_parallel)

    def _instance_worker(
        self,
        backend: str,
        receptor_path: str,
        lig_path: str,
        out_path: str,
        log_path: str,
        cfg_box: Dict[str, float],
        exhaustiveness: int,
        num_modes: int,
        cpu: int,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """
        Instance-level worker used for sequential (n_jobs == 1) runs so the
        test-suite can inject a lightweight dock class without patching.

        This mirrors the module-level worker but uses ``self._dock_cls``.
        """
        return _module_level_dock_worker(
            self._dock_cls,
            backend,
            receptor_path,
            lig_path,
            out_path,
            log_path,
            cfg_box,
            exhaustiveness,
            num_modes,
            cpu,
            seed,
        )

    def dock(
        self,
        receptor_path: str,
        ligand_dir: str,
        output_modes_dir: str,
        logs_dir: str,
        cfg_box: Dict[str, float],
        backend: str = "smina",
        exhaustiveness: int = 8,
        num_modes: int = 9,
        cpu: int = 4,
        seed: Optional[int] = None,
        ligand_glob: str = "*.pdbqt",
        batch_size: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 0,
        parallel_prefer: str = "threads",
    ) -> List[Dict[str, Any]]:
        """
        Dock all ligands in ``ligand_dir`` against the prepared receptor.

        :param receptor_path: Path to the prepared receptor (e.g., PDBQT).
        :param ligand_dir: Directory that contains ligand files matching ``ligand_glob``.
        :param output_modes_dir: Directory to write docked pose files.
        :param logs_dir: Directory to write textual logs for each ligand.
        :param cfg_box: Vina-style dictionary with keys ``center_x``, ``center_y``,
                        ``center_z`` and ``size_x``, ``size_y``, ``size_z``.
        :param backend: Docking engine name (defaults to ``"smina"``).
        :param exhaustiveness: Exhaustiveness/effort parameter (engine-specific).
        :param num_modes: Number of output poses to write per ligand.
        :param cpu: Number of CPU threads to request for the engine.
        :param seed: Optional random seed passed to the engine.
        :param ligand_glob: Glob pattern to select ligand files (default ``"*.pdbqt"``).
        :param batch_size: If provided, group ligands into batches of this size.
        :param n_jobs: Parallel workers (``1`` = sequential, >1 = parallel).
        :param verbose: 0=silent, 1=log-only, 2+=progress bars where available.
        :param parallel_prefer: ``"threads"`` or ``"processes"`` (joblib hint).
        :returns: List of dicts with docking results (one dict per ligand).
        :raises FileNotFoundError: If no ligand files match ``ligand_glob``.
        :raises RuntimeError: If ``cfg_box`` is empty or missing required keys.

        **Notes**
        - For unit tests / local development you can pass ``n_jobs=1`` and inject
          a lightweight dock class into the constructor. That avoids calling
          real binaries while exercising the pipeline end-to-end.

        **Example (sequential, testable)**

        >>> from tempfile import TemporaryDirectory
        >>> from pathlib import Path
        >>> # Minimal fake dock class used for testing (no external binaries)
        >>> class MiniSingleDock:
        ...     def __init__(self, engine):
        ...         self.engine = engine
        ...     def set_receptor(self, r): pass
        ...     def set_ligand(self, l): pass
        ...     def set_out(self, o): self._out = o
        ...     def set_log(self, L): self._log = L
        ...     def set_box(self, center, size): pass
        ...     def set_exhaustiveness(self, e): pass
        ...     def set_num_modes(self, m): pass
        ...     def set_cpu(self, c): pass
        ...     def run(self): Path(self._out).write_text("POSE"); Path(self._log).write_text("LOG")
        >>> tmp = TemporaryDirectory()
        >>> td = Path(tmp.name)
        >>> (td / "lig").mkdir()
        >>> lig = td / "lig" / "L1.pdbqt"; lig.write_text("LIG")
        >>> d = DockStep(dock_cls=MiniSingleDock)
        >>> cfg = {"center_x": 0, "center_y": 0, "center_z": 0, "size_x": 20, "size_y": 20, "size_z": 20}
        >>> res = d.dock(str(td / "receptor.pdbqt"), str(td / "lig"), str(td / "modes"),
        str(td / "logs"), cfg, n_jobs=1, verbose=0)
        >>> isinstance(res, list)
        True
        """
        receptor_path = Path(receptor_path)
        ligand_dir = Path(ligand_dir)
        output_modes_dir = Path(output_modes_dir)
        logs_dir = Path(logs_dir)

        output_modes_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        ligand_files = sorted(ligand_dir.glob(ligand_glob))
        if not ligand_files:
            raise FileNotFoundError(
                f"No ligand files matching {ligand_glob} in {ligand_dir}"
            )

        # Validate cfg_box
        required_keys = {
            "center_x",
            "center_y",
            "center_z",
            "size_x",
            "size_y",
            "size_z",
        }
        if not cfg_box or not required_keys.issubset(set(cfg_box.keys())):
            raise RuntimeError(
                "cfg_box must be provided with center_x/y/z and size_x/y/z keys."
            )

        # Utility to create worker tuple for module-level worker when needed
        def _make_module_worker_args(lig_path: Path):
            ligand_name = lig_path.stem
            out_path = output_modes_dir / f"{ligand_name}.pdbqt"
            log_path = logs_dir / f"{ligand_name}.txt"
            return (
                self._dock_cls,
                backend,
                str(receptor_path),
                str(lig_path),
                str(out_path),
                str(log_path),
                cfg_box,
                exhaustiveness,
                num_modes,
                cpu,
                seed,
            )

        # PER-LIGAND mode (preferred)
        if batch_size is None:
            worker_args = [_make_module_worker_args(lig) for lig in ligand_files]

            # If verbose >=2 and tqdm.contrib.concurrent available and requested, use mapping helpers
            can_use_tqdm_map = (
                n_jobs != 1
                and verbose >= 2
                and self._prefer_tqdm
                and (
                    (parallel_prefer == "processes" and process_map is not None)
                    or (parallel_prefer == "threads" and thread_map is not None)
                )
            )

            if can_use_tqdm_map:
                mapping = process_map if parallel_prefer == "processes" else thread_map
                # mapping expects func and iterable (we pass tuples), use unpack wrapper
                results = mapping(
                    _unpack_for_map, worker_args, max_workers=n_jobs, chunksize=1
                )
            else:
                if n_jobs == 1:
                    # sequential — use instance worker (so tests can inject lightweight dock_cls)
                    results = []
                    for lig in iter_progress(
                        ligand_files,
                        verbose=verbose,
                        desc="Docking ligands",
                        unit="ligand",
                    ):
                        args = _make_module_worker_args(lig)
                        # call instance worker to use self._dock_cls (testable)
                        results.append(
                            self._instance_worker(*args[1:])
                        )  # drop dock_cls param
                else:
                    # joblib parallel using module-level worker (picklable)
                    results = Parallel(n_jobs=n_jobs, prefer=parallel_prefer)(
                        delayed(_module_level_dock_worker)(*args)
                        for args in worker_args
                    )

        else:
            # BATCH mode: group ligand_files into batches without spaces around slice colon
            if batch_size >= len(ligand_files):
                batches = [ligand_files]
            else:
                # fmt: off
                batches = [
                    ligand_files[i: i + batch_size]
                    for i in range(0, len(ligand_files), batch_size)
                ]
                # fmt: on

            def _run_batch(batch_files: List[Path]) -> List[Dict[str, Any]]:
                outs = []
                for lig in batch_files:
                    outs.append(
                        self._instance_worker(
                            backend,
                            str(receptor_path),
                            str(lig),
                            str(output_modes_dir / f"{lig.stem}.pdbqt"),
                            str(logs_dir / f"{lig.stem}.txt"),
                            cfg_box,
                            exhaustiveness,
                            num_modes,
                            cpu,
                            seed,
                        )
                    )
                return outs

            can_use_tqdm_map = (
                n_jobs != 1
                and verbose >= 2
                and self._prefer_tqdm
                and (
                    (parallel_prefer == "processes" and process_map is not None)
                    or (parallel_prefer == "threads" and thread_map is not None)
                )
            )

            if can_use_tqdm_map:
                mapping = process_map if parallel_prefer == "processes" else thread_map
                results_nested = mapping(
                    _run_batch, batches, max_workers=n_jobs, chunksize=1
                )
                results = [item for sub in results_nested for item in sub]
            else:
                if n_jobs == 1:
                    results = []
                    for b in iter_progress(
                        batches, verbose=verbose, desc="Docking batches", unit="batch"
                    ):
                        results.extend(_run_batch(b))
                else:
                    nested = Parallel(n_jobs=n_jobs, prefer=parallel_prefer)(
                        delayed(_run_batch)(b) for b in batches
                    )
                    results = [item for sub in nested for item in sub]

        return list(results)
