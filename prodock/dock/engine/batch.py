"""
BatchDock (workers use SingleDock inside worker).

This module implements a ProcessPool-based parallel executor which instantiates
SingleDock *inside* each worker process. That avoids pickling complex engine
objects while reusing the SingleDock facade.

Usage example
-------------
>>> from prodock.dock.engine.batch import BatchDock
>>> bd = BatchDock(engine="smina", n_jobs=4, progress=True)
>>> rows = [{"id":"L1","receptor":"rec.pdbqt","ligand":"lig1.pdbqt","center":(10,10,10),"size":(20,20,20)}]
>>> results = bd.run(rows, out_dir="out/docked", log_dir="out/logs")
"""

from __future__ import annotations
import concurrent.futures
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


from .config import BatchConfig

PathLike = Union[str, Path]
Vec3 = Tuple[float, float, float]


@dataclass
class DockTask:
    job_id: str
    receptor: str
    ligand: str
    center: Optional[Vec3] = None
    size: Optional[Vec3] = None
    engine_name: str = "vina"
    engine_mode: Optional[str] = None
    engine_options: Dict[str, Any] = None
    exhaustiveness: Optional[int] = None
    n_poses: Optional[int] = None
    cpu: Optional[int] = None
    seed: Optional[int] = None
    autobox_ref: Optional[str] = None
    autobox_pad: Optional[float] = None
    out_path: Optional[str] = None
    log_path: Optional[str] = None
    retries: int = 1
    timeout: Optional[float] = None
    tmp_dir: Optional[str] = None


@dataclass
class DockResult:
    job_id: str
    success: bool
    out_path: Optional[str]
    log_path: Optional[str]
    called: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    elapsed: Optional[float] = None


def _resolve_engine_key(engine_name: str, engine_mode: Optional[str]) -> str:
    if engine_name.lower() == "vina" and engine_mode:
        m = engine_mode.lower()
        if m in ("binding", "py", "python"):
            return "vina_binding"
        return "vina"
    return engine_name


def worker_process_job_using_singledock(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker entrypoint that builds a SingleDock inside the worker and uses it.

    This avoids pickling backend instances while reusing SingleDock's chainable API.
    """
    start = time.time()
    task = DockTask(**task_dict)

    # Resolve engine key / mode
    engine_key = _resolve_engine_key(task.engine_name, task.engine_mode)

    # Import SingleDock inside worker (local import avoids pickling issues)
    try:
        from prodock.dock.engine.single import SingleDock  # local import
    except Exception as e:
        tb = traceback.format_exc()
        return asdict(
            DockResult(
                job_id=task.job_id,
                success=False,
                out_path=task.out_path,
                log_path=task.log_path,
                called=None,
                error=f"Failed to import SingleDock: {e}",
                traceback=tb,
                elapsed=time.time() - start,
            )
        )

    attempt = 0
    last_exc = None
    while attempt < max(1, task.retries):
        attempt += 1
        try:
            # create SingleDock inside worker
            sd = SingleDock(engine=engine_key)

            # engine_options best-effort: set attributes if present
            if task.engine_options:
                for k, v in task.engine_options.items():
                    try:
                        setattr(sd._backend, k, v)
                    except Exception:
                        # ignore unknown options — engines should provide setters
                        pass

            # chainable configuration via SingleDock facade
            sd.set_receptor(task.receptor, validate=False)
            sd.set_ligand(task.ligand)
            if task.center and task.size:
                sd.set_box(tuple(task.center), tuple(task.size))
            if task.exhaustiveness is not None:
                sd.set_exhaustiveness(task.exhaustiveness)
            if task.n_poses is not None:
                sd.set_num_modes(task.n_poses)
            if task.cpu is not None:
                sd.set_cpu(task.cpu)
            if task.seed is not None:
                sd.set_seed(task.seed)
            if task.out_path:
                Path(task.out_path).parent.mkdir(parents=True, exist_ok=True)
                sd.set_out(task.out_path)
            if task.log_path:
                Path(task.log_path).parent.mkdir(parents=True, exist_ok=True)
                sd.set_log(task.log_path)
            if task.autobox_ref is not None:
                # this will raise on engines that don't support autobox (qvina/vina)
                sd.enable_autobox(task.autobox_ref, padding=task.autobox_pad)

            # Run, using sd.run() — which returns a SingleResult with artifacts
            res = sd.run(exhaustiveness=task.exhaustiveness, n_poses=task.n_poses)
            called = getattr(res.artifacts, "called", None)

            return asdict(
                DockResult(
                    job_id=task.job_id,
                    success=True,
                    out_path=task.out_path,
                    log_path=task.log_path,
                    called=called,
                    error=None,
                    traceback=None,
                    elapsed=time.time() - start,
                )
            )
        except Exception as e:  # broad catch to enable retries
            last_exc = e
            # tiny backoff
            time.sleep(min(1.0, 0.1 * attempt))

    tb = traceback.format_exc()
    return asdict(
        DockResult(
            job_id=task.job_id,
            success=False,
            out_path=task.out_path,
            log_path=task.log_path,
            called=None,
            error=f"All {task.retries} attempts failed; last error: {last_exc}",
            traceback=tb,
            elapsed=time.time() - start,
        )
    )


class BatchDock:
    """
    Batch orchestration facade for parallel docking jobs.

    BatchDock creates per-worker SingleDock instances inside worker processes and runs
    many docking tasks in parallel using :class:`concurrent.futures.ProcessPoolExecutor`.

    Important semantics
    - Each worker constructs a fresh SingleDock (and backend) to avoid pickling
      live engine objects.
    - Tasks are independent: provide per-row receptor/ligand and optional per-row overrides.
    - The facade returns a list of :class:`DockResult` objects (one per submitted job)
      containing success flag, artifact paths, elapsed time, and any exception details.

    Rows schema (each row is a dict or dataclass LigandTask)
    - id (string)         : unique job id
    - receptor (string)   : receptor file path
    - ligand (string)     : ligand file path
    - box (dict/list)     : optional box; either {"center":[x,y,z],"size":[sx,sy,sz]} or [[center],[size]]
    - exhaustiveness, n_poses, cpu, seed, out, log, autobox_ref, autobox_pad, engine_options

    Resolution and binary handling
    - Engine resolution works like SingleDock: you can pass engine name (e.g., "smina") and
      the worker will locate a binary from PATH or the repository-local
      ``prodock/dock/binary/<exe>`` folder. Use :py:meth:`BatchDock.from_config` to set
      a global executable override if needed via engine_options or per-row out/executable fields.

    Examples
    --------
    1) Programmatic creation and run
    .. code-block:: python

        from prodock.dock.engine import BatchDock
        bd = BatchDock(engine="smina", n_jobs=4)
        rows = [
            {"id":"L1","receptor":"rec.pdbqt","ligand":"l1.pdbqt",
             "box":{"center":[32.5,13.0,133.75],"size":[22.5,23.5,22.5]}},
            {"id":"L2","receptor":"rec.pdbqt","ligand":"l2.pdbqt",
             "box":{"center":[32.5,13.0,133.75],"size":[22.5,23.5,22.5]}}
        ]
        results = bd.run(rows, out_dir="out/docked", log_dir="out/logs", exhaustiveness=8)
        for r in results:
            print(r.job_id, r.success, r.out_path, r.called)

    2) From a dataclass or config file (JSON/YAML)
    .. code-block:: python

        # Using dataclass-based BatchConfig (programmatic)
        from prodock.dock.engine.config_dataclass import BatchConfig, LigandTask, Box
        rows = [
            LigandTask(id="L1", receptor="rec.pdbqt", ligand="l1.pdbqt",
                       box=Box(center=(32.5,13.0,133.75), size=(22.5,23.5,22.5))),
            ...
        ]
        cfg = BatchConfig(engine="smina", n_jobs=4, rows=rows, out_dir="out/docked", log_dir="out/logs")
        results = BatchDock.run_from_config(cfg)

        # From a JSON/YAML file:
        results = BatchDock.run_from_config("configs/batch_smina.json")

    Return values
    ----------------
    The return value of :meth:`run` and :meth:`run_from_config` is a list of :class:`DockResult`
    dataclasses with fields:
      - job_id: str
      - success: bool
      - out_path: Optional[str]
      - log_path: Optional[str]
      - called: Optional[str]  # command used (best-effort from worker/backend)
      - error: Optional[str]   # error message if failed
      - traceback: Optional[str]
      - elapsed: Optional[float] # seconds

    Notes & best practices
    - Ensure bundled binaries are executable (``chmod +x prodock/dock/binary/smina``).
    - Use absolute paths for files when running with ProcessPoolExecutor to avoid cwd issues.
    - For heavy I/O workloads, tune ``n_jobs`` relative to CPU and disk speed.
    - BatchDock will attempt a small retry/backoff per-task controlled by the task ``retries`` field.

    API
    ---
    :param engine: registry key for backend (string)
    :param engine_mode: optional hint (e.g., "binding" to prefer vina_binding)
    :param n_jobs: number of worker processes
    :param progress: if True and tqdm is installed, show progress bar
    :param default_retries: per-task retry default
    """

    def __init__(
        self,
        engine: Union[str, Any] = "vina",
        *,
        engine_mode: Optional[str] = None,
        n_jobs: int = 1,
        progress: bool = True,
        default_retries: int = 1,
        timeout: Optional[float] = None,
        tmp_root: Optional[PathLike] = None,
    ):
        if callable(engine) and not isinstance(engine, str):
            raise ValueError(
                "BatchDock currently expects an engine registry name string."
            )
        self._engine_name = str(engine)
        self.engine_mode = engine_mode
        self.n_jobs = max(1, int(n_jobs or 1))
        self.progress = progress and (tqdm is not None)
        self.default_retries = max(1, int(default_retries or 1))
        self.timeout = timeout
        self.tmp_root = Path(tmp_root) if tmp_root else None

    def _default_out_for(self, ligand_id: str) -> str:
        return str(Path("docked") / f"{ligand_id}_docked.pdbqt")

    def _default_log_for(self, ligand_id: str) -> str:
        return str(Path("logs") / f"{ligand_id}.log")

    def create_tasks(
        self,
        rows: Iterable[Dict[str, Any]],
        *,
        ligand_id_key: str = "id",
        receptor_key: str = "receptor",
        ligand_key: str = "ligand",
        center_key: str = "center",
        size_key: str = "size",
        engine_options: Optional[Dict[str, Any]] = None,
        out_dir: Optional[PathLike] = None,
        log_dir: Optional[PathLike] = None,
        exhaustiveness: Optional[int] = None,
        n_poses: Optional[int] = None,
        cpu: Optional[int] = None,
        seed: Optional[int] = None,
        autobox_ref_key: Optional[str] = None,
        autobox_pad: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> List[DockTask]:
        out_dir = Path(out_dir) if out_dir else None
        log_dir = Path(log_dir) if log_dir else None
        retries = int(retries or self.default_retries)
        tasks: List[DockTask] = []
        for r in rows:
            ligand_id = str(r[ligand_id_key])
            receptor = str(r[receptor_key])
            ligand = str(r[ligand_key])
            center = tuple(r[center_key]) if center_key in r and r[center_key] else None
            size = tuple(r[size_key]) if size_key in r and r[size_key] else None
            autobox_ref = (
                str(r[autobox_ref_key])
                if autobox_ref_key and autobox_ref_key in r and r[autobox_ref_key]
                else None
            )
            out_path = str(
                (out_dir / f"{ligand_id}_docked.pdbqt")
                if out_dir
                else Path(self._default_out_for(ligand_id))
            )
            log_path = str(
                (log_dir / f"{ligand_id}.log")
                if log_dir
                else Path(self._default_log_for(ligand_id))
            )
            tasks.append(
                DockTask(
                    job_id=ligand_id,
                    receptor=receptor,
                    ligand=ligand,
                    center=center,
                    size=size,
                    engine_name=self._engine_name,
                    engine_mode=self.engine_mode,
                    engine_options=engine_options or {},
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                    cpu=cpu,
                    seed=seed,
                    autobox_ref=autobox_ref,
                    autobox_pad=autobox_pad,
                    out_path=out_path,
                    log_path=log_path,
                    retries=retries,
                    timeout=self.timeout,
                    tmp_dir=str(self.tmp_root) if self.tmp_root else None,
                )
            )
        return tasks

    def run_tasks(self, tasks: Iterable[DockTask]) -> List[DockResult]:
        tlist = list(tasks)
        dicts = [asdict(t) for t in tlist]
        results: List[DockResult] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as exe:
            futures = [
                exe.submit(worker_process_job_using_singledock, d) for d in dicts
            ]
            if self.progress and tqdm is not None:
                for fut in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="BatchDock",
                ):
                    results.append(DockResult(**fut.result()))
            else:
                for fut in concurrent.futures.as_completed(futures):
                    results.append(DockResult(**fut.result()))
        return results

    def run(self, rows: Iterable[Dict[str, Any]], **kwargs) -> List[DockResult]:
        tasks = self.create_tasks(rows, **kwargs)
        return self.run_tasks(tasks)

    # Config helpers using dataclasses
    @classmethod
    def from_config(
        cls, config: Union[str, Dict[str, Any], BatchConfig]
    ) -> "BatchDock":
        if isinstance(config, BatchConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = BatchConfig.from_dict(config)
        else:
            cfg = BatchConfig.from_file(config)
        bd = cls(
            engine=cfg.engine,
            engine_mode=cfg.engine_mode,
            n_jobs=cfg.n_jobs,
            progress=cfg.progress,
            default_retries=cfg.default_retries,
            timeout=cfg.timeout,
            tmp_root=cfg.tmp_root,
        )
        bd._config = cfg
        return bd

    @classmethod
    def run_from_config(
        cls, config: Union[str, Dict[str, Any], BatchConfig]
    ) -> List[DockResult]:
        if isinstance(config, BatchConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = BatchConfig.from_dict(config)
        else:
            cfg = BatchConfig.from_file(config)

        bd = cls.from_config(cfg)
        rows = cfg.rows
        rows_for_create = []
        for r in rows:
            rd = r.to_dict()
            if "id" not in rd:
                rd["id"] = r.id
            rows_for_create.append(rd)
        return bd.run(
            rows_for_create,
            out_dir=cfg.out_dir,
            log_dir=cfg.log_dir,
            engine_options=cfg.engine_options,
            exhaustiveness=cfg.exhaustiveness,
            n_poses=cfg.n_poses,
            cpu=cfg.cpu,
            seed=cfg.seed,
            autobox_ref_key=cfg.autobox_ref_key,
            autobox_pad=cfg.autobox_pad,
            retries=cfg.retries,
        )
