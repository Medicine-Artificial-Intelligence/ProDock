from __future__ import annotations

import concurrent.futures
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore

from .config import BatchConfig, DockRow, ReceptorSpec

PathLike = Union[str, Path]
Vec3 = Tuple[float, float, float]


@dataclass
class DockTask:
    """
    Concrete docking task consumed by the batch worker.

    A :class:`DockTask` is the normalized unit of work produced by
    :class:`BatchDock`. It contains fully resolved receptor, ligand, engine,
    path, and runtime parameters ready to be executed by
    :func:`worker_process_job_using_singledock`.

    :param job_id:
        Unique job identifier, typically composed from receptor id, engine name,
        and ligand id.
    :type job_id: str

    :param receptor_id:
        Logical receptor identifier for grouping and reporting.
    :type receptor_id: str

    :param engine_name:
        Docking engine key, for example ``"vina"`` or ``"qvina"``.
    :type engine_name: str

    :param ligand_id:
        Logical ligand identifier for grouping and reporting.
    :type ligand_id: str

    :param receptor:
        Receptor input file path.
    :type receptor: str

    :param ligand:
        Ligand input file path.
    :type ligand: str

    :param center:
        Optional docking box center.
    :type center: Optional[Vec3]

    :param size:
        Optional docking box size.
    :type size: Optional[Vec3]

    :param autobox_ref:
        Optional reference structure used for autoboxing when an explicit box is
        not provided.
    :type autobox_ref: Optional[str]

    :param autobox_pad:
        Optional padding added around the autobox reference.
    :type autobox_pad: Optional[float]

    :param exhaustiveness:
        Optional docking exhaustiveness value.
    :type exhaustiveness: Optional[int]

    :param n_poses:
        Optional number of output poses to request.
    :type n_poses: Optional[int]

    :param cpu:
        Optional CPU count.
    :type cpu: Optional[int]

    :param seed:
        Optional random seed.
    :type seed: Optional[int]

    :param executable:
        Optional explicit backend executable path.
    :type executable: Optional[str]

    :param engine_options:
        Additional engine-specific options to apply before running.
    :type engine_options: Dict[str, Any]

    :param out_path:
        Output docking pose path.
    :type out_path: Optional[str]

    :param log_path:
        Log file path.
    :type log_path: Optional[str]

    :param retries:
        Number of times the task may be retried upon failure.
    :type retries: int

    :param timeout:
        Optional timeout placeholder for future orchestration logic.
    :type timeout: Optional[float]
    """

    job_id: str
    receptor_id: str
    engine_name: str
    ligand_id: str
    receptor: str
    ligand: str
    center: Optional[Vec3] = None
    size: Optional[Vec3] = None
    autobox_ref: Optional[str] = None
    autobox_pad: Optional[float] = None
    exhaustiveness: Optional[int] = None
    n_poses: Optional[int] = None
    cpu: Optional[int] = None
    seed: Optional[int] = None
    executable: Optional[str] = None
    engine_options: Dict[str, Any] = field(default_factory=dict)
    out_path: Optional[str] = None
    log_path: Optional[str] = None
    retries: int = 1
    timeout: Optional[float] = None


@dataclass
class DockResult:
    """
    Result record returned for each completed docking task.

    :param job_id:
        Unique job identifier of the executed task.
    :type job_id: str

    :param receptor_id:
        Receptor identifier associated with the task.
    :type receptor_id: str

    :param engine_name:
        Docking engine key used for the task.
    :type engine_name: str

    :param ligand_id:
        Ligand identifier associated with the task.
    :type ligand_id: str

    :param success:
        Whether the task completed successfully.
    :type success: bool

    :param out_path:
        Output docking pose path, if defined.
    :type out_path: Optional[str]

    :param log_path:
        Log file path, if defined.
    :type log_path: Optional[str]

    :param called:
        Backend-reported command line or invocation summary, if available.
    :type called: Optional[str]

    :param error:
        Human-readable error message on failure.
    :type error: Optional[str]

    :param traceback:
        Python traceback captured on failure.
    :type traceback: Optional[str]

    :param elapsed:
        Elapsed wall-clock time in seconds.
    :type elapsed: Optional[float]

    :param metadata:
        Additional backend metadata emitted during the run.
    :type metadata: Dict[str, Any]
    """

    job_id: str
    receptor_id: str
    engine_name: str
    ligand_id: str
    success: bool
    out_path: Optional[str]
    log_path: Optional[str]
    called: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    elapsed: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def worker_process_job_using_singledock(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute one serialized :class:`DockTask` using :class:`SingleDock`.

    This function is intentionally top-level so it can be pickled and used by
    :class:`concurrent.futures.ProcessPoolExecutor`.

    The input and output are plain dictionaries to simplify cross-process
    transport. Internally, the function reconstructs a :class:`DockTask`,
    performs the docking run, and returns a serialized :class:`DockResult`.

    :param task_dict:
        Serialized docking task dictionary, typically produced by
        :func:`dataclasses.asdict`.
    :type task_dict: Dict[str, Any]

    :returns:
        Serialized docking result dictionary.
    :rtype: Dict[str, Any]

    Example
    -------
    .. code-block:: python

        task = DockTask(
            job_id="4WKQ:qvina:erlotinib",
            receptor_id="4WKQ",
            engine_name="qvina",
            ligand_id="erlotinib",
            receptor="4WKQ.pdbqt",
            ligand="erlotinib.pdbqt",
        )
        result = worker_process_job_using_singledock(asdict(task))
    """
    start = time.time()
    task = DockTask(**task_dict)

    try:
        from .single import SingleDock
    except Exception as exc:  # pragma: no cover - packaging failure path
        return asdict(
            DockResult(
                job_id=task.job_id,
                receptor_id=task.receptor_id,
                engine_name=task.engine_name,
                ligand_id=task.ligand_id,
                success=False,
                out_path=task.out_path,
                log_path=task.log_path,
                error=f"Failed to import SingleDock: {exc}",
                traceback=traceback.format_exc(),
                elapsed=time.time() - start,
            )
        )

    attempt = 0
    last_error: Optional[BaseException] = None
    last_trace: Optional[str] = None

    while attempt < max(1, task.retries):
        attempt += 1
        try:
            sd = SingleDock(engine=task.engine_name)
            sd.set_receptor(task.receptor, validate=False)
            sd.set_ligand(task.ligand)

            if task.center is not None and task.size is not None:
                sd.set_box(task.center, task.size)
            elif task.autobox_ref is not None:
                sd.enable_autobox(task.autobox_ref, padding=task.autobox_pad)

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
            if task.executable:
                sd.set_executable(task.executable)
            if task.engine_options:
                sd.apply_engine_options(task.engine_options)

            res = sd.run(exhaustiveness=task.exhaustiveness, n_poses=task.n_poses)
            return asdict(
                DockResult(
                    job_id=task.job_id,
                    receptor_id=task.receptor_id,
                    engine_name=task.engine_name,
                    ligand_id=task.ligand_id,
                    success=True,
                    out_path=task.out_path,
                    log_path=task.log_path,
                    called=res.artifacts.called,
                    elapsed=time.time() - start,
                    metadata=res.artifacts.metadata,
                )
            )
        except Exception as exc:  # pragma: no cover - runtime/backend dependent
            last_error = exc
            last_trace = traceback.format_exc()
            if attempt < max(1, task.retries):
                time.sleep(min(1.0, 0.15 * attempt))

    return asdict(
        DockResult(
            job_id=task.job_id,
            receptor_id=task.receptor_id,
            engine_name=task.engine_name,
            ligand_id=task.ligand_id,
            success=False,
            out_path=task.out_path,
            log_path=task.log_path,
            error=str(last_error) if last_error is not None else "Unknown error",
            traceback=last_trace,
            elapsed=time.time() - start,
        )
    )


class BatchDock:
    """
    Parallel docking orchestrator for flat and receptor-centric batch layouts.

    The class supports two input styles:

    - flat row-based batches using :class:`DockRow`
    - hierarchical receptor-centric batches using :class:`ReceptorSpec`

    Tasks are normalized into :class:`DockTask` instances and then executed
    either serially or in parallel through a process pool.

    :param engine:
        Default engine name used when a flat row batch does not override it.
    :type engine: str

    :param n_jobs:
        Number of worker processes. Values below 1 are coerced to 1.
    :type n_jobs: int

    :param progress:
        Whether a progress bar should be shown when ``tqdm`` is available.
    :type progress: bool

    :param default_retries:
        Default retry count used for tasks without an explicit retry override.
    :type default_retries: int

    :param timeout:
        Optional timeout placeholder stored on generated tasks.
    :type timeout: Optional[float]

    :param tmp_root:
        Optional temporary root directory placeholder for future extensions.
    :type tmp_root: Optional[PathLike]

    Example
    -------
    .. code-block:: python

        batch = BatchDock(engine="qvina", n_jobs=4)

        results = batch.run(
            [
                {
                    "id": "erlotinib",
                    "receptor": "4WKQ.pdbqt",
                    "ligand": "erlotinib.pdbqt",
                    "center": [2.865, 193.257, 21.367],
                    "size": [27.091, 27.091, 27.091],
                }
            ],
            out_dir="docked",
            log_dir="logs",
            exhaustiveness=8,
            n_poses=10,
        )
    """

    def __init__(
        self,
        engine: str = "vina",
        *,
        n_jobs: int = 1,
        progress: bool = True,
        default_retries: int = 1,
        timeout: Optional[float] = None,
        tmp_root: Optional[PathLike] = None,
    ) -> None:
        """
        Initialize a batch docking orchestrator.

        :param engine:
            Default engine name used for flat row batches.
        :type engine: str

        :param n_jobs:
            Number of worker processes. Values below 1 are coerced to 1.
        :type n_jobs: int

        :param progress:
            Whether progress reporting should be enabled when ``tqdm`` is
            installed.
        :type progress: bool

        :param default_retries:
            Default retry count for tasks lacking an explicit retry value.
        :type default_retries: int

        :param timeout:
            Optional timeout placeholder propagated to generated tasks.
        :type timeout: Optional[float]

        :param tmp_root:
            Optional temporary root directory placeholder.
        :type tmp_root: Optional[PathLike]
        """
        self.default_engine = engine
        self.n_jobs = max(1, int(n_jobs or 1))
        self.progress = bool(progress and tqdm is not None)
        self.default_retries = max(1, int(default_retries or 1))
        self.timeout = timeout
        self.tmp_root = Path(tmp_root) if tmp_root else None

    def _default_out_for(
        self, receptor_id: str, engine_name: str, ligand_id: str
    ) -> Path:
        """
        Build the default output pose path for a task.

        :param receptor_id:
            Receptor identifier.
        :type receptor_id: str

        :param engine_name:
            Engine name.
        :type engine_name: str

        :param ligand_id:
            Ligand identifier.
        :type ligand_id: str

        :returns:
            Default output pose path.
        :rtype: Path
        """
        return Path("docked") / receptor_id / engine_name / f"{ligand_id}_docked.pdbqt"

    def _default_log_for(
        self, receptor_id: str, engine_name: str, ligand_id: str
    ) -> Path:
        """
        Build the default log path for a task.

        :param receptor_id:
            Receptor identifier.
        :type receptor_id: str

        :param engine_name:
            Engine name.
        :type engine_name: str

        :param ligand_id:
            Ligand identifier.
        :type ligand_id: str

        :returns:
            Default log path.
        :rtype: Path
        """
        return Path("logs") / receptor_id / engine_name / f"{ligand_id}.log"

    @staticmethod
    def _first_not_none(*values: Any) -> Any:
        """
        Return the first value that is not ``None``.

        :param values:
            Candidate values to inspect in order.
        :type values: Any

        :returns:
            The first non-``None`` value, or ``None`` if all inputs are
            ``None``.
        :rtype: Any
        """
        for value in values:
            if value is not None:
                return value
        return None

    def _resolve_output_path(
        self,
        *,
        base_dir: Optional[str],
        global_dir: Optional[Path],
        receptor_id: str,
        engine_name: str,
        ligand_id: str,
        suffix: str,
        append_engine_to_base: bool = False,
    ) -> str:
        """
        Resolve a task-specific output or log path.

        Resolution precedence is:

        1. receptor/software-local base directory,
        2. global directory passed into the batch run,
        3. built-in default ``docked/`` or ``logs/`` layout.

        :param base_dir:
            Receptor- or software-specific base directory override.
        :type base_dir: Optional[str]

        :param global_dir:
            Global batch-level root directory.
        :type global_dir: Optional[Path]

        :param receptor_id:
            Receptor identifier.
        :type receptor_id: str

        :param engine_name:
            Engine name.
        :type engine_name: str

        :param ligand_id:
            Ligand identifier.
        :type ligand_id: str

        :param suffix:
            Filename suffix such as ``".log"`` or ``"_docked.pdbqt"``.
        :type suffix: str

        :param append_engine_to_base:
            Whether the engine name should be appended under ``base_dir``.
        :type append_engine_to_base: bool

        :returns:
            Resolved file path as a string.
        :rtype: str
        """
        if base_dir:
            root = Path(base_dir)
            if append_engine_to_base:
                root = root / engine_name
        elif global_dir is not None:
            root = global_dir / receptor_id / engine_name
        else:
            root = (
                Path("logs") / receptor_id / engine_name
                if suffix == ".log"
                else Path("docked") / receptor_id / engine_name
            )

        filename = f"{ligand_id}{suffix}"
        return str(root / filename)

    def create_tasks(
        self,
        rows: Iterable[Union[DockRow, Dict[str, Any]]],
        *,
        engine: Optional[str] = None,
        out_dir: Optional[PathLike] = None,
        log_dir: Optional[PathLike] = None,
        exhaustiveness: Optional[int] = None,
        n_poses: Optional[int] = None,
        cpu: Optional[int] = None,
        seed: Optional[int] = None,
        retries: Optional[int] = None,
        engine_options: Optional[Dict[str, Any]] = None,
        executable: Optional[str] = None,
    ) -> List[DockTask]:
        """
        Create normalized tasks from a flat row-based batch definition.

        Per-row values override method-level defaults when present.

        :param rows:
            Iterable of flat row definitions as :class:`DockRow` objects or raw
            dictionaries.
        :type rows: Iterable[Union[DockRow, Dict[str, Any]]]

        :param engine:
            Engine name override for the whole batch.
        :type engine: Optional[str]

        :param out_dir:
            Global output directory for generated pose files.
        :type out_dir: Optional[PathLike]

        :param log_dir:
            Global log directory.
        :type log_dir: Optional[PathLike]

        :param exhaustiveness:
            Default exhaustiveness value.
        :type exhaustiveness: Optional[int]

        :param n_poses:
            Default number of poses.
        :type n_poses: Optional[int]

        :param cpu:
            Default CPU count.
        :type cpu: Optional[int]

        :param seed:
            Default random seed.
        :type seed: Optional[int]

        :param retries:
            Default retry count.
        :type retries: Optional[int]

        :param engine_options:
            Default engine-specific options merged with per-row options.
        :type engine_options: Optional[Dict[str, Any]]

        :param executable:
            Optional executable override applied to every task.
        :type executable: Optional[str]

        :returns:
            List of normalized docking tasks.
        :rtype: List[DockTask]

        Example
        -------
        .. code-block:: python

            tasks = BatchDock(engine="qvina").create_tasks(
                [
                    {
                        "id": "erlotinib",
                        "receptor": "4WKQ.pdbqt",
                        "ligand": "erlotinib.pdbqt",
                        "center": [2.865, 193.257, 21.367],
                        "size": [27.091, 27.091, 27.091],
                    }
                ],
                out_dir="docked",
                log_dir="logs",
                exhaustiveness=8,
            )
        """
        resolved_engine = (engine or self.default_engine).lower()
        resolved_retries = int(retries or self.default_retries)
        out_root = Path(out_dir) if out_dir else None
        log_root = Path(log_dir) if log_dir else None

        tasks: List[DockTask] = []
        for item in rows:
            row = item if isinstance(item, DockRow) else DockRow.from_dict(item)
            box = row.resolved_box()
            receptor_id = Path(row.receptor).stem
            ligand_id = row.id

            out_path = row.out or str(
                (out_root / f"{ligand_id}_docked.pdbqt")
                if out_root
                else self._default_out_for(receptor_id, resolved_engine, ligand_id)
            )
            log_path = row.log or str(
                (log_root / f"{ligand_id}.log")
                if log_root
                else self._default_log_for(receptor_id, resolved_engine, ligand_id)
            )

            tasks.append(
                DockTask(
                    job_id=f"{receptor_id}:{resolved_engine}:{ligand_id}",
                    receptor_id=receptor_id,
                    engine_name=resolved_engine,
                    ligand_id=ligand_id,
                    receptor=row.receptor,
                    ligand=row.ligand,
                    center=box.center if box else row.center,
                    size=box.size if box else row.size,
                    autobox_ref=row.autobox_ref,
                    autobox_pad=row.autobox_pad,
                    exhaustiveness=(
                        row.exhaustiveness
                        if row.exhaustiveness is not None
                        else exhaustiveness
                    ),
                    n_poses=row.n_poses if row.n_poses is not None else n_poses,
                    cpu=row.cpu if row.cpu is not None else cpu,
                    seed=row.seed if row.seed is not None else seed,
                    executable=executable,
                    engine_options={
                        **(engine_options or {}),
                        **(row.engine_options or {}),
                    },
                    out_path=out_path,
                    log_path=log_path,
                    retries=int(row.retries or resolved_retries),
                    timeout=self.timeout,
                )
            )
        return tasks

    def create_tasks_from_receptors(
        self,
        receptors: Iterable[Union[ReceptorSpec, Dict[str, Any]]],
        *,
        out_dir: Optional[PathLike] = None,
        log_dir: Optional[PathLike] = None,
    ) -> List[DockTask]:
        """
        Create normalized tasks from a receptor-centric batch definition.

        This method expands the hierarchy:

        ``receptor -> software -> ligand -> DockTask``

        :param receptors:
            Iterable of receptor-centric definitions as :class:`ReceptorSpec`
            objects or raw dictionaries.
        :type receptors: Iterable[Union[ReceptorSpec, Dict[str, Any]]]

        :param out_dir:
            Optional global output directory used when receptor/software-local
            output roots are not defined.
        :type out_dir: Optional[PathLike]

        :param log_dir:
            Optional global log directory used when receptor/software-local
            log roots are not defined.
        :type log_dir: Optional[PathLike]

        :returns:
            List of normalized docking tasks.
        :rtype: List[DockTask]

        Example
        -------
        .. code-block:: python

            tasks = BatchDock().create_tasks_from_receptors(
                receptors,
                out_dir="docked",
                log_dir="logs",
            )
        """
        out_root = Path(out_dir) if out_dir else None
        log_root = Path(log_dir) if log_dir else None
        tasks: List[DockTask] = []

        for item in receptors:
            receptor = (
                item if isinstance(item, ReceptorSpec) else ReceptorSpec.from_dict(item)
            )
            box = receptor.box

            for software in receptor.softwares:
                engine_name = software.name.lower()

                software_out_dir = getattr(software, "out_dir", None)
                software_log_dir = getattr(software, "log_dir", None)

                receptor_out_dir = getattr(receptor, "out_dir", None)
                receptor_log_dir = getattr(receptor, "log_dir", None)

                base_out_dir = software_out_dir or receptor_out_dir
                base_log_dir = software_log_dir or receptor_log_dir

                append_engine_to_out_base = bool(receptor_out_dir) and not bool(
                    software_out_dir
                )
                append_engine_to_log_base = bool(receptor_log_dir) and not bool(
                    software_log_dir
                )

                for ligand in software.ligands:
                    ligand_id = ligand.id

                    task_out = ligand.out or self._resolve_output_path(
                        base_dir=base_out_dir,
                        global_dir=out_root,
                        receptor_id=receptor.id,
                        engine_name=engine_name,
                        ligand_id=ligand_id,
                        suffix="_docked.pdbqt",
                        append_engine_to_base=append_engine_to_out_base,
                    )
                    task_log = ligand.log or self._resolve_output_path(
                        base_dir=base_log_dir,
                        global_dir=log_root,
                        receptor_id=receptor.id,
                        engine_name=engine_name,
                        ligand_id=ligand_id,
                        suffix=".log",
                        append_engine_to_base=append_engine_to_log_base,
                    )

                    tasks.append(
                        DockTask(
                            job_id=f"{receptor.id}:{engine_name}:{ligand_id}",
                            receptor_id=receptor.id,
                            engine_name=engine_name,
                            ligand_id=ligand_id,
                            receptor=receptor.receptor,
                            ligand=ligand.ligand,
                            center=box.center if box else None,
                            size=box.size if box else None,
                            autobox_ref=receptor.autobox_ref,
                            autobox_pad=receptor.autobox_pad,
                            exhaustiveness=self._first_not_none(
                                ligand.exhaustiveness, software.exhaustiveness
                            ),
                            n_poses=self._first_not_none(
                                ligand.n_poses, software.n_poses
                            ),
                            cpu=self._first_not_none(ligand.cpu, software.cpu),
                            seed=self._first_not_none(ligand.seed, software.seed),
                            executable=software.executable,
                            engine_options={
                                **software.engine_options,
                                **ligand.engine_options,
                            },
                            out_path=task_out,
                            log_path=task_log,
                            retries=int(ligand.retries or self.default_retries),
                            timeout=self.timeout,
                        )
                    )
        return tasks

    def run_tasks(self, tasks: Iterable[DockTask]) -> List[DockResult]:
        """
        Execute a collection of normalized docking tasks.

        When ``n_jobs == 1``, tasks are executed serially in the current process.
        Otherwise they are distributed through a
        :class:`concurrent.futures.ProcessPoolExecutor`.

        :param tasks:
            Iterable of normalized docking tasks.
        :type tasks: Iterable[DockTask]

        :returns:
            List of docking results.
        :rtype: List[DockResult]

        Example
        -------
        .. code-block:: python

            tasks = batch.create_tasks(rows)
            results = batch.run_tasks(tasks)
        """
        task_list = list(tasks)
        if not task_list:
            return []

        payloads = [asdict(task) for task in task_list]

        if self.n_jobs == 1:
            return [
                DockResult(**worker_process_job_using_singledock(payload))
                for payload in payloads
            ]

        results: List[DockResult] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
            futures = [
                pool.submit(worker_process_job_using_singledock, payload)
                for payload in payloads
            ]
            iterator = concurrent.futures.as_completed(futures)
            if self.progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="BatchDock")
            for future in iterator:
                results.append(DockResult(**future.result()))
        return results

    def run(
        self,
        rows: Iterable[Union[DockRow, Dict[str, Any]]],
        **kwargs: Any,
    ) -> List[DockResult]:
        """
        Create and run tasks from a flat row-based batch definition.

        This is a convenience wrapper around :meth:`create_tasks` followed by
        :meth:`run_tasks`.

        :param rows:
            Iterable of flat row definitions.
        :type rows: Iterable[Union[DockRow, Dict[str, Any]]]

        :param kwargs:
            Additional keyword arguments forwarded to :meth:`create_tasks`.
        :type kwargs: Any

        :returns:
            List of docking results.
        :rtype: List[DockResult]
        """
        return self.run_tasks(self.create_tasks(rows, **kwargs))

    def run_receptors(
        self,
        receptors: Iterable[Union[ReceptorSpec, Dict[str, Any]]],
        *,
        out_dir: Optional[PathLike] = None,
        log_dir: Optional[PathLike] = None,
    ) -> List[DockResult]:
        """
        Create and run tasks from a receptor-centric batch definition.

        This is a convenience wrapper around :meth:`create_tasks_from_receptors`
        followed by :meth:`run_tasks`.

        :param receptors:
            Iterable of receptor-centric definitions.
        :type receptors: Iterable[Union[ReceptorSpec, Dict[str, Any]]]

        :param out_dir:
            Optional global output directory.
        :type out_dir: Optional[PathLike]

        :param log_dir:
            Optional global log directory.
        :type log_dir: Optional[PathLike]

        :returns:
            List of docking results.
        :rtype: List[DockResult]
        """
        tasks = self.create_tasks_from_receptors(
            receptors,
            out_dir=out_dir,
            log_dir=log_dir,
        )
        return self.run_tasks(tasks)

    @classmethod
    def from_config(
        cls, config: Union[str, Dict[str, Any], BatchConfig]
    ) -> "BatchDock":
        """
        Build a :class:`BatchDock` instance from a batch configuration.

        :param config:
            Batch configuration represented as a :class:`BatchConfig`, mapping,
            or configuration file path.
        :type config: Union[str, Dict[str, Any], BatchConfig]

        :returns:
            Configured batch orchestrator.
        :rtype: BatchDock

        Example
        -------
        .. code-block:: python

            batch = BatchDock.from_config("batch.json")
        """
        cfg = (
            config
            if isinstance(config, BatchConfig)
            else (
                BatchConfig.from_dict(config)
                if isinstance(config, dict)
                else BatchConfig.from_file(config)
            )
        )
        inst = cls(
            engine=cfg.engine or "vina",
            n_jobs=cfg.n_jobs,
            progress=cfg.progress,
            default_retries=cfg.default_retries,
            timeout=cfg.timeout,
            tmp_root=cfg.tmp_root,
        )
        inst._config = cfg
        return inst

    @classmethod
    def run_from_config(
        cls, config: Union[str, Dict[str, Any], BatchConfig]
    ) -> List[DockResult]:
        """
        Execute a full batch directly from configuration.

        The method automatically dispatches to either flat row execution or
        receptor-centric execution depending on whether ``cfg.receptors`` is
        populated.

        :param config:
            Batch configuration represented as a :class:`BatchConfig`, mapping,
            or configuration file path.
        :type config: Union[str, Dict[str, Any], BatchConfig]

        :returns:
            List of docking results.
        :rtype: List[DockResult]

        Example
        -------
        .. code-block:: python

            results = BatchDock.run_from_config("batch.json")
        """
        cfg = (
            config
            if isinstance(config, BatchConfig)
            else (
                BatchConfig.from_dict(config)
                if isinstance(config, dict)
                else BatchConfig.from_file(config)
            )
        )
        inst = cls.from_config(cfg)

        if cfg.receptors:
            return inst.run_receptors(
                cfg.receptors,
                out_dir=cfg.out_dir,
                log_dir=cfg.log_dir,
            )

        return inst.run(
            cfg.rows,
            engine=cfg.engine or inst.default_engine,
            out_dir=cfg.out_dir,
            log_dir=cfg.log_dir,
            exhaustiveness=cfg.exhaustiveness,
            n_poses=cfg.n_poses,
            cpu=cfg.cpu,
            seed=cfg.seed,
            retries=cfg.retries,
            engine_options=cfg.engine_options,
        )


MatrixDock = BatchDock
