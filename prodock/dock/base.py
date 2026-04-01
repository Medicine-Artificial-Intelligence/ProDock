from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union

Vec3 = Tuple[float, float, float]
PathLike = Union[str, Path]


class DockBackend(Protocol):
    """
    Structural protocol defining the common interface for docking backends.

    A backend implementing this protocol is expected to support receptor and
    ligand assignment, docking box configuration, runtime options, output file
    assignment, and execution of the docking job.

    The protocol is intentionally backend-agnostic so that engines such as
    AutoDock Vina, Smina, QuickVina, or custom wrappers can be used
    interchangeably by higher-level orchestration code.

    Example
    -------
    Basic usage in a backend-agnostic workflow::

        from pathlib import Path

        def prepare_and_run(backend: DockBackend) -> None:
            backend.set_receptor("receptor.pdbqt")
            backend.set_ligand("ligand.pdbqt")
            backend.set_box(
                center=(10.0, 12.5, -3.0),
                size=(20.0, 20.0, 20.0),
            )
            backend.set_exhaustiveness(16)
            backend.set_num_modes(10)
            backend.set_out("poses.pdbqt")
            backend.set_log("dock.log")
            backend.run()

    Notes
    -----
    This protocol only specifies the expected callable surface. It does not
    enforce how the backend stores state internally or how commands are
    executed.
    """

    def set_receptor(
        self, receptor_path: PathLike, *, validate: bool = False
    ) -> "DockBackend":
        """
        Set the receptor structure used for docking.

        :param receptor_path:
            Path to the receptor structure file. This is typically a prepared
            receptor file such as ``.pdbqt``, but the exact supported formats
            depend on the backend implementation.
        :type receptor_path: PathLike

        :param validate:
            Whether to validate the receptor file before accepting it. The
            meaning and strictness of validation are backend-specific.
        :type validate: bool

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend
        """
        ...

    def set_ligand(self, ligand_path: PathLike) -> "DockBackend":
        """
        Set the ligand structure used for docking.

        :param ligand_path:
            Path to the ligand structure file. In most workflows this is a
            prepared ligand file such as ``.pdbqt``.
        :type ligand_path: PathLike

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend
        """
        ...

    def set_box(self, center: Vec3, size: Vec3) -> "DockBackend":
        """
        Define the docking search box explicitly.

        :param center:
            Cartesian box center as ``(x, y, z)`` in Å.
        :type center: Vec3

        :param size:
            Box size along each axis as ``(sx, sy, sz)`` in Å.
        :type size: Vec3

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend

        Example
        -------
        ::

            backend.set_box(
                center=(12.0, 5.5, -1.2),
                size=(18.0, 20.0, 16.0),
            )
        """
        ...

    def enable_autobox(
        self, reference_file: PathLike, padding: Optional[float] = None
    ) -> "DockBackend":
        """
        Enable automatic docking box generation from a reference structure.

        The backend may infer the search region from a co-crystallized ligand,
        reference ligand, or another structure file used as an autobox source.

        :param reference_file:
            Path to the reference structure used for automatic box generation.
        :type reference_file: PathLike

        :param padding:
            Optional padding, in Å, to expand the inferred box dimensions.
            If ``None``, the backend default is used.
        :type padding: Optional[float]

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend
        """
        ...

    def set_exhaustiveness(self, value: Optional[int]) -> "DockBackend":
        """
        Set the docking search exhaustiveness.

        :param value:
            Exhaustiveness level to use. Higher values usually increase search
            effort and runtime. If ``None``, the backend default is retained.
        :type value: Optional[int]

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend
        """
        ...

    def set_num_modes(self, value: Optional[int]) -> "DockBackend":
        """
        Set the maximum number of output poses.

        :param value:
            Number of docking poses to retain. If ``None``, the backend default
            is retained.
        :type value: Optional[int]

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend
        """
        ...

    def set_cpu(self, value: Optional[int]) -> "DockBackend":
        """
        Set the number of CPU threads used by the backend.

        :param value:
            Number of CPU threads to use. If ``None``, the backend default is
            retained.
        :type value: Optional[int]

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend
        """
        ...

    def set_seed(self, value: Optional[int]) -> "DockBackend":
        """
        Set the random seed for reproducible docking runs.

        :param value:
            Random seed value. If ``None``, the backend may use its default
            non-deterministic behavior.
        :type value: Optional[int]

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend
        """
        ...

    def set_out(self, out_path: PathLike) -> "DockBackend":
        """
        Set the output path for docked poses.

        :param out_path:
            Path where the backend should write the docking pose file.
        :type out_path: PathLike

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend
        """
        ...

    def set_log(self, log_path: PathLike) -> "DockBackend":
        """
        Set the output path for the docking log file.

        :param log_path:
            Path where the backend should write its log or console capture.
        :type log_path: PathLike

        :returns:
            The backend instance itself, allowing fluent method chaining.
        :rtype: DockBackend
        """
        ...

    def run(
        self, *, exhaustiveness: Optional[int] = None, n_poses: Optional[int] = None
    ) -> "DockBackend":
        """
        Execute the docking calculation.

        Runtime keyword arguments may temporarily override previously assigned
        backend settings for the current run.

        :param exhaustiveness:
            Optional per-run override for search exhaustiveness.
        :type exhaustiveness: Optional[int]

        :param n_poses:
            Optional per-run override for the number of poses to generate or
            retain.
        :type n_poses: Optional[int]

        :returns:
            The backend instance itself, allowing fluent method chaining or
            post-run inspection.
        :rtype: DockBackend

        Example
        -------
        ::

            (
                backend
                .set_receptor("receptor.pdbqt")
                .set_ligand("ligand.pdbqt")
                .set_box((0.0, 0.0, 0.0), (20.0, 20.0, 20.0))
                .set_out("dock_out.pdbqt")
                .set_log("dock.log")
                .run(exhaustiveness=12, n_poses=9)
            )
        """
        ...


@dataclass
class RunArtifacts:
    """
    Container for files and metadata produced by a docking run.

    This object is useful as a lightweight summary of the tangible outputs of a
    completed or attempted docking job.

    :param out_path:
        Path to the generated docking pose file, if one was produced.
    :type out_path: Optional[Path]

    :param log_path:
        Path to the docking log file, if one was produced.
    :type log_path: Optional[Path]

    :param called:
        Optional textual representation of the executed command, backend call,
        or engine invocation.
    :type called: Optional[str]

    :param metadata:
        Additional backend-specific metadata associated with the run, such as
        timing, scores, configuration values, or parsed summaries.
    :type metadata: Dict[str, Any]

    Example
    -------
    ::

        artifacts = RunArtifacts(
            out_path=Path("poses.pdbqt"),
            log_path=Path("dock.log"),
            called="vina --receptor receptor.pdbqt --ligand ligand.pdbqt",
            metadata={"engine": "vina", "status": "ok"},
        )
    """

    out_path: Optional[Path]
    log_path: Optional[Path]
    called: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DockIdentity:
    """
    Human-readable identifiers describing one docking job.

    This dataclass is intended for labeling, reporting, manifest generation, or
    database insertion where a docking record should be associated with a
    receptor, an engine, and a ligand in a clear and stable way.

    :param receptor_id:
        Identifier for the receptor used in the docking job.
    :type receptor_id: Optional[str]

    :param engine_name:
        Name of the docking engine or backend, such as ``"vina"`` or
        ``"gnina"``.
    :type engine_name: Optional[str]

    :param ligand_id:
        Identifier for the ligand used in the docking job.
    :type ligand_id: Optional[str]

    Example
    -------
    ::

        identity = DockIdentity(
            receptor_id="1abc_chainA",
            engine_name="vina",
            ligand_id="erlotinib",
        )
    """

    receptor_id: Optional[str] = None
    engine_name: Optional[str] = None
    ligand_id: Optional[str] = None
