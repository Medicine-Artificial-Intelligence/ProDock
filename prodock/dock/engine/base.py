from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Tuple, Optional, Union

Vec3 = Tuple[float, float, float]
PathLike = Union[str, Path]


class DockBackend(Protocol):
    """
    Minimal common surface for docking backends.

    This protocol intentionally excludes any score parsing — engines are only responsible
    for preparing I/O, running the docking call, and writing artifacts (poses, logs).

    Methods are chainable to enable a fluent configuration style.

    Examples
    --------
    .. code-block:: python

        backend = SminaEngine() \
            .set_receptor("rec.pdbqt", validate=True) \
            .set_ligand("lig.pdbqt") \
            .set_box((10.0, 10.0, 10.0), (20.0, 20.0, 20.0)) \
            .set_exhaustiveness(8).set_num_modes(9).set_cpu(4).set_seed(42) \
            .set_out("out/lig_docked.pdbqt").set_log("out/lig.log")
        backend.run()
    """

    # Chainable setup
    def set_receptor(
        self, receptor_path: PathLike, *, validate: bool = False
    ) -> "DockBackend": ...
    def set_ligand(self, ligand_path: PathLike) -> "DockBackend": ...
    def set_box(self, center: Vec3, size: Vec3) -> "DockBackend": ...

    def enable_autobox(
        self, reference_file: PathLike, padding: Optional[float] = None
    ) -> "DockBackend": ...
    def set_exhaustiveness(self, value: Optional[int]) -> "DockBackend": ...
    def set_num_modes(self, value: Optional[int]) -> "DockBackend": ...
    def set_cpu(self, value: Optional[int]) -> "DockBackend": ...
    def set_seed(self, value: Optional[int]) -> "DockBackend": ...
    def set_out(self, out_path: PathLike) -> "DockBackend": ...
    def set_log(self, log_path: PathLike) -> "DockBackend": ...

    # Execute
    def run(
        self, *, exhaustiveness: Optional[int] = None, n_poses: Optional[int] = None
    ) -> "DockBackend": ...


@dataclass
class RunArtifacts:
    """
    Pointers to artifacts produced by a docking run.

    :param out_path: Path to the written pose file (e.g., PDBQT or SDF), if any.
    :param log_path: Path to the engine log file, if any.
    :param called: Debug string of the exact command (for CLI engines), if available.
    """

    out_path: Optional[Path]
    log_path: Optional[Path]
    called: Optional[str] = None
