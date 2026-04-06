from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Set

from rdkit import Chem

from prodock.structure.conversion import pdbqt_to_sdf

PathLike = str | Path


def _as_path(value: PathLike) -> Path:
    """
    Convert a path-like value to :class:`pathlib.Path`.

    This helper normalizes string paths and existing
    :class:`pathlib.Path` objects into a single internal representation used by
    the conversion utilities.

    :param value:
        Input path to normalize.
    :type value: str | pathlib.Path

    :returns:
        Normalized path object.
    :rtype: pathlib.Path

    Example
    -------
    .. code-block:: python

        path = _as_path(
            "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
        )
        assert isinstance(path, Path)
    """
    return value if isinstance(value, Path) else Path(value)


def _iter_pdbqt_files(
    roots: Sequence[PathLike],
    *,
    recursive: bool = True,
    engine: Optional[str] = None,
) -> List[Path]:
    """
    Discover ``.pdbqt`` files for conversion without importing pose readers.

    This helper is intentionally defined locally in ``convert.py`` so the
    conversion utilities do not depend on ``io.py``. This keeps the module
    dependency direction simple and avoids circular-import issues.

    Input roots may be individual files or directories. When ``engine`` is
    provided, it is applied as a filter against the parent directory name of
    each candidate ``.pdbqt`` file.

    Duplicate paths are removed after resolution, and the returned list is
    sorted for deterministic downstream processing.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param recursive:
        Whether to recurse into nested directories when a root is a directory.
    :type recursive: bool
    :param engine:
        Optional engine filter applied to the immediate parent directory name of
        each candidate pose file.
    :type engine: Optional[str]

    :returns:
        Sorted list of unique resolved ``.pdbqt`` files.
    :rtype: list[pathlib.Path]

    Example
    -------
    .. code-block:: python

        files = _iter_pdbqt_files(
            ["Data/testcase/post/1M17/results/docked"],
            recursive=True,
            engine="vina",
        )
    """
    discovered: Set[Path] = set()
    engine_token = engine.lower() if engine else None

    for root in roots:
        path = _as_path(root)
        if path.is_file():
            if path.suffix.lower() == ".pdbqt":
                if engine_token is None or path.parent.name.lower() == engine_token:
                    discovered.add(path.resolve())
            continue

        if not path.exists():
            continue

        globber = path.rglob if recursive else path.glob
        for candidate in globber("*.pdbqt"):
            if (
                engine_token is not None
                and candidate.parent.name.lower() != engine_token
            ):
                continue
            discovered.add(candidate.resolve())

    return sorted(discovered)


def save_pose_sdf(
    pdbqt_file: PathLike,
    *,
    backend: str = "obabel",
    overwrite: bool = False,
    out_file: Optional[PathLike] = None,
) -> Path:
    """
    Convert a docked ``.pdbqt`` pose file to ``.sdf`` and save it on disk.

    By default, the output SDF is written next to the input file using the same
    file stem. An explicit output path may also be supplied via ``out_file``.

    If the destination file already exists and ``overwrite`` is ``False``, the
    existing path is returned without performing a new conversion.

    :param pdbqt_file:
        Input docked ``.pdbqt`` file.
    :type pdbqt_file: str | pathlib.Path
    :param backend:
        Conversion backend passed to
        :func:`prodock.structure.conversion.pdbqt_to_sdf`.
    :type backend: str
    :param overwrite:
        Whether an existing output file may be overwritten.
    :type overwrite: bool
    :param out_file:
        Optional explicit destination path. When omitted, the output file is
        created beside the input file with suffix ``.sdf``.
    :type out_file: Optional[str | pathlib.Path]

    :returns:
        Path to the written or reused SDF file.
    :rtype: pathlib.Path

    Example
    -------
    .. code-block:: python

        sdf_path = save_pose_sdf(
            "Data/testcase/post/1M17/results/docked/qvina/erlotinib_docked.pdbqt",
            backend="obabel",
            overwrite=True,
        )
    """
    pdbqt_path = _as_path(pdbqt_file)
    sdf_path = (
        _as_path(out_file) if out_file is not None else pdbqt_path.with_suffix(".sdf")
    )

    if sdf_path.exists() and not overwrite:
        return sdf_path

    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdbqt_to_sdf(str(pdbqt_path), str(sdf_path), backend=backend)
    return sdf_path


def pdbqt_to_rdkit_mols(
    pdbqt_file: PathLike,
    *,
    backend: str = "obabel",
    sanitize: bool = True,
    remove_hs: bool = False,
) -> List[Chem.Mol]:
    """
    Convert a docked ``.pdbqt`` file into RDKit molecules via a temporary SDF.

    The function first converts the input ``.pdbqt`` file into a temporary SDF
    file using :func:`prodock.structure.conversion.pdbqt_to_sdf`, then loads
    the molecules with :class:`rdkit.Chem.SDMolSupplier`.

    Invalid molecules returned as ``None`` by the supplier are discarded.

    :param pdbqt_file:
        Input docked ``.pdbqt`` file.
    :type pdbqt_file: str | pathlib.Path
    :param backend:
        Conversion backend passed to the PDBQT-to-SDF converter.
    :type backend: str
    :param sanitize:
        Whether RDKit sanitization should be applied while reading the temporary
        SDF.
    :type sanitize: bool
    :param remove_hs:
        Whether hydrogens should be removed during SDF import.
    :type remove_hs: bool

    :returns:
        List of successfully loaded RDKit molecule objects.
    :rtype: list[rdkit.Chem.Mol]

    Example
    -------
    .. code-block:: python

        mols = pdbqt_to_rdkit_mols(
            "Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt",
            sanitize=True,
            remove_hs=False,
        )
    """
    pdbqt_path = _as_path(pdbqt_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        sdf_path = Path(tmpdir) / f"{pdbqt_path.stem}.sdf"
        pdbqt_to_sdf(str(pdbqt_path), str(sdf_path), backend=backend)

        supplier = Chem.SDMolSupplier(
            str(sdf_path),
            sanitize=sanitize,
            removeHs=remove_hs,
        )
        return [mol for mol in supplier if mol is not None]


def convert_pose_tree(
    roots: Sequence[PathLike],
    *,
    engine: Optional[str] = None,
    recursive: bool = True,
    backend: str = "obabel",
    overwrite: bool = False,
    out_dir: Optional[PathLike] = None,
) -> List[Path]:
    """
    Convert discovered ``.pdbqt`` pose files into ``.sdf`` files.

    When ``out_dir`` is omitted, each output SDF is written beside its source
    ``.pdbqt`` file. When ``out_dir`` is provided, all SDF files are written
    into that shared destination directory.

    If multiple input files share the same stem and a shared ``out_dir`` is
    used, unique filenames are generated by appending suffixes such as
    ``_2``, ``_3``, and so on.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engine:
        Optional engine filter applied to the parent directory name of
        discovered pose files.
    :type engine: Optional[str]
    :param recursive:
        Whether to recurse into nested directories during file discovery.
    :type recursive: bool
    :param backend:
        Conversion backend passed to the underlying PDBQT-to-SDF converter.
    :type backend: str
    :param overwrite:
        Whether existing SDF files may be overwritten.
    :type overwrite: bool
    :param out_dir:
        Optional shared output directory for all converted SDF files.
    :type out_dir: Optional[str | pathlib.Path]

    :returns:
        Paths to written or reused SDF files.
    :rtype: list[pathlib.Path]

    Example
    -------
    .. code-block:: python

        outputs = convert_pose_tree(
            ["Data/testcase/post/1M17/results/docked"],
            engine="vina",
            recursive=True,
            out_dir="Data/testcase/post/converted_sdf",
        )
    """
    paths = _iter_pdbqt_files(roots, recursive=recursive, engine=engine)
    outputs: List[Path] = []
    seen_names: Set[str] = set()
    shared_dir = _as_path(out_dir) if out_dir is not None else None

    for path in paths:
        if shared_dir is None:
            outputs.append(save_pose_sdf(path, backend=backend, overwrite=overwrite))
            continue

        shared_dir.mkdir(parents=True, exist_ok=True)
        name = f"{path.stem}.sdf"
        if name in seen_names:
            counter = 2
            while f"{path.stem}_{counter}.sdf" in seen_names:
                counter += 1
            name = f"{path.stem}_{counter}.sdf"
        seen_names.add(name)
        outputs.append(
            save_pose_sdf(
                path,
                backend=backend,
                overwrite=overwrite,
                out_file=shared_dir / name,
            )
        )

    return outputs
