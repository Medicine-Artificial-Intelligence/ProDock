from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .convert import convert_pose_tree
from .io import build_pose_mol_rows, build_pose_records
from .select import best_pose_per_group, pose_mols_to_dataframe, poses_to_dataframe

PathLike = str | Path


class PoseCrawler:
    """
    High-level helper for discovering, summarizing, converting, and loading
    docked poses.

    This class provides a compact interface over the lower-level pose utilities
    for:

    - discovering pose files
    - building :class:`PoseRecord` entries
    - converting records into DataFrames
    - loading RDKit molecules
    - selecting the best-scoring pose per group
    - converting discovered ``.pdbqt`` files to ``.sdf``

    Supported input layouts
    -----------------------
    1. A direct path to one ``.pdbqt`` file with ``engine=...``.
    2. A direct path to a flat folder of ``.pdbqt`` files with ``engine=...``.
    3. A higher-level ProDock tree such as
       ``<root>/<receptor>/results/docked/<engine>/*.pdbqt``, where receptor id
       and engine are inferred automatically.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engine:
        Optional engine hint for direct-file or flat-directory inputs, or an
        optional filter for hierarchical ProDock trees.
    :type engine: Optional[str]
    :param recursive:
        Whether nested directories should be searched recursively.
    :type recursive: bool

    Example
    -------
    .. code-block:: python

        from prodock.postprocess.pose.core import PoseCrawler

        crawler = PoseCrawler(["Data/testcase/post"])

        df = crawler.crawl()
        best_df = crawler.best()

        mol_df = crawler.crawl_mols(save_sdf=True)
        best_mol_df = crawler.best_mols()

        sdf_paths = crawler.convert(out_dir="Data/testcase/post/converted_sdf")

    A direct single-file workflow is also supported:

    .. code-block:: python

        crawler = PoseCrawler(
            ["Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"],
            engine="vina",
        )
        df = crawler.crawl()

    Common real input examples include:

    .. code-block:: python

        "Data/testcase/post/1M17/results/docked/qvina/erlotinib_docked.pdbqt"
        "Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"
        "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
    """

    def __init__(
        self,
        roots: Sequence[PathLike],
        *,
        engine: Optional[str] = None,
        recursive: bool = True,
    ) -> None:
        """
        Initialize a pose crawler.

        :param roots:
            Root files or directories to inspect.
        :type roots: Sequence[str | pathlib.Path]
        :param engine:
            Optional engine hint for direct-file or flat-directory inputs, or an
            optional filter for hierarchical ProDock trees.
        :type engine: Optional[str]
        :param recursive:
            Whether nested directories should be searched recursively.
        :type recursive: bool
        """
        self.roots = list(roots)
        self.engine = engine
        self.recursive = recursive

    def records(self):
        """
        Return discovered pose records.

        This method delegates to :func:`prodock.postprocess.pose.io.build_pose_records`
        using the crawler configuration captured at initialization.

        :returns:
            Discovered pose records.
        :rtype: list[prodock.postprocess.pose.model.PoseRecord]

        Example
        -------
        .. code-block:: python

            crawler = PoseCrawler(["Data/testcase/post"])
            records = crawler.records()
        """
        return build_pose_records(
            self.roots,
            engine=self.engine,
            recursive=self.recursive,
        )

    def crawl(self) -> pd.DataFrame:
        """
        Return discovered pose records as a standardized DataFrame.

        The returned table uses the public pose schema:

        - ``receptor_id``
        - ``ligand_id``
        - ``engine``
        - ``pose_rank``
        - ``affinity``

        :returns:
            Pose summary table.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            crawler = PoseCrawler(["Data/testcase/post"])
            df = crawler.crawl()
        """
        return poses_to_dataframe(self.records())

    def crawl_mols(
        self,
        *,
        backend: str = "obabel",
        sanitize: bool = True,
        remove_hs: bool = False,
        save_sdf: bool = False,
        overwrite_sdf: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame containing pose metadata and RDKit molecules.

        This method loads molecules from the discovered pose files and returns a
        standardized table with the public pose-plus-molecule schema:

        - ``receptor_id``
        - ``ligand_id``
        - ``engine``
        - ``pose_rank``
        - ``affinity``
        - ``mol``

        :param backend:
            Conversion backend used during PDBQT-to-SDF conversion.
        :type backend: str
        :param sanitize:
            Whether imported RDKit molecules should be sanitized.
        :type sanitize: bool
        :param remove_hs:
            Whether hydrogens should be removed during SDF import.
        :type remove_hs: bool
        :param save_sdf:
            Whether to also write an SDF file beside each source ``.pdbqt`` file.
        :type save_sdf: bool
        :param overwrite_sdf:
            Whether an existing neighboring SDF file may be overwritten.
        :type overwrite_sdf: bool

        :returns:
            Pose table with RDKit molecules.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            crawler = PoseCrawler(
                ["Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"],
                engine="smina",
            )
            mol_df = crawler.crawl_mols(save_sdf=True)
        """
        rows = build_pose_mol_rows(
            self.roots,
            engine=self.engine,
            recursive=self.recursive,
            backend=backend,
            sanitize=sanitize,
            remove_hs=remove_hs,
            save_sdf=save_sdf,
            overwrite_sdf=overwrite_sdf,
        )
        return pose_mols_to_dataframe(rows)

    def best(
        self,
        *,
        by: Sequence[str] = ("receptor_id", "ligand_id", "engine"),
    ) -> pd.DataFrame:
        """
        Return best-scoring pose rows per group.

        Lower affinity is treated as better. By default, one best row is
        selected for each ``(receptor_id, ligand_id, engine)`` group.

        :param by:
            Grouping columns that define independent selection groups.
        :type by: Sequence[str]

        :returns:
            Best-scoring rows per group.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            crawler = PoseCrawler(["Data/testcase/post"])
            best_df = crawler.best()
        """
        return best_pose_per_group(self.records(), by=by)

    def best_mols(
        self,
        *,
        by: Sequence[str] = ("receptor_id", "ligand_id", "engine"),
        backend: str = "obabel",
        sanitize: bool = True,
        remove_hs: bool = False,
        save_sdf: bool = False,
        overwrite_sdf: bool = False,
    ) -> pd.DataFrame:
        """
        Return best-scoring pose rows per group, including RDKit molecules.

        This method first builds a pose-plus-molecule DataFrame and then applies
        best-pose selection on top of it.

        :param by:
            Grouping columns that define independent selection groups.
        :type by: Sequence[str]
        :param backend:
            Conversion backend used during PDBQT-to-SDF conversion.
        :type backend: str
        :param sanitize:
            Whether imported RDKit molecules should be sanitized.
        :type sanitize: bool
        :param remove_hs:
            Whether hydrogens should be removed during SDF import.
        :type remove_hs: bool
        :param save_sdf:
            Whether to also write an SDF file beside each source ``.pdbqt`` file.
        :type save_sdf: bool
        :param overwrite_sdf:
            Whether an existing neighboring SDF file may be overwritten.
        :type overwrite_sdf: bool

        :returns:
            Best-scoring rows with molecule objects.
        :rtype: pandas.DataFrame

        Example
        -------
        .. code-block:: python

            crawler = PoseCrawler(["Data/testcase/post"])
            best_mol_df = crawler.best_mols(save_sdf=False)
        """
        df = self.crawl_mols(
            backend=backend,
            sanitize=sanitize,
            remove_hs=remove_hs,
            save_sdf=save_sdf,
            overwrite_sdf=overwrite_sdf,
        )
        return best_pose_per_group(df, by=by)

    def convert(
        self,
        *,
        backend: str = "obabel",
        overwrite: bool = False,
        out_dir: Optional[PathLike] = None,
    ):
        """
        Convert discovered PDBQT pose files into SDF files.

        When ``out_dir`` is omitted, each SDF file is written beside its source
        ``.pdbqt`` file. When ``out_dir`` is provided, converted files are
        written into that shared destination directory.

        :param backend:
            Conversion backend used for PDBQT-to-SDF conversion.
        :type backend: str
        :param overwrite:
            Whether existing output files may be overwritten.
        :type overwrite: bool
        :param out_dir:
            Optional shared output directory. When omitted, SDF files are saved
            beside the source ``.pdbqt`` files.
        :type out_dir: Optional[str | pathlib.Path]

        :returns:
            Written or reused SDF paths.
        :rtype: list[pathlib.Path]

        Example
        -------
        .. code-block:: python

            crawler = PoseCrawler(["Data/testcase/post"])
            sdf_paths = crawler.convert(
                out_dir="Data/testcase/post/converted_sdf",
                overwrite=True,
            )
        """
        return convert_pose_tree(
            self.roots,
            engine=self.engine,
            recursive=self.recursive,
            backend=backend,
            overwrite=overwrite,
            out_dir=out_dir,
        )


def crawl_poses(
    roots: Sequence[PathLike],
    *,
    engine: Optional[str] = None,
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper around :meth:`PoseCrawler.crawl`.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engine:
        Optional engine hint or filter.
    :type engine: Optional[str]
    :param recursive:
        Whether nested directories should be searched recursively.
    :type recursive: bool

    :returns:
        Standardized pose summary table.
    :rtype: pandas.DataFrame

    Example
    -------
    .. code-block:: python

        df = crawl_poses(["Data/testcase/post"])
    """
    return PoseCrawler(roots, engine=engine, recursive=recursive).crawl()


def crawl_pose_mols(
    roots: Sequence[PathLike],
    *,
    engine: Optional[str] = None,
    recursive: bool = True,
    backend: str = "obabel",
    sanitize: bool = True,
    remove_hs: bool = False,
    save_sdf: bool = False,
    overwrite_sdf: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper around :meth:`PoseCrawler.crawl_mols`.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engine:
        Optional engine hint or filter.
    :type engine: Optional[str]
    :param recursive:
        Whether nested directories should be searched recursively.
    :type recursive: bool
    :param backend:
        Conversion backend used during PDBQT-to-SDF conversion.
    :type backend: str
    :param sanitize:
        Whether imported RDKit molecules should be sanitized.
    :type sanitize: bool
    :param remove_hs:
        Whether hydrogens should be removed during SDF import.
    :type remove_hs: bool
    :param save_sdf:
        Whether to also write an SDF beside each source ``.pdbqt`` file.
    :type save_sdf: bool
    :param overwrite_sdf:
        Whether an existing neighboring SDF file may be overwritten.
    :type overwrite_sdf: bool

    :returns:
        Standardized pose-plus-molecule table.
    :rtype: pandas.DataFrame

    Example
    -------
    .. code-block:: python

        mol_df = crawl_pose_mols(
            ["Data/testcase/post/1M17/results/docked/qvina/erlotinib_docked.pdbqt"],
            engine="qvina",
            save_sdf=True,
        )
    """
    return PoseCrawler(roots, engine=engine, recursive=recursive).crawl_mols(
        backend=backend,
        sanitize=sanitize,
        remove_hs=remove_hs,
        save_sdf=save_sdf,
        overwrite_sdf=overwrite_sdf,
    )


def select_best_poses(
    roots: Sequence[PathLike],
    *,
    engine: Optional[str] = None,
    recursive: bool = True,
    by: Sequence[str] = ("receptor_id", "ligand_id", "engine"),
) -> pd.DataFrame:
    """
    Convenience wrapper around :meth:`PoseCrawler.best`.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engine:
        Optional engine hint or filter.
    :type engine: Optional[str]
    :param recursive:
        Whether nested directories should be searched recursively.
    :type recursive: bool
    :param by:
        Grouping columns that define independent selection groups.
    :type by: Sequence[str]

    :returns:
        Best-scoring pose rows per group.
    :rtype: pandas.DataFrame

    Example
    -------
    .. code-block:: python

        best_df = select_best_poses(["Data/testcase/post"])
    """
    return PoseCrawler(roots, engine=engine, recursive=recursive).best(by=by)


def select_best_pose_mols(
    roots: Sequence[PathLike],
    *,
    engine: Optional[str] = None,
    recursive: bool = True,
    by: Sequence[str] = ("receptor_id", "ligand_id", "engine"),
    backend: str = "obabel",
    sanitize: bool = True,
    remove_hs: bool = False,
    save_sdf: bool = False,
    overwrite_sdf: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper around :meth:`PoseCrawler.best_mols`.

    :param roots:
        Root files or directories to inspect.
    :type roots: Sequence[str | pathlib.Path]
    :param engine:
        Optional engine hint or filter.
    :type engine: Optional[str]
    :param recursive:
        Whether nested directories should be searched recursively.
    :type recursive: bool
    :param by:
        Grouping columns that define independent selection groups.
    :type by: Sequence[str]
    :param backend:
        Conversion backend used during PDBQT-to-SDF conversion.
    :type backend: str
    :param sanitize:
        Whether imported RDKit molecules should be sanitized.
    :type sanitize: bool
    :param remove_hs:
        Whether hydrogens should be removed during SDF import.
    :type remove_hs: bool
    :param save_sdf:
        Whether to also write an SDF beside each source ``.pdbqt`` file.
    :type save_sdf: bool
    :param overwrite_sdf:
        Whether an existing neighboring SDF file may be overwritten.
    :type overwrite_sdf: bool

    :returns:
        Best-scoring pose rows with molecule objects.
    :rtype: pandas.DataFrame

    Example
    -------
    .. code-block:: python

        best_mol_df = select_best_pose_mols(
            ["Data/testcase/post"],
            save_sdf=False,
        )
    """
    return PoseCrawler(roots, engine=engine, recursive=recursive).best_mols(
        by=by,
        backend=backend,
        sanitize=sanitize,
        remove_hs=remove_hs,
        save_sdf=save_sdf,
        overwrite_sdf=overwrite_sdf,
    )
