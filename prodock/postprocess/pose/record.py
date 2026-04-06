from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PoseRecord:
    """
    Lightweight metadata container for one docked pose.

    This record stores the minimal normalized metadata associated with a single
    pose extracted from a docked ``.pdbqt`` file. It is used internally by the
    pose postprocessing workflow before conversion to public DataFrame-based
    outputs.

    A single source ``.pdbqt`` file may produce multiple :class:`PoseRecord`
    objects, one for each pose rank discovered in that file.

    :param receptor_id:
        Optional receptor identifier. This is typically inferred from a
        hierarchical ProDock path such as
        ``Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt``.
        It may be ``None`` for direct-file and flat-folder inputs where receptor
        metadata is not encoded in the path.
    :type receptor_id: Optional[str]
    :param ligand_id:
        Ligand identifier inferred from the input pose filename stem after
        normalization of common suffixes such as ``_docked`` or ``_out``.
    :type ligand_id: str
    :param engine:
        Docking engine name, for example ``vina``, ``smina``, or ``qvina``.
    :type engine: str
    :param pose_rank:
        One-based rank of the pose inside the source ``.pdbqt`` file.
    :type pose_rank: int
    :param affinity:
        Parsed affinity score for the pose when available. This may be ``None``
        if the input file does not contain a recognized score annotation.
    :type affinity: Optional[float]
    :param source_file:
        Source ``.pdbqt`` file from which the pose metadata was extracted.
    :type source_file: pathlib.Path

    Example
    -------
    .. code-block:: python

        from pathlib import Path
        from prodock.postprocess.pose.model import PoseRecord

        record = PoseRecord(
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="vina",
            pose_rank=1,
            affinity=-7.2,
            source_file=Path(
                "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
            ),
        )

    Another real example for other engines:

    .. code-block:: python

        smina_record = PoseRecord(
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="smina",
            pose_rank=1,
            affinity=-7.4,
            source_file=Path(
                "Data/testcase/post/1M17/results/docked/smina/erlotinib_docked.pdbqt"
            ),
        )

        qvina_record = PoseRecord(
            receptor_id="1M17",
            ligand_id="erlotinib",
            engine="qvina",
            pose_rank=1,
            affinity=-7.1,
            source_file=Path(
                "Data/testcase/post/1M17/results/docked/qvina/erlotinib_docked.pdbqt"
            ),
        )
    """

    receptor_id: Optional[str]
    ligand_id: str
    engine: str
    pose_rank: int
    affinity: Optional[float]
    source_file: Path
