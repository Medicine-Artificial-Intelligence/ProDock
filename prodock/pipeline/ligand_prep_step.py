# prodock/pipeline/ligand_prep_step.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import logging
import pandas as pd
from joblib import Parallel, delayed

# Project imports must be at top (fixes E402)
from prodock.preprocess.ligand.ligand_prep import LigandProcess as DefaultLigandProcess
from prodock.pipeline.utils import iter_progress

logger = logging.getLogger("prodock.pipeline.ligand_prep_step")


class LigandPrepStep:
    """
    Batched ligand preparation wrapper around a configurable ``LigandProcess`` class.

    The default implementation uses :class:`prodock.ligand.ligand_prep.LigandProcess`,
    but you can inject a custom class for testing or extension.

    :example:

    >>> from tempfile import TemporaryDirectory
    >>> tmp = TemporaryDirectory()
    >>> step = LigandPrepStep()
    >>> ligs = [{"id": "L1", "smiles": "CCO"}, {"id": "L2", "smiles": "CCN"}]
    >>> step.prep(ligs, tmp.name, n_jobs=1, verbose=0)  # writes to tmp.name
    """

    def __init__(self, ligand_process_cls: Type = DefaultLigandProcess) -> None:
        """
        :param ligand_process_cls: Class implementing the LigandProcess API
                                   (``set_embed_method``, ``set_opt_method``,
                                   ``set_converter_backend``, ``from_list_of_dicts``,
                                   ``process_all``).
        """
        self._cls = ligand_process_cls

    @staticmethod
    def _normalize_ligands(
        ligands: Union[str, List[Dict[str, str]], pd.DataFrame],
        id_key: str,
        smiles_key: str,
    ) -> List[Dict[str, str]]:
        """
        Normalize various ligand inputs into a list of dictionaries.

        :param ligands: Either a single SMILES string, list of dictionaries,
                        or a DataFrame with the required columns.
        :param id_key: Column/key name holding ligand identifier.
        :param smiles_key: Column/key name holding SMILES.
        :returns: List of records with at least ``id_key`` and ``smiles_key``.
        """
        if isinstance(ligands, str):
            return [{id_key: ligands, smiles_key: ligands}]
        if isinstance(ligands, pd.DataFrame):
            return ligands.to_dict("records")
        return list(ligands)

    def prep(
        self,
        ligands: Union[str, List[Dict[str, str]], pd.DataFrame],
        output_dir: Union[str, Path],
        id_key: str = "id",
        smiles_key: str = "smiles",
        embed_algorithm: str = "ETKDGv3",
        conf_opt_method: str = "MMFF94s",
        backend: str = "meeko",
        n_jobs: int = 1,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        parallel_prefer: str = "threads",
    ) -> None:
        """
        Prepare ligands into 3D/conformer + PDBQT (depending on backend settings).

        :param ligands: SMILES (str), list of dicts, or DataFrame.
        :param output_dir: Destination directory for prepared ligands.
        :param id_key: Name of the identifier field in the inputs.
        :param smiles_key: Name of the SMILES field in the inputs.
        :param embed_algorithm: Embedding method label for the LigandProcess.
        :param conf_opt_method: Conformer optimization method label.
        :param backend: Converter backend (e.g., ``"meeko"`` or ``"obabel"``).
        :param n_jobs: Number of parallel batches (``1`` = sequential).
        :param batch_size: Batch size; if ``None`` or >= ``len(ligands)``,
                           processes in one batch.
        :param verbose: 0=silent, 1=log-only, 2+=progress bars where available.
        :param parallel_prefer: ``"threads"`` or ``"processes"`` (joblib hint).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ligs = self._normalize_ligands(ligands, id_key, smiles_key)
        if not ligs:
            logger.warning("No ligands provided to LigandPrepStep.prep()")
            return

        if batch_size is None or batch_size >= len(ligs):
            batches = [ligs]
        else:
            # fmt: off
            batches = [
                ligs[i: i + batch_size] for i in range(0, len(ligs), batch_size)
            ]
            # fmt: on

        def _process_batch(batch: List[Dict[str, str]]) -> None:
            lp = self._cls(output_dir=output_dir, name_key=id_key)
            lp.set_embed_method(embed_algorithm)
            lp.set_opt_method(conf_opt_method)
            lp.set_converter_backend(backend)
            lp.from_list_of_dicts(batch)
            lp.process_all()

        if n_jobs == 1:
            for batch in iter_progress(
                batches, verbose=verbose, desc="Preparing ligands", unit="batch"
            ):
                _process_batch(batch)
        else:
            Parallel(n_jobs=n_jobs, prefer=parallel_prefer)(
                delayed(_process_batch)(batch) for batch in batches
            )
