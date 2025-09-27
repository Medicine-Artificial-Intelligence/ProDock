from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, List, Dict, Any, Union

from prodock.io.logging import get_logger
from .orchestrator import PDBOrchestrator

logger = get_logger(__name__)


class PDBQuery:
    """
    Thin public wrapper around the PDB orchestrator.

    This wrapper preserves the old public API surface while delegating
    the implementation of individual steps to smaller modules
    (fetch, convert, selection, etc.). Use this class when you want
    a single-entry point that behaves like the original `PDBQuery`.

    :param str pdb_id: PDB identifier (case-insensitive), e.g. ``"5N2F"``.
    :param Union[str, pathlib.Path] output_dir:
        Base output directory where per-PDB subfolders will be created.
    :param Optional[Sequence[str]] chains:
        Sequence of chain identifiers to keep (e.g. ``["A"]``). If ``None`` or
        empty, all chains are preserved.
    :param str ligand_code:
        Three-letter ligand residue name to extract (e.g. ``"HEM"`` or ``"8HW"``).
    :param Optional[str] ligand_name:
        Friendly ligand name. Kept for compatibility; canonical filenames use ``ligand_code``.
    :param Optional[Sequence[str]] cofactors:
        Residue names to preserve when cleaning solvents (e.g. ``["HEM"]``).
    :param Optional[str] protein_name:
        Optional friendly protein name (not used for canonical filenames).
    :param bool auto_create_dirs:
        If ``True`` (default) the orchestrator will create the standard output
        sub-folders (``fetched_protein``, ``filtered_protein``, ``reference_ligand``,
        ``cocrystal``). Set to ``False`` to opt out of automatic directory creation.

    :returns: instance of :class:`PDBQuery`

    .. note::
        The machinery for fetching / selection / conversion is implemented in
        :mod:`prodock.structure.fetch`, :mod:`prodock.structure.selection` and
        :mod:`prodock.structure.convert`. The wrapper only exposes a compact,
        backward-compatible API.

    Examples
    --------
    Basic usage::

        .. code-block:: python

            from prodock.structure import PDBQuery

            pq = PDBQuery(
                pdb_id="5N2F",
                output_dir="out/5N2F",
                chains=["A"],
                ligand_code="HEM",
                cofactors=["HEM"],
                auto_create_dirs=True
            )
            pq.run_all()

    Batch usage (convenience helper)::

        .. code-block:: python

            items = [
                {"pdb_id": "5N2F", "ligand_code": "HEM", "chains": ["A"]},
                {"pdb_id": "1ABC", "ligand_code": "ABC", "chains": []},
            ]
            PDBQuery.process_batch(items, output_dir="out/batch")
    """

    def __init__(
        self,
        pdb_id: str,
        output_dir: Union[str, Path],
        chains: Optional[Sequence[str]] = None,
        ligand_code: str = "",
        ligand_name: Optional[str] = None,
        cofactors: Optional[Sequence[str]] = None,
        protein_name: Optional[str] = None,
        auto_create_dirs: bool = True,
    ) -> None:
        self._orchestrator = PDBOrchestrator(
            pdb_id=str(pdb_id),
            base_out=Path(output_dir),
            chains=list(chains) if chains else [],
            ligand_code=ligand_code,
            cofactors=list(cofactors) if cofactors else [],
            auto_create_dirs=bool(auto_create_dirs),
        )

    def validate(self):
        """
        Ensure runtime preconditions and (optionally) create output directories.

        Returns
        -------
        self
        """
        self._orchestrator.validate()
        return self

    def run_all(self):
        """
        Run the full pipeline:

          validate() -> fetch() -> filter_chains() -> extract_ligand() ->
          clean_solvents_and_cofactors() -> save_filtered_protein()

        Returns
        -------
        self
        """
        self._orchestrator.run_all()
        return self

    # Properties (string paths to remain backward-compatible)
    @property
    def pdb_path(self) -> Optional[str]:
        return str(self._orchestrator.pdb_path) if self._orchestrator.pdb_path else None

    @property
    def filtered_protein_path(self) -> Optional[str]:
        return (
            str(self._orchestrator.filtered_path)
            if self._orchestrator.filtered_path
            else None
        )

    @property
    def reference_ligand_path(self) -> Optional[str]:
        return str(self._orchestrator.ref_path) if self._orchestrator.ref_path else None

    @property
    def cocrystal_ligand_path(self) -> Optional[str]:
        return (
            str(self._orchestrator.cocrystal_path)
            if self._orchestrator.cocrystal_path
            else None
        )

    # batch helper kept as top-level convenience
    @classmethod
    def process_batch(
        cls,
        items: Union[List[Dict[str, Any]], Any],
        output_dir: Union[str, Path],
        **kwargs,
    ):
        """
        Batch helper (kept for backwards compatibility).

        Example
        -------
        items = [{"pdb_id": "5N2F", "ligand_code": "HEM", "chains": ["A"]}]
        PDBQuery.process_batch(items, output_dir="out/batch")
        """
        from .batch import process_batch as _process_batch

        return _process_batch(items=items, output_dir=output_dir, **kwargs)
