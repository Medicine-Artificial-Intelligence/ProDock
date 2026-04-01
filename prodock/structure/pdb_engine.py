from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from prodock.io.logging import get_logger

from .constants import DEFAULT_SOLVENTS
from .selection import chain_selection, resn_selection

logger = get_logger(__name__)

try:
    from pymol import cmd  # type: ignore
except Exception:
    cmd = None


class PDBEngine:
    """
    Step-wise backend engine for preparing a PDB structure for downstream use.

    This class orchestrates a typical receptor-preparation workflow around a
    PyMOL session. The workflow can include:

    - validating runtime requirements and output paths
    - fetching a structure by PDB identifier
    - filtering the structure to selected chains
    - extracting a bound ligand as reference and cocrystal files
    - removing solvents while optionally preserving cofactors
    - saving the filtered protein structure

    The engine is designed in a fluent style, so most public methods return
    the current instance and can be chained.

    :param pdb_id:
        PDB identifier of the structure to fetch and process.
    :type pdb_id: str

    :param base_out:
        Base output directory under which subdirectories for fetched proteins,
        filtered proteins, reference ligands, and cocrystal ligands are created.
    :type base_out: Path

    :param chains:
        Optional list of chain identifiers to keep. If empty or ``None``, all
        chains are retained.
    :type chains: Optional[List[str]]

    :param ligand_code:
        Residue name of the ligand to extract, for example ``"ATP"`` or
        ``"HEM"``. If empty, ligand extraction is skipped.
    :type ligand_code: str

    :param cofactors:
        Optional list of residue names that should be preserved even if they
        appear in the solvent-removal list.
    :type cofactors: Optional[List[str]]

    :param auto_create_dirs:
        Whether output directories should be created automatically when needed.
    :type auto_create_dirs: bool

    :raises RuntimeError:
        Raised later by :meth:`validate` or :meth:`fetch` if PyMOL is not
        available at runtime.

    Example
    -------
    Basic end-to-end usage:

    .. code-block:: python

        from pathlib import Path
        from prodock.structure.pdb_engine import PDBEngine

        engine = (
            PDBEngine(
                pdb_id="1ABC",
                base_out=Path("output"),
                chains=["A"],
                ligand_code="LIG",
                cofactors=["MG", "ZN"],
            )
            .run_all()
        )

        print(engine.filtered_path)
        print(engine.ref_path)
        print(engine.cocrystal_path)

    Example
    -------
    Step-wise usage for finer control:

    .. code-block:: python

        engine = PDBEngine(
            pdb_id="2XYZ",
            base_out=Path("output"),
            chains=["A", "B"],
            ligand_code="ATP",
        )

        (
            engine.validate()
            .fetch()
            .filter_chains()
            .extract_ligand()
            .clean_solvents_and_cofactors()
            .save_filtered_protein()
        )
    """

    def __init__(
        self,
        pdb_id: str,
        base_out: Path,
        chains: Optional[List[str]] = None,
        ligand_code: str = "",
        cofactors: Optional[List[str]] = None,
        auto_create_dirs: bool = True,
    ):
        """
        Initialize a :class:`PDBEngine` instance.

        The constructor stores user configuration and precomputes the standard
        output subdirectories used throughout the workflow. File paths for the
        fetched structure and derived outputs are assigned later during
        :meth:`validate` or :meth:`fetch`.

        :param pdb_id:
            PDB identifier of the target structure.
        :type pdb_id: str

        :param base_out:
            Base output directory.
        :type base_out: Path

        :param chains:
            Optional list of chain identifiers to keep.
        :type chains: Optional[List[str]]

        :param ligand_code:
            Residue name of the ligand to extract. If empty, ligand extraction
            is disabled.
        :type ligand_code: str

        :param cofactors:
            Optional list of residue names to preserve during solvent cleanup.
        :type cofactors: Optional[List[str]]

        :param auto_create_dirs:
            Whether required directories should be created automatically.
        :type auto_create_dirs: bool

        Example
        -------
        .. code-block:: python

            engine = PDBEngine(
                pdb_id="3PTB",
                base_out=Path("results"),
                chains=["A"],
                ligand_code="BEN",
                cofactors=["CA"],
            )
        """
        self.pdb_id = str(pdb_id)
        self.base_out = Path(base_out)
        self.chains = list(chains) if chains else []
        self.ligand_code = ligand_code
        self.cofactors = list(cofactors) if cofactors else []
        self.auto_create_dirs = bool(auto_create_dirs)

        self.fetch_dir = self.base_out / "fetched_protein"
        self.filtered_dir = self.base_out / "filtered_protein"
        self.ref_dir = self.base_out / "reference_ligand"
        self.cocrystal_dir = self.base_out / "cocrystal"

        self.pdb_path: Optional[Path] = None
        self.ref_path: Optional[Path] = None
        self.cocrystal_path: Optional[Path] = None
        self.filtered_path: Optional[Path] = None

    def _ensure_dir(self, p: Path) -> None:
        """
        Create a directory if automatic directory creation is enabled.

        :param p:
            Directory path to create.
        :type p: Path

        :returns:
            This method returns ``None``.
        :rtype: None
        """
        if self.auto_create_dirs:
            p.mkdir(parents=True, exist_ok=True)

    def validate(self) -> "PDBEngine":
        """
        Validate runtime requirements and initialize canonical output paths.

        This method verifies that the PyMOL ``cmd`` API is available, ensures
        that required output directories exist, and sets the expected output
        file paths for the fetched structure, filtered protein, reference
        ligand, and cocrystal ligand.

        :returns:
            The current engine instance.
        :rtype: PDBEngine

        :raises RuntimeError:
            If PyMOL ``cmd`` is not importable.

        Example
        -------
        .. code-block:: python

            engine = PDBEngine("1ABC", Path("out"), ligand_code="LIG")
            engine.validate()

            print(engine.pdb_path)
            print(engine.filtered_path)
            print(engine.ref_path)
            print(engine.cocrystal_path)
        """
        if cmd is None:
            raise RuntimeError(
                "PyMOL 'cmd' is not importable. Install PyMOL for runtime ops."
            )
        for d in (self.fetch_dir, self.filtered_dir, self.ref_dir, self.cocrystal_dir):
            self._ensure_dir(d)

        self.pdb_path = self.fetch_dir / f"{self.pdb_id}.pdb"
        self.filtered_path = self.filtered_dir / f"{self.pdb_id}.pdb"
        self.ref_path = (
            (self.ref_dir / f"{self.ligand_code}.sdf") if self.ligand_code else None
        )
        self.cocrystal_path = self.cocrystal_dir / f"{self.pdb_id}.sdf"
        return self

    def fetch(self) -> "PDBEngine":
        """
        Fetch the requested PDB structure and load it into the active PyMOL session.

        The structure file is retrieved via :func:`fetch_pdb_to_dir`, stored in
        the configured fetch directory, and then loaded into PyMOL.

        :returns:
            The current engine instance.
        :rtype: PDBEngine

        :raises RuntimeError:
            If PyMOL ``cmd`` is not available.

        Example
        -------
        .. code-block:: python

            engine = PDBEngine("1ABC", Path("out")).validate().fetch()
            print(engine.pdb_path)
        """
        if cmd is None:
            raise RuntimeError("PyMOL cmd is not available. Cannot fetch PDB.")
        self._ensure_dir(self.fetch_dir)

        from .fetch import fetch_pdb_to_dir

        chosen = fetch_pdb_to_dir(self.pdb_id, self.fetch_dir)
        self.pdb_path = Path(chosen)
        logger.debug("Loading %s into PyMOL session", chosen)
        cmd.load(str(chosen))
        return self

    def filter_chains(self) -> "PDBEngine":
        """
        Keep only the requested chains in the PyMOL session.

        If no chains were configured, the structure is left unchanged. When
        chains are provided, a PyMOL selection is built and all other atoms are
        removed from the current session.

        :returns:
            The current engine instance.
        :rtype: PDBEngine

        Example
        -------
        .. code-block:: python

            engine = (
                PDBEngine("1ABC", Path("out"), chains=["A", "B"])
                .validate()
                .fetch()
                .filter_chains()
            )
        """
        if not self.chains:
            logger.debug("No chains provided; keeping all chains.")
            return self
        sel = chain_selection(self.chains)
        cmd.select("kept_chains", sel)
        logger.info("Keeping chains selection: %s", sel)
        cmd.select("removed_complex", "all and not kept_chains")
        cmd.remove("removed_complex")
        return self

    def _ligand_selection(self, chain: Optional[str]) -> str:
        """
        Build the PyMOL selection string for the requested ligand.

        :param chain:
            Optional chain identifier. If provided, the ligand selection is
            restricted to that chain.
        :type chain: Optional[str]

        :returns:
            A PyMOL selection expression targeting the configured ligand.
        :rtype: str

        Example
        -------
        .. code-block:: python

            engine = PDBEngine("1ABC", Path("out"), ligand_code="ATP")
            selection = engine._ligand_selection("A")
            # "resn ATP and chain A"
        """
        return f"resn {self.ligand_code}" + (f" and chain {chain}" if chain else "")

    def _count_selected_atoms(self, selection_name: str = "ligand") -> int:
        """
        Count atoms in a PyMOL selection with a fallback strategy.

        The method first tries :func:`cmd.count_atoms`. If that fails, it falls
        back to ``cmd.get_model(...).atom``. If both approaches fail, ``0`` is
        returned.

        :param selection_name:
            Name of the PyMOL selection to inspect.
        :type selection_name: str

        :returns:
            Number of atoms in the selection, or ``0`` if the count cannot be
            determined.
        :rtype: int
        """
        try:
            return int(cmd.count_atoms(selection_name))
        except Exception:
            try:
                return len(cmd.get_model(selection_name).atom)
            except Exception:
                return 0

    def _tmp_ligand_pdb_path(self) -> Path:
        """
        Return the path of the temporary PDB file used for ligand conversion.

        :returns:
            Temporary ligand PDB path in the reference ligand directory.
        :rtype: Path
        """
        return self.ref_dir / f"{self.ligand_code}_tmp.pdb"

    def _cleanup_tmp_ligand_file(self, tmp_pdb: Path) -> None:
        """
        Remove the temporary ligand PDB file if it exists.

        Any filesystem errors are suppressed because this is a best-effort
        cleanup step.

        :param tmp_pdb:
            Temporary PDB file to remove.
        :type tmp_pdb: Path

        :returns:
            This method returns ``None``.
        :rtype: None
        """
        try:
            if tmp_pdb.exists():
                tmp_pdb.unlink()
        except Exception:
            pass

    def _cleanup_extra_ref_sdfs(self) -> None:
        """
        Remove extra SDF files in the reference ligand directory.

        The canonical reference ligand file stored in :attr:`ref_path` is kept,
        while other ``*.sdf`` files in the same directory are removed on a
        best-effort basis.

        :returns:
            This method returns ``None``.
        :rtype: None
        """
        if self.ref_path is None:
            return
        try:
            for p in self.ref_dir.glob("*.sdf"):
                try:
                    if p.resolve() != self.ref_path.resolve():
                        p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    def _save_selected_ligand_to_tmp(self, tmp_pdb: Path) -> bool:
        """
        Save the active ligand selection to a temporary PDB file.

        The current PyMOL selection named ``"ligand"`` is written to the
        provided temporary path. Any pre-existing temporary file is removed
        first.

        :param tmp_pdb:
            Target path of the temporary ligand PDB file.
        :type tmp_pdb: Path

        :returns:
            ``True`` if the temporary file exists after the save attempt,
            otherwise ``False``.
        :rtype: bool
        """
        self._cleanup_tmp_ligand_file(tmp_pdb)
        try:
            cmd.save(str(tmp_pdb), "ligand")
        except Exception as exc:
            logger.warning("PyMOL cmd.save to temporary PDB failed: %s", exc)
        return tmp_pdb.exists()

    def _convert_reference_ligand(self, tmp_pdb: Path) -> bool:
        """
        Convert a temporary ligand PDB into the configured reference SDF file.

        :param tmp_pdb:
            Temporary ligand PDB file to convert.
        :type tmp_pdb: Path

        :returns:
            ``True`` if conversion succeeded and the configured reference SDF
            exists, otherwise ``False``.
        :rtype: bool
        """
        assert self.ref_path is not None
        from .conversion import convert_with_obabel

        try:
            convert_with_obabel(tmp_pdb, self.ref_path, extra_args=("-h",))
        except Exception as exc:
            logger.warning("Reference ligand conversion failed: %s", exc)
            return False

        return self.ref_path.exists() and self.ref_path.stat().st_size > 0

    def _convert_cocrystal_ligand(self) -> bool:
        """
        Convert the reference ligand file into the cocrystal ligand SDF file.

        :returns:
            ``True`` if conversion succeeded and the cocrystal SDF exists,
            otherwise ``False``.
        :rtype: bool
        """
        assert self.ref_path is not None
        assert self.cocrystal_path is not None
        from .conversion import convert_with_obabel

        try:
            convert_with_obabel(
                self.ref_path,
                self.cocrystal_path,
                extra_args=("-h",),
            )
        except Exception as exc:
            logger.warning("Cocrystal ligand conversion failed: %s", exc)
            return False

        return self.cocrystal_path.exists() and self.cocrystal_path.stat().st_size > 0

    def _cleanup_partial_ligand_outputs(self) -> None:
        """
        Remove partially created ligand output files.

        This is a best-effort cleanup method used when ligand extraction or
        conversion fails after some intermediate files were already produced.

        :returns:
            This method returns ``None``.
        :rtype: None
        """
        for path in (self.ref_path, self.cocrystal_path):
            if path is None:
                continue
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

    def _try_extract_ligand_from_chain(self, chain: Optional[str]) -> bool:
        """
        Try to extract and convert the ligand for one chain candidate.

        This method performs one extraction attempt by selecting the configured
        ligand, counting atoms in the selection, saving the ligand to a
        temporary PDB file, converting it to the reference SDF, and then
        converting that result to the cocrystal SDF.

        :param chain:
            Chain identifier to try, or ``None`` to search without a chain
            restriction.
        :type chain: Optional[str]

        :returns:
            ``True`` if both reference and cocrystal ligand files were produced
            successfully, otherwise ``False``.
        :rtype: bool

        Example
        -------
        .. code-block:: python

            engine = PDBEngine(
                pdb_id="1ABC",
                base_out=Path("out"),
                ligand_code="LIG",
                chains=["A", "B"],
            ).validate().fetch()

            success = engine._try_extract_ligand_from_chain("A")
            print(success)
        """
        selection = self._ligand_selection(chain)
        cmd.select("ligand", selection)

        if self._count_selected_atoms("ligand") == 0:
            return False

        tmp_pdb = self._tmp_ligand_pdb_path()
        if not self._save_selected_ligand_to_tmp(tmp_pdb):
            return False

        try:
            if not self._convert_reference_ligand(tmp_pdb):
                return False

            if not self._convert_cocrystal_ligand():
                self._cleanup_partial_ligand_outputs()
                return False

            return True
        finally:
            self._cleanup_tmp_ligand_file(tmp_pdb)
            self._cleanup_extra_ref_sdfs()

    def extract_ligand(self) -> "PDBEngine":
        """
        Extract the requested ligand and save reference and cocrystal files.

        The ligand is written first as a temporary PDB file and then converted
        into the configured SDF outputs. If chain identifiers were provided,
        extraction is attempted chain by chain. If no chains were provided,
        extraction is attempted without chain restriction.

        If no ligand code was configured, this step is skipped.

        :returns:
            The current engine instance.
        :rtype: PDBEngine

        :raises RuntimeError:
            If ligand extraction was requested but no ligand could be saved.

        Example
        -------
        .. code-block:: python

            engine = (
                PDBEngine(
                    pdb_id="1ABC",
                    base_out=Path("out"),
                    ligand_code="ATP",
                    chains=["A"],
                )
                .validate()
                .fetch()
                .extract_ligand()
            )

            print(engine.ref_path)
            print(engine.cocrystal_path)
        """
        if not self.ligand_code:
            logger.debug("No ligand_code provided; skipping ligand extraction.")
            return self

        assert self.ref_path is not None
        assert self.cocrystal_path is not None

        self._ensure_dir(self.ref_dir)
        self._ensure_dir(self.cocrystal_dir)

        chain_candidates = self.chains if self.chains else [None]

        for chain in chain_candidates:
            if self._try_extract_ligand_from_chain(chain):
                try:
                    cmd.remove(f"resn {self.ligand_code}")
                except Exception:
                    pass
                return self

        raise RuntimeError(
            f"Failed to save reference ligand for PDB {self.pdb_id} "
            f"ligand_code={self.ligand_code}"
        )

    def clean_solvents_and_cofactors(self) -> "PDBEngine":
        """
        Remove configured solvent residues while optionally preserving cofactors.

        Solvent residue names are taken from :data:`DEFAULT_SOLVENTS`. If
        cofactors were configured, they are excluded from removal even if they
        overlap with the solvent list.

        :returns:
            The current engine instance.
        :rtype: PDBEngine

        Example
        -------
        .. code-block:: python

            engine = (
                PDBEngine(
                    pdb_id="1ABC",
                    base_out=Path("out"),
                    cofactors=["MG", "ZN"],
                )
                .validate()
                .fetch()
                .clean_solvents_and_cofactors()
            )
        """
        solvent_sel = resn_selection(DEFAULT_SOLVENTS)
        cmd.select("solvents", solvent_sel)
        if self.cofactors:
            cof_sel = resn_selection(self.cofactors)
            cmd.select("cofactors", cof_sel)
            cmd.select("removed_solvent", "solvents and not cofactors")
            logger.info("Preserving cofactors: %s", ", ".join(self.cofactors))
        else:
            cmd.select("removed_solvent", "solvents")
            logger.info("Removing all listed solvents (no cofactors provided).")
        cmd.remove("removed_solvent")
        return self

    def save_filtered_protein(self) -> "PDBEngine":
        """
        Save the current PyMOL session as the filtered protein structure.

        The structure is saved to :attr:`filtered_path` using the PyMOL
        selection ``"all"``. After saving, the PyMOL session is cleared with
        ``cmd.delete("all")`` on a best-effort basis.

        :returns:
            The current engine instance.
        :rtype: PDBEngine

        Example
        -------
        .. code-block:: python

            engine = (
                PDBEngine("1ABC", Path("out"))
                .validate()
                .fetch()
                .save_filtered_protein()
            )

            print(engine.filtered_path)
        """
        try:
            cmd.save(str(self.filtered_path), "all")
            logger.info("Saved filtered protein to: %s", self.filtered_path)
        except Exception as exc:
            logger.warning("PyMOL cmd.save for filtered protein failed: %s", exc)
        try:
            cmd.delete("all")
        except Exception:
            pass
        return self

    def run_all(self) -> "PDBEngine":
        """
        Execute the full PDB preparation workflow.

        The workflow consists of:

        1. :meth:`validate`
        2. :meth:`fetch`
        3. :meth:`filter_chains`
        4. :meth:`extract_ligand`
        5. :meth:`clean_solvents_and_cofactors`
        6. :meth:`save_filtered_protein`

        :returns:
            The current engine instance after all processing steps complete.
        :rtype: PDBEngine

        Example
        -------
        .. code-block:: python

            engine = PDBEngine(
                pdb_id="1ABC",
                base_out=Path("out"),
                chains=["A"],
                ligand_code="LIG",
                cofactors=["MG"],
            ).run_all()

            print("Filtered protein:", engine.filtered_path)
            print("Reference ligand:", engine.ref_path)
            print("Cocrystal ligand:", engine.cocrystal_path)
        """
        return (
            self.validate()
            .fetch()
            .filter_chains()
            .extract_ligand()
            .clean_solvents_and_cofactors()
            .save_filtered_protein()
        )
