# prodock/pipeline/pipeline.py
from __future__ import annotations

import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

import logging

# Project imports must be at top (fixes E402)
from prodock.pipeline.repr_helpers import ProDockReprMixin
from prodock.pipeline.receptor_step import ReceptorStep
from prodock.pipeline.ligand_prep_step import LigandPrepStep
from prodock.pipeline.dock_step import DockStep
from prodock.preprocess.gridbox import GridBox

logger = logging.getLogger("prodock.pipeline")
logger.addHandler(logging.NullHandler())

_PIPELINE_STATE_VERSION = 1


class ProDock(ProDockReprMixin):
    """
    High-level orchestration that wires receptor preparation, ligand preparation,
    and docking into a simple, reproducible workflow.

    The class is dependency-injection friendly; you can pass custom step classes
    for testing. State can be serialized to a small pickle (no heavy objects).

    :example:

    >>> from tempfile import TemporaryDirectory
    >>> tmp = TemporaryDirectory()
    >>> # Minimal pipeline using DI for steps in your tests, or the defaults here.
    >>> pp = ProDock(
    ...     target_path="Data/testcase/dock/receptor/5N2F.pdb",
    ...     project_dir=tmp.name,
    ...     cfg_box={"center_x": 0, "center_y": 0, "center_z": 0,
    ...              "size_x": 20, "size_y": 20, "size_z": 20},
    ... )
    >>> # In real use: pp.prepare_receptor(); pp.prep(...); pp.dock(...)
    """

    def __init__(
        self,
        target_path: Union[str, Path],
        crystal: bool = False,
        ligand_path: Optional[Union[str, Path]] = None,
        project_dir: Optional[Union[str, Path]] = None,
        cfg_box: Optional[Dict[str, float]] = None,
        receptor_step: Optional[ReceptorStep] = None,
        ligand_step: Optional[LigandPrepStep] = None,
        dock_step: Optional[DockStep] = None,
    ) -> None:
        """
        :param target_path: Input receptor (PDB/PDBQT) to prepare.
        :param crystal: Whether to use crystal restraints (passed into prep step
                        if your ReceptorPrep supports it).
        :param ligand_path: Optional reference ligand path used to auto-infer a
                            grid box when ``cfg_box`` is not provided.
        :param project_dir: Working directory; pipeline writes subfolders inside.
        :param cfg_box: Precomputed vina-style box dictionary; if absent and
                        ``ligand_path`` is provided, the constructor attempts to
                        infer it via :class:`~prodock.preprocess.gridbox.GridBox`.
        :param receptor_step: Optional injected :class:`ReceptorStep`.
        :param ligand_step: Optional injected :class:`LigandPrepStep`.
        :param dock_step: Optional injected :class:`DockStep`.
        """
        self.target_path = Path(target_path).resolve()
        self.crystal = bool(crystal)
        self.ligand_path = Path(ligand_path).resolve() if ligand_path else None
        self.project_dir = Path(project_dir or os.getcwd()).resolve()

        # Directories
        self.ligand_dir = self.project_dir / "ligand"
        self.receptor_dir = self.project_dir / "receptor"
        self.output_dir = self.project_dir / "output"
        self.logs_dir = self.output_dir / "Log"
        self.modes_dir = self.output_dir / "Mode"
        for d in (
            self.ligand_dir,
            self.receptor_dir,
            self.output_dir,
            self.logs_dir,
            self.modes_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Grid box configuration
        self.cfg_box: Optional[Dict[str, float]] = None
        if cfg_box is not None:
            if not isinstance(cfg_box, dict):
                raise TypeError("cfg_box must be dict")
            self.cfg_box = cfg_box
        else:
            if self.ligand_path:
                try:
                    fmt = self._guess_format_from_path(self.ligand_path)
                    gb = GridBox().load_ligand(str(self.ligand_path), fmt=fmt)
                    if hasattr(gb, "from_ligand_pad"):
                        gb = gb.from_ligand_pad(pad=4.0, isotropic=True)
                    self.cfg_box = gb.vina_dict
                except Exception as exc:  # pragma: no cover - depends on GridBox
                    logger.warning("Failed to infer cfg_box from ligand_path: %s", exc)

        self.target: Optional[Path] = None

        # Steps (DI)
        self.receptor_step = receptor_step or ReceptorStep()
        self.ligand_step = ligand_step or LigandPrepStep()
        self.dock_step = dock_step or DockStep()

    # Helpers
    def _guess_format_from_path(self, p: Union[str, Path]) -> str:
        """
        Guess ligand format from path suffix.

        :param p: Path to ligand-like file.
        :returns: Lowercase suffix without dot (e.g., ``"sdf"``).
        """
        p = Path(p)
        suffix = p.suffix.lower().lstrip(".")
        if suffix in {
            "sdf",
            "sd",
            "mol",
            "mol2",
            "pdb",
            "pdbqt",
            "smi",
            "smi.txt",
            "smi.gz",
        }:
            return suffix
        return "sdf"

    # --- public API ---
    def prepare_receptor(
        self,
        energy_diff: float = 10.0,
        max_minimization_steps: int = 5000,
        minimize_in_water: bool = False,
        out_fmt: str = "pdbqt",
        enable_logging: bool = True,
    ) -> Path:
        """
        Prepare the receptor via :class:`ReceptorStep`.

        :param energy_diff: Energy diff threshold (kcal/mol) for cleanup/filtering.
        :param max_minimization_steps: Maximum minimization iterations.
        :param minimize_in_water: Whether to minimize in implicit water (if supported).
        :param out_fmt: Output format (e.g., ``"pdbqt"``).
        :param enable_logging: Pass-through toggle for step logging.
        :returns: Path to prepared receptor.
        """
        self.target = self.receptor_step.prepare(
            input_pdb=self.target_path,
            output_dir=self.receptor_dir,
            energy_diff=energy_diff,
            max_minimization_steps=max_minimization_steps,
            minimize_in_water=minimize_in_water,
            out_fmt=out_fmt,
            enable_logging=enable_logging,
        )
        return self.target

    def prep(
        self,
        ligands: Union[str, List[Dict[str, str]], "pd.DataFrame"],
        smiles_key: str = "smiles",
        id_key: str = "id",
        embed_algorithm: str = "ETKDGv3",
        conf_opt_method: str = "MMFF94s",
        backend: str = "meeko",
        n_jobs: int = 1,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        parallel_prefer: str = "threads",
    ) -> None:
        """
        Prepare ligands via :class:`LigandPrepStep`.

        :param ligands: SMILES string, list of dicts, or DataFrame.
        :param smiles_key: Name of the SMILES field.
        :param id_key: Name of the identifier field.
        :param embed_algorithm: Embedding algorithm label.
        :param conf_opt_method: Conformer optimization method label.
        :param backend: Converter backend (e.g., meeko/obabel).
        :param n_jobs: Parallel batches (1 = sequential).
        :param batch_size: Optional batch size.
        :param verbose: 0=silent, 1=log-only, 2+=progress bars where available.
        :param parallel_prefer: ``"threads"`` or ``"processes"``.
        """
        self.ligand_step.prep(
            ligands=ligands,
            output_dir=self.ligand_dir,
            id_key=id_key,
            smiles_key=smiles_key,
            embed_algorithm=embed_algorithm,
            conf_opt_method=conf_opt_method,
            backend=backend,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            parallel_prefer=parallel_prefer,
        )

    def dock(
        self,
        backend: str = "smina",
        exhaustiveness: int = 8,
        num_modes: int = 9,
        cpu: int = 4,
        seed: Optional[int] = None,
        ligand_glob: str = "*.pdbqt",
        batch_size: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 1,
        parallel_prefer: str = "threads",
    ) -> List[Dict[str, Union[str, bool]]]:
        """
        Dock all prepared ligands against the prepared receptor.

        :param backend: Backend name (e.g., ``"smina"``).
        :param exhaustiveness: Search exhaustiveness.
        :param num_modes: Number of poses to output.
        :param cpu: CPU threads.
        :param seed: Optional random seed.
        :param ligand_glob: Pattern for prepared ligand files.
        :param batch_size: Optional batch size for docking.
        :param n_jobs: Parallelism level.
        :param verbose: 0=silent, 1=log-only, 2+=progress bars where available.
        :param parallel_prefer: ``"threads"`` or ``"processes"``.
        :returns: List of dictionaries with status for each ligand.
        :raises RuntimeError: If receptor not prepared or cfg_box missing.
        """
        if self.target is None:
            raise RuntimeError(
                "Target receptor not prepared. Call prepare_receptor() first."
            )
        if not self.cfg_box:
            raise RuntimeError(
                "cfg_box not set. Provide cfg_box or a reference ligand to infer it."
            )

        return self.dock_step.dock(
            receptor_path=str(self.target),
            ligand_dir=str(self.ligand_dir),
            output_modes_dir=str(self.modes_dir),
            logs_dir=str(self.logs_dir),
            cfg_box=self.cfg_box,
            backend=backend,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            cpu=cpu,
            seed=seed,
            ligand_glob=ligand_glob,
            batch_size=batch_size,
            n_jobs=n_jobs,
            verbose=verbose,
            parallel_prefer=parallel_prefer,
        )

    def summary(self) -> Dict[str, Union[str, bool, None]]:
        """
        Lightweight snapshot for human-readable reporting.

        :returns: Dict of key paths and flags (e.g., whether ``cfg_box`` is present).
        """
        return {
            "target_path": str(self.target_path),
            "project_dir": str(self.project_dir),
            "ligand_dir": str(self.ligand_dir),
            "receptor_dir": str(self.receptor_dir),
            "output_dir": str(self.output_dir),
            "has_ligand_path": bool(self.ligand_path),
            "cfg_box_present": bool(self.cfg_box),
            "prepared_target": str(self.target) if self.target else None,
        }

    # --- persistence (pickle minimal state) ---
    def state_dict(self) -> Dict[str, Union[str, bool, Dict[str, str], None]]:
        """
        Minimal, portable state suitable for pickling.

        :returns: Serializable dictionary (no heavy objects).
        """
        return {
            "version": _PIPELINE_STATE_VERSION,
            "target_path": str(self.target_path),
            "crystal": self.crystal,
            "ligand_path": str(self.ligand_path) if self.ligand_path else None,
            "project_dir": str(self.project_dir),
            "dirs": {
                "ligand_dir": str(self.ligand_dir),
                "receptor_dir": str(self.receptor_dir),
                "output_dir": str(self.output_dir),
                "logs_dir": str(self.logs_dir),
                "modes_dir": str(self.modes_dir),
            },
            "cfg_box": dict(self.cfg_box) if self.cfg_box else None,
            "prepared_target": str(self.target) if self.target else None,
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save pipeline state as a pickle.

        :param path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProDock":
        """
        Restore a pipeline from :meth:`save`.

        :param path: Pickle path produced by :meth:`save`.
        :returns: Reconstructed :class:`ProDock` with default steps.
        """
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls(
            target_path=state["target_path"],
            crystal=state.get("crystal", False),
            ligand_path=state.get("ligand_path"),
            project_dir=state.get("project_dir"),
            cfg_box=state.get("cfg_box"),
        )
        prepared = state.get("prepared_target")
        obj.target = Path(prepared) if prepared else None
        return obj
