# prodock/chem/optimize.py
"""
Optimizer: RDKit-only optimization utilities (OOP) for prodock.chem.

Exposed algorithms:
  - 'UFF'
  - 'MMFF'     (alias of 'MMFF94')
  - 'MMFF94'
  - 'MMFF94S'  (the 's' variant)

Works with RDKit Mol objects or MolBlock strings.
Writes energy tags as molecule properties (CONF_ENERGY_<confId>) when exporting to SDF.
"""

from __future__ import annotations
from typing import List, Dict, Iterable
from pathlib import Path
import logging

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    raise ImportError(
        "RDKit is required for prodock.chem.optimize: install rdkit from conda-forge"
    ) from e

# prodock logging utilities — unified import + robust fallback
try:
    from prodock.io.logging import get_logger, StructuredAdapter
except Exception:

    def get_logger(name: str):
        return logging.getLogger(name)

    class StructuredAdapter(logging.LoggerAdapter):
        def __init__(self, logger, extra):
            super().__init__(logger, extra)


logger = StructuredAdapter(
    get_logger("prodock.chem.optimize"), {"component": "optimize"}
)
logger._base_logger = getattr(logger, "_base_logger", getattr(logger, "logger", None))


class Optimizer:
    """
    Optimizer class for UFF / MMFF optimizations.

    Methods are chainable (return self). Use properties to access results.

    Example
    -------
    A minimal example showing how to create an Optimizer, load MolBlocks, run optimization
    and export results to an SDF:

    .. code-block:: python

        from prodock.chem.optimize import Optimizer
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # create a simple molecule and embed into 1 conformer, then convert to MolBlock
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol)
        molblock = Chem.MolToMolBlock(mol)

        opt = Optimizer(max_iters=250)
        opt.load_molblocks([molblock]) \\
           .optimize_all(method="UFF") \\
           .write_sdf("out_folder", per_mol_folder=False, write_energy_tags=True)

    Parameters
    ----------
    :param max_iters:
        Maximum number of iterations for internal minimizer calls (passed to RDKit
        force-field minimizers). Defaults to ``200``.
    :type max_iters: int

    Attributes
    ----------
    _molblocks_in : List[str]
        Internal list of input MolBlock strings loaded via :meth:`load_molblocks`.
    _optimized_blocks : List[str]
        MolBlock strings after optimization (updated in-place by :meth:`optimize_all`).
    _energies : List[Dict[int, float]]
        Per-molecule mapping of conformer id -> energy (filled by :meth:`optimize_all`).
    """

    def __init__(self, max_iters: int = 200) -> None:
        self.max_iters = int(max_iters)
        self._molblocks_in: List[str] = []
        self._optimized_blocks: List[str] = []
        self._energies: List[Dict[int, float]] = []  # per molecule: confId -> energy

    def __repr__(self) -> str:
        return (
            f"<Optimizer inputs={len(self._molblocks_in)}"
            + f" optimized={len(self._optimized_blocks)} max_iters={self.max_iters}>"
        )

    # ---------------- properties ----------------
    @property
    def optimized_molblocks(self) -> List[str]:
        """
        Return a shallow copy of the optimized MolBlock strings.

        :return: list of MolBlock strings representing optimized molecules
        :rtype: List[str]
        """
        return list(self._optimized_blocks)

    @property
    def energies(self) -> List[Dict[int, float]]:
        """
        Return a deep-ish copy of stored per-molecule energies.

        Each list element corresponds to one input molecule and is a mapping:
        conformer id (int) -> energy (float).

        :return: list of energy dictionaries
        :rtype: List[Dict[int, float]]
        """
        return [dict(e) for e in self._energies]

    # ---------------- loading ----------------
    def load_molblocks(self, molblocks: Iterable[str]) -> "Optimizer":
        """
        Load MolBlock strings for subsequent optimization.

        Each MolBlock is validated via RDKit parsing (``Chem.MolFromMolBlock``). Invalid
        or empty entries are skipped and a warning is emitted.

        :param molblocks: An iterable of MolBlock strings (RDKit MolBlock format).
        :type molblocks: Iterable[str]
        :returns: self (chainable)
        :rtype: Optimizer
        :raises TypeError: If the provided ``molblocks`` is not iterable of strings (standard
            Python TypeError will propagate).
        """
        blocks = []
        for mb in molblocks:
            if not mb:
                continue
            m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
            if m is None:
                logger.warning(
                    "Optimizer: failed to parse MolBlock, skipping one entry."
                )
                continue
            blocks.append(Chem.MolToMolBlock(m))
        self._molblocks_in = blocks
        logger.info(
            "Optimizer: loaded %d MolBlocks for optimization", len(self._molblocks_in)
        )
        return self

    # ---------------- single-molecule optimizers ----------------
    def _optimize_uff_single(self, mol: Chem.Mol) -> Dict[int, float]:
        """
        Optimize a single RDKit Mol using the UFF force field.

        The function handles molecules with zero, one or multiple conformers:
        - If zero conformers are present, an empty dict is returned.
        - If a single conformer is present, it is minimized in-place.
        - If multiple conformers are present, ``AllChem.UFFOptimizeMoleculeConfs`` is
          attempted and fallbacks are used if needed.

        :param mol: RDKit molecule object (may contain 0..N conformers)
        :type mol: rdkit.Chem.rdchem.Mol
        :return: mapping conformer id -> energy (in RDKit energy units)
        :rtype: Dict[int, float]
        """
        energies: Dict[int, float] = {}
        try:
            nconf = mol.GetNumConformers()
            if nconf == 0:
                return energies
            if nconf > 1:
                try:
                    # returns list of (converged, energy) or different tuple shapes by RDKit version
                    res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=self.max_iters)
                    for i, r in enumerate(res):
                        if isinstance(r, (tuple, list)) and len(r) >= 2:
                            energies[i] = float(r[1])
                        elif isinstance(r, (int, float)):
                            energies[i] = float(r)
                        else:
                            ff = AllChem.UFFGetMoleculeForceField(mol, confId=i)
                            energies[i] = float(ff.CalcEnergy())
                except Exception:
                    for cid in range(nconf):
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                        ff.Minimize(maxIts=self.max_iters)
                        energies[cid] = float(ff.CalcEnergy())
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
                ff.Minimize(maxIts=self.max_iters)
                energies[0] = float(ff.CalcEnergy())
        except Exception as e:
            logger.exception("Optimizer UFF failed: %s", e)
        return energies

    def _optimize_mmff_single(
        self, mol: Chem.Mol, variant: str = "MMFF94"
    ) -> Dict[int, float]:
        """
        Optimize a single RDKit Mol using the MMFF force field.

        The ``variant`` may be provided in different casings. The alias ``'MMFF'``
        is treated as ``'MMFF94'``.

        :param mol: RDKit molecule object (may contain 0..N conformers)
        :type mol: rdkit.Chem.rdchem.Mol
        :param variant: Variant string for MMFF: ``'MMFF94'`` or ``'MMFF94S'``.
                        ``'MMFF'`` is accepted as an alias for ``'MMFF94'``.
        :type variant: str
        :return: mapping conformer id -> energy (in RDKit energy units)
        :rtype: Dict[int, float]
        """
        v = (variant or "MMFF94").upper()
        if v == "MMFF":
            v = "MMFF94"
        energies: Dict[int, float] = {}
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=v)
            if props is None:
                return energies
            nconf = mol.GetNumConformers()
            if nconf == 0:
                return energies
            if nconf > 1:
                try:
                    res = AllChem.MMFFOptimizeMoleculeConfs(
                        mol, mmffVariant=v, maxIters=self.max_iters
                    )
                    for i, r in enumerate(res):
                        if isinstance(r, (tuple, list)) and len(r) >= 2:
                            energies[i] = float(r[1])
                        elif isinstance(r, (int, float)):
                            energies[i] = float(r)
                        else:
                            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=i)
                            energies[i] = float(ff.CalcEnergy())
                except Exception:
                    for cid in range(nconf):
                        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                        ff.Minimize(maxIts=self.max_iters)
                        energies[cid] = float(ff.CalcEnergy())
            else:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
                ff.Minimize(maxIts=self.max_iters)
                energies[0] = float(ff.CalcEnergy())
        except Exception as e:
            logger.exception("Optimizer MMFF(%s) failed: %s", v, e)
        return energies

    # ---------------- bulk optimization ----------------
    def optimize_all(self, method: str = "MMFF94") -> "Optimizer":
        """
        Optimize all loaded MolBlocks with the requested method/variant.

        The method operates in-place on loaded MolBlocks and fills
        :pyattr:`_optimized_blocks` and :pyattr:`_energies`.

        :param method:
            One of: ``'UFF'``, ``'MMFF'``, ``'MMFF94'``, ``'MMFF94S'``.
            Case-insensitive. ``'MMFF'`` is treated as ``'MMFF94'``.
        :type method: str
        :returns: self (chainable)
        :rtype: Optimizer
        :raises RuntimeError: if no MolBlocks were loaded prior to calling this method
            (call :meth:`load_molblocks` first).
        :raises ValueError: if an unsupported method string is provided.
        """
        if not self._molblocks_in:
            raise RuntimeError("Optimizer: no MolBlocks loaded (call load_molblocks).")

        choice = (method or "MMFF94").upper()
        self._optimized_blocks = []
        self._energies = []

        for mb in self._molblocks_in:
            mol = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
            if mol is None:
                logger.warning(
                    "Optimizer: failed to parse MolBlock during optimization; skipping."
                )
                continue

            if choice == "UFF":
                energies = self._optimize_uff_single(mol)
            elif choice in ("MMFF", "MMFF94", "MMFF94S"):
                energies = self._optimize_mmff_single(mol, variant=choice)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")

            try:
                opt_block = Chem.MolToMolBlock(mol)
            except Exception:
                opt_block = mb
            self._optimized_blocks.append(opt_block)
            self._energies.append(energies)

        logger.info(
            "Optimizer: finished optimization: %d succeeded",
            len(self._optimized_blocks),
        )
        return self

    # ---------------- write ----------------
    def write_sdf(
        self,
        out_folder: str,
        per_mol_folder: bool = True,
        write_energy_tags: bool = True,
    ) -> "Optimizer":
        """
        Write optimized molecules to SDF files.

        If ``write_energy_tags`` is True, per-conformer energies saved in
        :pyattr:`_energies` will be attached to each RDKit Mol as properties named
        ``CONF_ENERGY_<confId>`` (stringified floats).

        Files are written either as one SDF per-molecule in individual folders
        (``out_folder/ligand_i/ligand_i.sdf``) when ``per_mol_folder=True``, or as
        flat SDF files in ``out_folder`` (``out_folder/ligand_i.sdf``) when False.

        :param out_folder: destination folder where SDF(s) will be written.
        :type out_folder: str
        :param per_mol_folder:
            If True (default), create a folder per molecule and write the SDF inside
            as ``<folder>/<folder>.sdf``. If False, write ``ligand_<i>.sdf`` directly
            into ``out_folder``.
        :type per_mol_folder: bool
        :param write_energy_tags:
            If True (default) attach per-conformer energies to each RDKit Mol as
            properties named ``CONF_ENERGY_<confId>`` before writing the SDF.
        :type write_energy_tags: bool
        :returns: self (chainable)
        :rtype: Optimizer
        :raises OSError: if the output folder cannot be created due to filesystem errors
            (the underlying Path.mkdir will raise).
        """
        out = Path(out_folder)
        out.mkdir(parents=True, exist_ok=True)
        for i, block in enumerate(self._optimized_blocks):
            mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            if mol is None:
                logger.warning(
                    "Optimizer.write_sdf: could not parse molblock for index %d", i
                )
                continue

            if write_energy_tags and i < len(self._energies):
                energies = self._energies[i]
                for cid, e in energies.items():
                    try:
                        mol.SetProp(f"CONF_ENERGY_{cid}", str(e))
                    except Exception:
                        logger.debug(
                            "Optimizer.write_sdf: failed to set CONF_ENERGY_%s for mol %d",
                            cid,
                            i,
                        )

            if per_mol_folder:
                folder = out / f"ligand_{i}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{folder.name}.sdf"
            else:
                path = out / f"ligand_{i}.sdf"

            writer = Chem.SDWriter(str(path))
            writer.write(mol)
            writer.close()
            logger.debug("Optimizer: wrote SDF for ligand %d -> %s", i, path)
        logger.info(
            "Optimizer.write_sdf: wrote %d files to %s",
            len(self._optimized_blocks),
            out,
        )
        return self
