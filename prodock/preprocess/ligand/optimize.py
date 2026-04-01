# prodock/chem/optimize.py
"""
RDKit-based optimization utilities for ``prodock.chem``.

This module provides :class:`Optimizer`, a lightweight object-oriented wrapper
around RDKit force-field optimization for molecules represented as RDKit Mol
objects or MolBlock strings.

Supported force fields
----------------------
The optimizer exposes the following method names:

- ``"UFF"``
- ``"MMFF"``
- ``"MMFF94"``
- ``"MMFF94S"``

The alias ``"MMFF"`` is treated as ``"MMFF94"``.

The class is designed for sequential use and integrates naturally with higher-
level workflow managers such as :class:`prodock.chem.conformer.Conformer`.

Energy export
-------------
When writing SDF outputs, conformer energies can be stored as molecule
properties named ``CONF_ENERGY_<confId>``.

Example
-------
.. code-block:: python

    from prodock.chem.optimize import Optimizer
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol)
    molblock = Chem.MolToMolBlock(mol)

    opt = Optimizer(max_iters=250)
    opt.load_molblocks([molblock])
    opt.optimize_all(method="UFF")
    opt.write_sdf("out_folder", per_mol_folder=False, write_energy_tags=True)
"""

from __future__ import annotations
from typing import List, Dict, Iterable
from pathlib import Path
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    raise ImportError(
        "RDKit is required for prodock.chem.optimize: install rdkit from conda-forge"
    ) from e

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
    RDKit force-field optimizer for molecular conformers.

    This class supports UFF and MMFF-based minimization of molecules loaded as
    MolBlock strings. Results are stored internally as optimized MolBlocks and
    per-conformer energy mappings.

    Methods are chainable and return ``self``.

    Typical workflow
    ----------------
    1. Create an optimizer instance.
    2. Load MolBlocks with :meth:`load_molblocks`.
    3. Run :meth:`optimize_all`.
    4. Access optimized MolBlocks and energies through properties or write them
       to SDF with :meth:`write_sdf`.

    :param max_iters:
        Maximum number of iterations passed to RDKit force-field minimizers.
        Defaults to ``200``.
    :type max_iters: int

    Example
    -------
    .. code-block:: python

        from prodock.chem.optimize import Optimizer
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol)
        molblock = Chem.MolToMolBlock(mol)

        opt = Optimizer(max_iters=250)
        opt.load_molblocks([molblock])
        opt.optimize_all(method="UFF")
        opt.write_sdf("out_folder", per_mol_folder=False, write_energy_tags=True)
    """

    def __init__(self, max_iters: int = 200) -> None:
        self.max_iters = int(max_iters)
        self._molblocks_in: List[str] = []
        self._optimized_blocks: List[str] = []
        self._energies: List[Dict[int, float]] = []

    def __repr__(self) -> str:
        return (
            f"<Optimizer inputs={len(self._molblocks_in)}"
            f" optimized={len(self._optimized_blocks)} max_iters={self.max_iters}>"
        )

    @property
    def optimized_molblocks(self) -> List[str]:
        """
        Return a copy of optimized MolBlock strings.

        :returns:
            Optimized MolBlock strings.
        :rtype: List[str]
        """
        return list(self._optimized_blocks)

    @property
    def energies(self) -> List[Dict[int, float]]:
        """
        Return stored per-molecule conformer energies.

        Each list element corresponds to one optimized molecule and maps
        conformer id to energy.

        :returns:
            Per-molecule conformer-energy mappings.
        :rtype: List[Dict[int, float]]
        """
        return [dict(e) for e in self._energies]

    def load_molblocks(self, molblocks: Iterable[str]) -> "Optimizer":
        """
        Load MolBlock strings for subsequent optimization.

        Each non-empty MolBlock is parsed with RDKit for validation. Invalid
        entries are skipped and a warning is logged.

        :param molblocks:
            Iterable of MolBlock strings in RDKit MolBlock format.
        :type molblocks: Iterable[str]

        :returns:
            The current optimizer instance.
        :rtype: Optimizer

        :raises TypeError:
            Propagated if ``molblocks`` is not iterable.

        Example
        -------
        .. code-block:: python

            opt = Optimizer()
            opt.load_molblocks([molblock_1, molblock_2])
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

    def _optimize_uff_single(self, mol: Chem.Mol) -> Dict[int, float]:
        """
        Optimize a single molecule using the UFF force field.

        Molecules with multiple conformers are optimized using
        :func:`rdkit.Chem.AllChem.UFFOptimizeMoleculeConfs` when available.
        Fallback per-conformer minimization is used if bulk optimization fails.

        :param mol:
            RDKit molecule object containing zero or more conformers.
        :type mol: rdkit.Chem.rdchem.Mol

        :returns:
            Mapping from conformer id to calculated energy.
        :rtype: Dict[int, float]
        """
        energies: Dict[int, float] = {}
        try:
            nconf = mol.GetNumConformers()
            if nconf == 0:
                return energies
            if nconf > 1:
                try:
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
        Optimize a single molecule using an MMFF force field.

        The accepted variants are ``"MMFF94"`` and ``"MMFF94S"``. The alias
        ``"MMFF"`` is normalized internally to ``"MMFF94"``.

        :param mol:
            RDKit molecule object containing zero or more conformers.
        :type mol: rdkit.Chem.rdchem.Mol
        :param variant:
            MMFF variant name. Accepted values include ``"MMFF"``,
            ``"MMFF94"``, and ``"MMFF94S"``.
        :type variant: str

        :returns:
            Mapping from conformer id to calculated energy.
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

    def optimize_all(self, method: str = "MMFF94") -> "Optimizer":
        """
        Optimize all loaded MolBlocks using the selected force field.

        This method parses each loaded MolBlock into an RDKit molecule, performs
        optimization in place, stores the optimized MolBlock, and records
        conformer energies.

        Supported methods are ``"UFF"``, ``"MMFF"``, ``"MMFF94"``, and
        ``"MMFF94S"``. Method matching is case-insensitive.

        :param method:
            Optimization method or MMFF variant.
        :type method: str

        :returns:
            The current optimizer instance.
        :rtype: Optimizer

        :raises RuntimeError:
            If no MolBlocks have been loaded.
        :raises ValueError:
            If ``method`` is not supported.

        Example
        -------
        .. code-block:: python

            opt = Optimizer()
            opt.load_molblocks([molblock])
            opt.optimize_all(method="MMFF94")
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

    def write_sdf(
        self,
        out_folder: str,
        per_mol_folder: bool = True,
        write_energy_tags: bool = True,
    ) -> "Optimizer":
        """
        Write optimized molecules to SDF files.

        If energy data are available and ``write_energy_tags`` is enabled,
        conformer energies are added as molecule properties named
        ``CONF_ENERGY_<confId>``.

        Output is written either as one SDF per molecule in separate folders or
        as a flat directory of SDF files.

        :param out_folder:
            Destination directory where output SDF files will be written.
        :type out_folder: str
        :param per_mol_folder:
            If ``True``, write each molecule to
            ``out_folder/ligand_i/ligand_i.sdf``. If ``False``, write
            ``out_folder/ligand_i.sdf``.
        :type per_mol_folder: bool
        :param write_energy_tags:
            Whether to write ``CONF_ENERGY_<confId>`` molecule properties.
        :type write_energy_tags: bool

        :returns:
            The current optimizer instance.
        :rtype: Optimizer

        :raises OSError:
            Propagated if the output directory cannot be created.

        Example
        -------
        .. code-block:: python

            opt.write_sdf(
                "out_folder",
                per_mol_folder=False,
                write_energy_tags=True,
            )
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
