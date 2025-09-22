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

    :param max_iters: maximum iterations for minimizer calls (default 200).
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
        return list(self._optimized_blocks)

    @property
    def energies(self) -> List[Dict[int, float]]:
        return [dict(e) for e in self._energies]

    # ---------------- loading ----------------
    def load_molblocks(self, molblocks: Iterable[str]) -> "Optimizer":
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
        :param variant: 'MMFF94' or 'MMFF94S' (case-insensitive). 'MMFF' is treated as 'MMFF94'.
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

        :param method: 'UFF' | 'MMFF' | 'MMFF94' | 'MMFF94S'
        :return: self
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
        Write optimized molecules to SDF. Optionally add CONF_ENERGY_<id> tags.

        :param out_folder: destination folder
        :param per_mol_folder: if True, write to ligand_i/ligand_i.sdf
        :param write_energy_tags: include per-conformer energies as properties
        :return: self
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
