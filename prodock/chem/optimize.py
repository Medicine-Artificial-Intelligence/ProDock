# prodock/chem/optimize.py
"""
Optimizer: RDKit-only optimization utilities (OOP) for prodock.chem.

Single-process UFF / MMFF optimizers. Works with RDKit Mol objects or MolBlock strings.
Writes energy tags as molecule properties (CONF_ENERGY_<id>).

Logging:
  Uses prodock.io.logging StructuredAdapter and log_step for high-level ops.
"""

from __future__ import annotations
from typing import List, Dict, Iterable
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    raise ImportError(
        "RDKit is required for prodock.chem.optimize: install rdkit from conda-forge"
    ) from e

from prodock.io.logging import get_logger, StructuredAdapter


logger = StructuredAdapter(
    get_logger("prodock.chem.optimize"), {"component": "optimize"}
)
logger._base_logger = getattr(logger, "_base_logger", getattr(logger, "logger", None))


class Optimizer:
    """
    Optimizer class for UFF / MMFF optimizations.

    Methods are chainable (return self). Use properties to access optimized molblocks and energies.

    :param max_iters: maximum iterations for minimizer calls (default 200).
    """

    def __init__(self, max_iters: int = 200) -> None:
        self.max_iters = int(max_iters)
        self._molblocks_in: List[str] = []
        self._optimized_blocks: List[str] = []
        self._energies: List[Dict[int, float]] = []  # per molecule: confId -> energy

    def __repr__(self) -> str:
        return (
            f"<Optimizer inputs={len(self._molblocks_in)} "
            + f"optimized={len(self._optimized_blocks)} max_iters={self.max_iters}>"
        )

    def help(self) -> None:
        print(
            "Optimizer: load_molblocks -> optimize_all(method='UFF'|'MMFF') -> "
            "access .optimized_molblocks and .energies\n"
            "Methods: optimize_uff_single / optimize_mmff_single for single-Mol operations."
        )

    # ---------------- properties ----------------
    @property
    def optimized_molblocks(self) -> List[str]:
        """Return optimized MolBlock strings (copy)."""
        return list(self._optimized_blocks)

    @property
    def energies(self) -> List[Dict[int, float]]:
        """Return list of energy maps (confId -> energy) for each optimized molecule."""
        return [dict(e) for e in self._energies]

    # ---------------- loading ----------------
    def load_molblocks(self, molblocks: Iterable[str]) -> "Optimizer":
        """
        Load MolBlock strings to be optimized.

        :param molblocks: iterable of MolBlock strings
        :return: self
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
            # reserialize to normalized MolBlock for consistent output
            blocks.append(Chem.MolToMolBlock(m))
        self._molblocks_in = blocks
        return self

    # ---------------- single-molecule optimizers ----------------
    def optimize_uff_single(self, mol: Chem.Mol) -> Dict[int, float]:
        """
        Optimize all conformers of an RDKit Mol with UFF in-place.

        :param mol: RDKit Mol (should have conformers).
        :return: mapping confId -> energy (float)
        """
        energies: Dict[int, float] = {}
        if mol.GetNumConformers() == 0:
            return energies
        try:
            if mol.GetNumConformers() > 1:
                # best-effort: use batch optimizer if available
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
                    # fallback: per-conformer minimize
                    for cid in range(mol.GetNumConformers()):
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

    def optimize_mmff_single(self, mol: Chem.Mol) -> Dict[int, float]:
        """
        Optimize all conformers of an RDKit Mol with MMFF in-place.

        :param mol: RDKit Mol with conformers.
        :return: mapping confId -> energy (float)
        """
        energies: Dict[int, float] = {}
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
        except Exception as e:
            logger.exception("Optimizer: MMFF properties creation failed: %s", e)
            return energies

        try:
            if mol.GetNumConformers() > 1:
                try:
                    res = AllChem.MMFFOptimizeMoleculeConfs(
                        mol, maxIters=self.max_iters
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
                    for cid in range(mol.GetNumConformers()):
                        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                        ff.Minimize(maxIts=self.max_iters)
                        energies[cid] = float(ff.CalcEnergy())
            else:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
                ff.Minimize(maxIts=self.max_iters)
                energies[0] = float(ff.CalcEnergy())
        except Exception as e:
            logger.exception("Optimizer MMFF failed: %s", e)
        return energies

    # ---------------- bulk optimization ----------------
    def optimize_all(self, method: str = "MMFF") -> "Optimizer":
        """
        Optimize all loaded MolBlocks sequentially using the chosen method.

        :param method: 'UFF' or 'MMFF'
        :return: self
        """
        if not self._molblocks_in:
            raise RuntimeError("Optimizer: no MolBlocks loaded (call load_molblocks).")

        method_u = method.upper()
        self._optimized_blocks = []
        self._energies = []

        for mb in self._molblocks_in:
            mol = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
            if mol is None:
                logger.warning(
                    "Optimizer: failed to parse MolBlock during optimization; skipping."
                )
                continue

            if method_u == "UFF":
                energies = self.optimize_uff_single(mol)
            elif method_u == "MMFF":
                energies = self.optimize_mmff_single(mol)
            else:
                raise ValueError("Unsupported optimization method: %s" % method)

            # reserialize optimized mol to MolBlock
            try:
                opt_block = Chem.MolToMolBlock(mol)
            except Exception:
                opt_block = mb  # fallback to original if serialization fails
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
        Write optimized MolBlocks to SDF files.

        :param out_folder: destination folder
        :param per_mol_folder: place each SDF in ligand_i/ligand_i.sdf if True
        :param write_energy_tags: annotate molecule with CONF_ENERGY_<id> if energy available
        :return: self
        """
        out = Path(out_folder)
        out.mkdir(parents=True, exist_ok=True)
        for i, block in enumerate(self._optimized_blocks):
            mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            if mol is None:
                continue
            if write_energy_tags and i < len(self._energies):
                energies = self._energies[i]
                for cid, e in energies.items():
                    mol.SetProp(f"CONF_ENERGY_{cid}", str(e))

            if per_mol_folder:
                folder = out / f"ligand_{i}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"ligand_{i}.sdf"
            else:
                path = out / f"ligand_{i}.sdf"

            writer = Chem.SDWriter(str(path))
            writer.write(mol)
            writer.close()
        return self
