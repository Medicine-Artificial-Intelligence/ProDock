# prodock/postprocess/metrics/dock_power.py
"""
Docking pose structural evaluation utilities using RDKit, OpenBabel, or PyMOL.

This module prefers RDKit but supports modern OpenBabel wheels (importable as
``openbabel`` / ``openbabel.openbabel``). If OpenBabel exposes ``openbabel.pybel``
we use that; otherwise we use OBConversion + OBMol low-level API.

Sphinx-style docstrings are used on the main class and public methods.

Examples
--------

Using RDKit objects (recommended):

>>> from rdkit import Chem
>>> from rdkit.Chem import AllChem
>>> m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
>>> AllChem.EmbedMolecule(m, AllChem.ETKDG())
>>> AllChem.UFFOptimizeMolecule(m)
>>> de = DockEvaluator(engine="rdkit")
>>> rmsd_val = de.rmsd(m, m)  # identical -> near 0

Using file paths (SDF/PDB):

>>> de = DockEvaluator(engine="rdkit")
>>> rmsd_val = de.rmsd("ref.sdf", "probe.sdf")

Using OpenBabel backend (if installed via openbabel-wheel):

>>> de = DockEvaluator(engine="openbabel")
>>> rmsd_val = de.rmsd("ref.sdf", "probe.sdf")

Using PyMOL (if PyMOL is importable in your environment):

>>> de = DockEvaluator(engine="pymol")
>>> rmsd_val = de.rmsd("ref.pdb", "probe.pdb")

Notes
-----
- RDKit backend requires 3D conformers on both molecules.
- OpenBabel backend will attempt to read files or convert RDKit mols to SDF for loading.
- PyMOL backend will load objects into the PyMOL session; the objects are deleted after alignment.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import math
from pathlib import Path

import numpy as np

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

# PyMOL (we only import the command API to avoid importing unused module name)
try:
    from pymol import cmd as _pymol_cmd  # type: ignore

    _HAS_PYMOL = True
except Exception:
    _HAS_PYMOL = False

# Modern OpenBabel (openbabel-wheel)
_HAS_OPENBABEL = False
_HAS_OB_PYBEL = False
_openbabel = None
_ob = None
_ob_pybel = None

try:
    # Attempt to import openbabel package from the wheel
    import openbabel  # type: ignore

    try:
        from openbabel import openbabel as ob  # type: ignore

        _ob = ob
        _openbabel = openbabel
        _HAS_OPENBABEL = True
    except Exception:
        try:
            _ob = openbabel.openbabel  # type: ignore
            _openbabel = openbabel
            _HAS_OPENBABEL = True
        except Exception:
            _HAS_OPENBABEL = False

    # pybel wrapper may be provided under openbabel.pybel
    try:
        from openbabel import pybel  # type: ignore

        _ob_pybel = pybel
        _HAS_OB_PYBEL = True
    except Exception:
        _HAS_OB_PYBEL = False

except Exception:
    _HAS_OPENBABEL = False
    _HAS_OB_PYBEL = False
    _openbabel = None
    _ob = None
    _ob_pybel = None


MolLike = Union["Chem.Mol", Path, str]  # type: ignore


class DockEvaluator:
    """
    Evaluate docking poses and compute RMSD using a chosen backend.

    :param engine: Backend engine to use for RMSD calculation. One of
                   ``"rdkit"`` (default), ``"openbabel"`` (uses openbabel wheel),
                   or ``"pymol"``.
    :type engine: str
    :raises ImportError: if the requested engine backend is not available.
    """

    def __init__(self, engine: str = "rdkit"):
        self.engine = engine.lower()
        if self.engine == "rdkit":
            if not _HAS_RDKIT:
                raise ImportError(
                    "RDKit backend requested but RDKit is not importable."
                )
        elif self.engine == "openbabel":
            if not _HAS_OPENBABEL:
                raise ImportError(
                    "openbabel backend requested but openbabel is not importable."
                )
        elif self.engine == "pymol":
            if not _HAS_PYMOL:
                raise ImportError(
                    "pymol backend requested but PyMOL is not importable."
                )
        else:
            raise ValueError(
                f"Unknown engine '{engine}'. Supported: rdkit, openbabel, pymol."
            )

    # ---------- loaders ----------
    def _load_rdkit(self, obj: MolLike):
        """
        Load an RDKit molecule from a path or return if already an RDKit Mol.

        :param obj: RDKit Mol or path to molecule file (SDF/PDB/ MOL).
        :type obj: rdkit.Chem.Mol or Path or str
        :returns: RDKit Mol
        :raises ValueError: if the file cannot be parsed.
        """
        if hasattr(obj, "GetNumAtoms"):
            return obj  # already RDKit Mol

        p = Path(str(obj))
        if not p.exists():
            raise ValueError(f"Path does not exist: {p}")

        # SDF/MOL
        if p.suffix.lower() in {".sdf", ".mol", ".sd"}:
            suppl = Chem.SDMolSupplier(str(p), removeHs=False)
            if len(suppl) == 0 or suppl[0] is None:
                raise ValueError(f"Failed to read SDF file: {p}")
            return suppl[0]
        # PDB
        if p.suffix.lower() in {".pdb", ".ent"}:
            m = Chem.MolFromPDBFile(str(p), removeHs=False)
            if m is None:
                raise ValueError(f"Failed to read PDB file: {p}")
            return m
        # generic molfile fallback
        m = Chem.MolFromMolFile(str(p), removeHs=False)
        if m is None:
            raise ValueError(f"Failed to read molecule file: {p}")
        return m

    def _load_obmol(self, obj: MolLike):
        """
        Load an OpenBabel OBMol from a path or convert RDKit Mol to OBMol.

        :param obj: RDKit Mol or path or str
        :returns: openbabel.OBMol instance
        :raises ValueError: if reading/conversion fails
        """
        if not _HAS_OPENBABEL:
            raise ImportError("openbabel not available")

        # If pybel wrapper exists, prefer returning OBMol from pybel
        if _HAS_OB_PYBEL:
            # pybel accepts filenames or OBMol; if RDKit Mol provided, write tmp SDF
            if hasattr(obj, "GetNumAtoms"):
                rd_mol = obj  # type: ignore
                tmp = Path("__tmp_ob_sdf.sdf")
                from rdkit import Chem as _Chem

                w = _Chem.SDWriter(str(tmp))
                w.write(rd_mol)
                w.close()
                gen = _ob_pybel.readfile("sdf", str(tmp))
                mol = next(gen, None)
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                if mol is None:
                    raise ValueError(
                        "openbabel.pybel failed to read converted RDKit SDF"
                    )
                return mol.OBMol  # return OBMol
            # else it's a filename or pybel.Molecule
            p = Path(str(obj))
            if p.exists():
                fmt = p.suffix.lstrip(".")
                gen = _ob_pybel.readfile(fmt, str(p))
                mol = next(gen, None)
                if mol is None:
                    raise ValueError(f"openbabel.pybel failed to read {p}")
                return mol.OBMol
            raise ValueError("Cannot load object into openbabel.pybel")

        # Otherwise use OBConversion low-level API
        # If obj is an RDKit Mol -> write temp sdf and read
        if hasattr(obj, "GetNumAtoms"):
            rd_mol = obj  # type: ignore
            tmp = Path("__tmp_ob_sdf.sdf")
            from rdkit import Chem as _Chem

            w = _Chem.SDWriter(str(tmp))
            w.write(rd_mol)
            w.close()
            obj = str(tmp)

        # Now obj should be a filename
        p = Path(str(obj))
        if not p.exists():
            raise ValueError(f"Path does not exist: {p}")

        conv = _ob.OBConversion()
        fmt = p.suffix.lstrip(".").lower() if p.suffix else "sdf"
        if not conv.SetInFormat(fmt):
            if not conv.SetInFormat("sdf"):
                raise ValueError(f"OpenBabel couldn't set input format for {p}")
        obmol = _ob.OBMol()
        ok = conv.ReadFile(obmol, str(p))
        # cleanup tmp if we created one
        if str(obj).endswith("__tmp_ob_sdf.sdf"):
            try:
                Path(obj).unlink(missing_ok=True)
            except Exception:
                pass
        if not ok:
            raise ValueError(f"OpenBabel failed to read file {p}")
        return obmol

    def _obmol_coords(self, obmol):
        """
        Extract Nx3 numpy coordinate array from OBMol object.
        """
        coords = []
        for atom in _ob.OBMolAtomIter(obmol):
            coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
        return np.asarray(coords, dtype=float)

    def _kabsch_rmsd(self, coords_ref, coords_prb):
        """
        Compute best-fit RMSD using Kabsch algorithm between two Nx3 arrays.
        """
        if coords_ref.shape != coords_prb.shape:
            raise ValueError("Coordinate arrays must have same shape for Kabsch RMSD.")
        ref_c = coords_ref.mean(axis=0)
        prb_c = coords_prb.mean(axis=0)
        A = coords_prb - prb_c
        B = coords_ref - ref_c
        H = A.T @ B
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # reflection correction
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        A_rot = A @ R
        diff = A_rot - B
        rmsd = math.sqrt((diff**2).sum() / coords_ref.shape[0])
        return float(rmsd)

    def _load_pymol(self, obj: MolLike, name: str = "tmp"):
        """
        Load molecule into PyMOL session and return object name.

        :param obj: RDKit Mol or path or str
        :param name: object name to use in PyMOL
        :returns: pymol object name (str)
        """
        if hasattr(obj, "GetNumAtoms"):
            tmp = Path(f"__pymol_tmp_{name}.pdb")
            from rdkit import Chem as _Chem

            _Chem.MolToPDBFile(obj, str(tmp))
            _pymol_cmd.load(str(tmp), name)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return name
        p = Path(str(obj))
        if not p.exists():
            raise ValueError(f"Path {p} does not exist for pymol load")
        _pymol_cmd.load(str(p), name)
        return name

    # ---------- RMSD implementations ----------
    def rmsd(
        self,
        ref: MolLike,
        probe: MolLike,
        atom_map: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> float:
        """
        Compute RMSD between ``ref`` and ``probe`` molecules using the selected engine.

        :param ref: Reference molecule (RDKit Mol or file path)
        :type ref: rdkit.Chem.Mol or Path or str
        :param probe: Probe molecule (RDKit Mol or file path)
        :type probe: rdkit.Chem.Mol or Path or str
        :param atom_map: optional list of (ref_idx, probe_idx) atom index pairs to constrain alignment
        :type atom_map: sequence of tuple(int,int) or None
        :returns: RMSD in Angstroms (float)
        :raises ValueError: on parsing/align errors or if backend is not available
        """
        if self.engine == "rdkit":
            if not _HAS_RDKIT:
                raise ImportError("RDKit not available")
            ref_m = self._load_rdkit(ref)
            probe_m = self._load_rdkit(probe)
            if ref_m.GetNumConformers() == 0 or probe_m.GetNumConformers() == 0:
                raise ValueError("Both molecules must have 3D conformers for RMSD.")
            # remove hydrogens for alignment to reduce artifacts
            try:
                Chem.RemoveHs(ref_m)
                Chem.RemoveHs(probe_m)
            except Exception:
                # ignore if RemoveHs not supported in-place
                pass
            try:
                if atom_map is None:
                    rmsd_val = AllChem.AlignMol(probe_m, ref_m)
                else:
                    rmsd_val = AllChem.AlignMol(probe_m, ref_m, atomMap=list(atom_map))
            except Exception as exc:
                raise ValueError(f"RDKit alignment failed: {exc}")
            return float(rmsd_val)

        elif self.engine == "openbabel":
            if not _HAS_OPENBABEL:
                raise ImportError("openbabel not available")
            ob_ref = self._load_obmol(ref)
            ob_prb = self._load_obmol(probe)
            # Try to use OBAlign/OB functions if available
            try:
                if hasattr(_ob, "OBAlign"):
                    aligner = _ob.OBAlign()
                    aligner.Setup(ob_ref, ob_prb)
                    rms = aligner.RMSD()
                    if rms is not None:
                        return float(rms)
            except Exception:
                # ignore and fallback
                pass
            coords_ref = self._obmol_coords(ob_ref)
            coords_prb = self._obmol_coords(ob_prb)
            return float(self._kabsch_rmsd(coords_ref, coords_prb))

        else:  # pymol
            if not _HAS_PYMOL:
                raise ImportError("pymol is not available")
            ref_name = self._load_pymol(ref, name="ref_tmp")
            prb_name = self._load_pymol(probe, name="prb_tmp")
            try:
                out = _pymol_cmd.align(prb_name, ref_name)
                # cmd.align often returns (rms, matched_pairs, ...) — use first element when present
                if isinstance(out, (list, tuple)) and len(out) > 0:
                    rms = float(out[0])
                else:
                    # fallback to rms_cur if align didn't return RMS directly
                    rms = float(_pymol_cmd.rms_cur(prb_name, ref_name))
            except Exception as exc:
                raise ValueError(f"PyMOL alignment failed: {exc}")
            finally:
                # cleanup PyMOL objects
                try:
                    _pymol_cmd.delete(ref_name)
                    _pymol_cmd.delete(prb_name)
                except Exception:
                    pass
            return float(rms)
