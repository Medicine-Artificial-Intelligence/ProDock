# prodock/postprocess/interaction/io.py
from __future__ import annotations

"""
Input preparation helpers for ProLIF-based interaction analysis.

This module provides utilities to:

- reduce noisy MDAnalysis warnings and info logs,
- lazily import optional dependencies,
- normalize ligand input from several supported formats,
- load a receptor PDB into a ProLIF molecule,
- convert ligand inputs into ProLIF-ready ligand molecules.

The functions are designed to keep optional third-party imports local so that
the module can still be imported in partially configured environments.

Example
-------
.. code-block:: python

    from prodock.postprocess.interaction.io import (
        load_receptor_molecule,
        prepare_ligands,
    )

    protein = load_receptor_molecule(
        "Data/receptor/EGFR_prepared.pdb",
        selection="protein",
        guess_bonds=True,
    )

    mol_names, rdkit_mols, prolif_mols = prepare_ligands(
        "Data/poses/poses.sdf",
        resname="LIG",
        resnumber=1,
    )
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Sequence, Tuple
import logging
import warnings

from .exceptions import InvalidLigandInputError, MissingDependencyError
from prodock.structure.conversion import load_sdf_for_interactions

PathLike = str | Path
NamedMol = Tuple[str, Any]


@contextmanager
def quiet_mdanalysis(
    *,
    suppress_warnings: bool = True,
    suppress_info_logs: bool = True,
) -> Iterator[None]:
    """
    Temporarily silence common MDAnalysis warning and info-level log noise.
    """
    logger_names = [
        "MDAnalysis",
        "MDAnalysis.coordinates.AMBER",
        "MDAnalysis.core.universe",
        "MDAnalysis.topology.PDBParser",
        "MDAnalysis.guesser.base",
        "MDAnalysis.converters.RDKit",
    ]
    original_levels: dict[str, int] = {}
    if suppress_info_logs:
        for name in logger_names:
            logger = logging.getLogger(name)
            original_levels[name] = logger.level
            logger.setLevel(logging.WARNING)

    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.filterwarnings(
                "ignore",
                message=r".*MDAnalysis\.topology\.tables has been moved to MDAnalysis\.guesser\.tables.*",
                category=DeprecationWarning,
            )
        try:
            yield
        finally:
            if suppress_info_logs:
                for name, level in original_levels.items():
                    logging.getLogger(name).setLevel(level)


def _import_rdkit_chem() -> Any:
    """
    Lazily import :mod:`rdkit.Chem`.
    """
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover
        raise MissingDependencyError(
            "RDKit is required for ligand input handling."
        ) from exc
    return Chem


def _import_prolif() -> Any:
    """
    Lazily import :mod:`prolif`.
    """
    try:
        import prolif as plf
    except Exception as exc:  # pragma: no cover
        raise MissingDependencyError(
            "ProLIF is required for interaction extraction. Install `prolif`."
        ) from exc
    return plf


def _import_mdanalysis() -> Any:
    """
    Lazily import :mod:`MDAnalysis`.
    """
    try:
        import MDAnalysis as mda
    except Exception as exc:  # pragma: no cover
        raise MissingDependencyError(
            "MDAnalysis is required for receptor PDB loading. Install `MDAnalysis`."
        ) from exc
    return mda


def _is_rdkit_mol(value: Any) -> bool:
    """
    Check whether an object is an RDKit molecule.
    """
    try:
        Chem = _import_rdkit_chem()
    except MissingDependencyError:  # pragma: no cover
        return False
    return isinstance(value, Chem.Mol)


def _extract_mol_name(mol: Any, index: int) -> str:
    """
    Extract a stable display name from an RDKit molecule.
    """
    for prop in ("mol_name", "pose_name", "PoseName", "ID", "id", "_Name"):
        try:
            if mol.HasProp(prop):
                value = str(mol.GetProp(prop)).strip()
                if value:
                    return value
        except Exception:
            continue
    return f"mol_{index:04d}"


def _set_default_mol_props(mol: Any, name: str) -> Any:
    """
    Ensure standard name properties exist on an RDKit molecule.
    """
    try:
        if not mol.HasProp("mol_name"):
            mol.SetProp("mol_name", name)
        if not mol.HasProp("_Name"):
            mol.SetProp("_Name", name)
    except Exception:
        pass
    return mol


def mol_to_smiles(mol: Any) -> str:
    """
    Convert an RDKit molecule into a SMILES string.
    """
    try:
        Chem = _import_rdkit_chem()
        return str(Chem.MolToSmiles(mol))
    except Exception:
        return ""


def _has_bonds(atom_group: Any) -> bool:
    """
    Check whether an MDAnalysis atom group exposes bond topology.
    """
    try:
        return len(atom_group.bonds) > 0
    except Exception:
        return False


def _guess_bonds(
    universe: Any,
    atom_group: Any,
    vdwradii: Mapping[str, float] | None = None,
) -> bool:
    """
    Attempt to populate missing bond topology for a receptor atom group.
    """
    if _has_bonds(atom_group):
        return True

    if hasattr(universe, "guess_TopologyAttrs"):
        guess_kwargs: dict[str, Any] = {"to_guess": ["bonds"]}
        if vdwradii is not None:
            guess_kwargs["vdwradii"] = dict(vdwradii)
        try:
            universe.guess_TopologyAttrs(**guess_kwargs)
            if _has_bonds(atom_group):
                return True
        except Exception:
            pass

    if hasattr(atom_group, "guess_bonds"):
        try:
            if vdwradii is None:
                atom_group.guess_bonds()
            else:
                atom_group.guess_bonds(vdwradii=dict(vdwradii))
            return _has_bonds(atom_group)
        except Exception:
            pass

    return _has_bonds(atom_group)


def load_receptor_molecule(
    receptor_pdb: PathLike,
    selection: str | None = None,
    use_segid: bool | None = None,
    warn_if_no_hydrogens: bool = True,
    guess_bonds: bool = True,
    vdwradii: Mapping[str, float] | None = None,
    suppress_mdanalysis_warnings: bool = True,
    suppress_mdanalysis_info_logs: bool = True,
) -> Any:
    """
    Load a receptor PDB file and convert it into a ProLIF molecule.
    """
    path = Path(receptor_pdb)
    if not path.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {path}")

    mda = _import_mdanalysis()
    plf = _import_prolif()

    with quiet_mdanalysis(
        suppress_warnings=suppress_mdanalysis_warnings,
        suppress_info_logs=suppress_mdanalysis_info_logs,
    ):
        universe_kwargs: dict[str, Any] = {"to_guess": ["types", "masses"]}
        try:
            universe = mda.Universe(str(path), **universe_kwargs)
        except TypeError:
            universe = mda.Universe(str(path))

        atom_group = universe.select_atoms(selection) if selection else universe.atoms

        if guess_bonds and not _has_bonds(atom_group):
            _guess_bonds(universe, atom_group, vdwradii=vdwradii)

        if warn_if_no_hydrogens and len(atom_group) > 0:
            hydrogen_count = sum(
                1 for atom in atom_group if getattr(atom, "element", "").upper() == "H"
            )
            if hydrogen_count == 0:
                warnings.warn(
                    (
                        "The receptor PDB appears to contain no explicit hydrogens. "
                        "ProLIF recommends explicit hydrogens for accurate interaction "
                        "detection, especially hydrogen bonds and charge-dependent terms."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

        return plf.Molecule.from_mda(atom_group, use_segid=use_segid)


def iter_named_rdkit_molecules(
    ligands: PathLike | Any | Sequence[Any] | Iterable[Any] | Mapping[str, Any],
    *,
    sdf_sanitize: bool = True,
    resname: str = "LIG",
    resnumber: int = 1,
    chain: str = "",
) -> Iterator[NamedMol]:
    """
    Yield named RDKit molecules from several supported ligand input styles.

    For SDF input, the file is loaded as an RDKit supplier and valid molecules
    are yielded one by one. Invalid records returned as ``None`` are skipped.
    """
    Chem = _import_rdkit_chem()

    if isinstance(ligands, (str, Path)):
        path = Path(ligands)
        if not path.exists():
            raise FileNotFoundError(f"Ligand file not found: {path}")
        if path.suffix.lower() != ".sdf":
            raise InvalidLigandInputError(
                f"Unsupported ligand file format for interaction extraction: {path.suffix}. "
                "Currently supported file input is `.sdf`."
            )

        try:
            supplier = Chem.SDMolSupplier(
                str(path),
                removeHs=False,
                sanitize=sdf_sanitize,
            )
        except Exception as exc:
            raise InvalidLigandInputError(
                f"No valid ligand molecule was found in {path}"
            ) from exc

        found_any = False
        for index, mol in enumerate(supplier):
            if mol is None:
                continue
            found_any = True
            name = _extract_mol_name(mol, index)
            yield name, _set_default_mol_props(mol, name)

        if not found_any:
            raise InvalidLigandInputError(
                f"No valid ligand molecule was found in {path}"
            )
        return

    if _is_rdkit_mol(ligands):
        name = _extract_mol_name(ligands, 0)
        yield name, _set_default_mol_props(ligands, name)
        return

    if isinstance(ligands, Mapping):
        for index, (name, mol) in enumerate(ligands.items()):
            if not _is_rdkit_mol(mol):
                raise InvalidLigandInputError(
                    "Encountered an unsupported ligand entry inside the ligand mapping."
                )
            clean_name = str(name).strip() or f"mol_{index:04d}"
            yield clean_name, _set_default_mol_props(mol, clean_name)
        return

    try:
        iterator = iter(ligands)
    except TypeError as exc:
        raise InvalidLigandInputError(
            "Ligands must be an SDF path, an RDKit Mol, a mapping of name -> RDKit Mol, "
            "an iterable of RDKit Mol, or an iterable of (name, mol) pairs."
        ) from exc

    for index, item in enumerate(iterator):
        if _is_rdkit_mol(item):
            name = _extract_mol_name(item, index)
            yield name, _set_default_mol_props(item, name)
            continue
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and _is_rdkit_mol(item[1])
        ):
            name = item[0].strip() or f"mol_{index:04d}"
            mol = _set_default_mol_props(item[1], name)
            yield name, mol
            continue
        raise InvalidLigandInputError(
            "Encountered an unsupported ligand entry. Expected an RDKit Mol or a "
            "(name, mol) tuple."
        )

    if _is_rdkit_mol(ligands):
        name = _extract_mol_name(ligands, 0)
        yield name, _set_default_mol_props(ligands, name)
        return

    if isinstance(ligands, Mapping):
        for index, (name, mol) in enumerate(ligands.items()):
            if not _is_rdkit_mol(mol):
                raise InvalidLigandInputError(
                    "Encountered an unsupported ligand entry inside the ligand mapping."
                )
            clean_name = str(name).strip() or f"mol_{index:04d}"
            yield clean_name, _set_default_mol_props(mol, clean_name)
        return

    try:
        iterator = iter(ligands)
    except TypeError as exc:
        raise InvalidLigandInputError(
            "Ligands must be an SDF path, an RDKit Mol, a mapping of name -> RDKit Mol, "
            "an iterable of RDKit Mol, or an iterable of (name, mol) pairs."
        ) from exc

    for index, item in enumerate(iterator):
        if _is_rdkit_mol(item):
            name = _extract_mol_name(item, index)
            yield name, _set_default_mol_props(item, name)
            continue
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and _is_rdkit_mol(item[1])
        ):
            name = item[0].strip() or f"mol_{index:04d}"
            mol = _set_default_mol_props(item[1], name)
            yield name, mol
            continue
        raise InvalidLigandInputError(
            "Encountered an unsupported ligand entry. Expected an RDKit Mol or a "
            "(name, mol) tuple."
        )


def prepare_ligands(
    ligands: PathLike | Any | Sequence[Any] | Iterable[Any] | Mapping[str, Any],
    *,
    resname: str = "LIG",
    resnumber: int = 1,
    chain: str = "",
    use_segid: bool = False,
    sdf_sanitize: bool = True,
) -> Tuple[List[str], List[Any], List[Any]]:
    """
    Convert ligand inputs into ProLIF ligand molecules.
    """
    plf = _import_prolif()

    mol_names: List[str] = []
    rdkit_mols: List[Any] = []
    prolif_mols: List[Any] = []

    for index, (name, mol) in enumerate(
        iter_named_rdkit_molecules(
            ligands,
            sdf_sanitize=sdf_sanitize,
            resname=resname,
            resnumber=resnumber,
            chain=chain,
        )
    ):
        if mol is None:
            continue
        clean_name = name or f"mol_{index:04d}"
        mol_names.append(clean_name)
        rdkit_mols.append(mol)
        prolif_mols.append(
            plf.Molecule.from_rdkit(
                mol,
                resname=resname,
                resnumber=resnumber,
                chain=chain,
                use_segid=use_segid,
            )
        )

    if not prolif_mols:
        raise InvalidLigandInputError("No valid ligand poses were found.")

    return mol_names, rdkit_mols, prolif_mols
