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

    This helper is useful when loading structures repeatedly, especially in
    pipelines where MDAnalysis may emit non-actionable deprecation warnings or
    repetitive parser messages.

    The context manager restores the original logger levels after the wrapped
    block exits, even if an exception is raised inside the block.

    :param suppress_warnings:
        Whether to suppress selected known non-actionable MDAnalysis warnings.
    :type suppress_warnings: bool
    :param suppress_info_logs:
        Whether to temporarily raise the logging threshold of common
        MDAnalysis loggers to ``logging.WARNING``.
    :type suppress_info_logs: bool

    :returns:
        Context manager yielding control to the wrapped block.
    :rtype: Iterator[None]

    Example
    -------
    .. code-block:: python

        with quiet_mdanalysis():
            protein = load_receptor_molecule("receptor.pdb")
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

    This helper keeps RDKit as an optional runtime dependency until ligand
    handling is actually needed.

    :returns:
        Imported ``rdkit.Chem`` module.
    :rtype: Any

    :raises MissingDependencyError:
        If RDKit is not installed or cannot be imported.

    Example
    -------
    .. code-block:: python

        Chem = _import_rdkit_chem()
        mol = Chem.MolFromSmiles("CCO")
    """
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "RDKit is required for ligand input handling."
        ) from exc
    return Chem


def _import_prolif() -> Any:
    """
    Lazily import :mod:`prolif`.

    This helper delays the ProLIF dependency until interaction-specific
    conversion is required.

    :returns:
        Imported ProLIF module.
    :rtype: Any

    :raises MissingDependencyError:
        If ProLIF is not installed or cannot be imported.

    Example
    -------
    .. code-block:: python

        plf = _import_prolif()
        print(plf.__name__)
    """
    try:
        import prolif as plf
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "ProLIF is required for interaction extraction. Install `prolif`."
        ) from exc
    return plf


def _import_mdanalysis() -> Any:
    """
    Lazily import :mod:`MDAnalysis`.

    This helper delays the MDAnalysis dependency until receptor loading is
    required.

    :returns:
        Imported MDAnalysis module.
    :rtype: Any

    :raises MissingDependencyError:
        If MDAnalysis is not installed or cannot be imported.

    Example
    -------
    .. code-block:: python

        mda = _import_mdanalysis()
        print(mda.__name__)
    """
    try:
        import MDAnalysis as mda
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "MDAnalysis is required for receptor PDB loading. Install `MDAnalysis`."
        ) from exc
    return mda


def _is_rdkit_mol(value: Any) -> bool:
    """
    Check whether an object is an RDKit molecule.

    This function returns ``False`` if RDKit is unavailable, which makes it safe
    to use in mixed environments.

    :param value:
        Object to test.
    :type value: Any

    :returns:
        ``True`` if ``value`` is an RDKit ``Mol`` instance, otherwise ``False``.
    :rtype: bool

    Example
    -------
    .. code-block:: python

        Chem = _import_rdkit_chem()
        mol = Chem.MolFromSmiles("CCO")
        assert _is_rdkit_mol(mol) is True
    """
    try:
        Chem = _import_rdkit_chem()
    except MissingDependencyError:  # pragma: no cover - dependency missing
        return False
    return isinstance(value, Chem.Mol)


def _extract_mol_name(mol: Any, index: int) -> str:
    """
    Extract a stable display name from an RDKit molecule.

    The function searches a list of common string properties in priority order
    and falls back to a generated positional name when none are available.

    Checked properties are:

    - ``mol_name``
    - ``pose_name``
    - ``PoseName``
    - ``ID``
    - ``id``
    - ``_Name``

    :param mol:
        RDKit molecule from which a name should be extracted.
    :type mol: Any
    :param index:
        Positional fallback index used when no name-like property is present.
    :type index: int

    :returns:
        Extracted or generated molecule name.
    :rtype: str

    Example
    -------
    .. code-block:: python

        Chem = _import_rdkit_chem()
        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("_Name", "ethanol")
        name = _extract_mol_name(mol, 0)
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

    The function tries to populate ``mol_name`` and ``_Name`` if they are not
    already present. Failures are ignored so that unusual molecule wrappers do
    not break the pipeline.

    :param mol:
        RDKit molecule to annotate.
    :type mol: Any
    :param name:
        Name to store on the molecule.
    :type name: str

    :returns:
        The same molecule object, potentially with updated properties.
    :rtype: Any

    Example
    -------
    .. code-block:: python

        Chem = _import_rdkit_chem()
        mol = Chem.MolFromSmiles("CCO")
        mol = _set_default_mol_props(mol, "ethanol")
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

    The function returns an empty string if RDKit conversion fails for any
    reason.

    :param mol:
        RDKit molecule to convert.
    :type mol: Any

    :returns:
        Canonicalizable SMILES string when conversion succeeds, otherwise an
        empty string.
    :rtype: str

    Example
    -------
    .. code-block:: python

        Chem = _import_rdkit_chem()
        mol = Chem.MolFromSmiles("CCO")
        smiles = mol_to_smiles(mol)
    """
    try:
        Chem = _import_rdkit_chem()
        return str(Chem.MolToSmiles(mol))
    except Exception:
        return ""


def _has_bonds(atom_group: Any) -> bool:
    """
    Check whether an MDAnalysis atom group exposes bond topology.

    :param atom_group:
        MDAnalysis atom group or compatible object.
    :type atom_group: Any

    :returns:
        ``True`` if the object exposes at least one bond, otherwise ``False``.
    :rtype: bool

    Example
    -------
    .. code-block:: python

        has_connectivity = _has_bonds(atom_group)
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

    The function first tries the newer ``Universe.guess_TopologyAttrs`` API and
    then falls back to ``AtomGroup.guess_bonds`` if available.

    :param universe:
        MDAnalysis universe containing the receptor structure.
    :type universe: Any
    :param atom_group:
        Atom group whose bond topology should be inferred.
    :type atom_group: Any
    :param vdwradii:
        Optional van der Waals radii mapping forwarded to MDAnalysis bond
        guessing routines.
    :type vdwradii: Mapping[str, float] | None

    :returns:
        ``True`` if bond topology is available after the guessing attempts,
        otherwise ``False``.
    :rtype: bool

    Example
    -------
    .. code-block:: python

        ok = _guess_bonds(universe, atom_group, vdwradii={"ZN": 1.39})
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

    MDAnalysis is used intentionally because ProLIF recommends MDAnalysis-based
    protein parsing to better preserve residue chemistry and non-standard
    protonation states.

    When requested, the function also tries to infer missing bond topology and
    warns if the selected atoms appear to contain no explicit hydrogens.

    :param receptor_pdb:
        Path to the receptor PDB file.
    :type receptor_pdb: str | pathlib.Path
    :param selection:
        Optional MDAnalysis atom selection string. If ``None``, all atoms in the
        universe are used.
    :type selection: str | None
    :param use_segid:
        Whether ProLIF should use segment identifiers instead of chain
        identifiers when building residue labels.
    :type use_segid: bool | None
    :param warn_if_no_hydrogens:
        Whether to emit a runtime warning when the selected receptor atoms
        contain no explicit hydrogens.
    :type warn_if_no_hydrogens: bool
    :param guess_bonds:
        Whether to proactively guess receptor bonds before ProLIF conversion.
    :type guess_bonds: bool
    :param vdwradii:
        Optional VdW radii overrides used during MDAnalysis bond guessing.
    :type vdwradii: Mapping[str, float] | None
    :param suppress_mdanalysis_warnings:
        Whether to suppress selected known non-actionable MDAnalysis warnings
        during loading.
    :type suppress_mdanalysis_warnings: bool
    :param suppress_mdanalysis_info_logs:
        Whether to suppress repeated MDAnalysis info log messages during
        loading.
    :type suppress_mdanalysis_info_logs: bool

    :returns:
        Receptor converted to a ProLIF molecule.
    :rtype: Any

    :raises FileNotFoundError:
        If ``receptor_pdb`` does not exist.

    Example
    -------
    .. code-block:: python

        protein = load_receptor_molecule(
            "Data/receptor/EGFR_prepared.pdb",
            selection="protein",
            guess_bonds=True,
            use_segid=False,
        )
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
) -> Iterator[NamedMol]:
    """
    Yield named RDKit molecules from several supported ligand input styles.

    Supported inputs include:

    - path to an ``.sdf`` file,
    - a single RDKit molecule,
    - a mapping of ``name -> mol``,
    - an iterable of RDKit molecules,
    - an iterable of ``(name, mol)`` tuples.

    Molecules read from SDF keep hydrogens because ``removeHs=False`` is used.

    :param ligands:
        Ligand input source.
    :type ligands: str | pathlib.Path | Any | Sequence[Any] | Iterable[Any] | Mapping[str, Any]
    :param sdf_sanitize:
        Whether to sanitize molecules while reading an SDF file.
    :type sdf_sanitize: bool

    :returns:
        Iterator of ``(molecule_name, rdkit_molecule)`` pairs.
    :rtype: Iterator[Tuple[str, Any]]

    :raises FileNotFoundError:
        If an SDF path is provided and the file does not exist.
    :raises InvalidLigandInputError:
        If the ligand input format is unsupported or contains unsupported
        entries.

    Example
    -------
    .. code-block:: python

        for name, mol in iter_named_rdkit_molecules("poses.sdf"):
            print(name, mol_to_smiles(mol))

    .. code-block:: python

        Chem = _import_rdkit_chem()
        mol = Chem.MolFromSmiles("CCO")
        named = list(iter_named_rdkit_molecules({"ethanol": mol}))
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
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=sdf_sanitize)
        for index, mol in enumerate(supplier):
            if mol is None:
                continue
            name = _extract_mol_name(mol, index)
            yield name, _set_default_mol_props(mol, name)
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

    The function first normalizes the input with
    :func:`iter_named_rdkit_molecules`, then converts each RDKit molecule into a
    ProLIF molecule using ``plf.Molecule.from_rdkit``.

    :param ligands:
        Ligand input source.
    :type ligands: str | pathlib.Path | Any | Sequence[Any] | Iterable[Any] | Mapping[str, Any]
    :param resname:
        Default residue name applied when a ligand molecule lacks residue
        information.
    :type resname: str
    :param resnumber:
        Default residue number applied when a ligand molecule lacks residue
        information.
    :type resnumber: int
    :param chain:
        Default chain identifier applied when a ligand molecule lacks residue
        information.
    :type chain: str
    :param use_segid:
        Whether ProLIF should use segment identifiers instead of chain
        identifiers.
    :type use_segid: bool
    :param sdf_sanitize:
        Whether to sanitize molecules while reading an SDF file.
    :type sdf_sanitize: bool

    :returns:
        Tuple of ``(molecule_names, rdkit_molecules, prolif_molecules)``.
    :rtype: Tuple[List[str], List[Any], List[Any]]

    :raises InvalidLigandInputError:
        If no valid ligand poses are found after input normalization.

    Example
    -------
    .. code-block:: python

        mol_names, rdkit_mols, prolif_mols = prepare_ligands(
            "poses.sdf",
            resname="LIG",
            resnumber=1,
            chain="A",
        )

    .. code-block:: python

        Chem = _import_rdkit_chem()
        mol = Chem.MolFromSmiles("CCO")
        mol_names, rdkit_mols, prolif_mols = prepare_ligands(
            {"ethanol": mol},
            resname="UNL",
        )
    """
    plf = _import_prolif()

    mol_names: List[str] = []
    rdkit_mols: List[Any] = []
    prolif_mols: List[Any] = []

    for index, (name, mol) in enumerate(
        iter_named_rdkit_molecules(ligands, sdf_sanitize=sdf_sanitize)
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
