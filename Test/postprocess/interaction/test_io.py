from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prodock.postprocess.interaction.exceptions import (
    InvalidLigandInputError,
    MissingDependencyError,
)
from prodock.postprocess.interaction.io import (
    _extract_mol_name,
    _guess_bonds,
    _has_bonds,
    _is_rdkit_mol,
    _set_default_mol_props,
    iter_named_rdkit_molecules,
    mol_to_smiles,
    prepare_ligands,
    quiet_mdanalysis,
)


class FakeMol:
    def __init__(self) -> None:
        self.props = {}

    def HasProp(self, key: str) -> bool:
        return key in self.props

    def GetProp(self, key: str) -> str:
        return self.props[key]

    def SetProp(self, key: str, value: str) -> None:
        self.props[key] = value


class FakeChemModule:
    class Mol(FakeMol):
        pass

    @staticmethod
    def MolToSmiles(mol) -> str:
        if getattr(mol, "fail_smiles", False):
            raise ValueError("cannot convert")
        return "CCO"

    @staticmethod
    def SDMolSupplier(path, removeHs=False, sanitize=True):
        mol1 = FakeChemModule.Mol()
        mol1.SetProp("_Name", "pose_1")
        mol2 = FakeChemModule.Mol()
        return [mol1, None, mol2]


class FakePLFMoleculeFactory:
    @staticmethod
    def from_rdkit(
        mol,
        resname="LIG",
        resnumber=1,
        chain="",
        use_segid=False,
    ):
        return {
            "mol": mol,
            "resname": resname,
            "resnumber": resnumber,
            "chain": chain,
            "use_segid": use_segid,
        }


class FakePLF:
    Molecule = FakePLFMoleculeFactory


class FakeAtomGroupNoBonds:
    def __init__(self) -> None:
        self.bonds = []

    def guess_bonds(self, vdwradii=None) -> None:
        self.bonds = [("a", "b")]


class FakeAtomGroupWithBrokenBonds:
    @property
    def bonds(self):
        raise RuntimeError("broken")


class FakeUniverseGuessAttrs:
    def __init__(self) -> None:
        self.called = False

    def guess_TopologyAttrs(self, **kwargs) -> None:
        self.called = True


class FakeAtomGroupForUniverseGuess:
    def __init__(self) -> None:
        self.bonds = []


class TestPrepareHelpers(unittest.TestCase):
    def test_quiet_mdanalysis_restores_logger_level(self) -> None:
        import logging

        logger = logging.getLogger("MDAnalysis")
        original = logger.level
        logger.setLevel(logging.INFO)

        with quiet_mdanalysis():
            self.assertEqual(logger.level, logging.WARNING)

        self.assertEqual(logger.level, logging.INFO)
        logger.setLevel(original)

    def test_is_rdkit_mol_true(self) -> None:
        mol = FakeChemModule.Mol()
        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            self.assertTrue(_is_rdkit_mol(mol))

    def test_is_rdkit_mol_false_when_missing_dependency(self) -> None:
        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            side_effect=MissingDependencyError("missing"),
        ):
            self.assertFalse(_is_rdkit_mol(object()))

    def test_extract_mol_name_prefers_named_prop(self) -> None:
        mol = FakeMol()
        mol.SetProp("_Name", "lig1")
        self.assertEqual(_extract_mol_name(mol, 2), "lig1")

    def test_extract_mol_name_fallback(self) -> None:
        mol = FakeMol()
        self.assertEqual(_extract_mol_name(mol, 7), "mol_0007")

    def test_set_default_mol_props(self) -> None:
        mol = FakeMol()
        out = _set_default_mol_props(mol, "ethanol")
        self.assertIs(out, mol)
        self.assertEqual(mol.GetProp("mol_name"), "ethanol")
        self.assertEqual(mol.GetProp("_Name"), "ethanol")

    def test_mol_to_smiles_success(self) -> None:
        mol = FakeChemModule.Mol()
        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            self.assertEqual(mol_to_smiles(mol), "CCO")

    def test_mol_to_smiles_failure_returns_empty(self) -> None:
        mol = FakeChemModule.Mol()
        mol.fail_smiles = True
        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            self.assertEqual(mol_to_smiles(mol), "")

    def test_has_bonds_true(self) -> None:
        atom_group = type("AtomGroup", (), {"bonds": [1]})()
        self.assertTrue(_has_bonds(atom_group))

    def test_has_bonds_false_on_exception(self) -> None:
        self.assertFalse(_has_bonds(FakeAtomGroupWithBrokenBonds()))

    def test_guess_bonds_via_universe(self) -> None:
        universe = FakeUniverseGuessAttrs()
        atom_group = FakeAtomGroupForUniverseGuess()

        def fake_has_bonds(obj):
            return len(obj.bonds) > 0

        def fake_guess(**kwargs):
            universe.called = True
            atom_group.bonds = [("x", "y")]

        universe.guess_TopologyAttrs = fake_guess

        with patch(
            "prodock.postprocess.interaction.io._has_bonds",
            side_effect=fake_has_bonds,
        ):
            self.assertTrue(_guess_bonds(universe, atom_group))
            self.assertTrue(universe.called)

    def test_guess_bonds_via_atom_group_method(self) -> None:
        universe = object()
        atom_group = FakeAtomGroupNoBonds()
        self.assertTrue(_guess_bonds(universe, atom_group))


class TestIterNamedRDKitMolecules(unittest.TestCase):
    def test_single_rdkit_mol(self) -> None:
        mol = FakeChemModule.Mol()
        mol.SetProp("_Name", "pose_a")

        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            items = list(iter_named_rdkit_molecules(mol))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][0], "pose_a")

    def test_mapping_input(self) -> None:
        mol = FakeChemModule.Mol()

        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            items = list(iter_named_rdkit_molecules({"ligA": mol}))

        self.assertEqual(items[0][0], "ligA")
        self.assertEqual(items[0][1].GetProp("mol_name"), "ligA")

    def test_iterable_of_mols(self) -> None:
        mol1 = FakeChemModule.Mol()
        mol2 = FakeChemModule.Mol()
        mol2.SetProp("_Name", "mol_b")

        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            items = list(iter_named_rdkit_molecules([mol1, mol2]))

        self.assertEqual(items[0][0], "mol_0000")
        self.assertEqual(items[1][0], "mol_b")

    def test_iterable_of_named_pairs(self) -> None:
        mol = FakeChemModule.Mol()

        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            items = list(iter_named_rdkit_molecules([("ligX", mol)]))

        self.assertEqual(items[0][0], "ligX")

    def test_invalid_mapping_entry_raises(self) -> None:
        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            with self.assertRaises(InvalidLigandInputError):
                list(iter_named_rdkit_molecules({"bad": object()}))

    def test_invalid_iterable_entry_raises(self) -> None:
        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            with self.assertRaises(InvalidLigandInputError):
                list(iter_named_rdkit_molecules([object()]))

    def test_non_iterable_input_raises(self) -> None:
        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            with self.assertRaises(InvalidLigandInputError):
                list(iter_named_rdkit_molecules(12345))

    def test_missing_sdf_path_raises(self) -> None:
        with patch(
            "prodock.postprocess.interaction.io._import_rdkit_chem",
            return_value=FakeChemModule,
        ):
            with self.assertRaises(FileNotFoundError):
                list(iter_named_rdkit_molecules("missing_file.sdf"))

    def test_wrong_file_extension_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ligands.txt"
            path.write_text("dummy", encoding="utf-8")

            with patch(
                "prodock.postprocess.interaction.io._import_rdkit_chem",
                return_value=FakeChemModule,
            ):
                with self.assertRaises(InvalidLigandInputError):
                    list(iter_named_rdkit_molecules(path))

    def test_sdf_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "poses.sdf"
            path.write_text("", encoding="utf-8")

            with patch(
                "prodock.postprocess.interaction.io._import_rdkit_chem",
                return_value=FakeChemModule,
            ):
                items = list(iter_named_rdkit_molecules(path))

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][0], "pose_1")
        self.assertEqual(items[1][0], "mol_0002")


class TestPrepareLigands(unittest.TestCase):
    def test_prepare_ligands_success(self) -> None:
        mol = FakeChemModule.Mol()
        mol.SetProp("_Name", "lig1")

        with (
            patch(
                "prodock.postprocess.interaction.io._import_rdkit_chem",
                return_value=FakeChemModule,
            ),
            patch(
                "prodock.postprocess.interaction.io._import_prolif",
                return_value=FakePLF,
            ),
        ):
            names, rdkit_mols, prolif_mols = prepare_ligands(
                {"lig1": mol},
                resname="LIG",
                resnumber=5,
                chain="A",
                use_segid=True,
            )

        self.assertEqual(names, ["lig1"])
        self.assertEqual(len(rdkit_mols), 1)
        self.assertEqual(len(prolif_mols), 1)
        self.assertEqual(prolif_mols[0]["resname"], "LIG")
        self.assertEqual(prolif_mols[0]["resnumber"], 5)
        self.assertEqual(prolif_mols[0]["chain"], "A")
        self.assertTrue(prolif_mols[0]["use_segid"])

    def test_prepare_ligands_no_valid_molecules_raises(self) -> None:
        with (
            patch(
                "prodock.postprocess.interaction.io._import_prolif",
                return_value=FakePLF,
            ),
            patch(
                "prodock.postprocess.interaction.io.iter_named_rdkit_molecules",
                return_value=iter(()),
            ),
        ):
            with self.assertRaises(InvalidLigandInputError):
                prepare_ligands([])
