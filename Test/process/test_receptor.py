# tests/process/test_receptor_process.py
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import importlib
from prodock.process.receptor import ReceptorProcess

rp_mod = importlib.import_module("prodock.process.receptor")


# ----------------------
# Lightweight fakes
# ----------------------
class FakeAtom:
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index


class FakeTopology:
    def __init__(self, n_atoms=3):
        # produce three backbone atoms N, CA, C
        self._atoms = [FakeAtom(n, i) for i, n in enumerate(("N", "CA", "C"))]

    def atoms(self):
        return list(self._atoms)


class FakePosition:
    """Simple position object supporting value_in_unit or indexing."""

    def __init__(self, x: float, y: float, z: float):
        self._v = (x, y, z)

    def value_in_unit(self, unit):
        # ignore unit, just return tuple-of-values
        return self._v

    def __getitem__(self, idx):
        return self._v[idx]

    @property
    def x(self):
        return self._v[0]

    @property
    def y(self):
        return self._v[1]

    @property
    def z(self):
        return self._v[2]


class FakePDBFixer:
    def __init__(self, filename=None):
        # pretend to parse file; set topology and initial positions
        self.topology = FakeTopology()
        self.positions = [FakePosition(0.1, 0.2, 0.3) for _ in range(3)]

    def removeHeterogens(self, keepWater=False):
        pass

    def findMissingResidues(self):
        pass

    def findMissingAtoms(self):
        pass

    def addMissingAtoms(self):
        pass

    def addMissingHydrogens(self, pH=7.4):
        pass


class FakeModeller:
    def __init__(self, topology, positions):
        # copy topology and positions to mimic OpenMM Modeller
        self.topology = topology
        self.positions = positions

    def addSolvent(self, ff, model="tip3p", padding=None, ionicStrength=None):
        # Add some solvent by appending positions (makes positions longer)
        self.positions = self.positions + [FakePosition(1.0, 1.1, 1.2)]


class FakeSystem:
    def __init__(self):
        self.forces = []

    def addForce(self, f):
        self.forces.append(f)


class FakeForceField:
    def __init__(self, *args, **kwargs):
        pass

    def createSystem(self, topology, nonbondedMethod=None, constraints=None):
        return FakeSystem()


class FakePlatform:
    @staticmethod
    def getName():
        return "FAKE"


class FakeContext:
    def __init__(self):
        self._positions = []

    def setPositions(self, positions):
        # store positions (list-like)
        self._positions = positions

    def getState(self, getPositions=False):
        # return a state object with getPositions
        class S:
            def __init__(self, positions):
                self._p = positions

            def getPositions(self):
                return self._p

        return S(self._positions)


class FakeSimulation:
    def __init__(self, topology, system, integrator, platform, props):
        self.topology = topology
        self._context = FakeContext()

    @property
    def context(self):
        return self._context

    def minimizeEnergy(self, tolerance=None, maxIterations=None):
        # pretend to minimize (no-op)
        pass


class FakeLangevinIntegrator:
    def __init__(self, *args, **kwargs):
        pass

    def setRandomNumberSeed(self, s):
        pass


class FakePDBFile:
    """Fake PDBFile that can be constructed from a path and used for writing/reading."""

    def __init__(self, path_or_str=None):
        # ignore input; create topology and positions similar to above
        self.topology = FakeTopology()
        self.positions = [FakePosition(0.2, 0.3, 0.4) for _ in range(3)]

    @staticmethod
    def writeFile(topology, positions, fh):
        fh.write("RENDERED PDB\n")


# ----------------------
# Tests
# ----------------------
class TestReceptorProcess(unittest.TestCase):
    def setUp(self):
        # temp dir for files and temporary executables
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

        # save original PATH so we can restore in tearDown
        self._orig_path = os.environ.get("PATH", "")

        # keep originals to restore later
        self._orig_modules = {
            "GridBox": getattr(rp_mod, "GridBox", None),
            "cmd": getattr(rp_mod, "cmd", None),
            "PDBFixer": getattr(rp_mod, "PDBFixer", None),
            "Modeller": getattr(rp_mod, "Modeller", None),
            "ForceField": getattr(rp_mod, "ForceField", None),
            "Platform": getattr(rp_mod, "Platform", None),
            "Simulation": getattr(rp_mod, "Simulation", None),
            "LangevinIntegrator": getattr(rp_mod, "LangevinIntegrator", None),
            "PDBFile": getattr(rp_mod, "PDBFile", None),
        }

        # inject fakes into module
        class FakeGridBoxObj:
            def __init__(self):
                self.center = (1.0, 2.0, 3.0)
                self.size = (25.0, 25.0, 25.0)

            def preset(self, *args, **kwargs):
                return self

            def from_ligand_pad_adv(self, *args, **kwargs):
                self.size = (30.0, 25.0, 25.0)
                return self

            def from_ligand_pad(self, *args, **kwargs):
                self.size = (28.0, 25.0, 25.0)
                return self

            def from_ligand_scale(self, scale=1.0, isotropic=True):
                self.size = tuple(s * scale for s in self.size)
                return self

            def from_center_size(self, center, size):
                self.center = center
                self.size = size
                return self

            @property
            def vina_dict(self):
                return {
                    "center_x": self.center[0],
                    "center_y": self.center[1],
                    "center_z": self.center[2],
                    "size_x": self.size[0],
                    "size_y": self.size[1],
                    "size_z": self.size[2],
                }

        class FakeGridBoxFactory:
            def load_ligand(self, path):
                # ignore path, return object with methods
                return FakeGridBoxObj()

        rp_mod.GridBox = FakeGridBoxFactory

        # PyMOL cmd: simple no-op object
        rp_mod.cmd = SimpleNamespace(
            load=lambda *a, **k: None,
            alter=lambda *a, **k: None,
            select=lambda *a, **k: None,
            remove=lambda *a, **k: None,
            save=lambda *a, **k: None,
            delete=lambda *a, **k: None,
        )

        rp_mod.PDBFixer = FakePDBFixer
        rp_mod.Modeller = FakeModeller
        rp_mod.ForceField = FakeForceField
        rp_mod.Platform = SimpleNamespace(getPlatformByName=lambda name: FakePlatform())
        rp_mod.Simulation = FakeSimulation
        rp_mod.LangevinIntegrator = FakeLangevinIntegrator
        rp_mod.PDBFile = FakePDBFile

        # create a tiny dummy PDB to feed into ReceptorProcess
        self.input_pdb = self.tmp / "input.pdb"
        self.input_pdb.write_text(
            "HEADER DUMMY\nATOM 1 N\nATOM 2 CA\nATOM 3 C\n", encoding="utf-8"
        )

    def tearDown(self):
        # restore PATH first
        os.environ["PATH"] = self._orig_path

        # restore originals
        for k, v in self._orig_modules.items():
            if v is None:
                if hasattr(rp_mod, k):
                    delattr(rp_mod, k)
            else:
                setattr(rp_mod, k, v)

        self.tmpdir.cleanup()

    def _make_fake_executable(self, name: str):
        """
        Create a small executable script in tmp bin that writes any requested outputs.
        The script writes files passed after flags (--write_pdbqt,'--write_gpf') or after '-o'.
        Returns path to the script and ensures PATH is patched to find it.
        """
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir(exist_ok=True)
        script = bin_dir / name
        # Python script that writes files passed after known flags (supports '-o' for ADT)
        py = (
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "argv = sys.argv[1:]\n"
            "outpaths = []\n"
            "for i,a in enumerate(argv):\n"
            "    if a in ('--write_pdbqt','--write_gpf') and i+1 < len(argv):\n"
            "        outpaths.append(argv[i+1])\n"
            "    if a == '-o' and i+1 < len(argv):\n"
            "        outpaths.append(argv[i+1])\n"
            "for p in outpaths:\n"
            "    try:\n"
            "        open(p,'w').write('CREATED ' + p + '\\n')\n"
            "    except Exception:\n"
            "        pass\n"
            "print('STDOUT: wrote', outpaths)\n"
            "sys.exit(0)\n"
        )
        script.write_text(py, encoding="utf-8")
        st = script.stat().st_mode
        script.chmod(st | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        # prepend bin_dir to PATH so shutil.which can find it
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        return str(script)

    def test_compute_gridbox_from_ligand_success(self):
        rp = ReceptorProcess()
        # should return a dict with center/size keys
        res = rp.compute_gridbox_from_ligand(str(self.input_pdb))
        self.assertIsInstance(res, dict)
        for k in ("center_x", "size_x"):
            self.assertIn(k, res)

    def test_call_mekoo_not_found(self):
        rp = ReceptorProcess(mekoo_cmd="nonexistent_mekoo_cmd")
        in_pdb = Path(self.input_pdb)
        out_basename = self.tmp / "outbase"
        info = rp._call_mekoo(input_pdb=in_pdb, out_basename=out_basename)
        self.assertIn("mekoo", info.get("stderr", "").lower())
        self.assertEqual(info["produced"], [])

    def test_call_adt_and_fix_and_minimize_pdb_pdbqt_via_adt_fallback(self):
        # Build a fake ADT in tmp bin that will create the requested pdbqt when invoked.
        _ = self._make_fake_executable("fake_adt.py")
        rp = ReceptorProcess(mekoo_cmd="nonexistent_mekoo_cmd", adt_cmd="fake_adt.py")
        outdir = self.tmp / "out"
        outdir.mkdir()
        # ask for pdbqt output (mekoo missing -> ADT fallback)
        rp.fix_and_minimize_pdb(
            input_pdb=str(self.input_pdb),
            output_dir=str(outdir),
            minimize_in_water=False,
            pdb_id="DUMMY",
            protein_name="DUMMY",
            out_fmt="pdbqt",
            input_ligand=None,  # not required for pdbqt
        )
        # after running, last_simulation_report should exist
        self.assertIsNotNone(rp.last_simulation_report)
        final = Path(rp.last_simulation_report["final_artifact"])
        self.assertTrue(final.exists())

    def test_fix_and_minimize_pdb_minimize_in_water_true_produces_pdb(self):
        # For this test, use out_fmt='pdb' and minimize_in_water True to exercise water path
        rp = ReceptorProcess(
            mekoo_cmd="nonexistent_mekoo_cmd", adt_cmd="nonexistent_adt"
        )
        outdir = self.tmp / "out2"
        outdir.mkdir()
        rp.fix_and_minimize_pdb(
            input_pdb=str(self.input_pdb),
            output_dir=str(outdir),
            minimize_in_water=True,
            pdb_id="D2",
            protein_name="D2",
            out_fmt="pdb",
        )
        final = Path(rp.last_simulation_report["final_artifact"])
        self.assertTrue(final.exists())
        self.assertEqual(final.suffix.lower(), ".pdb")
        self.assertEqual(rp.last_simulation_report["minimized_stage"], "solvent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
