import os
import tempfile
import shutil
import unittest
from pathlib import Path

from prodock.dock.engine.registry import register
from prodock.dock.engine.common_binary import BaseBinaryEngine
from prodock.dock.engine.single import SingleDock  # note: import concrete path


class _SminaFake(BaseBinaryEngine):
    # Force resolution failure to avoid calling any real binary
    exe_name = "___definitely_missing_binary___"
    supports_autobox = True


class TestSingleDock(unittest.TestCase):
    def setUp(self):
        # temp workspace + expected Data tree
        self.td = Path(tempfile.mkdtemp())
        self._cwd = Path.cwd()
        os.chdir(self.td)  # keep relative paths simple, like in your snippet

        # Create the expected input tree
        rec = self.td / "Data/testcase/dock/receptor"
        lig = self.td / "Data/testcase/dock/ligand"
        rec.mkdir(parents=True, exist_ok=True)
        lig.mkdir(parents=True, exist_ok=True)
        (rec / "5N2F.pdbqt").write_text("RECEPTOR")
        (lig / "8HW.pdbqt").write_text("LIGAND")

        # Register our minimal CLI backend under the name "smina"
        register("smina", lambda: _SminaFake())

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.td, ignore_errors=True)

    def test_chain_and_run_raises_missing_binary(self):
        # Use exactly your snippet (works under cwd=self.td thanks to created Data/)
        sd = (
            SingleDock("smina")
            .set_receptor("Data/testcase/dock/receptor/5N2F.pdbqt", validate=True)
            .set_ligand("Data/testcase/dock/ligand/8HW.pdbqt")
            .set_box((12, 8, 5), (20, 20, 20))
            .set_exhaustiveness(8)
            .set_num_modes(9)
            .set_cpu(4)
            .set_seed(42)
            .set_out("out/lig_docked.pdbqt")
            .set_log("out/lig.log")
        )

        # Running attempts to resolve the binary and must fail cleanly
        with self.assertRaises(FileNotFoundError):
            sd.run()

        # Ensure our intended output paths were staged on the facade
        self.assertEqual(sd._out, Path("out/lig_docked.pdbqt"))
        self.assertEqual(sd._log, Path("out/lig.log"))

    def test_run_with_config_prefer_instance_still_uses_this_engine(self):
        # Prepare a JSON config that says "engine": "smina" (any value is OK here)
        cfg_path = self.td / "single.json"
        cfg_path.write_text(
            '{"engine":"smina","receptor":"Data/testcase/dock/receptor/5N2F.pdbqt",'
            '"ligand":"Data/testcase/dock/ligand/8HW.pdbqt",'
            '"box":{"center":[12,8,5],"size":[20,20,20]},'
            '"exhaustiveness":8,"n_poses":9,"cpu":4,"seed":42,'
            '"out":"out/lig_docked.pdbqt","log":"out/lig.log","validate_receptor":true}'
        )

        # Also register another name to ensure registry works (both use same fake)
        from prodock.dock.engine.registry import register as _reg

        _reg("qvina", lambda: _SminaFake())

        sd = SingleDock("qvina")
        # Prefer the instance engine ("qvina") even if config says "smina"
        with self.assertRaises(FileNotFoundError):
            sd.run_with_config(str(cfg_path), prefer="instance")

        self.assertEqual(sd.engine, "qvina")  # engine on instance unchanged


if __name__ == "__main__":
    unittest.main()
