import unittest
from pathlib import Path

from prodock.dock.base import DockIdentity, RunArtifacts


class DummyBackend:
    """
    Minimal concrete implementation compatible with DockBackend for testing.
    """

    def __init__(self):
        self.receptor_path = None
        self.validate = False
        self.ligand_path = None
        self.center = None
        self.size = None
        self.reference_file = None
        self.padding = None
        self.exhaustiveness = None
        self.num_modes = None
        self.cpu = None
        self.seed = None
        self.out_path = None
        self.log_path = None
        self.run_called = False
        self.run_kwargs = {}

    def set_receptor(self, receptor_path, *, validate=False):
        self.receptor_path = Path(receptor_path)
        self.validate = validate
        return self

    def set_ligand(self, ligand_path):
        self.ligand_path = Path(ligand_path)
        return self

    def set_box(self, center, size):
        self.center = center
        self.size = size
        return self

    def enable_autobox(self, reference_file, padding=None):
        self.reference_file = Path(reference_file)
        self.padding = padding
        return self

    def set_exhaustiveness(self, value):
        self.exhaustiveness = value
        return self

    def set_num_modes(self, value):
        self.num_modes = value
        return self

    def set_cpu(self, value):
        self.cpu = value
        return self

    def set_seed(self, value):
        self.seed = value
        return self

    def set_out(self, out_path):
        self.out_path = Path(out_path)
        return self

    def set_log(self, log_path):
        self.log_path = Path(log_path)
        return self

    def run(self, *, exhaustiveness=None, n_poses=None):
        self.run_called = True
        self.run_kwargs = {
            "exhaustiveness": exhaustiveness,
            "n_poses": n_poses,
        }
        return self


class TestRunArtifacts(unittest.TestCase):
    def test_defaults(self):
        artifacts = RunArtifacts(out_path=None, log_path=None)
        self.assertIsNone(artifacts.out_path)
        self.assertIsNone(artifacts.log_path)
        self.assertIsNone(artifacts.called)
        self.assertEqual(artifacts.metadata, {})

    def test_explicit_values(self):
        artifacts = RunArtifacts(
            out_path=Path("poses.pdbqt"),
            log_path=Path("dock.log"),
            called="vina --config conf.txt",
            metadata={"engine": "vina", "status": "ok"},
        )
        self.assertEqual(artifacts.out_path, Path("poses.pdbqt"))
        self.assertEqual(artifacts.log_path, Path("dock.log"))
        self.assertEqual(artifacts.called, "vina --config conf.txt")
        self.assertEqual(artifacts.metadata["engine"], "vina")
        self.assertEqual(artifacts.metadata["status"], "ok")

    def test_metadata_is_not_shared(self):
        a1 = RunArtifacts(out_path=None, log_path=None)
        a2 = RunArtifacts(out_path=None, log_path=None)

        a1.metadata["x"] = 1
        self.assertNotIn("x", a2.metadata)


class TestDockIdentity(unittest.TestCase):
    def test_defaults(self):
        identity = DockIdentity()
        self.assertIsNone(identity.receptor_id)
        self.assertIsNone(identity.engine_name)
        self.assertIsNone(identity.ligand_id)

    def test_explicit_values(self):
        identity = DockIdentity(
            receptor_id="recA",
            engine_name="vina",
            ligand_id="lig1",
        )
        self.assertEqual(identity.receptor_id, "recA")
        self.assertEqual(identity.engine_name, "vina")
        self.assertEqual(identity.ligand_id, "lig1")


class TestDockBackendProtocolCompatibility(unittest.TestCase):
    def test_dummy_backend_is_instance_of_protocol_at_runtime(self):
        """
        Protocols are mainly for static typing, but this test ensures our dummy
        backend exposes the expected method names and basic behavior.
        """
        backend = DummyBackend()

        required_methods = [
            "set_receptor",
            "set_ligand",
            "set_box",
            "enable_autobox",
            "set_exhaustiveness",
            "set_num_modes",
            "set_cpu",
            "set_seed",
            "set_out",
            "set_log",
            "run",
        ]

        for method_name in required_methods:
            self.assertTrue(hasattr(backend, method_name))
            self.assertTrue(callable(getattr(backend, method_name)))

    def test_fluent_chaining(self):
        backend = DummyBackend()

        result = (
            backend.set_receptor("receptor.pdbqt", validate=True)
            .set_ligand("ligand.pdbqt")
            .set_box((1.0, 2.0, 3.0), (10.0, 12.0, 14.0))
            .enable_autobox("ref.sdf", padding=4.0)
            .set_exhaustiveness(16)
            .set_num_modes(9)
            .set_cpu(4)
            .set_seed(123)
            .set_out("out.pdbqt")
            .set_log("dock.log")
            .run(exhaustiveness=32, n_poses=5)
        )

        self.assertIs(result, backend)
        self.assertTrue(backend.run_called)

    def test_values_are_stored_correctly(self):
        backend = DummyBackend()
        backend.set_receptor("receptor.pdbqt", validate=True)
        backend.set_ligand("ligand.pdbqt")
        backend.set_box((1.0, 2.0, 3.0), (20.0, 20.0, 20.0))
        backend.enable_autobox("ref.sdf", padding=3.5)
        backend.set_exhaustiveness(8)
        backend.set_num_modes(7)
        backend.set_cpu(2)
        backend.set_seed(42)
        backend.set_out("poses.pdbqt")
        backend.set_log("dock.log")
        backend.run(exhaustiveness=10, n_poses=4)

        self.assertEqual(backend.receptor_path, Path("receptor.pdbqt"))
        self.assertTrue(backend.validate)
        self.assertEqual(backend.ligand_path, Path("ligand.pdbqt"))
        self.assertEqual(backend.center, (1.0, 2.0, 3.0))
        self.assertEqual(backend.size, (20.0, 20.0, 20.0))
        self.assertEqual(backend.reference_file, Path("ref.sdf"))
        self.assertEqual(backend.padding, 3.5)
        self.assertEqual(backend.exhaustiveness, 8)
        self.assertEqual(backend.num_modes, 7)
        self.assertEqual(backend.cpu, 2)
        self.assertEqual(backend.seed, 42)
        self.assertEqual(backend.out_path, Path("poses.pdbqt"))
        self.assertEqual(backend.log_path, Path("dock.log"))
        self.assertEqual(backend.run_kwargs["exhaustiveness"], 10)
        self.assertEqual(backend.run_kwargs["n_poses"], 4)


if __name__ == "__main__":
    unittest.main()
