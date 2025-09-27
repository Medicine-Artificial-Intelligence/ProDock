import unittest

from prodock.dock.engine.registry import register, factory


class TestRegistry(unittest.TestCase):
    def test_register_and_factory(self):
        class Dummy:  # simple factory product
            pass

        register("MyEngine", lambda: Dummy())
        f = factory("myengine")
        inst = f()
        self.assertIsInstance(inst, Dummy)

    def test_unknown_engine_raises(self):
        with self.assertRaises(KeyError):
            factory("does_not_exist")


if __name__ == "__main__":
    unittest.main()
