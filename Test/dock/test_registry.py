import unittest

from prodock.dock import registry
from prodock.dock.registry import available, factory, register, register_many


class TestRegistry(unittest.TestCase):
    def setUp(self):
        """
        Preserve global registry state so each test remains isolated.
        """
        self._old_registry = dict(registry._REGISTRY)
        registry._REGISTRY.clear()

    def tearDown(self):
        """
        Restore original global registry contents after each test.
        """
        registry._REGISTRY.clear()
        registry._REGISTRY.update(self._old_registry)

    def test_register_and_factory(self):
        class Dummy:
            pass

        register("MyEngine", lambda: Dummy())
        f = factory("myengine")
        inst = f()

        self.assertTrue(callable(f))
        self.assertIsInstance(inst, Dummy)

    def test_register_is_case_insensitive(self):
        class Dummy:
            pass

        register("MyEngine", lambda: Dummy())

        f1 = factory("myengine")
        f2 = factory("MYENGINE")
        f3 = factory("MyEnGiNe")

        self.assertTrue(callable(f1))
        self.assertIs(f1, f2)
        self.assertIs(f2, f3)

    def test_register_stores_lowercase_key(self):
        class Dummy:
            pass

        register("MyEngine", lambda: Dummy())

        self.assertIn("myengine", registry._REGISTRY)
        self.assertNotIn("MyEngine", registry._REGISTRY)

    def test_factory_returns_exact_registered_callable(self):
        def make_dummy():
            return object()

        register("engine_x", make_dummy)
        self.assertIs(factory("ENGINE_X"), make_dummy)

    def test_unknown_engine_raises(self):
        with self.assertRaises(KeyError) as ctx:
            factory("does_not_exist")

        msg = str(ctx.exception)
        self.assertIn("does_not_exist", msg)
        self.assertIn("<empty>", msg)

    def test_unknown_engine_lists_available(self):
        register("vina", lambda: object())
        register("smina", lambda: object())

        with self.assertRaises(KeyError) as ctx:
            factory("gnina")

        msg = str(ctx.exception)
        self.assertIn("Unknown docking engine", msg)
        self.assertIn("gnina", msg)
        self.assertIn("smina", msg)
        self.assertIn("vina", msg)

    def test_available_empty(self):
        self.assertEqual(available(), [])

    def test_available_returns_sorted_lowercase_names(self):
        register("Smina", lambda: object())
        register("vina", lambda: object())
        register("GNINA", lambda: object())

        self.assertEqual(available(), ["gnina", "smina", "vina"])

    def test_register_many(self):
        class A:
            pass

        class B:
            pass

        register_many(
            [
                ("EngineA", lambda: A()),
                ("engineb", lambda: B()),
            ]
        )

        self.assertEqual(available(), ["enginea", "engineb"])
        self.assertIsInstance(factory("ENGINEA")(), A)
        self.assertIsInstance(factory("engineB")(), B)

    def test_register_many_overwrites_existing_entry(self):
        class A:
            pass

        class B:
            pass

        register("test", lambda: A())
        register_many([("TEST", lambda: B())])

        inst = factory("test")()
        self.assertIsInstance(inst, B)

    def test_register_overwrites_existing_factory(self):
        class A:
            pass

        class B:
            pass

        register("dup", lambda: A())
        first = factory("dup")()
        self.assertIsInstance(first, A)

        register("DUP", lambda: B())
        second = factory("dup")()
        self.assertIsInstance(second, B)

    def test_register_many_accepts_iterable(self):
        class Dummy:
            pass

        items = ((name, lambda cls=Dummy: cls()) for name in ["one", "two", "three"])
        register_many(items)

        self.assertEqual(available(), ["one", "three", "two"])
        self.assertIsInstance(factory("one")(), Dummy)
        self.assertIsInstance(factory("two")(), Dummy)
        self.assertIsInstance(factory("three")(), Dummy)


if __name__ == "__main__":
    unittest.main()
