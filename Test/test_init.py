# Test/test_init.py
import unittest


class TestProDockImport(unittest.TestCase):
    def test_import_prodock_root(self):
        """
        Verify that `from prodock import ProDock` succeeds and the symbol is callable.
        This test avoids importing the entire heavy pipeline until ProDock is accessed.
        """
        from prodock import ProDock  # should trigger lazy load in __getattr__

        # ProDock should be a class or callable factory for pipeline objects
        self.assertTrue(
            callable(ProDock), "ProDock should be callable (a class or factory)"
        )

    def test_dir_contains_prodock(self):
        # Also ensure `dir(prodock)` exposes the alias after import
        import prodock

        # Access attribute to populate module cache if lazy
        _ = getattr(prodock, "ProDock", None)
        self.assertIn("ProDock", dir(prodock))


if __name__ == "__main__":
    unittest.main()
