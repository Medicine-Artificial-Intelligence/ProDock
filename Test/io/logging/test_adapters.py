import logging
import unittest

from prodock.io.logging.adapters import StructuredAdapter


class ListHandler(logging.Handler):
    """Simple handler that stores received LogRecord objects."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class TestStructuredAdapter(unittest.TestCase):
    def test_structured_adapter_merges_extras(self):
        logger = logging.getLogger("test.adapters")
        logger.setLevel(logging.DEBUG)
        lh = ListHandler()
        logger.addHandler(lh)

        adapter = StructuredAdapter(logger, {"run_id": "r1", "pdb": "5N2F"})
        adapter.info("msg", extra={"step": "prepare"})

        # expect one record
        self.assertEqual(len(lh.records), 1)
        rec = lh.records[0]

        # Adapter merges context into record attributes
        self.assertTrue(hasattr(rec, "run_id"))
        self.assertTrue(hasattr(rec, "pdb"))
        self.assertTrue(hasattr(rec, "step"))
        self.assertEqual(getattr(rec, "run_id"), "r1")
        self.assertEqual(getattr(rec, "pdb"), "5N2F")
        self.assertEqual(getattr(rec, "step"), "prepare")

        # cleanup handlers
        logger.removeHandler(lh)


if __name__ == "__main__":
    unittest.main()
