import logging
import unittest

from prodock.io.logging.timing import Timer


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class TestTimer(unittest.TestCase):
    def test_timer_logs_elapsed(self):
        logger = logging.getLogger("test.timer")
        logger.setLevel(logging.DEBUG)
        lh = ListHandler()
        logger.addHandler(lh)

        with Timer("unit-test", logger=logger) as _:
            # quick block
            pass

        # ensure a debug record was emitted
        found = False
        for r in lh.records:
            # message is 'timer.elapsed' in the implementation
            if r.getMessage() == "timer.elapsed":
                found = True
                # the adapter adds extra 'elapsed' attribute
                self.assertTrue(hasattr(r, "elapsed"))
                # elapsed should be a float >= 0
                self.assertIsInstance(getattr(r, "elapsed"), float)
                self.assertGreaterEqual(getattr(r, "elapsed"), 0.0)
                break

        self.assertTrue(found, "timer.elapsed record was not emitted")

        logger.removeHandler(lh)


if __name__ == "__main__":
    unittest.main()
