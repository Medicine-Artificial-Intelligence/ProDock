import logging
import unittest

from prodock.io.logging.decorators import log_step


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class Dummy:
    def __init__(self, logger):
        self.logger = logger
        self.log_context = {"run": "r1"}

    @log_step("dostep")
    def step(self):
        # returning a dict so _try_summary_from may pick it up (but not required)
        return {"ok": True}


class TestLogStepDecorator(unittest.TestCase):
    def test_log_step_sync(self):
        logger = logging.getLogger("test.decorators")
        logger.setLevel(logging.INFO)
        lh = ListHandler()
        logger.addHandler(lh)

        d = Dummy(logger)
        d.step()

        # Expect at least two messages: step.start and step.finish
        messages = [r.getMessage() for r in lh.records]
        self.assertTrue(
            any("step.start" == m for m in messages), f"start not in {messages}"
        )
        self.assertTrue(
            any("step.finish" == m for m in messages), f"finish not in {messages}"
        )

        # check that finish record has elapsed attribute
        finish_recs = [r for r in lh.records if r.getMessage() == "step.finish"]
        self.assertTrue(len(finish_recs) >= 1)
        fin = finish_recs[0]
        self.assertTrue(hasattr(fin, "elapsed"))
        # elapsed should be numeric
        self.assertIsInstance(getattr(fin, "elapsed"), float)

        logger.removeHandler(lh)


if __name__ == "__main__":
    unittest.main()
