import logging
import os
import tempfile
import unittest

from prodock.io.logging.manager import setup_logging, get_logger


class TestManager(unittest.TestCase):
    def test_setup_writes_file(self):
        # Create temporary directory
        with tempfile.TemporaryDirectory() as td:
            # Configure logging to write a JSON file
            setup_logging(
                log_dir=td, log_file="test.log", level="DEBUG", colored=False, json=True
            )
            logger = get_logger("prodock.test.manager")
            # Log something
            logger.info("hello world", extra={"k": "v"})
            # Ensure file exists and contains content
            p = os.path.join(td, "test.log")
            self.assertTrue(os.path.exists(p), "Log file was not created")
            # Some handlers may buffer, flush by shutting down logging
            logging.shutdown()
            with open(p, "r", encoding="utf-8") as fh:
                content = fh.read()
            self.assertTrue(len(content.strip()) > 0, "Log file is empty")

    def tearDown(self):
        # Reset root handlers to avoid interference between tests/runs
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)


if __name__ == "__main__":
    unittest.main()
