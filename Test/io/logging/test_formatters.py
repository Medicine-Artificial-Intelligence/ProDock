import logging
import json
import unittest

from prodock.io.logging.formatters import JSONFormatter, SimpleColorFormatter


class TestFormatters(unittest.TestCase):
    def test_json_formatter_basic(self):
        fmt = JSONFormatter()
        r = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        out = fmt.format(r)
        data = json.loads(out)
        self.assertEqual(data["message"], "hello")
        self.assertEqual(data["level"], "INFO")
        self.assertIn("ts", data)
        self.assertIn("logger", data)

    def test_simple_color_formatter_no_error(self):
        fmt = SimpleColorFormatter(force_color=False)
        r = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hi",
            args=(),
            exc_info=None,
        )
        out = fmt.format(r)
        self.assertIn("hi", out)


if __name__ == "__main__":
    unittest.main()
