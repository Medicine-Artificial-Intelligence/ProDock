# tests/test_repr_helpers.py
import unittest
from pathlib import Path

from prodock.preprocess.receptor.repr_helpers import ReprMixin


class Dummy(ReprMixin):
    pass


class TestReprHelpers(unittest.TestCase):
    def test_basic_info_and_converter(self):
        d = Dummy()
        d._final_artifact = Path("/tmp/hello.pdb")
        d._used_obabel = True
        d._minimized_stage = "obabel"
        d._use_meeko = False
        d._last_simulation_report = {"out_fmt": "pdb"}

        info = d._repr_basic_info()
        self.assertEqual(info["artifact_name"], "hello.pdb")
        self.assertTrue(info["used_obabel"])
        self.assertEqual(info["out_fmt"], "pdb")

        conv = d._repr_converter_status(info)
        # when used_obabel is True, converter status should indicate OpenBabel
        self.assertTrue(isinstance(conv, str))
        self.assertIn("open", conv.lower())

    def test_repr_and_str(self):
        d = Dummy()
        d._final_artifact = None
        d._used_obabel = False
        d._last_simulation_report = None
        s = str(d)
        r = repr(d)
        self.assertIsInstance(s, str)
        self.assertIsInstance(r, str)
        self.assertIn("ReceptorPrep Summary", r)


if __name__ == "__main__":
    unittest.main()
