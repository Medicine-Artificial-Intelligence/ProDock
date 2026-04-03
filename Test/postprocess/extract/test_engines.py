from __future__ import annotations

import unittest

from prodock.postprocess.extract.engines import (
    ENGINE_HINTS,
    GNINA_ROW_RE,
    GNINA_TABLE_HEADER,
    VINA_ROW_RE,
    VINA_TABLE_HEADER,
    detect_engine,
)


class TestEngineHints(unittest.TestCase):
    def test_engine_hints_is_not_empty(self) -> None:
        self.assertTrue(len(ENGINE_HINTS) > 0)

    def test_engine_hints_entries_are_pairs(self) -> None:
        for item in ENGINE_HINTS:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_more_specific_patterns_appear_before_generic_vina(self) -> None:
        names = [name for _, name in ENGINE_HINTS]
        self.assertLess(names.index("gnina"), names.index("vina"))
        self.assertLess(names.index("smina"), names.index("vina"))
        self.assertLess(names.index("qvina"), names.index("vina"))


class TestVinaRegexes(unittest.TestCase):
    def test_vina_table_header_matches_standard_header(self) -> None:
        text = "mode | affinity | dist from best mode"
        self.assertIsNotNone(VINA_TABLE_HEADER.search(text))

    def test_vina_table_header_matches_case_insensitively(self) -> None:
        text = "MODE | AFFINITY | DIST FROM BEST MODE"
        self.assertIsNotNone(VINA_TABLE_HEADER.search(text))

    def test_vina_row_re_matches_standard_row(self) -> None:
        line = "  1   -7.5   0.000   0.000"
        m = VINA_ROW_RE.match(line)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.group(1), "1")
        self.assertEqual(m.group(2), "-7.5")
        self.assertEqual(m.group(3), "0.000")
        self.assertEqual(m.group(4), "0.000")

    def test_vina_row_re_matches_exponential_values(self) -> None:
        line = "2 -7.5E+00 1.2e+00 2.4E-01"
        m = VINA_ROW_RE.match(line)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.group(1), "2")
        self.assertEqual(m.group(2), "-7.5E+00")
        self.assertEqual(m.group(3), "1.2e+00")
        self.assertEqual(m.group(4), "2.4E-01")

    def test_vina_row_re_rejects_invalid_row(self) -> None:
        self.assertIsNone(VINA_ROW_RE.match("mode affinity rmsd"))


class TestGninaRegexes(unittest.TestCase):
    def test_gnina_table_header_matches_standard_header(self) -> None:
        text = "mode |  affinity  |    CNN     |   CNN"
        self.assertIsNotNone(GNINA_TABLE_HEADER.search(text))

    def test_gnina_table_header_matches_case_insensitively(self) -> None:
        text = "MODE |  AFFINITY  |    CNN     |   CNN"
        self.assertIsNotNone(GNINA_TABLE_HEADER.search(text))

    def test_gnina_row_re_matches_standard_row(self) -> None:
        line = "  1   -8.2   0.71   7.45"
        m = GNINA_ROW_RE.match(line)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.group(1), "1")
        self.assertEqual(m.group(2), "-8.2")
        self.assertEqual(m.group(3), "0.71")
        self.assertEqual(m.group(4), "7.45")

    def test_gnina_row_re_matches_signed_and_exponential_values(self) -> None:
        line = "2 -8.2E+00 +7.1e-01 -1.2E+01"
        m = GNINA_ROW_RE.match(line)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.group(1), "2")
        self.assertEqual(m.group(2), "-8.2E+00")
        self.assertEqual(m.group(3), "+7.1e-01")
        self.assertEqual(m.group(4), "-1.2E+01")

    def test_gnina_row_re_rejects_invalid_row(self) -> None:
        self.assertIsNone(GNINA_ROW_RE.match("cnn row invalid"))


class TestDetectEngine(unittest.TestCase):
    def test_detect_engine_returns_none_for_none_input(self) -> None:
        self.assertIsNone(detect_engine(None))  # type: ignore[arg-type]

    def test_detect_engine_returns_none_when_no_match(self) -> None:
        self.assertIsNone(detect_engine("plain text without known engine markers"))

    def test_detect_engine_detects_gnina_banner(self) -> None:
        self.assertEqual(detect_engine("GNINA 1.0 docking run"), "gnina")

    def test_detect_engine_detects_smina_banner(self) -> None:
        self.assertEqual(detect_engine("smina execution started"), "smina")

    def test_detect_engine_detects_qvina_banner(self) -> None:
        self.assertEqual(detect_engine("qvina run"), "qvina")

    def test_detect_engine_detects_quickvina2_banner(self) -> None:
        self.assertEqual(detect_engine("QuickVina2 docking"), "qvina")

    def test_detect_engine_detects_quick_vina_2_gpu_banner(self) -> None:
        self.assertEqual(detect_engine("Quick Vina 2 GPU"), "qvina-gpu")

    def test_detect_engine_detects_quickvina2_gpu_banner(self) -> None:
        self.assertEqual(detect_engine("quickvina2-gpu"), "qvina-gpu")

    def test_detect_engine_detects_autodock_vina_banner(self) -> None:
        self.assertEqual(detect_engine("AutoDock Vina 1.2.5"), "vina")

    def test_detect_engine_detects_vina_gpu_banner(self) -> None:
        self.assertEqual(detect_engine("vina-gpu started"), "vina-gpu")

    def test_detect_engine_order_prefers_gnina_over_vina(self) -> None:
        self.assertEqual(detect_engine("gnina built on top of vina"), "gnina")

    def test_detect_engine_quick_vina_hyphenated_text_currently_matches_generic_vina(
        self,
    ) -> None:
        text = "This log mentions quick-vina backend"
        self.assertEqual(detect_engine(text), "vina")

    def test_detect_engine_quick_vina_gpu_free_text_currently_matches_generic_vina(
        self,
    ) -> None:
        text = "Using quick vina backend with gpu acceleration"
        self.assertEqual(detect_engine(text), "vina")

    def test_detect_engine_vina_and_gpu_free_text_currently_matches_generic_vina(
        self,
    ) -> None:
        text = "vina was launched with gpu acceleration"
        self.assertEqual(detect_engine(text), "vina")

    def test_detect_engine_header_based_gnina_detection(self) -> None:
        text = """
        some preamble
        mode | affinity | cnn pose | cnn affinity
           1   -8.2   0.71   7.45
        """
        self.assertEqual(detect_engine(text), "gnina")

    def test_detect_engine_header_based_vina_detection(self) -> None:
        text = """
        other preamble
        mode | affinity | dist from best mode
           1   -7.5   0.000   0.000
        """
        self.assertEqual(detect_engine(text), "vina")


if __name__ == "__main__":
    unittest.main()
