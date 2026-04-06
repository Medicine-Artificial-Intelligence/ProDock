from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prodock.postprocess.extract.utils import (
    LOG_SUFFIXES,
    TABLE_SUFFIXES,
    build_engine_pattern,
    engine_matches,
    is_log_path,
    is_table_path,
    normalize_engine_token,
)


class TestNormalizeEngineToken(unittest.TestCase):
    def test_normalize_engine_token_basic(self) -> None:
        self.assertEqual(normalize_engine_token("Vina"), "vina")

    def test_normalize_engine_token_strips_whitespace(self) -> None:
        self.assertEqual(normalize_engine_token("  SMINA  "), "smina")

    def test_normalize_engine_token_handles_non_string_like_input(self) -> None:
        self.assertEqual(normalize_engine_token(123), "123")  # type: ignore[arg-type]


class TestBuildEnginePattern(unittest.TestCase):
    def test_build_engine_pattern_multiple_values(self) -> None:
        pattern = build_engine_pattern(["vina", "smina", "qvina"])
        self.assertEqual(pattern, "vina|smina|qvina")

    def test_build_engine_pattern_normalizes_and_skips_empty(self) -> None:
        pattern = build_engine_pattern([" Vina ", "", "   ", "SMINA"])
        self.assertEqual(pattern, "vina|smina")

    def test_build_engine_pattern_empty_input(self) -> None:
        self.assertEqual(build_engine_pattern([]), "")

    def test_build_engine_pattern_escapes_regex_special_characters(self) -> None:
        pattern = build_engine_pattern(["qvina+gpu", "vina.test"])
        self.assertIn(r"qvina\+gpu", pattern)
        self.assertIn(r"vina\.test", pattern)


class TestEngineMatches(unittest.TestCase):
    def test_engine_matches_returns_true_when_pattern_empty(self) -> None:
        self.assertTrue(engine_matches("vina", ""))
        self.assertTrue(engine_matches("", ""))

    def test_engine_matches_returns_false_for_empty_engine_when_pattern_given(
        self,
    ) -> None:
        pattern = build_engine_pattern(["vina"])
        self.assertFalse(engine_matches("", pattern))

    def test_engine_matches_case_insensitive(self) -> None:
        pattern = build_engine_pattern(["vina", "smina"])
        self.assertTrue(engine_matches("VINA", pattern))
        self.assertTrue(engine_matches("Smina", pattern))

    def test_engine_matches_non_matching_value(self) -> None:
        pattern = build_engine_pattern(["vina", "smina"])
        self.assertFalse(engine_matches("gnina", pattern))

    def test_engine_matches_substring_behavior(self) -> None:
        pattern = build_engine_pattern(["vina"])
        self.assertTrue(engine_matches("qvina-w", pattern))


class TestPathHelpers(unittest.TestCase):
    def test_log_suffixes_defined(self) -> None:
        self.assertIn(".log", LOG_SUFFIXES)
        self.assertIn(".txt", LOG_SUFFIXES)

    def test_table_suffixes_defined(self) -> None:
        self.assertIn(".csv", TABLE_SUFFIXES)
        self.assertIn(".tsv", TABLE_SUFFIXES)
        self.assertIn(".tab", TABLE_SUFFIXES)

    def test_is_log_path_true_for_existing_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.log"
            path.write_text("content", encoding="utf-8")
            self.assertTrue(is_log_path(path))

    def test_is_log_path_true_for_existing_txt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.txt"
            path.write_text("content", encoding="utf-8")
            self.assertTrue(is_log_path(path))

    def test_is_log_path_false_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.log"
            self.assertFalse(is_log_path(path))

    def test_is_log_path_false_for_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            self.assertFalse(is_log_path(path))

    def test_is_log_path_false_for_unsupported_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.csv"
            path.write_text("content", encoding="utf-8")
            self.assertFalse(is_log_path(path))

    def test_is_log_path_suffix_check_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "RUN.LOG"
            path.write_text("content", encoding="utf-8")
            self.assertTrue(is_log_path(path))

    def test_is_table_path_true_for_existing_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.csv"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            self.assertTrue(is_table_path(path))

    def test_is_table_path_true_for_existing_tsv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.tsv"
            path.write_text("a\tb\n1\t2\n", encoding="utf-8")
            self.assertTrue(is_table_path(path))

    def test_is_table_path_true_for_existing_tab_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.tab"
            path.write_text("a\tb\n1\t2\n", encoding="utf-8")
            self.assertTrue(is_table_path(path))

    def test_is_table_path_false_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.csv"
            self.assertFalse(is_table_path(path))

    def test_is_table_path_false_for_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            self.assertFalse(is_table_path(path))

    def test_is_table_path_false_for_unsupported_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.log"
            path.write_text("content", encoding="utf-8")
            self.assertFalse(is_table_path(path))

    def test_is_table_path_suffix_check_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "SCORES.CSV"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            self.assertTrue(is_table_path(path))


if __name__ == "__main__":
    unittest.main()
