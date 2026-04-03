from __future__ import annotations

import unittest

from prodock.postprocess.extract.engines import (
    GNINA_ROW_RE,
    GNINA_TABLE_HEADER,
    VINA_ROW_RE,
    VINA_TABLE_HEADER,
)
from prodock.postprocess.extract.reader import (
    _iter_lines,
    _parse_gnina,
    _parse_vina_family,
    parse_log_text,
)


def _pick_matching_line(pattern, candidates: list[str]) -> str:
    """
    Return the first candidate line that matches the provided regex.

    :param pattern:
        Compiled regex pattern.
    :param candidates:
        Candidate lines to test.
    :returns:
        First matching candidate line.
    :raises AssertionError:
        If no candidate matches.
    """
    for line in candidates:
        if pattern.search(line) or pattern.match(line):
            return line
    raise AssertionError(
        f"No candidate matched pattern {getattr(pattern, 'pattern', pattern)!r}"
    )


def _build_vina_text() -> str:
    """
    Build a small Vina-like log text compatible with the actual regex objects.

    :returns:
        Log text containing a valid Vina-family header and two valid rows.
    :rtype: str
    """
    header_candidates = [
        "mode | affinity | dist from best mode",
        "mode |   affinity | dist from best mode",
        "mode | affinity | rmsd l.b. | rmsd u.b.",
        "-----+------------+----------+----------",
    ]
    row_candidates_1 = [
        "1 -7.5 0.000 0.000",
        "  1       -7.5      0.000      0.000",
        "1    -7.5    0.000    0.000",
    ]
    row_candidates_2 = [
        "2 -7.1 1.200 2.400",
        "  2       -7.1      1.200      2.400",
        "2    -7.1    1.200    2.400",
    ]

    header = _pick_matching_line(VINA_TABLE_HEADER, header_candidates)
    row1 = _pick_matching_line(VINA_ROW_RE, row_candidates_1)
    row2 = _pick_matching_line(VINA_ROW_RE, row_candidates_2)

    return "\n".join(
        [
            "Random preamble",
            header,
            row1,
            row2,
            "Done",
        ]
    )


def _build_gnina_text() -> str:
    """
    Build a small GNINA log text compatible with the actual regex objects.

    :returns:
        Log text containing a valid GNINA header and two valid rows.
    :rtype: str
    """
    header_candidates = [
        "mode | affinity | cnn_pose | cnn_affinity",
        "mode | affinity   | cnn_pose   | cnn_affinity",
        "mode | affinity | CNN pose score | CNN affinity",
        "mode | affinity | intramol | cnn_pose | cnn_affinity",
    ]
    row_candidates_1 = [
        "1 -8.2 0.71 7.45",
        "  1      -8.2        0.71          7.45",
        "1    -8.2    0.71    7.45",
    ]
    row_candidates_2 = [
        "2 -7.8 0.66 7.10",
        "  2      -7.8        0.66          7.10",
        "2    -7.8    0.66    7.10",
    ]

    header = _pick_matching_line(GNINA_TABLE_HEADER, header_candidates)
    row1 = _pick_matching_line(GNINA_ROW_RE, row_candidates_1)
    row2 = _pick_matching_line(GNINA_ROW_RE, row_candidates_2)

    return "\n".join(
        [
            "GNINA output",
            header,
            row1,
            row2,
        ]
    )


def _build_nonmatching_text() -> str:
    """
    Build a text block without supported docking tables.

    :returns:
        Unsupported log text.
    :rtype: str
    """
    return "\n".join(
        [
            "This file does not contain any supported docking table.",
            "Just plain text.",
        ]
    )


class TestIterLines(unittest.TestCase):
    def test_iter_lines_basic(self) -> None:
        self.assertEqual(list(_iter_lines("a\nb\nc\n")), ["a", "b", "c"])

    def test_iter_lines_empty_text(self) -> None:
        self.assertEqual(list(_iter_lines("")), [])

    def test_iter_lines_preserves_spaces(self) -> None:
        self.assertEqual(list(_iter_lines("  a  \n b\t\n")), ["  a  ", " b\t"])


class TestParseVinaFamily(unittest.TestCase):
    def test_parse_vina_family_returns_rows(self) -> None:
        text = _build_vina_text()
        rows = _parse_vina_family(text)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["mode"], 1)
        self.assertAlmostEqual(rows[0]["affinity_kcal_mol"], -7.5)
        self.assertAlmostEqual(rows[0]["rmsd_lb"], 0.0)
        self.assertAlmostEqual(rows[0]["rmsd_ub"], 0.0)

        self.assertEqual(rows[1]["mode"], 2)
        self.assertAlmostEqual(rows[1]["affinity_kcal_mol"], -7.1)
        self.assertAlmostEqual(rows[1]["rmsd_lb"], 1.2)
        self.assertAlmostEqual(rows[1]["rmsd_ub"], 2.4)

    def test_parse_vina_family_returns_empty_when_header_missing(self) -> None:
        rows = _parse_vina_family(_build_nonmatching_text())
        self.assertEqual(rows, [])

    def test_parse_vina_family_skips_non_matching_lines(self) -> None:
        text = _build_vina_text() + "\nthis is not a valid score row\n"
        rows = _parse_vina_family(text)
        self.assertEqual(len(rows), 2)


class TestParseGnina(unittest.TestCase):
    def test_parse_gnina_returns_rows(self) -> None:
        text = _build_gnina_text()
        rows = _parse_gnina(text)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["mode"], 1)
        self.assertAlmostEqual(rows[0]["affinity_kcal_mol"], -8.2)
        self.assertAlmostEqual(rows[0]["cnn_pose"], 0.71)
        self.assertAlmostEqual(rows[0]["cnn_affinity"], 7.45)

        self.assertEqual(rows[1]["mode"], 2)
        self.assertAlmostEqual(rows[1]["affinity_kcal_mol"], -7.8)
        self.assertAlmostEqual(rows[1]["cnn_pose"], 0.66)
        self.assertAlmostEqual(rows[1]["cnn_affinity"], 7.10)

    def test_parse_gnina_returns_empty_when_header_missing(self) -> None:
        rows = _parse_gnina(_build_nonmatching_text())
        self.assertEqual(rows, [])

    def test_parse_gnina_skips_non_matching_lines(self) -> None:
        text = _build_gnina_text() + "\ninvalid row\n"
        rows = _parse_gnina(text)
        self.assertEqual(len(rows), 2)


class TestParseLogText(unittest.TestCase):
    def test_parse_log_text_vina_engine(self) -> None:
        rows = parse_log_text(_build_vina_text(), engine="vina")
        self.assertEqual(len(rows), 2)
        self.assertIn("rmsd_lb", rows[0])
        self.assertIn("rmsd_ub", rows[0])

    def test_parse_log_text_gnina_engine(self) -> None:
        rows = parse_log_text(_build_gnina_text(), engine="gnina")
        self.assertEqual(len(rows), 2)
        self.assertIn("cnn_pose", rows[0])
        self.assertIn("cnn_affinity", rows[0])

    def test_parse_log_text_returns_empty_for_unknown_content(self) -> None:
        rows = parse_log_text(_build_nonmatching_text(), engine="vina")
        self.assertEqual(rows, [])

    def test_parse_log_text_custom_vina_regex(self) -> None:
        text = "ROW 1 -7.5 0.0 0.0\nROW 2 -7.1 1.2 2.4\n"
        regex = {
            "vina_row": r"^ROW\s+(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)$"
        }
        rows = parse_log_text(text, engine="vina", regex=regex)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["mode"], 1)
        self.assertAlmostEqual(rows[1]["rmsd_ub"], 2.4)

    def test_parse_log_text_custom_gnina_regex(self) -> None:
        text = "POSE 1 -8.2 0.71 7.45\nPOSE 2 -7.8 0.66 7.10\n"
        regex = {
            "gnina_row": r"^POSE\s+(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)$"
        }
        rows = parse_log_text(text, engine="gnina", regex=regex)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["mode"], 1)
        self.assertAlmostEqual(rows[0]["cnn_pose"], 0.71)
        self.assertAlmostEqual(rows[1]["cnn_affinity"], 7.10)

    def test_parse_log_text_custom_regex_ignored_when_no_match(self) -> None:
        regex = {
            "vina_row": r"^ROW\s+(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)$"
        }
        rows = parse_log_text(_build_vina_text(), engine="vina", regex=regex)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["mode"], 1)

    def test_parse_log_text_engine_name_is_canonicalized(self) -> None:
        rows = parse_log_text(_build_gnina_text(), engine=" GNINA ")
        self.assertEqual(len(rows), 2)
        self.assertIn("cnn_pose", rows[0])

    def test_parse_log_text_without_engine_uses_default_parser_path(self) -> None:
        rows = parse_log_text(_build_vina_text())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["mode"], 1)


if __name__ == "__main__":
    unittest.main()
