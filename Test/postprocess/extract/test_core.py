from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prodock.postprocess.extract.core import (
    Extractor,
    _collect_log_files,
    _crawl_auto,
    _crawl_engine_tree,
    _crawl_flat_dir,
    _crawl_single_file,
    _finalize_parts,
    _normalize_table_df,
    _parse_table_file,
    _read_csv_flexible,
    _require_engine,
    _rows_to_frame,
    _to_float_or_none,
    _to_int_or_none,
    crawl_scores,
    extract_engine_folders,
    extract_log_file,
    extract_logs_dir,
    extract_scores,
    list_engines,
)

BASE = Path("Data/testcase/post")
QVINA_DIR = Path("Data/testcase/post/1M17/results/logs/qvina")
QVINA_LOG = Path("Data/testcase/post/1M17/results/logs/qvina/erlotinib.log")


class TestScalarHelpers(unittest.TestCase):
    def test_to_float_or_none_valid_string(self) -> None:
        self.assertEqual(_to_float_or_none("-7.5"), -7.5)

    def test_to_float_or_none_valid_int(self) -> None:
        self.assertEqual(_to_float_or_none(3), 3.0)

    def test_to_float_or_none_invalid(self) -> None:
        self.assertIsNone(_to_float_or_none("bad"))

    def test_to_float_or_none_none(self) -> None:
        self.assertIsNone(_to_float_or_none(None))

    def test_to_int_or_none_valid_string(self) -> None:
        self.assertEqual(_to_int_or_none("2"), 2)

    def test_to_int_or_none_valid_float_string(self) -> None:
        self.assertEqual(_to_int_or_none("2.0"), 2)

    def test_to_int_or_none_invalid(self) -> None:
        self.assertIsNone(_to_int_or_none("bad"))

    def test_to_int_or_none_none(self) -> None:
        self.assertIsNone(_to_int_or_none(None))


class TestCsvHelpers(unittest.TestCase):
    def test_read_csv_flexible_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.csv"
            path.write_text("ligand_id,score\nligA,-7.5\n", encoding="utf-8")

            df = _read_csv_flexible(path)

            self.assertIsNotNone(df)
            assert df is not None
            self.assertEqual(df.shape, (1, 2))
            self.assertEqual(df.loc[0, "ligand_id"], "ligA")
            self.assertEqual(df.loc[0, "score"], -7.5)

    def test_read_csv_flexible_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.tsv"
            path.write_text("ligand_id\tscore\nligA\t-7.5\n", encoding="utf-8")

            df = _read_csv_flexible(path)

            self.assertIsNotNone(df)
            assert df is not None
            self.assertEqual(df.shape, (1, 2))
            self.assertEqual(df.loc[0, "ligand_id"], "ligA")

    def test_read_csv_flexible_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.tab"
            path.write_text("ligand_id\tscore\nligA\t-7.5\n", encoding="utf-8")

            df = _read_csv_flexible(path)

            self.assertIsNotNone(df)
            assert df is not None
            self.assertEqual(df.shape, (1, 2))

    def test_normalize_table_df_aliases(self) -> None:
        raw = pd.DataFrame(
            {
                "Ligand": ["lig1"],
                "Affinity": [-7.5],
                "Rank": [1],
                "Engine": ["vina"],
            }
        )

        norm = _normalize_table_df(raw)

        self.assertIn("ligand_id", norm.columns)
        self.assertIn("score", norm.columns)
        self.assertIn("rank", norm.columns)
        self.assertIn("engine", norm.columns)
        self.assertEqual(norm.loc[0, "ligand_id"], "lig1")

    def test_parse_table_file_adds_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.csv"
            path.write_text(
                "ligand,affinity,engine\nligA,-7.5,vina\n", encoding="utf-8"
            )

            df = _parse_table_file(path)

            self.assertIsNotNone(df)
            assert df is not None
            self.assertIn("source_file", df.columns)
            self.assertEqual(df.loc[0, "source_file"], str(path))
            self.assertEqual(df.loc[0, "ligand_id"], "ligA")
            self.assertEqual(df.loc[0, "score"], -7.5)


class TestFrameHelpers(unittest.TestCase):
    def test_finalize_parts_returns_none_for_empty(self) -> None:
        self.assertIsNone(_finalize_parts([]))

    def test_finalize_parts_enforces_core_columns(self) -> None:
        part1 = pd.DataFrame({"ligand_id": ["lig1"], "score": ["-7.5"]})
        part2 = pd.DataFrame({"ligand_id": ["lig2"], "rank": ["2"]})

        df = _finalize_parts([part1, part2])

        self.assertIsNotNone(df)
        assert df is not None
        for col in ["ligand_id", "score", "rank", "engine", "source_file"]:
            self.assertIn(col, df.columns)

        self.assertEqual(df.loc[0, "score"], -7.5)
        self.assertEqual(df.loc[1, "rank"], 2)
        self.assertEqual(df.loc[0, "engine"], "")

    def test_rows_to_frame_returns_none_for_empty_rows(self) -> None:
        df = _rows_to_frame(Path("lig1.log"), [], "vina")
        self.assertIsNone(df)

    def test_rows_to_frame_maps_fields(self) -> None:
        rows = [
            {
                "mode": 1,
                "affinity_kcal_mol": -8.2,
                "rmsd_lb": 0.0,
                "rmsd_ub": 0.0,
            },
            {
                "mode": 2,
                "affinity_kcal_mol": -7.5,
                "cnn_pose": 0.71,
                "cnn_affinity": 7.45,
            },
        ]

        df = _rows_to_frame(Path("erlotinib.log"), rows, "qvina")

        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(df.loc[0, "ligand_id"], "erlotinib")
        self.assertEqual(df.loc[0, "score"], -8.2)
        self.assertEqual(df.loc[0, "rank"], 1)
        self.assertEqual(df.loc[0, "engine"], "qvina")
        self.assertEqual(df.loc[0, "source_file"], "erlotinib.log")
        self.assertIn("rmsd_lb", df.columns)
        self.assertIn("cnn_pose", df.columns)


class TestCollectionHelpers(unittest.TestCase):
    def test_collect_log_files_missing_root(self) -> None:
        files = _collect_log_files(Path("does_not_exist"), recursive=True)
        self.assertEqual(files, [])

    def test_collect_log_files_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a.log"
            path.write_text("x", encoding="utf-8")

            files = _collect_log_files(path)

            self.assertEqual(files, [path])

    def test_collect_log_files_recursive_and_non_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            top = root / "a.log"
            subdir = root / "sub"
            subdir.mkdir()
            nested = subdir / "b.log"

            top.write_text("x", encoding="utf-8")
            nested.write_text("x", encoding="utf-8")

            files_nonrec = _collect_log_files(root, recursive=False)
            files_rec = _collect_log_files(root, recursive=True)

            self.assertEqual(files_nonrec, [top])
            self.assertEqual(files_rec, [top, nested])

    def test_require_engine_accepts_valid(self) -> None:
        self.assertEqual(_require_engine("qvina", "single_file"), "qvina")

    def test_require_engine_rejects_missing(self) -> None:
        with self.assertRaises(ValueError):
            _require_engine(None, "single_file")


class TestLayoutCrawlersSynthetic(unittest.TestCase):
    def test_crawl_single_file_requires_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a.csv"
            path.write_text("x,y\n1,2\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                _crawl_single_file([path], engine_hint="vina")

    def test_crawl_single_file_real_log(self) -> None:
        if not QVINA_LOG.exists():
            self.skipTest(f"Missing integration log: {QVINA_LOG}")

        df = _crawl_single_file([QVINA_LOG], engine_hint="qvina")

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)
        self.assertTrue((df["engine"].str.lower() == "qvina").any())
        self.assertTrue((df["ligand_id"] == "erlotinib").any())

    def test_crawl_flat_dir_real(self) -> None:
        if not QVINA_DIR.exists():
            self.skipTest(f"Missing integration dir: {QVINA_DIR}")

        df = _crawl_flat_dir([QVINA_DIR], engine_hint="qvina", recursive=True)

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)
        self.assertTrue((df["engine"].str.lower() == "qvina").any())

    def test_crawl_engine_tree_rejects_file_root(self) -> None:
        if not QVINA_LOG.exists():
            self.skipTest(f"Missing integration log: {QVINA_LOG}")

        with self.assertRaises(ValueError):
            _crawl_engine_tree([QVINA_LOG], recursive=True)

    def test_crawl_auto_table_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.csv"
            path.write_text(
                "ligand,affinity,engine\nligA,-7.5,vina\n", encoding="utf-8"
            )

            df = _crawl_auto(
                [path],
                include_logs=("**/*.log", "**/*.txt"),
                include_tables=("**/*.csv", "**/*.tsv", "**/*.tab"),
                engine_hint=None,
            )

            self.assertIsNotNone(df)
            assert df is not None
            self.assertFalse(df.empty)
            self.assertEqual(df.loc[0, "ligand_id"], "ligA")
            self.assertEqual(df.loc[0, "score"], -7.5)

    def test_crawl_auto_real_log_file(self) -> None:
        if not QVINA_LOG.exists():
            self.skipTest(f"Missing integration log: {QVINA_LOG}")

        df = _crawl_auto(
            [QVINA_LOG],
            include_logs=("**/*.log", "**/*.txt"),
            include_tables=("**/*.csv", "**/*.tsv", "**/*.tab"),
            engine_hint="qvina",
        )

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)
        self.assertTrue((df["engine"].str.lower() == "qvina").any())


class TestCrawlScoresPublicAPI(unittest.TestCase):
    def test_crawl_scores_auto_real_tree(self) -> None:
        if not BASE.exists():
            self.skipTest(f"Missing integration base: {BASE}")

        df = crawl_scores([BASE], engine_hint="qvina", layout="auto")

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)
        for col in ["ligand_id", "score", "rank", "engine", "source_file"]:
            self.assertIn(col, df.columns)

    def test_crawl_scores_single_file_real(self) -> None:
        if not QVINA_LOG.exists():
            self.skipTest(f"Missing integration log: {QVINA_LOG}")

        df = crawl_scores([QVINA_LOG], engine_hint="qvina", layout="single_file")

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)
        self.assertTrue((df["engine"].str.lower() == "qvina").any())

    def test_crawl_scores_flat_dir_real(self) -> None:
        if not QVINA_DIR.exists():
            self.skipTest(f"Missing integration dir: {QVINA_DIR}")

        df = crawl_scores([QVINA_DIR], engine_hint="qvina", layout="flat_dir")

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)

    def test_crawl_scores_engine_tree_real(self) -> None:
        logs_root = Path("Data/testcase/post/1M17/results/logs")
        if not logs_root.exists():
            self.skipTest(f"Missing engine-tree root: {logs_root}")

        df = crawl_scores([logs_root], layout="engine_tree")

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)
        self.assertIn("engine", df.columns)


class TestExtractorWithStubCrawler(unittest.TestCase):
    def _stub_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ligand_id": "ligA", "score": -7.5, "rank": 1, "engine": "vina"},
                {"ligand_id": "ligB", "score": -7.2, "rank": 1, "engine": "qvina"},
                {"ligand_id": "ligC", "score": -8.1, "rank": 1, "engine": "gnina"},
                {"ligand_id": "ligD", "score": -7.9, "rank": 1, "engine": "vina-gpu"},
            ]
        )

    def test_call_crawl_passes_defaults(self) -> None:
        seen = {}

        def fake_crawl(
            roots,
            include_logs=None,
            include_tables=None,
            engine_hint=None,
            layout="auto",
            recursive=True,
        ):
            seen["roots"] = roots
            seen["include_logs"] = include_logs
            seen["include_tables"] = include_tables
            seen["engine_hint"] = engine_hint
            seen["layout"] = layout
            seen["recursive"] = recursive
            return self._stub_df()

        extractor = Extractor(crawl_func=fake_crawl)
        df = extractor._call_crawl(
            ["x"],
            engine_hint="vina",
            layout="flat_dir",
            recursive=False,
        )

        self.assertIsNotNone(df)
        self.assertEqual(seen["roots"], ["x"])
        self.assertEqual(seen["engine_hint"], "vina")
        self.assertEqual(seen["layout"], "flat_dir")
        self.assertFalse(seen["recursive"])
        self.assertEqual(seen["include_logs"], ("**/*.log", "**/*.txt"))
        self.assertEqual(seen["include_tables"], ("**/*.csv", "**/*.tsv", "**/*.tab"))

    def test_extract_scores_returns_all_when_engines_none(self) -> None:
        extractor = Extractor(crawl_func=lambda *args, **kwargs: self._stub_df())

        df = extractor.extract_scores(["x"], engines=None)

        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(len(df), 4)

    def test_extract_scores_exact_match(self) -> None:
        extractor = Extractor(
            match_mode="exact",
            crawl_func=lambda *args, **kwargs: self._stub_df(),
        )

        df = extractor.extract_scores(["x"], engines=["vina"])

        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(set(df["engine"].tolist()), {"vina"})

    def test_extract_scores_substring_match(self) -> None:
        extractor = Extractor(
            match_mode="substring",
            crawl_func=lambda *args, **kwargs: self._stub_df(),
        )

        df = extractor.extract_scores(["x"], engines=["vina"])

        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(set(df["engine"].tolist()), {"vina", "qvina", "vina-gpu"})

    def test_extract_scores_regex_match(self) -> None:
        extractor = Extractor(
            match_mode="regex",
            crawl_func=lambda *args, **kwargs: self._stub_df(),
        )

        df = extractor.extract_scores(["x"], engines=[r"^gni"])

        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(set(df["engine"].tolist()), {"gnina"})

    def test_extract_scores_engine_map(self) -> None:
        extractor = Extractor(
            match_mode="exact",
            crawl_func=lambda *args, **kwargs: self._stub_df(),
            engine_map={"vina-family": ["vina", "qvina", "vina-gpu"]},
        )

        df = extractor.extract_scores(["x"], engines=["vina-family"])

        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(set(df["engine"].tolist()), {"vina", "qvina", "vina-gpu"})

    def test_extract_scores_empty_requested_returns_empty(self) -> None:
        extractor = Extractor(
            match_mode="exact",
            crawl_func=lambda *args, **kwargs: self._stub_df(),
        )

        df = extractor.extract_scores(["x"], engines=["", "   "])

        self.assertIsNotNone(df)
        assert df is not None
        self.assertTrue(df.empty)

    def test_list_engines_returns_unique_lowercase(self) -> None:
        df_in = pd.DataFrame(
            [
                {"engine": "VINA"},
                {"engine": "qvina"},
                {"engine": "QVINA"},
                {"engine": None},
            ]
        )
        extractor = Extractor(crawl_func=lambda *args, **kwargs: df_in)

        engines = extractor.list_engines(["x"])

        self.assertEqual(engines, {"vina", "qvina"})

    def test_list_engines_empty_when_none(self) -> None:
        extractor = Extractor(crawl_func=lambda *args, **kwargs: None)
        self.assertEqual(extractor.list_engines(["x"]), set())

    def test_extract_log_file_calls_single_file_layout(self) -> None:
        seen = {}

        def fake_crawl(
            roots,
            include_logs=None,
            include_tables=None,
            engine_hint=None,
            layout="auto",
            recursive=True,
        ):
            seen["roots"] = roots
            seen["engine_hint"] = engine_hint
            seen["layout"] = layout
            return self._stub_df()

        extractor = Extractor(crawl_func=fake_crawl)
        df = extractor.extract_log_file("a.log", engine="vina")

        self.assertIsNotNone(df)
        self.assertEqual(seen["roots"], ["a.log"])
        self.assertEqual(seen["engine_hint"], "vina")
        self.assertEqual(seen["layout"], "single_file")

    def test_extract_logs_dir_calls_flat_dir_layout(self) -> None:
        seen = {}

        def fake_crawl(
            roots,
            include_logs=None,
            include_tables=None,
            engine_hint=None,
            layout="auto",
            recursive=True,
        ):
            seen["roots"] = roots
            seen["engine_hint"] = engine_hint
            seen["layout"] = layout
            seen["recursive"] = recursive
            return self._stub_df()

        extractor = Extractor(crawl_func=fake_crawl)
        df = extractor.extract_logs_dir("logs", engine="qvina", recursive=False)

        self.assertIsNotNone(df)
        self.assertEqual(seen["roots"], ["logs"])
        self.assertEqual(seen["engine_hint"], "qvina")
        self.assertEqual(seen["layout"], "flat_dir")
        self.assertFalse(seen["recursive"])

    def test_extract_engine_folders_calls_engine_tree_layout(self) -> None:
        seen = {}

        def fake_crawl(
            roots,
            include_logs=None,
            include_tables=None,
            engine_hint=None,
            layout="auto",
            recursive=True,
        ):
            seen["roots"] = roots
            seen["layout"] = layout
            seen["recursive"] = recursive
            return self._stub_df()

        extractor = Extractor(crawl_func=fake_crawl)
        df = extractor.extract_engine_folders("logs", recursive=False)

        self.assertIsNotNone(df)
        self.assertEqual(seen["roots"], ["logs"])
        self.assertEqual(seen["layout"], "engine_tree")
        self.assertFalse(seen["recursive"])


class TestDefaultWrappers(unittest.TestCase):
    def test_extract_scores_wrapper_real(self) -> None:
        if not QVINA_LOG.exists():
            self.skipTest(f"Missing integration log: {QVINA_LOG}")

        df = extract_scores([QVINA_LOG], engine_hint="qvina", layout="single_file")

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)

    def test_list_engines_wrapper_real(self) -> None:
        if not QVINA_LOG.exists():
            self.skipTest(f"Missing integration log: {QVINA_LOG}")

        engines = list_engines([QVINA_LOG], engine_hint="qvina", layout="single_file")

        self.assertIn("qvina", {e.lower() for e in engines})

    def test_extract_log_file_wrapper_real(self) -> None:
        if not QVINA_LOG.exists():
            self.skipTest(f"Missing integration log: {QVINA_LOG}")

        df = extract_log_file(QVINA_LOG, engine="qvina")

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)

    def test_extract_logs_dir_wrapper_real(self) -> None:
        if not QVINA_DIR.exists():
            self.skipTest(f"Missing integration dir: {QVINA_DIR}")

        df = extract_logs_dir(QVINA_DIR, engine="qvina", recursive=True)

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)

    def test_extract_engine_folders_wrapper_real(self) -> None:
        logs_root = Path("Data/testcase/post/1M17/results/logs")
        if not logs_root.exists():
            self.skipTest(f"Missing engine-tree root: {logs_root}")

        df = extract_engine_folders(logs_root, recursive=True)

        self.assertIsNotNone(df)
        assert df is not None
        self.assertFalse(df.empty)


if __name__ == "__main__":
    unittest.main()
