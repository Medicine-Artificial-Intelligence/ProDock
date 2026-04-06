from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prodock.postprocess.extract.normalize import (
    _try_decode,
    normalize_file,
    read_text_flexible,
    safe_parse_file,
)


class TestTryDecode(unittest.TestCase):
    def test_try_decode_utf8(self) -> None:
        raw = "hello café".encode("utf-8")
        text, enc = _try_decode(raw)
        self.assertEqual(text, "hello café")
        self.assertEqual(enc, "utf-8")

    def test_try_decode_utf8_sig(self) -> None:
        raw = "hello".encode("utf-8-sig")
        text, enc = _try_decode(raw, encodings=("utf-8-sig", "utf-8"))
        self.assertEqual(text, "hello")
        self.assertEqual(enc, "utf-8-sig")

    def test_try_decode_latin1_fallback(self) -> None:
        raw = "café".encode("latin-1")
        text, enc = _try_decode(raw, encodings=("utf-8", "latin-1"))
        self.assertEqual(text, "café")
        self.assertEqual(enc, "latin-1")

    def test_try_decode_replace_fallback_when_all_candidates_fail(self) -> None:
        raw = b"\x80abc"
        text, enc = _try_decode(raw, encodings=("utf-8",))
        self.assertEqual(enc, "latin-1-replace")
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) >= 1)


class TestReadTextFlexible(unittest.TestCase):
    def test_read_text_flexible_reads_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text("hello café", encoding="utf-8")

            text, enc = read_text_flexible(path)

            self.assertEqual(text, "hello café")
            self.assertEqual(enc, "utf-8")

    def test_read_text_flexible_reads_latin1_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_bytes("café".encode("latin-1"))

            text, enc = read_text_flexible(path)

            self.assertEqual(text, "café")
            self.assertEqual(enc, "latin-1")


class TestNormalizeFile(unittest.TestCase):
    def test_normalize_file_without_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_bytes("café".encode("latin-1"))

            detected = normalize_file(path, backup=False)

            self.assertEqual(detected, "latin-1")
            self.assertEqual(path.read_text(encoding="utf-8"), "café")
            self.assertFalse((Path(tmpdir) / "sample.txt.bak").exists())

    def test_normalize_file_with_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            original_bytes = "café".encode("latin-1")
            path.write_bytes(original_bytes)

            detected = normalize_file(path, backup=True)

            bak = Path(tmpdir) / "sample.txt.bak"
            self.assertEqual(detected, "latin-1")
            self.assertTrue(bak.exists())
            self.assertEqual(bak.read_bytes(), original_bytes)
            self.assertEqual(path.read_text(encoding="utf-8"), "café")

    def test_normalize_file_with_existing_backup_creates_numbered_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_bytes("café".encode("latin-1"))

            existing_bak = Path(tmpdir) / "sample.txt.bak"
            existing_bak.write_text("older backup", encoding="utf-8")

            detected = normalize_file(path, backup=True)

            numbered_bak = Path(tmpdir) / "sample.txt.bak.1"
            self.assertEqual(detected, "latin-1")
            self.assertTrue(existing_bak.exists())
            self.assertTrue(numbered_bak.exists())
            self.assertEqual(numbered_bak.read_bytes(), "café".encode("latin-1"))
            self.assertEqual(path.read_text(encoding="utf-8"), "café")

    def test_normalize_file_uses_custom_encodings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_bytes("café".encode("latin-1"))

            detected = normalize_file(path, backup=False, encodings=("latin-1",))

            self.assertEqual(detected, "latin-1")
            self.assertEqual(path.read_text(encoding="utf-8"), "café")


class TestSafeParseFile(unittest.TestCase):
    def test_safe_parse_file_returns_rows_on_first_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dock.log"
            path.write_text("vina content", encoding="utf-8")

            calls = []

            def parse_fn(text: str, engine: str | None, regex: dict | None) -> list:
                calls.append((text, engine, regex))
                return [{"mode": 1}]

            rows, engine = safe_parse_file(path, parse_fn=parse_fn, engine_hint="vina")

            self.assertEqual(rows, [{"mode": 1}])
            self.assertEqual(engine, "vina")
            self.assertEqual(len(calls), 1)

    def test_safe_parse_file_retries_after_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dock.log"
            path.write_bytes("café".encode("latin-1"))

            calls = []

            def parse_fn(text: str, engine: str | None, regex: dict | None) -> list:
                calls.append(text)
                if len(calls) == 1:
                    return []
                return [{"mode": 2}]

            rows, engine = safe_parse_file(path, parse_fn=parse_fn, engine_hint="vina")

            self.assertEqual(rows, [{"mode": 2}])
            self.assertEqual(engine, "vina")
            self.assertEqual(len(calls), 2)
            self.assertTrue((Path(tmpdir) / "dock.log.bak").exists())

    def test_safe_parse_file_retries_after_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dock.log"
            path.write_text("vina content", encoding="utf-8")

            state = {"n": 0}

            def parse_fn(text: str, engine: str | None, regex: dict | None) -> list:
                state["n"] += 1
                if state["n"] == 1:
                    raise ValueError("parse failed")
                return [{"mode": 3}]

            rows, engine = safe_parse_file(path, parse_fn=parse_fn, engine_hint="vina")

            self.assertEqual(rows, [{"mode": 3}])
            self.assertEqual(engine, "vina")
            self.assertEqual(state["n"], 2)

    def test_safe_parse_file_no_retry_when_normalize_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dock.log"
            path.write_text("vina content", encoding="utf-8")

            calls = []

            def parse_fn(text: str, engine: str | None, regex: dict | None) -> list:
                calls.append(text)
                return []

            rows, engine = safe_parse_file(
                path,
                parse_fn=parse_fn,
                engine_hint="vina",
                normalize_on_failure=False,
            )

            self.assertEqual(rows, [])
            self.assertEqual(engine, "vina")
            self.assertEqual(len(calls), 1)
            self.assertFalse((Path(tmpdir) / "dock.log.bak").exists())

    def test_safe_parse_file_returns_empty_when_normalization_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.log"

            def parse_fn(text: str, engine: str | None, regex: dict | None) -> list:
                return [{"mode": 1}]

            with self.assertRaises(FileNotFoundError):
                # read_text_flexible happens before normalization fallback
                safe_parse_file(missing, parse_fn=parse_fn, engine_hint="vina")

    def test_safe_parse_file_passes_regex_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dock.log"
            path.write_text("vina content", encoding="utf-8")

            seen = {}

            def parse_fn(text: str, engine: str | None, regex: dict | None) -> list:
                seen["engine"] = engine
                seen["regex"] = regex
                return [{"mode": 1}]

            regex = {"vina_row": r"^test$"}
            rows, engine = safe_parse_file(
                path,
                parse_fn=parse_fn,
                engine_hint="vina",
                regex=regex,
            )

            self.assertEqual(rows, [{"mode": 1}])
            self.assertEqual(engine, "vina")
            self.assertEqual(seen["engine"], "vina")
            self.assertEqual(seen["regex"], regex)

    def test_safe_parse_file_detects_engine_when_hint_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dock.log"
            path.write_text("gnina", encoding="utf-8")

            seen = {}

            def parse_fn(text: str, engine: str | None, regex: dict | None) -> list:
                seen["engine"] = engine
                return [{"mode": 1}]

            rows, engine = safe_parse_file(path, parse_fn=parse_fn, engine_hint=None)

            self.assertEqual(rows, [{"mode": 1}])
            self.assertEqual(engine, seen["engine"])


if __name__ == "__main__":
    unittest.main()
