from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from typing import Any

from prodock.dock import BatchDock

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MULTI = REPO_ROOT / "Data" / "testcase" / "Multi"


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return whether ``path`` is inside ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_key_values(obj: Any, target_keys: set[str]) -> list[str]:
    """
    Recursively collect string values for matching keys from a JSON-like object.
    """
    values: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in target_keys and isinstance(value, str):
                values.append(value)
            values.extend(_iter_key_values(value, target_keys))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_iter_key_values(item, target_keys))

    return values


def _collect_result_dirs_from_campaign(campaign_json: Path) -> set[Path]:
    """
    Collect receptor-level results directories from a campaign JSON.

    This function recursively scans the whole payload for ``out_dir`` and
    ``log_dir`` keys, then converts entries like::

        /.../4WKQ/results/docked
        /.../4WKQ/results/logs

    into the receptor-level result directory::

        /.../4WKQ/results
    """
    payload = _load_json(campaign_json)
    dir_values = _iter_key_values(payload, {"out_dir", "log_dir"})

    result_dirs: set[Path] = set()
    for value in dir_values:
        p = Path(value).resolve()
        result_dirs.add(p.parent)

    return result_dirs


def _safe_rmtree(path: Path, *, allowed_root: Path) -> None:
    """
    Remove a directory tree only if it is inside ``allowed_root``.
    """
    resolved = path.resolve()
    allowed = allowed_root.resolve()

    if not _is_relative_to(resolved, allowed):
        raise RuntimeError(
            f"Refusing to delete directory outside allowed root: {resolved}"
        )

    if resolved.exists():
        shutil.rmtree(resolved)


def _cleanup_result_dirs(result_dirs: set[Path], *, allowed_root: Path) -> None:
    """
    Remove all result directories collected from a campaign.
    """
    for path in sorted(result_dirs):
        _safe_rmtree(path, allowed_root=allowed_root)


class TestBatchDockReal(unittest.TestCase):
    """
    Real integration test for :class:`prodock.dock.BatchDock`.

    This test uses the repository campaign fixture directly, but cleans the
    generated receptor ``results`` folders before and after execution so the
    fixture tree remains reusable.
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.campaign = SOURCE_MULTI / "campaign.json"
        cls.allowed_root = SOURCE_MULTI
        cls.result_dirs: set[Path] = set()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.result_dirs:
            _cleanup_result_dirs(cls.result_dirs, allowed_root=cls.allowed_root)

    def setUp(self) -> None:
        self.assertTrue(
            self.campaign.exists(),
            f"Missing campaign file: {self.campaign}",
        )

        self.result_dirs = _collect_result_dirs_from_campaign(self.campaign)
        self.__class__.result_dirs = self.result_dirs

        self.assertTrue(
            len(self.result_dirs) > 0,
            f"No result directories could be inferred from campaign.json: {self.campaign}",
        )

        _cleanup_result_dirs(self.result_dirs, allowed_root=self.allowed_root)

    def tearDown(self) -> None:
        if self.result_dirs:
            _cleanup_result_dirs(self.result_dirs, allowed_root=self.allowed_root)

    def _assert_result_paths_inside_allowed_root(self, results: list[Any]) -> None:
        """
        Assert that all produced output and log paths stay inside the fixture root.
        """
        for res in results:
            if not res.success:
                continue

            self.assertIsNotNone(res.out_path)
            self.assertIsNotNone(res.log_path)

            out_path = Path(res.out_path).resolve()
            log_path = Path(res.log_path).resolve()

            self.assertTrue(
                _is_relative_to(out_path, self.allowed_root),
                f"Output path is outside allowed root: {out_path}",
            )
            self.assertTrue(
                _is_relative_to(log_path, self.allowed_root),
                f"Log path is outside allowed root: {log_path}",
            )

    def test_run_from_real_campaign(self) -> None:
        """
        Real integration test using the actual campaign file and docking backend.
        """
        runner = BatchDock(n_jobs=4, progress=True)
        results = runner.run_from_config(str(self.campaign))

        self.assertTrue(len(results) > 0, "No docking results were produced")

        success_count = 0
        for res in results:
            if res.success:
                success_count += 1

                self.assertIsNotNone(res.out_path)
                self.assertIsNotNone(res.log_path)

                out_path = Path(res.out_path)
                log_path = Path(res.log_path)

                self.assertTrue(out_path.exists(), f"Missing output file: {out_path}")
                self.assertTrue(log_path.exists(), f"Missing log file: {log_path}")

                out_text = out_path.read_text(errors="ignore")
                log_text = log_path.read_text(errors="ignore")

                self.assertTrue(
                    len(out_text.strip()) > 0,
                    f"Empty output file: {out_path}",
                )
                self.assertTrue(
                    len(log_text.strip()) > 0,
                    f"Empty log file: {log_path}",
                )

        self.assertGreater(
            success_count,
            0,
            "No successful docking jobs were produced",
        )

        self._assert_result_paths_inside_allowed_root(results)

        for result_dir in self.result_dirs:
            self.assertTrue(
                result_dir.exists(),
                f"Expected result directory to exist after docking: {result_dir}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
