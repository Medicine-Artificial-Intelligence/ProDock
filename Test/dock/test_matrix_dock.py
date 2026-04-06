from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from typing import Any

from prodock.dock import BatchDock

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MULTI = REPO_ROOT / "Data" / "testcase" / "Multi"

# This real integration test runs only when explicitly enabled locally.
# Example:
#   RUN_REAL_DOCKING_TESTS=1 python -m unittest -v
RUN_REAL_DOCKING_TESTS = os.environ.get("RUN_REAL_DOCKING_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


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

    Parameters
    ----------
    obj
        JSON-like object composed of nested dictionaries, lists, and scalar
        values.
    target_keys
        Keys to match while traversing the object.

    Returns
    -------
    list[str]
        All string values associated with any key in ``target_keys``.
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

    Parameters
    ----------
    campaign_json
        Path to the campaign configuration JSON file.

    Returns
    -------
    set[Path]
        Set of receptor-level ``results`` directories inferred from the
        campaign payload.
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

    Parameters
    ----------
    path
        Directory to remove.
    allowed_root
        Root directory under which deletion is permitted.

    Raises
    ------
    RuntimeError
        If ``path`` is outside ``allowed_root``.
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

    Parameters
    ----------
    result_dirs
        Result directories to remove.
    allowed_root
        Root directory under which deletion is permitted.
    """
    for path in sorted(result_dirs):
        _safe_rmtree(path, allowed_root=allowed_root)


@unittest.skipUnless(
    RUN_REAL_DOCKING_TESTS,
    "Skipping real docking test. Set RUN_REAL_DOCKING_TESTS=1 to run locally.",
)
class TestBatchDockReal(unittest.TestCase):
    """
    Real integration test for :class:`prodock.dock.BatchDock`.

    This test uses the repository campaign fixture directly, but cleans the
    generated receptor ``results`` folders before and after execution so the
    fixture tree remains reusable.

    Notes
    -----
    - This test is intended for local execution only.
    - It is skipped by default unless ``RUN_REAL_DOCKING_TESTS=1`` is set.
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize shared campaign paths for the test class."""
        cls.campaign = SOURCE_MULTI / "campaign.json"
        cls.allowed_root = SOURCE_MULTI
        cls.result_dirs: set[Path] = set()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean any generated result directories after all tests finish."""
        if cls.result_dirs:
            _cleanup_result_dirs(cls.result_dirs, allowed_root=cls.allowed_root)

    def setUp(self) -> None:
        """Validate fixture availability and remove stale outputs before each test."""
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
        """Remove generated outputs after each test."""
        if self.result_dirs:
            _cleanup_result_dirs(self.result_dirs, allowed_root=self.allowed_root)

    def _assert_result_paths_inside_allowed_root(self, results: list[Any]) -> None:
        """
        Assert that all produced output and log paths stay inside the fixture root.

        Parameters
        ----------
        results
            Result objects returned by :meth:`BatchDock.run_from_config`.
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
        Run a real docking campaign from the repository fixture.

        This test verifies that:

        - at least one docking result is produced
        - at least one job succeeds
        - successful jobs create non-empty output and log files
        - produced files remain inside the allowed fixture tree
        - expected receptor-level ``results`` directories exist after execution
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
