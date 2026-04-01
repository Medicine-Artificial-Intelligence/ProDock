import unittest
from pathlib import Path

from prodock.dock import BatchDock


class TestBatchDockReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qvina_exe = True
        cls.campaign = Path("Data/testcase/Multi/campaign.json")

    def setUp(self) -> None:
        if self.qvina_exe is None:
            self.skipTest("qvina executable not found in PATH")
        if not self.campaign.exists():
            self.skipTest(f"Missing campaign file: {self.campaign}")

    def test_run_from_real_campaign(self) -> None:
        """
        Real integration test using the actual qvina backend and campaign file.

        Expected usage pattern:

            from prodock.dock import BatchDock

            runner = BatchDock(n_jobs=4, progress=True)
            results = runner.run_from_config("Data/testcase/Multi/campaign.json")
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
                    len(out_text.strip()) > 0, f"Empty output file: {out_path}"
                )
                self.assertTrue(
                    len(log_text.strip()) > 0, f"Empty log file: {log_path}"
                )

        self.assertGreater(success_count, 0, "No successful docking jobs were produced")


if __name__ == "__main__":
    unittest.main()
