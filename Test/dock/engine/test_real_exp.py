# test_single_real_smina_verbose.py
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import shutil as sh

from prodock.dock.engine import SingleDock
from prodock.dock.engine.registry import register
from prodock.dock.engine.smina import SminaEngine
from prodock.dock.engine.qvina import QVinaEngine
from prodock.dock.engine.qvina_w import QVinaWEngine


def find_repo_root_with_data(start: Optional[Path] = None) -> Optional[Path]:
    """Search upward for the repo root that contains Data/testcase/dock/receptor/5N2F.pdbqt."""
    if start is None:
        start = Path.cwd()
    for d in [start] + list(start.parents):
        if (d / "Data" / "testcase" / "dock" / "receptor" / "5N2F.pdbqt").exists():
            return d
    return None


class TestSingleDockSmina(unittest.TestCase):
    def setUp(self):
        self._orig_cwd = Path.cwd()
        # find repo root (by looking for the Data/testcase tree) and smina executable
        self.repo_root = find_repo_root_with_data(self._orig_cwd)
        self.smina_path = sh.which("smina")  # may be None

    def tearDown(self):
        try:
            os.chdir(self._orig_cwd)
        finally:
            pass

    def test_user_snippet_with_real_smina(self):

        tmpdir = Path(tempfile.mkdtemp(prefix="smina"))
        try:
            workdir = tmpdir / "work"
            workdir.mkdir(parents=True, exist_ok=True)

            # copy repo inputs into isolated workdir
            src_rec = (
                self.repo_root
                / "Data"
                / "testcase"
                / "dock"
                / "receptor"
                / "5N2F.pdbqt"
            )
            src_lig = (
                self.repo_root / "Data" / "testcase" / "dock" / "ligand" / "8HW.pdbqt"
            )
            (workdir / "Data" / "testcase" / "dock" / "receptor").mkdir(
                parents=True, exist_ok=True
            )
            (workdir / "Data" / "testcase" / "dock" / "ligand").mkdir(
                parents=True, exist_ok=True
            )
            shutil.copy2(
                src_rec,
                workdir / "Data" / "testcase" / "dock" / "receptor" / "5N2F.pdbqt",
            )
            shutil.copy2(
                src_lig, workdir / "Data" / "testcase" / "dock" / "ligand" / "8HW.pdbqt"
            )

            # change cwd so relative paths in the snippet work as-is
            os.chdir(workdir)

            # register a factory for "smina" that returns a SminaEngine configured to use the real binary
            def factory():
                e = SminaEngine()
                # only call set_executable if we actually have a resolved path
                if self.smina_path:
                    e.set_executable(self.smina_path)
                return e

            register("smina", factory)

            sd = (
                SingleDock("smina")
                .set_receptor("Data/testcase/dock/receptor/5N2F.pdbqt", validate=True)
                .set_ligand("Data/testcase/dock/ligand/8HW.pdbqt")
                .set_box((32.500, 13.0, 133.750), (22.5, 23.5, 22.5))
                .set_exhaustiveness(8)
                .set_num_modes(9)
                .set_cpu(4)
                .set_seed(42)
                .set_out("out/lig_docked.pdbqt")
                .set_log("out/lig.log")
            )

            # run — allow exceptions from smina to surface as test failures
            res = sd.run()

            # assertions: out & log exist, and result references them
            out_path = Path("out/lig_docked.pdbqt")
            log_path = Path("out/lig.log")
            self.assertTrue(out_path.exists(), f"Expected docking output at {out_path}")
            self.assertTrue(log_path.exists(), f"Expected log at {log_path}")

            # check returned artifact paths
            self.assertEqual(res.artifacts.out_path, out_path)
            self.assertEqual(res.artifacts.log_path, log_path)

            # optional: assert the command contained the smina path (if recorded)
            called = getattr(res.artifacts, "called", None)
            if called:
                called_str = str(called)
                # If we resolved an absolute path earlier, prefer that exact check.
                if self.smina_path:
                    self.assertIn(
                        self.smina_path,
                        called_str,
                        msg=f"expected smina_path {self.smina_path!r} in called: {called_str!r}",
                    )
                else:
                    # fallback: at minimum, the binary name should appear in the command string
                    self.assertIn(
                        "smina",
                        called_str,
                        msg=f"expected 'smina' in called: {called_str!r}",
                    )

        finally:
            # restore cwd and cleanup workspace
            try:
                os.chdir(self._orig_cwd)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)


class TestSingleDockQvina(unittest.TestCase):
    def setUp(self):
        self._orig_cwd = Path.cwd()
        # find repo root (by looking for the Data/testcase tree) and smina executable
        self.repo_root = find_repo_root_with_data(self._orig_cwd)
        self.qvina_path = sh.which("qvina")  # may be None

    def tearDown(self):
        try:
            os.chdir(self._orig_cwd)
        finally:
            pass

    def test_user_snippet_with_real_smina(self):

        tmpdir = Path(tempfile.mkdtemp(prefix="qvina"))
        try:
            workdir = tmpdir / "work"
            workdir.mkdir(parents=True, exist_ok=True)

            # copy repo inputs into isolated workdir
            src_rec = (
                self.repo_root
                / "Data"
                / "testcase"
                / "dock"
                / "receptor"
                / "5N2F.pdbqt"
            )
            src_lig = (
                self.repo_root / "Data" / "testcase" / "dock" / "ligand" / "8HW.pdbqt"
            )
            (workdir / "Data" / "testcase" / "dock" / "receptor").mkdir(
                parents=True, exist_ok=True
            )
            (workdir / "Data" / "testcase" / "dock" / "ligand").mkdir(
                parents=True, exist_ok=True
            )
            shutil.copy2(
                src_rec,
                workdir / "Data" / "testcase" / "dock" / "receptor" / "5N2F.pdbqt",
            )
            shutil.copy2(
                src_lig, workdir / "Data" / "testcase" / "dock" / "ligand" / "8HW.pdbqt"
            )

            # change cwd so relative paths in the snippet work as-is
            os.chdir(workdir)

            # register a factory for "qvina" that returns a SminaEngine configured to use the real binary
            def factory():
                e = QVinaEngine()
                # only call set_executable if we actually have a resolved path
                if self.qvina_path:
                    e.set_executable(self.qvina_path)
                return e

            register("qvina", factory)

            sd = (
                SingleDock("qvina")
                .set_receptor("Data/testcase/dock/receptor/5N2F.pdbqt", validate=True)
                .set_ligand("Data/testcase/dock/ligand/8HW.pdbqt")
                .set_box((32.500, 13.0, 133.750), (22.5, 23.5, 22.5))
                .set_exhaustiveness(8)
                .set_num_modes(9)
                .set_cpu(4)
                .set_seed(42)
                .set_out("out/lig_docked.pdbqt")
                .set_log("out/lig.log")
            )

            # run — allow exceptions from qvina to surface as test failures
            res = sd.run()

            # assertions: out & log exist, and result references them
            out_path = Path("out/lig_docked.pdbqt")
            log_path = Path("out/lig.log")
            self.assertTrue(out_path.exists(), f"Expected docking output at {out_path}")
            self.assertTrue(log_path.exists(), f"Expected log at {log_path}")

            # check returned artifact paths
            self.assertEqual(res.artifacts.out_path, out_path)
            self.assertEqual(res.artifacts.log_path, log_path)

            # optional: assert the command contained the qvina path (if recorded)
            called = getattr(res.artifacts, "called", None)
            if called:
                called_str = str(called)
                # If we resolved an absolute path earlier, prefer that exact check.
                if self.qvina_path:
                    self.assertIn(
                        self.qvina_path,
                        called_str,
                        msg=f"expected smina_path {self.qvina_path!r} in called: {called_str!r}",
                    )
                else:
                    # fallback: at minimum, the binary name should appear in the command string
                    self.assertIn(
                        "qvina",
                        called_str,
                        msg=f"expected 'qvina' in called: {called_str!r}",
                    )

        finally:
            # restore cwd and cleanup workspace
            try:
                os.chdir(self._orig_cwd)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)


class TestSingleDockQvinaW(unittest.TestCase):
    def setUp(self):
        self._orig_cwd = Path.cwd()
        # find repo root (by looking for the Data/testcase tree) and smina executable
        self.repo_root = find_repo_root_with_data(self._orig_cwd)
        self.qvina_path = sh.which("qvina-w")  # may be None

    def tearDown(self):
        try:
            os.chdir(self._orig_cwd)
        finally:
            pass

    def test_user_snippet_with_real_smina(self):

        tmpdir = Path(tempfile.mkdtemp(prefix="qvina-w"))
        try:
            workdir = tmpdir / "work"
            workdir.mkdir(parents=True, exist_ok=True)

            # copy repo inputs into isolated workdir
            src_rec = (
                self.repo_root
                / "Data"
                / "testcase"
                / "dock"
                / "receptor"
                / "5N2F.pdbqt"
            )
            src_lig = (
                self.repo_root / "Data" / "testcase" / "dock" / "ligand" / "8HW.pdbqt"
            )
            (workdir / "Data" / "testcase" / "dock" / "receptor").mkdir(
                parents=True, exist_ok=True
            )
            (workdir / "Data" / "testcase" / "dock" / "ligand").mkdir(
                parents=True, exist_ok=True
            )
            shutil.copy2(
                src_rec,
                workdir / "Data" / "testcase" / "dock" / "receptor" / "5N2F.pdbqt",
            )
            shutil.copy2(
                src_lig, workdir / "Data" / "testcase" / "dock" / "ligand" / "8HW.pdbqt"
            )

            # change cwd so relative paths in the snippet work as-is
            os.chdir(workdir)

            # register a factory for "qvina-w" that returns a SminaEngine configured to use the real binary
            def factory():
                e = QVinaWEngine()
                # only call set_executable if we actually have a resolved path
                if self.qvina_path:
                    e.set_executable(self.qvina_path)
                return e

            register("qvina", factory)

            sd = (
                SingleDock("qvina-w")
                .set_receptor("Data/testcase/dock/receptor/5N2F.pdbqt", validate=True)
                .set_ligand("Data/testcase/dock/ligand/8HW.pdbqt")
                .set_box((32.500, 13.0, 133.750), (22.5, 23.5, 22.5))
                .set_exhaustiveness(8)
                .set_num_modes(9)
                .set_cpu(4)
                .set_seed(42)
                .set_out("out/lig_docked.pdbqt")
                .set_log("out/lig.log")
            )

            # run — allow exceptions from qvina to surface as test failures
            res = sd.run()

            # assertions: out & log exist, and result references them
            out_path = Path("out/lig_docked.pdbqt")
            log_path = Path("out/lig.log")
            self.assertTrue(out_path.exists(), f"Expected docking output at {out_path}")
            self.assertTrue(log_path.exists(), f"Expected log at {log_path}")

            # check returned artifact paths
            self.assertEqual(res.artifacts.out_path, out_path)
            self.assertEqual(res.artifacts.log_path, log_path)

            # optional: assert the command contained the qvina path (if recorded)
            called = getattr(res.artifacts, "called", None)
            if called:
                called_str = str(called)
                # If we resolved an absolute path earlier, prefer that exact check.
                if self.qvina_path:
                    self.assertIn(
                        self.qvina_path,
                        called_str,
                        msg=f"expected smina_path {self.qvina_path!r} in called: {called_str!r}",
                    )
                else:
                    # fallback: at minimum, the binary name should appear in the command string
                    self.assertIn(
                        "qvina-w",
                        called_str,
                        msg=f"expected 'qvina-w' in called: {called_str!r}",
                    )

        finally:
            # restore cwd and cleanup workspace
            try:
                os.chdir(self._orig_cwd)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
