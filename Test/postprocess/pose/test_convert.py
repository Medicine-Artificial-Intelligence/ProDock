from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prodock.postprocess.pose import convert


class TestConvertHelpers(unittest.TestCase):
    def test_as_path_from_str_and_path(self) -> None:
        path1 = convert._as_path(
            "Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt"
        )
        path2 = convert._as_path(
            Path("Data/testcase/post/1M17/results/docked/vina/erlotinib_docked.pdbqt")
        )

        self.assertIsInstance(path1, Path)
        self.assertEqual(path1, path2)

    def test_iter_pdbqt_files_direct_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = (
                Path(tmpdir)
                / "Data"
                / "testcase"
                / "post"
                / "1M17"
                / "results"
                / "docked"
                / "vina"
                / "erlotinib_docked.pdbqt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("content", encoding="utf-8")

            files = convert._iter_pdbqt_files([path])

        self.assertEqual(files, [path.resolve()])

    def test_iter_pdbqt_files_direct_file_engine_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = (
                Path(tmpdir)
                / "Data"
                / "testcase"
                / "post"
                / "1M17"
                / "results"
                / "docked"
                / "qvina"
                / "erlotinib_docked.pdbqt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("content", encoding="utf-8")

            files = convert._iter_pdbqt_files([path], engine="qvina")

        self.assertEqual(files, [path.resolve()])

    def test_iter_pdbqt_files_direct_file_engine_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = (
                Path(tmpdir)
                / "Data"
                / "testcase"
                / "post"
                / "1M17"
                / "results"
                / "docked"
                / "vina"
                / "erlotinib_docked.pdbqt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("content", encoding="utf-8")

            files = convert._iter_pdbqt_files([path], engine="smina")

        self.assertEqual(files, [])

    def test_iter_pdbqt_files_directory_non_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "Data" / "testcase" / "post"
            top = root / "top_level.pdbqt"
            sub = (
                root / "1M17" / "results" / "docked" / "vina" / "erlotinib_docked.pdbqt"
            )

            top.parent.mkdir(parents=True, exist_ok=True)
            sub.parent.mkdir(parents=True, exist_ok=True)

            top.write_text("a", encoding="utf-8")
            sub.write_text("b", encoding="utf-8")

            files = convert._iter_pdbqt_files([root], recursive=False)

        self.assertEqual(files, [top.resolve()])

    def test_iter_pdbqt_files_directory_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "Data" / "testcase" / "post"
            vina = (
                root / "1M17" / "results" / "docked" / "vina" / "erlotinib_docked.pdbqt"
            )
            smina = (
                root
                / "1M17"
                / "results"
                / "docked"
                / "smina"
                / "erlotinib_docked.pdbqt"
            )

            vina.parent.mkdir(parents=True, exist_ok=True)
            smina.parent.mkdir(parents=True, exist_ok=True)

            vina.write_text("vina", encoding="utf-8")
            smina.write_text("smina", encoding="utf-8")

            files = convert._iter_pdbqt_files([root], recursive=True)

        self.assertEqual(files, sorted([vina.resolve(), smina.resolve()]))

    def test_iter_pdbqt_files_engine_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = (
                Path(tmpdir)
                / "Data"
                / "testcase"
                / "post"
                / "1M17"
                / "results"
                / "docked"
            )
            vina = root / "vina" / "erlotinib_docked.pdbqt"
            smina = root / "smina" / "erlotinib_docked.pdbqt"
            qvina = root / "qvina" / "erlotinib_docked.pdbqt"

            vina.parent.mkdir(parents=True, exist_ok=True)
            smina.parent.mkdir(parents=True, exist_ok=True)
            qvina.parent.mkdir(parents=True, exist_ok=True)

            vina.write_text("vina", encoding="utf-8")
            smina.write_text("smina", encoding="utf-8")
            qvina.write_text("qvina", encoding="utf-8")

            files = convert._iter_pdbqt_files([root], engine="vina")

        self.assertEqual(files, [vina.resolve()])

    def test_iter_pdbqt_files_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "Data" / "testcase" / "post"
            path = (
                root / "1M17" / "results" / "docked" / "vina" / "erlotinib_docked.pdbqt"
            )

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("a", encoding="utf-8")

            files = convert._iter_pdbqt_files([root, path], recursive=True)

        self.assertEqual(files, [path.resolve()])


class TestSavePoseSDF(unittest.TestCase):
    def test_save_pose_sdf_default_neighbor_output(self) -> None:
        original = convert.pdbqt_to_sdf
        calls = []

        def fake_pdbqt_to_sdf(src: str, dst: str, backend: str = "obabel") -> None:
            calls.append((src, dst, backend))
            Path(dst).write_text("fake sdf", encoding="utf-8")

        convert.pdbqt_to_sdf = fake_pdbqt_to_sdf
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdbqt = (
                    Path(tmpdir)
                    / "Data"
                    / "testcase"
                    / "post"
                    / "1M17"
                    / "results"
                    / "docked"
                    / "qvina"
                    / "erlotinib_docked.pdbqt"
                )
                pdbqt.parent.mkdir(parents=True, exist_ok=True)
                pdbqt.write_text("pdbqt", encoding="utf-8")

                out = convert.save_pose_sdf(pdbqt, backend="obabel", overwrite=False)

                self.assertEqual(out, pdbqt.with_suffix(".sdf"))
                self.assertTrue(out.exists())
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][2], "obabel")
        finally:
            convert.pdbqt_to_sdf = original

    def test_save_pose_sdf_explicit_out_file(self) -> None:
        original = convert.pdbqt_to_sdf

        def fake_pdbqt_to_sdf(src: str, dst: str, backend: str = "obabel") -> None:
            Path(dst).write_text("fake sdf", encoding="utf-8")

        convert.pdbqt_to_sdf = fake_pdbqt_to_sdf
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdbqt = (
                    Path(tmpdir)
                    / "Data"
                    / "testcase"
                    / "post"
                    / "1M17"
                    / "results"
                    / "docked"
                    / "vina"
                    / "erlotinib_docked.pdbqt"
                )
                out_file = Path(tmpdir) / "converted" / "erlotinib_from_vina.sdf"

                pdbqt.parent.mkdir(parents=True, exist_ok=True)
                pdbqt.write_text("pdbqt", encoding="utf-8")

                out = convert.save_pose_sdf(
                    pdbqt,
                    out_file=out_file,
                    overwrite=True,
                )

                self.assertEqual(out, out_file)
                self.assertTrue(out.exists())
        finally:
            convert.pdbqt_to_sdf = original

    def test_save_pose_sdf_reuses_existing_when_no_overwrite(self) -> None:
        original = convert.pdbqt_to_sdf
        calls = []

        def fake_pdbqt_to_sdf(src: str, dst: str, backend: str = "obabel") -> None:
            calls.append((src, dst, backend))
            Path(dst).write_text("new sdf", encoding="utf-8")

        convert.pdbqt_to_sdf = fake_pdbqt_to_sdf
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdbqt = (
                    Path(tmpdir)
                    / "Data"
                    / "testcase"
                    / "post"
                    / "1M17"
                    / "results"
                    / "docked"
                    / "smina"
                    / "erlotinib_docked.pdbqt"
                )
                sdf = pdbqt.with_suffix(".sdf")

                pdbqt.parent.mkdir(parents=True, exist_ok=True)
                pdbqt.write_text("pdbqt", encoding="utf-8")
                sdf.write_text("existing sdf", encoding="utf-8")

                out = convert.save_pose_sdf(pdbqt, overwrite=False)

                self.assertEqual(out, sdf)
                self.assertEqual(len(calls), 0)
                self.assertEqual(sdf.read_text(encoding="utf-8"), "existing sdf")
        finally:
            convert.pdbqt_to_sdf = original

    def test_save_pose_sdf_overwrites_existing_when_requested(self) -> None:
        original = convert.pdbqt_to_sdf
        calls = []

        def fake_pdbqt_to_sdf(src: str, dst: str, backend: str = "obabel") -> None:
            calls.append((src, dst, backend))
            Path(dst).write_text("overwritten sdf", encoding="utf-8")

        convert.pdbqt_to_sdf = fake_pdbqt_to_sdf
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdbqt = (
                    Path(tmpdir)
                    / "Data"
                    / "testcase"
                    / "post"
                    / "1M17"
                    / "results"
                    / "docked"
                    / "vina"
                    / "erlotinib_docked.pdbqt"
                )
                sdf = pdbqt.with_suffix(".sdf")

                pdbqt.parent.mkdir(parents=True, exist_ok=True)
                pdbqt.write_text("pdbqt", encoding="utf-8")
                sdf.write_text("existing sdf", encoding="utf-8")

                out = convert.save_pose_sdf(pdbqt, overwrite=True)

                self.assertEqual(out, sdf)
                self.assertEqual(len(calls), 1)
                self.assertEqual(sdf.read_text(encoding="utf-8"), "overwritten sdf")
        finally:
            convert.pdbqt_to_sdf = original


class FakeSupplier(list):
    def __init__(self, path: str, sanitize: bool = True, removeHs: bool = False):
        self.path = path
        self.sanitize = sanitize
        self.removeHs = removeHs
        super().__init__(["mol1", None, "mol2"])


class TestPDBQTToRDKitMols(unittest.TestCase):
    def test_pdbqt_to_rdkit_mols_filters_none_and_passes_flags(self) -> None:
        original_converter = convert.pdbqt_to_sdf
        original_supplier = convert.Chem.SDMolSupplier

        converter_calls = []
        supplier_calls = []

        def fake_pdbqt_to_sdf(src: str, dst: str, backend: str = "obabel") -> None:
            converter_calls.append((src, dst, backend))
            Path(dst).write_text("fake sdf", encoding="utf-8")

        def fake_supplier(path: str, sanitize: bool = True, removeHs: bool = False):
            supplier_calls.append((path, sanitize, removeHs))
            return FakeSupplier(path, sanitize=sanitize, removeHs=removeHs)

        convert.pdbqt_to_sdf = fake_pdbqt_to_sdf
        convert.Chem.SDMolSupplier = fake_supplier

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdbqt = (
                    Path(tmpdir)
                    / "Data"
                    / "testcase"
                    / "post"
                    / "1M17"
                    / "results"
                    / "docked"
                    / "smina"
                    / "erlotinib_docked.pdbqt"
                )
                pdbqt.parent.mkdir(parents=True, exist_ok=True)
                pdbqt.write_text("pdbqt", encoding="utf-8")

                mols = convert.pdbqt_to_rdkit_mols(
                    pdbqt,
                    backend="obabel",
                    sanitize=False,
                    remove_hs=True,
                )

                self.assertEqual(mols, ["mol1", "mol2"])
                self.assertEqual(len(converter_calls), 1)
                self.assertEqual(converter_calls[0][2], "obabel")
                self.assertEqual(len(supplier_calls), 1)
                self.assertFalse(supplier_calls[0][1])
                self.assertTrue(supplier_calls[0][2])
        finally:
            convert.pdbqt_to_sdf = original_converter
            convert.Chem.SDMolSupplier = original_supplier


class TestConvertPoseTree(unittest.TestCase):
    def test_convert_pose_tree_neighbor_outputs(self) -> None:
        original = convert.save_pose_sdf
        calls = []

        def fake_save_pose_sdf(
            pdbqt_file, backend="obabel", overwrite=False, out_file=None
        ):
            path = Path(pdbqt_file)
            dst = Path(out_file) if out_file is not None else path.with_suffix(".sdf")
            calls.append((path, backend, overwrite, out_file))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("sdf", encoding="utf-8")
            return dst

        convert.save_pose_sdf = fake_save_pose_sdf
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = (
                    Path(tmpdir)
                    / "Data"
                    / "testcase"
                    / "post"
                    / "1M17"
                    / "results"
                    / "docked"
                )
                a = root / "vina" / "erlotinib_docked.pdbqt"
                b = root / "vina" / "gefitinib_docked.pdbqt"

                a.parent.mkdir(parents=True, exist_ok=True)
                a.write_text("a", encoding="utf-8")
                b.write_text("b", encoding="utf-8")

                outputs = convert.convert_pose_tree([root], engine="vina")

                self.assertEqual(
                    [p.resolve() for p in outputs],
                    [
                        a.with_suffix(".sdf").resolve(),
                        b.with_suffix(".sdf").resolve(),
                    ],
                )
                self.assertEqual(len(calls), 2)
        finally:
            convert.save_pose_sdf = original

    def test_convert_pose_tree_shared_out_dir(self) -> None:
        original = convert.save_pose_sdf
        calls = []

        def fake_save_pose_sdf(
            pdbqt_file, backend="obabel", overwrite=False, out_file=None
        ):
            dst = Path(out_file)
            calls.append((Path(pdbqt_file), backend, overwrite, dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("sdf", encoding="utf-8")
            return dst

        convert.save_pose_sdf = fake_save_pose_sdf
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = (
                    Path(tmpdir)
                    / "Data"
                    / "testcase"
                    / "post"
                    / "1M17"
                    / "results"
                    / "docked"
                )
                out_dir = Path(tmpdir) / "converted_sdf"

                a = root / "vina" / "erlotinib_docked.pdbqt"
                b = root / "vina" / "gefitinib_docked.pdbqt"

                a.parent.mkdir(parents=True, exist_ok=True)
                a.write_text("a", encoding="utf-8")
                b.write_text("b", encoding="utf-8")

                outputs = convert.convert_pose_tree(
                    [root],
                    engine="vina",
                    out_dir=out_dir,
                )

                self.assertEqual(
                    outputs,
                    [
                        out_dir / "erlotinib_docked.sdf",
                        out_dir / "gefitinib_docked.sdf",
                    ],
                )
                self.assertEqual(len(calls), 2)
        finally:
            convert.save_pose_sdf = original

    def test_convert_pose_tree_shared_out_dir_duplicate_stems(self) -> None:
        original = convert.save_pose_sdf
        calls = []

        def fake_save_pose_sdf(
            pdbqt_file, backend="obabel", overwrite=False, out_file=None
        ):
            dst = Path(out_file)
            calls.append(dst.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("sdf", encoding="utf-8")
            return dst

        convert.save_pose_sdf = fake_save_pose_sdf
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = (
                    Path(tmpdir)
                    / "Data"
                    / "testcase"
                    / "post"
                    / "1M17"
                    / "results"
                    / "docked"
                )
                out_dir = Path(tmpdir) / "converted_sdf"

                a = root / "vina" / "erlotinib_docked.pdbqt"
                b = root / "smina" / "erlotinib_docked.pdbqt"
                c = root / "qvina" / "erlotinib_docked.pdbqt"

                a.parent.mkdir(parents=True, exist_ok=True)
                b.parent.mkdir(parents=True, exist_ok=True)
                c.parent.mkdir(parents=True, exist_ok=True)

                a.write_text("a", encoding="utf-8")
                b.write_text("b", encoding="utf-8")
                c.write_text("c", encoding="utf-8")

                outputs = convert.convert_pose_tree(
                    [root],
                    recursive=True,
                    out_dir=out_dir,
                )

                self.assertEqual(
                    outputs,
                    [
                        out_dir / "erlotinib_docked.sdf",
                        out_dir / "erlotinib_docked_2.sdf",
                        out_dir / "erlotinib_docked_3.sdf",
                    ],
                )
                self.assertEqual(
                    calls,
                    [
                        "erlotinib_docked.sdf",
                        "erlotinib_docked_2.sdf",
                        "erlotinib_docked_3.sdf",
                    ],
                )
        finally:
            convert.save_pose_sdf = original


if __name__ == "__main__":
    unittest.main()
