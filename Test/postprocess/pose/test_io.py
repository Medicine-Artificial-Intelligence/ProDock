from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prodock.postprocess.pose import io


class TestPoseIOHelpers(unittest.TestCase):
    def test_as_path_from_str_and_path(self) -> None:
        path1 = io._as_path("abc/file.pdbqt")
        path2 = io._as_path(Path("abc/file.pdbqt"))

        self.assertIsInstance(path1, Path)
        self.assertIsInstance(path2, Path)
        self.assertEqual(path1, path2)

    def test_to_float(self) -> None:
        self.assertEqual(io._to_float("-7.5"), -7.5)
        self.assertIsNone(io._to_float(None))
        self.assertIsNone(io._to_float("abc"))

    def test_strip_ligand_suffix(self) -> None:
        self.assertEqual(io._strip_ligand_suffix("lig_docked"), "lig")
        self.assertEqual(io._strip_ligand_suffix("lig_poses"), "lig")
        self.assertEqual(io._strip_ligand_suffix("lig_pose"), "lig")
        self.assertEqual(io._strip_ligand_suffix("lig_out"), "lig")
        self.assertEqual(io._strip_ligand_suffix("ligand"), "ligand")

    def test_infer_hierarchical_metadata_results_docked_layout(self) -> None:
        path = Path("demo/4WKQ/results/docked/vina/erlotinib.pdbqt")
        receptor_id, engine = io._infer_hierarchical_metadata(path)
        self.assertEqual(receptor_id, "4WKQ")
        self.assertEqual(engine, "vina")

    def test_infer_hierarchical_metadata_simple_layout(self) -> None:
        path = Path("demo/4WKQ/vina/erlotinib.pdbqt")
        receptor_id, engine = io._infer_hierarchical_metadata(path)
        self.assertEqual(receptor_id, "4WKQ")
        self.assertEqual(engine, "vina")

    def test_infer_hierarchical_metadata_unknown(self) -> None:
        receptor_id, engine = io._infer_hierarchical_metadata(Path("x.pdbqt"))
        self.assertIsNone(receptor_id)
        self.assertIsNone(engine)


class TestParsePDBQTPoseScores(unittest.TestCase):
    def test_parse_vina_without_models(self) -> None:
        text = "REMARK VINA RESULT: -7.2 0.0 0.0\n" "REMARK VINA RESULT: -6.8 1.2 2.1\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vina.pdbqt"
            path.write_text(text, encoding="utf-8")

            rows = io.parse_pdbqt_pose_scores(path)

        self.assertEqual(
            rows,
            [
                {"pose_rank": 1, "affinity": -7.2},
                {"pose_rank": 2, "affinity": -6.8},
            ],
        )

    def test_parse_smina_without_models(self) -> None:
        text = "REMARK minimizedAffinity -8.1\n" "REMARK minimizedAffinity -7.4\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "smina.pdbqt"
            path.write_text(text, encoding="utf-8")

            rows = io.parse_pdbqt_pose_scores(path)

        self.assertEqual(
            rows,
            [
                {"pose_rank": 1, "affinity": -8.1},
                {"pose_rank": 2, "affinity": -7.4},
            ],
        )

    def test_parse_mode_table_without_models(self) -> None:
        text = "  1   -7.0   0.0   0.0\n" "  2   -6.5   1.1   2.2\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.pdbqt"
            path.write_text(text, encoding="utf-8")

            rows = io.parse_pdbqt_pose_scores(path)

        self.assertEqual(
            rows,
            [
                {"pose_rank": 1, "affinity": -7.0},
                {"pose_rank": 2, "affinity": -6.5},
            ],
        )

    def test_parse_models_with_vina_and_smina(self) -> None:
        text = (
            "MODEL 1\n"
            "REMARK VINA RESULT: -7.9 0.0 0.0\n"
            "ATOM\n"
            "MODEL 2\n"
            "REMARK minimizedAffinity -6.2\n"
            "ATOM\n"
            "MODEL 3\n"
            "ATOM\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "models.pdbqt"
            path.write_text(text, encoding="utf-8")

            rows = io.parse_pdbqt_pose_scores(path)

        self.assertEqual(
            rows,
            [
                {"pose_rank": 1, "affinity": -7.9},
                {"pose_rank": 2, "affinity": -6.2},
                {"pose_rank": 3, "affinity": None},
            ],
        )

    def test_parse_fallback_when_no_scores(self) -> None:
        text = "ATOM something only\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unknown.pdbqt"
            path.write_text(text, encoding="utf-8")

            rows = io.parse_pdbqt_pose_scores(path)

        self.assertEqual(rows, [{"pose_rank": 1, "affinity": None}])


class TestDiscoverAndBuildPoseRecords(unittest.TestCase):
    def test_discover_pose_files_direct_file_requires_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ligand.pdbqt"
            path.write_text("MODEL 1\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                io.discover_pose_files([path])

    def test_discover_pose_files_flat_folder_requires_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "a.pdbqt").write_text("MODEL 1\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                io.discover_pose_files([folder])

    def test_discover_pose_files_flat_folder_with_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            a = folder / "a.pdbqt"
            b = folder / "b.pdbqt"
            c = folder / "note.txt"
            a.write_text("MODEL 1\n", encoding="utf-8")
            b.write_text("MODEL 1\n", encoding="utf-8")
            c.write_text("ignore\n", encoding="utf-8")

            files = io.discover_pose_files([folder], engine="vina")

        self.assertEqual(files, sorted([a.resolve(), b.resolve()]))

    def test_discover_pose_files_recursive_hierarchical_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vina_file = root / "4WKQ" / "results" / "docked" / "vina" / "lig1.pdbqt"
            smina_file = root / "4WKQ" / "results" / "docked" / "smina" / "lig2.pdbqt"
            vina_file.parent.mkdir(parents=True, exist_ok=True)
            smina_file.parent.mkdir(parents=True, exist_ok=True)
            vina_file.write_text("MODEL 1\n", encoding="utf-8")
            smina_file.write_text("MODEL 1\n", encoding="utf-8")

            files = io.discover_pose_files([root], engine="vina", recursive=True)

        self.assertEqual(files, [vina_file.resolve()])

    def test_build_pose_records_from_direct_file(self) -> None:
        text = "REMARK VINA RESULT: -7.0 0.0 0.0\n" "REMARK VINA RESULT: -6.0 0.0 0.0\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "erlotinib_docked.pdbqt"
            path.write_text(text, encoding="utf-8")

            records = io.build_pose_records([path], engine="vina")

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].ligand_id, "erlotinib")
        self.assertEqual(records[0].engine, "vina")
        self.assertIsNone(records[0].receptor_id)
        self.assertEqual(records[0].pose_rank, 1)
        self.assertEqual(records[0].affinity, -7.0)

    def test_build_pose_records_recursive_hierarchical(self) -> None:
        text = (
            "MODEL 1\n"
            "REMARK VINA RESULT: -8.0 0.0 0.0\n"
            "MODEL 2\n"
            "REMARK VINA RESULT: -7.2 0.0 0.0\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "4WKQ" / "results" / "docked" / "vina" / "erlotinib_out.pdbqt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

            records = io.build_pose_records([root], recursive=True)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].receptor_id, "4WKQ")
        self.assertEqual(records[0].engine, "vina")
        self.assertEqual(records[0].ligand_id, "erlotinib")
        self.assertEqual(records[0].pose_rank, 1)
        self.assertEqual(records[1].pose_rank, 2)


class TestBuildPoseMolRows(unittest.TestCase):
    def test_build_pose_mol_rows_with_stubbed_conversion(self) -> None:
        original_loader = io.pdbqt_to_rdkit_mols
        original_save_sdf = io.save_pose_sdf

        saved_calls = []
        load_calls = []

        def fake_save_pose_sdf(path, backend="obabel", overwrite=False):
            saved_calls.append((Path(path), backend, overwrite))
            return Path(path).with_suffix(".sdf")

        def fake_pdbqt_to_rdkit_mols(
            path, backend="obabel", sanitize=True, remove_hs=False
        ):
            load_calls.append((Path(path), backend, sanitize, remove_hs))
            return ["mol1", "mol2"]

        io.save_pose_sdf = fake_save_pose_sdf
        io.pdbqt_to_rdkit_mols = fake_pdbqt_to_rdkit_mols

        try:
            text = (
                "REMARK VINA RESULT: -7.3 0.0 0.0\n"
                "REMARK VINA RESULT: -6.4 0.0 0.0\n"
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "ligand_pose.pdbqt"
                path.write_text(text, encoding="utf-8")

                rows = io.build_pose_mol_rows(
                    [path],
                    engine="vina",
                    backend="obabel",
                    sanitize=False,
                    remove_hs=True,
                    save_sdf=True,
                    overwrite_sdf=True,
                )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["ligand_id"], "ligand")
            self.assertEqual(rows[0]["engine"], "vina")
            self.assertEqual(rows[0]["pose_rank"], 1)
            self.assertEqual(rows[0]["affinity"], -7.3)
            self.assertEqual(rows[0]["mol"], "mol1")
            self.assertEqual(rows[1]["mol"], "mol2")

            self.assertEqual(len(saved_calls), 1)
            self.assertEqual(saved_calls[0][1], "obabel")
            self.assertTrue(saved_calls[0][2])

            self.assertEqual(len(load_calls), 1)
            self.assertEqual(load_calls[0][1], "obabel")
            self.assertFalse(load_calls[0][2])
            self.assertTrue(load_calls[0][3])

        finally:
            io.pdbqt_to_rdkit_mols = original_loader
            io.save_pose_sdf = original_save_sdf

    def test_build_pose_mol_rows_when_fewer_mols_than_records(self) -> None:
        original_loader = io.pdbqt_to_rdkit_mols
        original_save_sdf = io.save_pose_sdf

        def fake_save_pose_sdf(path, backend="obabel", overwrite=False):
            return Path(path).with_suffix(".sdf")

        def fake_pdbqt_to_rdkit_mols(
            path, backend="obabel", sanitize=True, remove_hs=False
        ):
            return ["only_one_mol"]

        io.save_pose_sdf = fake_save_pose_sdf
        io.pdbqt_to_rdkit_mols = fake_pdbqt_to_rdkit_mols

        try:
            text = (
                "REMARK VINA RESULT: -7.3 0.0 0.0\n"
                "REMARK VINA RESULT: -6.4 0.0 0.0\n"
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "ligand_docked.pdbqt"
                path.write_text(text, encoding="utf-8")

                rows = io.build_pose_mol_rows(
                    [path],
                    engine="vina",
                    save_sdf=False,
                )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["mol"], "only_one_mol")
            self.assertIsNone(rows[1]["mol"])

        finally:
            io.pdbqt_to_rdkit_mols = original_loader
            io.save_pose_sdf = original_save_sdf


if __name__ == "__main__":
    unittest.main()
