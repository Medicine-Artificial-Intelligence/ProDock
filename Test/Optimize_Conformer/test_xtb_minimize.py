import unittest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock
from ProDock.Optimize_Conformer.xtb_minimize import XTBMinimize
from ProDock.Optimize_Conformer.ligand_conformer import Ligand_Conformer


class TestXTBMinimize(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.sdf_path = os.path.join(self.test_dir, "sdf_folder")
        Ligand_Conformer().write_sdf(
            smi_filename="Data/smi_testcase.txt", sdf_output_folder=self.sdf_path
        )

    def test_retrive_filename(self):

        with patch(
            "glob.glob",
            return_value=[
                os.path.join(self.sdf_path, "ligand_0"),
                os.path.join(self.sdf_path, "ligand_1"),
            ],
        ):
            folder_paths, file_names = XTBMinimize.retrive_filename(self.sdf_path)
            self.assertEqual(
                folder_paths,
                [
                    os.path.join(self.sdf_path, "ligand_0"),
                    os.path.join(self.sdf_path, "ligand_1"),
                ],
            )
            self.assertEqual(
                file_names,
                [
                    os.path.join(self.sdf_path, "ligand_0", "ligand_0.sdf"),
                    os.path.join(self.sdf_path, "ligand_1", "ligand_1.sdf"),
                ],
            )

    @patch("subprocess.run")
    def test_xtb_optimize(self, mock_subprocess):
        # Mock successful xtb execution
        mock_subprocess.return_value = MagicMock(returncode=0)
        XTBMinimize.xtb_optimize("ligand_0.sdf")
        mock_subprocess.assert_called_once_with(
            ["xtb", "ligand_0.sdf", "--opt", "--silent", "tight"], check=True
        )

    def test_fit(self):
        with patch(
            "ProDock.Optimize_Conformer.xtb_minimize.XTBMinimize.retrive_filename",
            return_value=(
                [os.path.join(self.sdf_path, "ligand_0")],
                [os.path.join(self.sdf_path, "ligand_0", "ligand_0.sdf")],
            ),
        ), patch(
            "ProDock.Optimize_Conformer.xtb_minimize.XTBMinimize.xtb_optimize"
        ) as mock_optimize:
            minimize = XTBMinimize()
            minimize.fit({self.sdf_path})
            mock_optimize.assert_called_once_with(
                os.path.join(self.sdf_path, "ligand_0", "ligand_0.sdf")
            )

    def tearDown(self):
        # Cleanup temporary directory after test
        shutil.rmtree(self.sdf_path)


if __name__ == "__main__":
    unittest.main()
