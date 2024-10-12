import unittest
from unittest.mock import patch, MagicMock
from ProDock.Optimize_Conformer.xtb_minimize import XTBMinimize


class TestXTBMinimize(unittest.TestCase):
    def test_retrive_filename(self):
        folder = "/Users/anpham/Documents/GitHub/ProDock/delete"
        with patch(
            "glob.glob",
            return_value=[
                "/Users/anpham/Documents/GitHub/ProDock/delete/ligand_0",
                "/Users/anpham/Documents/GitHub/ProDock/delete/ligand_1",
            ],
        ):
            folder_paths, file_names = XTBMinimize.retrive_filename(folder)
            self.assertEqual(
                folder_paths,
                [
                    "/Users/anpham/Documents/GitHub/ProDock/delete/ligand_0",
                    "/Users/anpham/Documents/GitHub/ProDock/delete/ligand_1",
                ],
            )
            self.assertEqual(
                file_names,
                [
                    "/Users/anpham/Documents/GitHub/ProDock/delete/ligand_0/ligand_0.sdf",
                    "/Users/anpham/Documents/GitHub/ProDock/delete/ligand_1/ligand_1.sdf",
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
                ["/Users/anpham/Documents/GitHub/ProDock/delete/ligand_0"],
                ["/Users/anpham/Documents/GitHub/ProDock/delete/ligand_0.sdf"],
            ),
        ), patch(
            "ProDock.Optimize_Conformer.xtb_minimize.XTBMinimize.xtb_optimize"
        ) as mock_optimize:
            minimize = XTBMinimize()
            minimize.fit("/Users/anpham/Documents/GitHub/ProDock/delete")
            mock_optimize.assert_called_once_with(
                "/Users/anpham/Documents/GitHub/ProDock/delete/ligand_0.sdf"
            )


if __name__ == "__main__":
    unittest.main()
