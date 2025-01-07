import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from rdkit import Chem
from ProDock.Optimize_Conformer.ligand_conformer import Ligand_Conformer


class TestLigandConformer(unittest.TestCase):

    @patch(
        "ProDock.Optimize_Conformer.ligand_conformer.Ligand_Conformer.list_smi",
        return_value=[MagicMock()],
    )
    @patch(
        "ProDock.Optimize_Conformer.ligand_conformer.Ligand_Conformer.mol_embbeding_3d",
        return_value=MagicMock(),
    )
    @patch("os.makedirs")
    @patch("shutil.move")
    @patch("rdkit.Chem.SDWriter")
    # def test_write_sdf(
    #     self, mock_writer, mock_move, mock_makedirs, mock_embedding, mock_list_smi
    # ):
    #     ligand_conformer = Ligand_Conformer()
    #     """Set up a temporary directory and create a test CSV file."""
    #     test_dir = tempfile.mkdtemp()
    #     ligand_conformer.write_sdf("Data/smi_testcase.txt", test_dir)
    #     mock_list_smi.assert_called_once_with("Data/smi_testcase.txt")
    #     mock_embedding.assert_called()
    #     mock_writer.assert_called()
    #     mock_move.assert_called()
    #     shutil.rmtree(test_dir)

    def test_write_sdf(
        self, mock_writer_class, mock_move, mock_makedirs, mock_embedding, mock_list_smi
    ):
        """
        Test the `write_sdf` method of the Ligand_Conformer class.
        """
        # Arrange
        ligand_conformer = Ligand_Conformer()
        test_dir = tempfile.mkdtemp()
        test_smi_file = "Data/smi_testcase.txt"

        # Mock SDWriter object behavior
        mock_writer_instance = MagicMock()
        mock_writer_class.return_value = mock_writer_instance

        # Act
        ligand_conformer.write_sdf(test_smi_file, test_dir)

        # Assert
        mock_list_smi.assert_called_once_with(test_smi_file)
        mock_embedding.assert_called()
        mock_writer_class.assert_called()
        mock_writer_instance.write.assert_called()
        mock_writer_instance.close.assert_called()
        mock_move.assert_called()
        mock_makedirs.assert_called()

        # Cleanup
        shutil.rmtree(test_dir)
if __name__ == "__main__":
    unittest.main()
