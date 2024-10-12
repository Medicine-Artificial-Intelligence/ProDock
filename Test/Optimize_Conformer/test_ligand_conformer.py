import unittest
from unittest.mock import patch, MagicMock
from ProDock.Optimize_Conformer.ligand_conformer import Ligand_Conformer

class TestLigandConformer(unittest.TestCase):

    @patch('ProDock.Optimize_Conformer.ligand_conformer.Ligand_Conformer.list_smi', return_value=[MagicMock()])
    @patch('ProDock.Optimize_Conformer.ligand_conformer.Ligand_Conformer.mol_embbeding_3d', return_value=MagicMock())
    @patch('os.makedirs')
    @patch("shutil.move")
    @patch('rdkit.Chem.SDWriter')
    def test_write_sdf(self, mock_writer, mock_move,mock_makedirs, mock_embedding, mock_list_smi):
        ligand_conformer = Ligand_Conformer()
        ligand_conformer.write_sdf("Data/smi_testcase.txt", "/Users/anpham/Documents/GitHub/ProDock/delete")
        mock_list_smi.assert_called_once_with("Data/smi_testcase.txt")
        mock_embedding.assert_called()
        mock_writer.assert_called()
        mock_move.assert_called()

if __name__ == "__main__":
    unittest.main()
