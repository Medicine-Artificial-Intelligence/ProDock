import unittest
from prodock.structure.selection import join_selection, chain_selection, resn_selection


class TestSelection(unittest.TestCase):
    def test_join_selection_basic(self):
        self.assertEqual(join_selection("chain", ["A", "B"]), "chain A or chain B")
        self.assertEqual(chain_selection(["A"]), "chain A")
        self.assertEqual(resn_selection(["HOH", "DOD"]), "resn HOH or resn DOD")


if __name__ == "__main__":
    unittest.main()
