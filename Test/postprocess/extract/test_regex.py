import unittest

from prodock.postprocess.extract import detect_engine, parse_log_text


VINA_TXT = """\
AutoDock Vina v1.2.4
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1        -10.0      0.000      0.000
   2         -8.2      2.300      5.835
"""

SMINA_TXT = """\
smina is based off AutoDock Vina. Please cite appropriately.
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
1       -10.0      0.000      0.000
2       -8.2       2.300      5.834
"""

QVINA_TXT = """\
QuickVina2
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1        -10.0      0.000      0.000
   2         -8.2      2.300      5.835
"""

VINAGPU_TXT = """\
Vina-GPU
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1        -10.0      0.000      0.000
   2         -8.2      2.300      5.832
"""

GNINA_TXT = """\
gnina v1.0.2
mode |  affinity  |    CNN     |   CNN
     | (kcal/mol) | pose score | affinity
-----+------------+------------+----------
    1       -9.99       0.8833      7.573
    2        8.69       0.7619      5.719
"""


class TestRegexEngines(unittest.TestCase):
    def test_detect_engine_banners(self):
        self.assertEqual(detect_engine(VINA_TXT), "vina")
        self.assertEqual(detect_engine(SMINA_TXT), "smina")
        self.assertEqual(detect_engine(QVINA_TXT), "qvina")
        self.assertEqual(detect_engine(VINAGPU_TXT), "vina-gpu")
        self.assertEqual(detect_engine(GNINA_TXT), "gnina")

    def test_parse_vina_family(self):
        for txt in (VINA_TXT, SMINA_TXT, QVINA_TXT, VINAGPU_TXT):
            rows = parse_log_text(txt)
            self.assertEqual(len(rows), 2)
            self.assertIn("affinity_kcal_mol", rows[0])
            self.assertIn("rmsd_lb", rows[0])
            self.assertIn("rmsd_ub", rows[0])

    def test_parse_gnina(self):
        rows = parse_log_text(GNINA_TXT)
        self.assertEqual(len(rows), 2)
        self.assertIn("cnn_pose", rows[0])
        self.assertIn("cnn_affinity", rows[0])

    def test_custom_regex_fallback(self):
        # Simulate an unknown engine but vina-like rows; force custom pattern
        unknown_txt = """\
MyNewDock v0.1
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1        -10.0      0.000      0.000
"""
        custom = {
            "vina_row": r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$"
        }
        rows = parse_log_text(unknown_txt, engine=None, regex=custom)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["affinity_kcal_mol"], -10.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
