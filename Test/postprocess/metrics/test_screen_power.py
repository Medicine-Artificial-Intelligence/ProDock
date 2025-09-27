import unittest
import numpy as np

from prodock.postprocess.metrics.screen_power import ScreenEvaluator


class TestScreenEvaluator(unittest.TestCase):
    """Unit tests for ScreenEvaluator with edge cases and extra checks."""

    def test_auc_pr_ef_bedroc_and_curves(self):
        se = ScreenEvaluator(higher_is_better=False)
        scores = np.array([-10.0, -9.0, -8.0, -7.0, -2.0, -1.0])
        labels = np.array([1, 0, 1, 0, 0, 0], dtype=int)

        # curves
        fpr, tpr, thr = se.roc_curve(scores, labels)
        self.assertTrue(isinstance(fpr, np.ndarray) and isinstance(tpr, np.ndarray))
        self.assertTrue(thr.size > 0)

        prec, rec, pthr = se.precision_recall_curve(scores, labels)
        self.assertTrue(isinstance(prec, np.ndarray) and isinstance(rec, np.ndarray))
        self.assertTrue(pthr.size > 0)

        auc = se.auc_roc(scores, labels)
        self.assertTrue(0.0 <= auc <= 1.0)

        pr = se.pr_auc(scores, labels)
        self.assertTrue(np.isfinite(pr))
        self.assertGreaterEqual(pr, 0.0)

        ef = se.enrichment_factor(scores, labels, fraction=0.5)
        self.assertGreaterEqual(ef, 0.0)

        efk = se.enrichment_at_k(scores, labels, k=3)
        # efk should equal ef when fraction == k/n (n=6 -> fraction=0.5)
        self.assertAlmostEqual(efk, ef, places=6)

        bd = se.bedroc(scores, labels, alpha=20.0)
        if np.isfinite(bd):
            self.assertGreaterEqual(bd, 0.0)
            self.assertLessEqual(bd, 1.0)

    def test_precision_recall_at_k_and_recall_at_k(self):
        se = ScreenEvaluator(higher_is_better=False)
        scores = np.array([-5.0, -4.0, -6.0, -3.0, -1.0])
        labels = np.array([1, 0, 0, 1, 0], dtype=int)  # two actives at idx0 and idx3
        # order after conversion (higher-is-better): [-1, -3, -4, -5, -6] -> indices [4,3,1,0,2]
        # top3 should correspond to indices [4,3,1] -> labels [0,1,0] => precision = 1/3
        prec_top3 = se.precision_at_k(scores, labels, k=3)
        self.assertAlmostEqual(prec_top3, 1.0 / 3.0, places=6)

        recall_top3 = se.recall_at_k(scores, labels, k=3)
        # two actives in dataset; top3 found 1 active -> recall = 1/2
        self.assertAlmostEqual(recall_top3, 0.5, places=6)

    def test_average_precision_alias_and_topn(self):
        se = ScreenEvaluator()
        scores = np.array([-10.0, -9.0, -5.0, -4.0])
        labels = np.array([1, 0, 1, 0])
        ap = se.average_precision(scores, labels)
        pr = se.pr_auc(scores, labels)
        # alias should match
        self.assertAlmostEqual(ap, pr, places=12)

        ligand_ids = ["L1", "L1", "L2", "L2"]
        scores2 = [-5.0, -4.0, -6.0, -3.0]
        labels2 = [1, 0, 0, 1]
        top1 = se.topn_success(ligand_ids, scores2, labels2, n=1)
        self.assertTrue(0.0 <= top1 <= 1.0)

    def test_mcc_search_and_threshold(self):
        se = ScreenEvaluator()
        scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
        labels = np.array([1, 1, 0, 0, 0], dtype=int)
        best_mcc = se.mcc(scores, labels)
        self.assertTrue(np.isfinite(best_mcc))
        # if we pick threshold 0.5, predictions are [1,1,0,0,0] -> perfect classification -> mcc=1
        mcc_at_05 = se.mcc(scores, labels, threshold=0.5)
        self.assertAlmostEqual(mcc_at_05, 1.0, places=6)

    def test_degenerate_inputs_return_nan(self):
        se = ScreenEvaluator()
        scores = np.array([0.1, 0.2, 0.3])
        all_active = np.array([1, 1, 1])
        all_decoy = np.array([0, 0, 0])
        self.assertTrue(np.isnan(se.auc_roc(scores, all_active)))
        self.assertTrue(np.isnan(se.auc_roc(scores, all_decoy)))
        self.assertTrue(np.isnan(se.mcc(scores, all_active)))
        self.assertTrue(np.isnan(se.mcc(scores, all_decoy)))

    def test_input_length_mismatch_raises(self):
        se = ScreenEvaluator()
        with self.assertRaises(ValueError):
            se.roc_curve([1, 2], [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
