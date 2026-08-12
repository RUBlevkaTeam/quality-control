import unittest

import numpy as np

from src.ultra_model import best_f1_threshold, combine_text_and_retrieval


class UltraModelTest(unittest.TestCase):
    def test_threshold_search_checks_every_unique_score(self):
        # More than 1001 distinct probabilities guards against the old sampled
        # quantile search.  The exact routine must equal brute force.
        probabilities = np.linspace(0.001, 0.999, 1503)
        labels = np.zeros(len(probabilities), dtype=np.int8)
        labels[-17:] = 1
        labels[-31] = 1
        threshold, score = best_f1_threshold(labels, probabilities)

        brute = []
        for candidate in np.unique(probabilities):
            prediction = probabilities >= candidate
            tp = int(np.sum((labels == 1) & prediction))
            fp = int(np.sum((labels == 0) & prediction))
            fn = int(np.sum((labels == 1) & ~prediction))
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            brute.append((f1, candidate))
        expected_score, expected_threshold = max(brute)
        self.assertAlmostEqual(score, expected_score)
        self.assertAlmostEqual(threshold, expected_threshold)

    def test_tier_modes_only_apply_allowed_label(self):
        evidence = [{"tier_a_label": 1}, {"tier_a_label": 0}]
        base = np.asarray([0.2, 0.8])
        negative = combine_text_and_retrieval(base, evidence, mode="tier_a_negative")
        positive = combine_text_and_retrieval(base, evidence, mode="tier_a_positive")
        self.assertAlmostEqual(negative[0], base[0])
        self.assertLess(negative[1], 0.001)
        self.assertGreater(positive[0], 0.999)
        self.assertAlmostEqual(positive[1], base[1])


if __name__ == "__main__":
    unittest.main()
