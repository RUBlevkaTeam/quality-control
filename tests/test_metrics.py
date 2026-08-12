import unittest

import numpy as np

from src.metrics import apply_thresholds, competition_metrics, tune_thresholds


class CompetitionMetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        result = competition_metrics(
            [0, 1, 0, 1], [0, 1, 0, 1], ["БАД", "БАД", "ЛВЖ", "ЛВЖ"]
        )
        self.assertEqual(result["mean_f1"], 1.0)

    def test_all_zero_predictions_do_not_fail(self):
        result = competition_metrics(
            [0, 1, 0, 1], [0, 0, 0, 0], ["БАД", "БАД", "ЛВЖ", "ЛВЖ"]
        )
        self.assertEqual(result["mean_f1"], 0.0)

    def test_mean_is_unweighted_between_categories(self):
        result = competition_metrics(
            [1, 0, 1, 1], [1, 0, 0, 0], ["БАД", "БАД", "ЛВЖ", "ЛВЖ"]
        )
        self.assertEqual(result["per_category"]["БАД"]["f1"], 1.0)
        self.assertEqual(result["per_category"]["ЛВЖ"]["f1"], 0.0)
        self.assertEqual(result["mean_f1"], 0.5)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            competition_metrics([0, 1], [0], ["БАД", "БАД"])

    def test_threshold_tuning_and_application(self):
        categories = np.array(["A", "A", "B", "B"])
        thresholds = tune_thresholds(
            [0, 1, 0, 1], [0.2, 0.8, 0.6, 0.7], categories, grid=[0.5, 0.65, 0.75]
        )
        predictions = apply_thresholds([0.2, 0.8, 0.6, 0.7], categories, thresholds)
        np.testing.assert_array_equal(predictions, [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()

