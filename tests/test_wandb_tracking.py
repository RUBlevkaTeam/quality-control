import unittest

from src.wandb_tracking import (
    category_metric_name,
    metrics_payload,
    model_selection_payload,
)


class WandbTrackingTest(unittest.TestCase):
    def test_category_names_are_stable_ascii(self):
        self.assertEqual(category_metric_name("БАД"), "bad")
        self.assertEqual(
            category_metric_name("Легковоспламеняющиеся"), "flammable"
        )
        self.assertEqual(category_metric_name("Some category"), "some_category")

    def test_metrics_are_flattened_for_dashboard(self):
        at_05 = {
            "mean_f1": 0.7,
            "per_category": {},
        }
        tuned = {
            "mean_f1": 0.8,
            "per_category": {
                "БАД": {
                    "f1": 0.9,
                    "precision": 0.8,
                    "recall": 1.0,
                    "count": 100,
                }
            },
        }
        payload = metrics_payload(at_05, tuned, {"БАД": 0.35}, 12.5, 7)
        self.assertEqual(payload["metrics/mean_f1_tuned"], 0.8)
        self.assertEqual(payload["metrics/bad/f1"], 0.9)
        self.assertEqual(payload["thresholds/bad"], 0.35)
        self.assertEqual(payload["runtime/total_seconds"], 12.5)
        self.assertEqual(payload["errors/count"], 7)

    def test_model_selection_is_flattened(self):
        payload = model_selection_payload(
            {"БАД": {"c": 0.1, "normalize": True}},
            [
                {
                    "c": 0.1,
                    "normalize": True,
                    "metrics_tuned": {"mean_f1": 0.8},
                }
            ],
        )

        self.assertEqual(payload["hyperparameters/bad/c"], 0.1)
        self.assertEqual(payload["hyperparameters/bad/normalize"], 1)
        self.assertEqual(payload["search/candidate_0/mean_f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
