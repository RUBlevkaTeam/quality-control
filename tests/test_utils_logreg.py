import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils_logreg import ProductQualityPredictor


class ProductQualityPredictorTest(unittest.TestCase):
    def test_trained_artifact_can_be_loaded_for_submission(self):
        embeddings = np.array(
            [[-2.0], [-1.0], [1.0], [2.0], [-3.0], [-1.5], [1.5], [3.0]]
        )
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        categories = np.array(["A"] * 4 + ["B"] * 4)
        predictor = ProductQualityPredictor().fit(
            embeddings, labels, categories, random_state=42
        )
        predictor.set_thresholds({"A": 0.5, "B": 0.5})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.joblib"
            predictor.save(path)
            loaded = ProductQualityPredictor.load(path)
            probabilities, predictions = loaded.predict(embeddings, categories)

        self.assertEqual(probabilities.shape, (8,))
        np.testing.assert_array_equal(predictions, labels)

    def test_npz_inference_matches_sklearn_with_normalization(self):
        embeddings = np.array(
            [[-4.0, 1.0], [-2.0, 0.5], [2.0, 0.5], [4.0, 1.0]] * 2
        )
        labels = np.array([0, 0, 1, 1] * 2)
        categories = np.array(["A"] * 4 + ["B"] * 4)
        predictor = ProductQualityPredictor().fit(
            embeddings,
            labels,
            categories,
            c={"A": 0.1, "B": 10.0},
            normalize={"A": False, "B": True},
        )
        predictor.set_thresholds({"A": 0.4, "B": 0.6})
        expected_probabilities, expected_predictions = predictor.predict(
            embeddings, categories
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.npz"
            predictor.export_npz(path)
            loaded = ProductQualityPredictor.load(path)
            probabilities, predictions = loaded.predict(embeddings, categories)

        np.testing.assert_allclose(probabilities, expected_probabilities, atol=1e-12)
        np.testing.assert_array_equal(predictions, expected_predictions)
        self.assertIn("normalize=True", loaded.summary())

    def test_unknown_category_is_rejected(self):
        predictor = ProductQualityPredictor().fit(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            np.array(["A", "A"]),
        )

        with self.assertRaisesRegex(ValueError, "unknown categories"):
            predictor.predict(np.array([[0.0]]), np.array(["B"]))


if __name__ == "__main__":
    unittest.main()
