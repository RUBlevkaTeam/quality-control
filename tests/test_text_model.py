import unittest

import numpy as np
import pandas as pd

from src.text_model import TextQualityModel


class TextModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TextQualityModel.load("text_model.joblib")

    def test_only_flammable_category_uses_rich_char_features(self):
        bad = self.model.category_models["БАД"]
        flame = self.model.category_models["Легковоспламеняющиеся"]

        self.assertEqual(bad["feature_mode"], "word_rules")
        self.assertEqual(set(bad["features"]), {"word"})
        self.assertEqual(flame["feature_mode"], "word_char_title_rules")
        self.assertEqual(set(flame["features"]), {"word", "char", "title"})

    def test_artifact_predicts_both_categories(self):
        frame = pd.DataFrame(
            {
                "name": ["Витаминный комплекс", "Газ для заправки зажигалок"],
                "description": [
                    "Биологически активная добавка к пище.",
                    "Баллон с горючим газом для портативных устройств.",
                ],
                "category": ["БАД", "Легковоспламеняющиеся"],
            }
        )
        probabilities, predictions = self.model.predict(frame)

        self.assertEqual(len(probabilities), 2)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(all(value in (0, 1) for value in predictions))


if __name__ == "__main__":
    unittest.main()
