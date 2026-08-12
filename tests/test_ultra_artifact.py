import unittest
from pathlib import Path

import pandas as pd

from src.ultra_predictor import load_ultra_artifact, predict_fast


ROOT = Path(__file__).resolve().parents[1]


class UltraArtifactIntegrationTest(unittest.TestCase):
    def test_checked_in_artifact_matches_runtime_schema(self):
        artifact = load_ultra_artifact(ROOT / "ultra_quality.joblib")
        rows = pd.DataFrame(
            {
                "id": ["bad", "fire"],
                "name": ["Биологически активная добавка", "Газовая горелка"],
                "description": ["Добавка к пище", "Баллон не входит в комплект"],
                "category": ["БАД", "Легковоспламеняющиеся"],
                "image_paths": [[], []],
            }
        )
        result = predict_fast(artifact, rows)
        self.assertEqual(len(result["predictions"]), len(rows))
        self.assertEqual(len(result["retrieval"]), len(rows))


if __name__ == "__main__":
    unittest.main()
