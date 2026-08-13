import unittest

import numpy as np
import pandas as pd

from src.evaluation import (
    apply_thresholds,
    build_folds,
    competition_metrics,
    tune_thresholds,
)
from src.utils_data_prep import build_group_key, build_text, prepare_dataframe


class EvaluationTest(unittest.TestCase):
    def test_prepare_dataframe_requires_id_and_category(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame({"name": ["товар"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "required columns"):
                prepare_dataframe(path, Path(directory) / "images")

    def test_text_preprocessing(self):
        df = pd.DataFrame(
            {
                "name": ["Товар&nbsp;  один"],
                "category": ["БАД"],
                "description": ["<p>Полезное</p>   описание"],
            }
        )

        self.assertEqual(
            build_text(df).iloc[0],
            "Название: Товар один\nКатегория: БАД\nОписание: Полезное описание",
        )

    def test_group_key_ignores_markup_and_case(self):
        df = pd.DataFrame(
            {
                "name": ["Товар Ё", "товар е"],
                "description": ["<b>Описание</b>", "описание"],
            }
        )

        self.assertEqual(build_group_key(df).nunique(), 1)

    def test_group_key_keeps_name_and_description_separate(self):
        df = pd.DataFrame(
            {
                "name": ["ab", "a"],
                "description": ["c", "bc"],
            }
        )

        self.assertEqual(build_group_key(df).nunique(), 2)

    def test_competition_metrics(self):
        metrics = competition_metrics(
            y_true=[1, 0, 1, 1],
            y_pred=[1, 0, 0, 0],
            categories=["БАД", "БАД", "ЛВЖ", "ЛВЖ"],
        )

        self.assertEqual(metrics["per_category"]["БАД"]["f1"], 1.0)
        self.assertEqual(metrics["per_category"]["ЛВЖ"]["f1"], 0.0)
        self.assertEqual(metrics["mean_f1"], 0.5)

    def test_thresholds(self):
        categories = np.array(["A", "A", "B", "B"])
        probabilities = [0.2, 0.8, 0.6, 0.7]
        thresholds = tune_thresholds(
            [0, 1, 0, 1],
            probabilities,
            categories,
            grid=[0.5, 0.65, 0.75],
        )

        predictions = apply_thresholds(probabilities, categories, thresholds)
        np.testing.assert_array_equal(predictions, [0, 1, 0, 1])

    def test_threshold_tie_uses_middle_of_plateau(self):
        threshold = tune_thresholds(
            [0, 1],
            [0.2, 0.8],
            ["A", "A"],
            grid=[0.5, 0.6, 0.7],
        )["A"]

        self.assertEqual(threshold, 0.6)

    def test_duplicates_stay_in_the_same_fold(self):
        rows = []
        for index in range(24):
            rows.append(
                {
                    "id": index,
                    "name": f"товар {index // 2}",
                    "description": f"описание {index // 2}",
                    "category": "A" if index % 4 < 2 else "B",
                    "label": (index // 4) % 2,
                }
            )

        folds = build_folds(pd.DataFrame(rows), n_splits=3).set_index("id")["fold"]

        for first_id in range(0, 24, 2):
            self.assertEqual(folds[first_id], folds[first_id + 1])


if __name__ == "__main__":
    unittest.main()
