import unittest

import pandas as pd

from src.folds import build_folds


class FoldsTest(unittest.TestCase):
    def test_duplicate_text_stays_in_one_fold(self):
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
        result = build_folds(pd.DataFrame(rows), n_splits=3)
        self.assertEqual(result.groupby("group_key")["fold"].nunique().max(), 1)
        self.assertEqual(set(result["fold"]), {0, 1, 2})

    def test_duplicate_ids_are_rejected(self):
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "name": ["a", "b"],
                "description": ["a", "b"],
                "category": ["A", "A"],
                "label": [0, 1],
            }
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            build_folds(df, n_splits=2)


if __name__ == "__main__":
    unittest.main()

