import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from run import resolve_classifier_path, resolve_models_dir
from scripts.validate_submission import validate_submission


class SubmissionValidationTest(unittest.TestCase):
    def test_models_path_precedence(self):
        with patch.dict("os.environ", {"SHARED_MODELS_PATH": "/mounted/models"}):
            self.assertEqual(resolve_models_dir(), Path("/mounted/models"))
            self.assertEqual(
                resolve_models_dir(Path("/explicit/models")),
                Path("/explicit/models"),
            )

    def test_classifier_falls_back_to_existing_joblib(self):
        self.assertTrue(resolve_classifier_path().is_file())

    def test_valid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            output_path = root / "output.csv"
            pd.DataFrame({"id": [1, 2]}).to_csv(input_path, index=False)
            comment = "Это достаточно подробное и конкретное объяснение решения модели."
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "result": [
                        f"<комментарий>{comment}<вердикт>бан",
                        f"<комментарий>{comment}<вердикт>не бан",
                    ],
                }
            ).to_csv(output_path, index=False)
            validate_submission(input_path, output_path)

    def test_short_comment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            output_path = root / "output.csv"
            pd.DataFrame({"id": [1]}).to_csv(input_path, index=False)
            pd.DataFrame(
                {"id": [1], "result": ["<комментарий>коротко<вердикт>бан"]}
            ).to_csv(output_path, index=False)
            with self.assertRaisesRegex(ValueError, "comment length"):
                validate_submission(input_path, output_path)


if __name__ == "__main__":
    unittest.main()
