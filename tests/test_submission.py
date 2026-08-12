import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.validate_submission import validate_submission


class SubmissionValidationTest(unittest.TestCase):
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

