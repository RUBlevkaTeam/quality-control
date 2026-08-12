import csv
import tempfile
import unittest
from pathlib import Path

from src.output_validation import validate_csv_file, validate_result
from src.ultra_explain import (
    build_comment,
    build_comment_variants,
    choose_comment_variant,
    format_result,
)


class UltraOutputTest(unittest.TestCase):
    def test_every_template_is_valid(self):
        categories = ["БАД", "Легковоспламеняющиеся"]
        for category in categories:
            for prediction in (0, 1):
                comment = build_comment(
                    category=category,
                    prediction=prediction,
                    flags={},
                    reason="text_rules",
                )
                validate_result(format_result(comment, prediction))

    def test_file_round_trip_with_commas_quotes_and_cyrillic(self):
        result = format_result(
            'Комментарий содержит запятую, слово "цитата" и кириллицу для проверки CSV.',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "result"])
                writer.writerow([7, result])
            validate_csv_file(path, [7])

    def test_invalid_reserved_tag_is_rejected(self):
        bad = "<комментарий>" + "а" * 60 + "</комментарий><вердикт>бан"
        with self.assertRaises(ValueError):
            validate_result(bad)

    def test_llm_can_only_select_prevalidated_variant(self):
        variants = build_comment_variants(
            category="БАД",
            prediction=1,
            flags={"bad_marker": True},
            reason="text_rules",
        )
        self.assertEqual(choose_comment_variant("A", variants), variants[0])
        self.assertEqual(choose_comment_variant("B", variants), variants[1])
        for unsafe in (
            "Карточку следует отклонить",
            "A потому что товар нужно заблокировать",
            "b",
            "<вердикт>бан",
        ):
            with self.subTest(unsafe=unsafe):
                self.assertEqual(choose_comment_variant(unsafe, variants), variants[0])
        for variant in variants:
            validate_result(format_result(variant, 1))


if __name__ == "__main__":
    unittest.main()
