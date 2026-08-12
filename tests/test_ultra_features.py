import unittest

import numpy as np

from src.ultra_features import (
    RULE_FEATURE_NAMES,
    evidence_flags,
    normalize_text,
    rule_feature_vector,
)


class UltraFeaturesTest(unittest.TestCase):
    def test_normalization_handles_html_unicode_and_nan(self):
        self.assertEqual(normalize_text("<b>БАД&nbsp;Ёж</b>"), "бад еж")
        self.assertEqual(normalize_text(float("nan")), "")
        self.assertEqual(normalize_text(None), "")

    def test_absence_veto_and_refill_exception(self):
        self.assertTrue(evidence_flags("Горелка", "баллон не входит в комплект")["empirical_absence_veto"])
        self.assertFalse(
            evidence_flags(
                "Перезаправляемая горелка",
                "Поставляется без газа, газ заправляется самостоятельно",
            )["empirical_absence_veto"]
        )

    def test_rule_vector_schema_is_stable(self):
        vector = rule_feature_vector("БАД", "Биологически активная добавка к пище", "БАД")
        self.assertEqual(len(vector), len(RULE_FEATURE_NAMES))
        features = dict(zip(RULE_FEATURE_NAMES, vector))
        self.assertEqual(features["strict_bad_marker"], 1.0)
        self.assertTrue(np.isfinite(vector).all())

    def test_strict_positive_family_is_shared_with_evidence(self):
        name = "Хлопушка праздничная артикул ТР 101"
        vector = rule_feature_vector(name, "Праздничный товар", "Легковоспламеняющиеся")
        features = dict(zip(RULE_FEATURE_NAMES, vector))
        self.assertEqual(features["strict_positive_family"], 1.0)
        self.assertTrue(evidence_flags(name, "Праздничный товар")["strict_positive_family"])


if __name__ == "__main__":
    unittest.main()
