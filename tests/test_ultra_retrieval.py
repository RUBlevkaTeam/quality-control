import unittest

from src.ultra_retrieval import build_retrieval_index, query_retrieval


class UltraRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.digest_a = bytes.fromhex("11" * 32)
        self.digest_b = bytes.fromhex("22" * 32)
        self.index = build_retrieval_index(
            names=["Один товар", "Один товар", "Другой"],
            descriptions=["Одинаковое описание", "Одинаковое описание", "Иное"],
            categories=["БАД", "БАД", "БАД"],
            labels=[1, 1, 0],
            image_hashes=[(self.digest_a,), (self.digest_a,), (self.digest_b,)],
        )

    def test_two_sources_create_tier_a_evidence(self):
        result = query_retrieval(
            self.index,
            category="БАД",
            name="Один товар",
            description="Одинаковое описание",
            image_hashes=(self.digest_a,),
        )
        self.assertEqual(result["tier_a_label"], 1)
        self.assertEqual(result["support"], 2)
        self.assertEqual(result["matched_image_count"], 1)

    def test_conflicting_neighbours_abstain(self):
        index = build_retrieval_index(
            names=["x", "x"],
            descriptions=["same", "same"],
            categories=["БАД", "БАД"],
            labels=[0, 1],
            image_hashes=[(), ()],
        )
        result = query_retrieval(
            index,
            category="БАД",
            name="x",
            description="same",
            image_hashes=(),
        )
        self.assertIsNone(result["hard_label"])
        self.assertIsNone(result["tier_a_label"])
        self.assertEqual(result["purity"], 0.5)

    def test_mixed_third_source_cancels_tier_a(self):
        digest = bytes.fromhex("33" * 32)
        index = build_retrieval_index(
            names=["same", "same", "other"],
            descriptions=["same desc", "same desc", "other desc"],
            categories=["БАД", "БАД", "БАД"],
            labels=[1, 1, 0],
            image_hashes=[(digest,), (), (digest,)],
        )
        result = query_retrieval(
            index,
            category="БАД",
            name="same",
            description="same desc",
            image_hashes=(digest,),
        )
        self.assertIsNone(result["tier_a_label"])


if __name__ == "__main__":
    unittest.main()
