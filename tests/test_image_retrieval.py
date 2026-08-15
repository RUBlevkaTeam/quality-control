import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.image_retrieval import HammingMatcher, fingerprint_file, retrieval_decision


class ImageRetrievalTest(unittest.TestCase):
    def test_dhash_is_stable_and_exact_hash_changes_with_bytes(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            pixels = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
            Image.fromarray(pixels, mode="L").save(first)
            Image.fromarray(pixels, mode="L").save(second, optimize=True)
            first_result = fingerprint_file(first)
            second_result = fingerprint_file(second)
            self.assertIsNotNone(first_result)
            self.assertIsNotNone(second_result)
            self.assertTrue(np.array_equal(first_result[1], second_result[1]))

    def test_numpy_topk_orders_by_distance_then_index(self):
        matcher = HammingMatcher(allow_native=False)
        train = np.zeros((4, 4), dtype=np.uint64)
        train[1, 0] = 1
        train[2, 0] = 3
        train[3, 0] = 7
        query = np.zeros((1, 4), dtype=np.uint64)
        indices, distances = matcher.topk(train, query, max_distance=2, top_k=4)
        self.assertEqual(indices[0].tolist(), [0, 1, 2, -1])
        self.assertEqual(distances[0, :3].tolist(), [0, 1, 2])

    def test_native_and_numpy_match_when_native_is_available(self):
        native = HammingMatcher(allow_native=True)
        if native.backend != "cpp":
            self.skipTest("native library is unavailable on this host")
        rng = np.random.default_rng(2026)
        train = rng.integers(0, np.iinfo(np.uint64).max, size=(100, 4), dtype=np.uint64)
        query = train[[3, 17, 44]].copy()
        query[1, 0] ^= np.uint64(7)
        expected = HammingMatcher(allow_native=False).topk(
            train, query, max_distance=6, top_k=8
        )
        actual = native.topk(train, query, max_distance=6, top_k=8)
        self.assertTrue(np.array_equal(actual[0], expected[0]))
        self.assertTrue(np.array_equal(actual[1], expected[1]))

    def test_retrieval_policy_is_asymmetric_and_conflict_safe(self):
        exact_positive = {
            "exact_products": ((1, 1, None, True), (2, 1, None, False)),
            "perceptual_products": (),
        }
        self.assertEqual(
            retrieval_decision("БАД", exact_positive),
            (1, "exact_sha256_multi_name"),
        )
        weak_exact = {"exact_products": ((1, 1, None, True),), "perceptual_products": ()}
        self.assertEqual(
            retrieval_decision("БАД", weak_exact),
            (None, "exact_sha256_observe"),
        )

        dhash_positive = {
            "exact_products": (),
            "perceptual_products": ((1, 1, 2, True),),
        }
        self.assertEqual(
            retrieval_decision("БАД", dhash_positive),
            (1, "dhash_name_positive"),
        )
        self.assertEqual(
            retrieval_decision("Легковоспламеняющиеся", dhash_positive),
            (1, "dhash_name_positive"),
        )

        conflict = {
            "exact_products": ((1, 0, None, False),),
            "perceptual_products": ((2, 1, 1, True),),
        }
        self.assertEqual(retrieval_decision("Легковоспламеняющиеся", conflict)[0], None)

        mixed_exact = {
            "exact_products": ((1, 0, None, False), (2, 1, None, False)),
            "perceptual_products": ((3, 1, 1, True),),
        }
        self.assertEqual(
            retrieval_decision("Легковоспламеняющиеся", mixed_exact),
            (None, "exact_sha256_conflict"),
        )


if __name__ == "__main__":
    unittest.main()
