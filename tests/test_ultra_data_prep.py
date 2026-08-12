import tempfile
import unittest
from pathlib import Path

from src.utils_data_prep import _find_images_for_id


class UltraDataPrepTest(unittest.TestCase):
    def test_relative_image_root_is_not_duplicated(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            product = root / "42"
            product.mkdir()
            image = product / "front.webp"
            image.touch()
            paths = _find_images_for_id(42, root)
            self.assertEqual(paths, [str(image)])
            self.assertTrue(Path(paths[0]).is_file())


if __name__ == "__main__":
    unittest.main()
