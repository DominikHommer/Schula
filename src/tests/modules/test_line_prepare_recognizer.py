import unittest
import os
import cv2
import numpy as np

from modules.line_prepare_recognizer import LinePrepareRecognizer

TEST_IMAGE_PATH = os.path.join("tests", "fixtures", "line_crop_handwriting.png")

class TestLinePrepareRecognizer(unittest.TestCase):
    def setUp(self):
        self.module = LinePrepareRecognizer(debug=True)
        if not os.path.exists(TEST_IMAGE_PATH):
            self.skipTest(f"Testbild nicht gefunden: {TEST_IMAGE_PATH}")

    def test_process_returns_prepared_images(self):
        image = cv2.imread(TEST_IMAGE_PATH)
        self.assertIsInstance(image, np.ndarray)

        data = {"line-cropper": [image]}
        result = self.module.process(data)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        for processed in result:
            self.assertIsInstance(processed, np.ndarray)
            self.assertEqual(processed.shape[0], image.shape[0])  # gleiche Höhe
            self.assertEqual(processed.shape[2], 3)  # RGB

    def test_debug_output_created(self):
        image = cv2.imread(TEST_IMAGE_PATH)
        self.module.process({"line-cropper": [image]})

        debug_image = os.path.join(self.module.debug_folder, "debug_section_0.png")
        self.assertTrue(os.path.exists(debug_image))
