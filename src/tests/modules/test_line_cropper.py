# pylint: skip-file
import unittest
import os
import cv2
import numpy as np

from modules.line_cropper import LineCropper

TEST_INPUT_PATH = os.path.join("tests", "fixtures", "horizontal_cut_section.png")  # Abschnitt mit Text o.Ä.

class TestLineCropper(unittest.TestCase):
    def setUp(self):
        self.line_cropper = LineCropper(debug=False)
        if not os.path.exists(TEST_INPUT_PATH):
            self.skipTest(f"Testbild nicht gefunden: {TEST_INPUT_PATH}")

    def test_process_returns_cropped_images(self):
        image = cv2.imread(TEST_INPUT_PATH)
        self.assertIsInstance(image, np.ndarray)

        data = {"horizontal-cutter": [image]}
        result = self.line_cropper.process(data)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        for cropped in result:
            self.assertIsInstance(cropped, np.ndarray)
            h, w = cropped.shape[:2]
            self.assertGreater(h, 0)
            self.assertGreater(w, 0)
            self.assertNotEqual(cropped.shape[0], image.shape[0])  # ungleiche Höhe
            self.assertNotEqual(cropped.shape[1], image.shape[1])  # ungleiche Breite

    def test_debug_images_created(self):
        self.line_cropper = LineCropper(debug=True)
        image = cv2.imread(TEST_INPUT_PATH)
        self.line_cropper.process({"horizontal-cutter": [image]})

        debug_path = os.path.join(self.line_cropper.debug_folder, "debug_section_0.png")
        self.assertTrue(os.path.exists(debug_path))
