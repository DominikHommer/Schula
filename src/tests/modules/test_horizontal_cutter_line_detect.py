# pylint: skip-file
import unittest
import os
import numpy as np
import cv2

from modules.horizontal_cutter_line_detect import HorizontalCutterLineDetect

TEST_IMAGE_PATH = os.path.join("tests", "fixtures", "test_image_cut.png")

class TestHorizontalCutterLineDetect(unittest.TestCase):
    def setUp(self):
        self.module = HorizontalCutterLineDetect(debug=False)
        if not os.path.exists(TEST_IMAGE_PATH):
            self.skipTest(f"Testbild nicht gefunden: {TEST_IMAGE_PATH}")
    
    def test_process_returns_sections(self):
        image = cv2.imread(TEST_IMAGE_PATH)
        self.assertIsInstance(image, np.ndarray)

        result = self.module.process({"red-remover": image})
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(len(result), 32)

        for section in result:
            self.assertIsInstance(section, np.ndarray)
            h, w = section.shape[:2]
            self.assertGreater(h, self.module.min_height)
            self.assertEqual(image.shape[1], w)

    def test_debug_image_exists(self):
        self.module = HorizontalCutterLineDetect(debug=True)
        
        image = cv2.imread(TEST_IMAGE_PATH)
        self.module.process({"red-remover": image})
        debug_image_path = os.path.join(self.module.debug_folder, "debug_horizontalCutterLineDetect.png")
        self.assertTrue(os.path.exists(debug_image_path))

