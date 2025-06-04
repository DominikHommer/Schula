import unittest
import os
import cv2
import numpy as np
from modules.text_recognizer import TextRecognizer

TEST_IMAGE_PATH = os.path.join("tests", "fixtures", "line_crop_handwriting.png")

class TestTextRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = TextRecognizer(debug=False)
        self.recognizer._warmup()

        if not os.path.exists(TEST_IMAGE_PATH):
            self.skipTest(f"Testbild nicht gefunden: {TEST_IMAGE_PATH}")

    def test_process_text_recognition(self):
        image = cv2.imread(TEST_IMAGE_PATH)
        self.assertIsInstance(image, np.ndarray)

        result = self.recognizer.process({"line-prepared": [image]})

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        for text in result:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text.strip()), 0)
            self.assertEqual('selbstbewusst', text)

    def test_debug_file_created(self):
        self.recognizer = TextRecognizer(debug=True)
        self.recognizer._warmup()

        image = cv2.imread(TEST_IMAGE_PATH)
        self.recognizer.process({"line-prepared": [image]})

        debug_path = os.path.join(self.recognizer.debug_folder, "debug_textrecognizer.txt")
        self.assertTrue(os.path.exists(debug_path))


