import unittest
import os
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from modules.line_denoiser import LineDenoiser

TEST_IMAGE_PATH = os.path.join("tests", "fixtures", "denoiser_input.png")

class TestLineDenoiser(unittest.TestCase):
    def setUp(self):
        self.denoiser = LineDenoiser(debug=True)
        if not os.path.exists(TEST_IMAGE_PATH):
            self.skipTest(f"Testbild nicht gefunden: {TEST_IMAGE_PATH}")
    
    @patch("modules.line_denoiser.load_model")
    def test_process_returns_cleaned_images(self, mock_load_model):
        # Fake model mit predict-Mock
        mock_model = MagicMock()
        mock_model.predict.return_value = np.ones((1, 80, 384, 1)) * 255  # Dummy-Ausgabe
        mock_load_model.return_value = mock_model

        image = cv2.imread(TEST_IMAGE_PATH)
        self.assertIsInstance(image, np.ndarray)

        result = self.denoiser.process({"strike-through-cleaner": [image]})

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)

        h, w = result[0].shape[:2]
        self.assertEqual(h, image.shape[0])
        self.assertEqual(w, image.shape[1])

    @patch("modules.line_denoiser.load_model")
    def test_debug_output_created(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.ones((1, 80, 384, 1)) * 255
        mock_load_model.return_value = mock_model

        image = cv2.imread(TEST_IMAGE_PATH)
        self.denoiser.process({"strike-through-cleaner": [image]})

        debug_path = os.path.join(self.denoiser.debug_folder, "debug_section_0.png")
        self.assertTrue(os.path.exists(debug_path))

