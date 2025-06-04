# pylint: skip-file
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

from modules.strikethrough_cleaner import StrikeThroughCleaner  

class TestStrikeThroughCleaner(unittest.TestCase):

    @patch("modules.strikethrough_cleaner.YOLO")
    def test_process_removes_strikethrough(self, mock_yolo_class):
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            MagicMock(boxes=MagicMock(xyxy=np.array([[10, 10, 50, 50]])))
        ]
        mock_yolo_class.return_value = mock_model

        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 127
        data = {"horizontal-cutter": [test_img.copy()]}

        cleaner = StrikeThroughCleaner(debug=False)
        output = cleaner.process(data)

        self.assertEqual(len(output), 1)
        cleaned = output[0]
        self.assertTrue(np.all(cleaned[10:50, 10:50] == 255))

