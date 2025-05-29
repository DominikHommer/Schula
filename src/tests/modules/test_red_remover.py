import unittest
import os
import cv2
import numpy as np
from modules.red_remover import RedRemover

test_image_path = os.path.join("tests", "fixtures", "test_image.png")
result_image_path = os.path.join("tests", "fixtures", "result_redremover.png")
RedRemoverInstance = RedRemover()

# From: https://stackoverflow.com/a/45577032
def image_compare(image_1, image_2) -> bool:
    arr1 = np.array(image_1)
    arr2 = np.array(image_2)
    if arr1.shape != arr2.shape:
        return False
    
    maxdiff = np.max(np.abs(arr1 - arr2))

    return maxdiff == 0

class TestRedRemover(unittest.TestCase):
    def testProcess(self):
        image = cv2.imread(test_image_path)
        res_img = cv2.imread(result_image_path)
        
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(res_img, np.ndarray)

        result = RedRemoverInstance.process(data = dict({
            "input": image,
        }))

        self.assertFalse(result is None)
        self.assertIsInstance(result, np.ndarray)

        self.assertTrue(image_compare(result, res_img))

