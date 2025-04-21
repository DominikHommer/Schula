import os
import numpy as np
import cv2

class RedRemover:
    """
    Entfernt in einem Bild rote Bereiche, indem Pixel, die gewisse Kriterien 
    (rote Dominanz) erfüllen, auf Weiß gesetzt werden.
    Optional kann ein Debug-Bild gespeichert werden.
    """
    def __init__(self, debug=False, debug_folder="debug/debug_redremover"):
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 1]
        red_mask = (r > 120) & (r > g + 20) & (r > b + 20)
        img[red_mask] = [255, 255, 255]
        if self.debug:
            debug_path = os.path.join(self.debug_folder, "debug_redremover.png")
            cv2.imwrite(debug_path, img)
            print(f"[RedRemover] Debug-Bild gespeichert in: {debug_path}")
        return img