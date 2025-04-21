import os
import numpy as np
import cv2
from .module_base import Module

class RedRemover(Module):
    """
    Entfernt in einem Bild rote Bereiche, indem Pixel, die gewisse Kriterien 
    (rote Dominanz) erfüllen, auf Weiß gesetzt werden.
    Optional kann ein Debug-Bild gespeichert werden.
    """
    def __init__(self, debug=False, debug_folder="debug/debug_redremover"):
        super().__init__("red-remover")

        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)
    
    def get_preconditions(self) -> list[str]:
        return ['input']
    
    def process(self, data: dict) -> np.ndarray:
        image: np.ndarray = data['input']

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