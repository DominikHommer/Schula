import os
import numpy as np
import cv2
from .module_base import Module

class RedRemover(Module):
    """
    Entfernt in einem Bild rote Bereiche via Inpainting:
    Pixel, bei denen der Rot-Kanal größer ist als Grün und Blau
    (mit optionaler Dominanz und Schwelle), werden aufgefüllt.
    Optional kann ein Debug-Bild gespeichert werden.
    """
    def __init__(self,
                 thr: int = 0,
                 dom: int = 0,
                 inpaint_radius: int = 5,
                 debug: bool = False,
                 debug_folder: str = "debug/debug_redremover"):
        super().__init__("red-remover")
        self.thr = thr
        self.dom = dom
        self.inpaint_radius = inpaint_radius
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['input']

    def process(self, data: dict) -> np.ndarray:
        image: np.ndarray = data['input']
        img = image.copy()

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        mask = ((r > self.thr) & (r > g + self.dom) & (r > b + self.dom)).astype(np.uint8) * 255

        result = cv2.inpaint(img, mask, self.inpaint_radius, cv2.INPAINT_TELEA)

        if self.debug:
            debug_path = os.path.join(self.debug_folder, "debug_redremover.png")
            cv2.imwrite(debug_path, result)
            print(f"[RedRemover] Debug-Bild gespeichert in: {debug_path}")
        return result

