import os
import cv2
import numpy as np

class LineCropper:
    """
    Schneidet in jedem Bildausschnitt den relevanten Bereich zu, 
    indem die Konturen (z.B. des Textes) ermittelt werden.
    Optional wird ein Debug-Bild (mit eingezeichneten Konturen und 
    Bounding-Box) gespeichert.
    """
    def __init__(self, padding=10, debug=False, debug_folder="debug/debug_linecropper"):
        self.padding = padding
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def process(self, sections: list) -> list:
        cropped_images = []
        for idx, img in enumerate(sections):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                print(f"[LineCropper] Keine Konturen gefunden in Abschnitt {idx}")
                continue

            all_contours = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_contours)
            height, width = img.shape[:2]
            x1 = max(x - self.padding, 0)
            y1 = max(y - self.padding, 0)
            x2 = min(x + w + self.padding, width)
            y2 = min(y + h + self.padding, height)
            cropped = img[y1:y2, x1:x2]

            if self.debug:
                debug_img = img.copy()
                cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                debug_path = os.path.join(self.debug_folder, f"debug_section_{idx}.png")
                cv2.imwrite(debug_path, debug_img)
                print(f"[LineCropper] Debug-Bild gespeichert: {debug_path}")

            cropped_images.append(cropped)
        return cropped_images