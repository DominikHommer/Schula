import os
import cv2
from .module_base import Module
from ultralytics import YOLO

class StrikeThroughCleaner(Module):
    """
    Entfernt in jedem Bildausschnitt den durchgestrichene Stellen.
    Optional wird ein Debug-Bild (mit eingezeichneter Bounding-Box) gespeichert.
    """
    def __init__(self, debug=False, debug_folder="debug/debug_strikethrough_cleaner"):
        super().__init__("strike-through-cleaner")
        
        self.confidence_threshold = 0.3
        self.model = YOLO("models/strikethrough/best.pt")
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['horizontal-cutter']
    
    def process(self, data: dict) -> list:
        sections: list = data.get('horizontal-cutter')
    
        cleaned_images = []
        for idx, img in enumerate(sections):
            results = self.model.predict(img, conf=self.confidence_threshold, verbose=self.debug)

            debug_img = img.copy()
            for box in results[0].boxes.xyxy:  # (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = map(int, box)

                # Sicherheits-Check: Rand prüfen
                h, w, _ = img.shape
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                if self.debug:
                    cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
                # Durchstreichung ausweißen
                img[y_min:y_max, x_min:x_max] = (255, 255, 255)

            if self.debug:
                debug_path_1 = os.path.join(self.debug_folder, f"debug_section_detected_{idx}.png")
                debug_path_2 = os.path.join(self.debug_folder, f"debug_section_cleaned_{idx}.png")
                cv2.imwrite(debug_path_1, debug_img)
                cv2.imwrite(debug_path_2, img)
                print(f"[StrikeThroughCleaner] Debug-Bild gespeichert: {debug_path_1}")

            cleaned_images.append(img)
        return cleaned_images