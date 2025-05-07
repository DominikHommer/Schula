import os
import cv2
import fleep
from pdf2image import convert_from_path

from .pipeline import Pipeline

class CVPipeline(Pipeline):
    """
    Stellt eine modulare Pipeline zusammen, in der verschiedene Verarbeitungsschritte
    (Klassen mit einer process()-Methode) sequentiell ausgeführt werden.
    """
    def __init__(self, input_data: dict = {}):
        super().__init__(input_data)

    def _is_pdf(self, file_path) -> bool:
        with open(file_path, "rb") as file:
            info = fleep.get(file.read(128))

            return info.extension_matches("pdf")
        
        return False
    
    def run_and_save_text(self, image_path: str, output_txt: str):
        inputs = [image_path]
        if self._is_pdf(image_path):
            images = convert_from_path(image_path)
            
            inputs = []
            for i, img in enumerate(images):
                path = f"{os.path.dirname(image_path)}/image_{i}.png"
                img.save(path)
                inputs.append(path)

        ret = []
        for input in inputs:
            image = cv2.imread(input)
            if image is None:
                raise ValueError(f"Bild konnte nicht geladen werden: {input}")
            
            data = self.run(image)
            
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                with open(output_txt, "w", encoding="utf-8") as f:
                    for text in data:
                        f.write(text + "\n")
                print(f"[CVPipeline] Erkannt Texte gespeichert in: {output_txt}")
            else:
                raise ValueError("Das Endergebnis der Pipeline entspricht nicht der erwarteten Textliste.")
            
            ret.append(data)
        
        if len(ret) == 1:
            return ret[0]

        return ret