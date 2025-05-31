import os
import cv2
import fleep
from pdf2image import convert_from_path
import re

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
    
    def run_and_save_text(self, paths: list[str], output_txt: str|None = None):
        path_inputs = []
        for input in paths:
            if self._is_pdf(input):
                images = convert_from_path(input)
            
                for i, img in enumerate(images):
                    path = os.path.join("data", "local", f"image_{i}.png")
                    img.save(path)
                    path_inputs.append(path)
            else:
                path_inputs.append(input)

        ret = []
        for input in path_inputs:
            image = cv2.imread(input)
            if image is None:
                raise ValueError(f"Bild konnte nicht geladen werden: {input}")
            
            data = self.run(image)

            ret.append(data)

        full_text = ""
        for page in ret:
            if not isinstance(page, list):
                raise ValueError("Das Endergebnis der Pipeline entspricht nicht der erwarteten Textliste.")

            for word in page:
                if not isinstance(word, str):
                    raise ValueError("Das Endergebnis der Pipeline entspricht nicht der erwarteten Textliste.")
                
                full_text += " " + word
            
            full_text += "\n"
        
        # Wrong place but fine for now -> better: have post-process modules
        full_text = re.sub(r'\s*([\n])\s*', r' \1', full_text)
        #full_text = re.sub(r'\s*([.,!?;:()\[\]{}"“”])\s*', r'\1 ', full_text)
        
        if output_txt is not None:
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(full_text)

                print(f"[CVPipeline] Erkannt Texte gespeichert in: {output_txt}")
            
        if len(ret) == 1:
            return ret[0]

        return ret