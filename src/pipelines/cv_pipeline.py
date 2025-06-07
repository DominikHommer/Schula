import os
import re
import cv2
import fleep
from pdf2image import convert_from_path

from .pipeline import Pipeline

class CVPipeline(Pipeline):
    """
    Stellt eine modulare Pipeline zusammen, in der verschiedene Verarbeitungsschritte
    (Klassen mit einer process()-Methode) sequentiell ausgeführt werden.
    """
    def __init__(self, input_data: dict | None = None):
        super().__init__(input_data)

    def _is_pdf(self, file_path) -> bool:
        with open(file_path, "rb") as file:
            info = fleep.get(file.read(128))

            return info.extension_matches("pdf")
        
        return False
    
    def run_and_save_text(self, paths: list[str], output_txt: str|None = None):
        """
        Runs pipeline and saves final text in output file
        """
        path_inputs = []
        for p_i, _input in enumerate(paths):
            if self._is_pdf(_input):
                images = convert_from_path(_input)
            
                for i, img in enumerate(images):
                    path = os.path.join("data", "local", f"image_{p_i}_{i}.png")
                    img.save(path)
                    path_inputs.append(path)
            else:
                path_inputs.append(_input)

        ret = []
        for _input in path_inputs:
            image = cv2.imread(_input)
            if image is None:
                raise ValueError(f"Bild konnte nicht geladen werden: {_input}")
            
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
