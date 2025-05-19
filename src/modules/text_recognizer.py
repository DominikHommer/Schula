import torch
import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from .module_base import Module
import os

class TextRecognizer(Module):
    """
    Nutzt das TrOCR-Modell zur Handschriftenerkennung in Bildausschnitten.
    Erkennt den Text und erstellt optional ein Debug-Log, das die 
    erkannten Ergebnisse auflistet.
    """
    def __init__(self, model_name="fhswf/TrOCR_german_handwritten", debug=False, debug_folder="debug/debug_textrecognizer"):
        super().__init__("text-recognizer")

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(torch.device("mps"))
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)
    
    def get_preconditions(self) -> list[str]:
        return ['line-cropper']
    
    def process(self, data: dict) -> list:
        #images: list = data.get('line-cropper', [])
        images: list = data.get('line-prepared', [])

        texts = []
        debug_log = []

        for idx, img in enumerate(images):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(torch.device("mps"))
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            texts.append(text)
            print(f"[TextRecognizer] Erkannt für Bild {idx}: {text}")
            if self.debug:
                debug_log.append(f"Bild {idx}: {text}")

        if self.debug:
            debug_path = os.path.join(self.debug_folder, "debug_textrecognizer.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write("\n".join(debug_log))
            print(f"[TextRecognizer] Debug-Log gespeichert in: {debug_path}")

        return texts