import os
import torch
import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from .module_base import Module

class TextRecognizer(Module):
    """
    Nutzt das TrOCR-Modell zur Handschriftenerkennung in Bildausschnitten.
    Erkennt den Text und erstellt optional ein Debug-Log, das die 
    erkannten Ergebnisse auflistet.
    """
    _model_cache: dict[str, tuple[TrOCRProcessor, VisionEncoderDecoderModel]] = {}

    def __init__(self, model_name="fhswf/TrOCR_german_handwritten", debug=False, debug_folder="debug/debug_textrecognizer"):
        super().__init__("text-recognizer")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.debug = debug
        self.debug_folder = debug_folder
        self.model = None
        self.processor = None

        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def _warmup(self):
        self.processor, self.model = self._get_processor_and_model(self.model_name)

    @classmethod
    def _get_processor_and_model(cls, model_name: str) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
        """
        Lädt Prozessor und Modell nur einmal pro model_name und cached sie.
        """
        if model_name not in cls._model_cache:
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            cls._model_cache[model_name] = (processor, model)

        return cls._model_cache[model_name]
    
    def get_preconditions(self) -> list[str]:
        return ['line-prepared']
    
    def process(self, data: dict) -> list:
        images: list = data.get('line-prepared', [])

        texts = []
        debug_log = []

        if self.model is None or self.processor is None:
            raise Exception('Missing warmup phase')

        self.model.to(self.device)

        for idx, img in enumerate(images):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(torch.device(self.device))

            with torch.no_grad():
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
