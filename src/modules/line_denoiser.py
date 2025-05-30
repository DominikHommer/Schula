import os
from typing_extensions import deprecated
import cv2
import numpy as np
import tensorflow
from .module_base import Module
from tensorflow.keras.models import load_model

def weighted_mse(y_true, y_pred):
    img_height, img_width = y_true.shape[1], y_true.shape[2]
    
    # Erzeugen Sie ein Gitter an Koordinaten
    x = np.linspace(0, 1, img_width)
    y = np.linspace(0, 1, img_height)
    xv, yv = np.meshgrid(x, y)
    
    # Definieren Sie eine Gewichtsfunktion; in diesem Beispiel nehmen wir an,
    # dass das Zentrum (0.5, 0.5) den höchsten Wert hat und die Ränder einen niedrigeren.
    # Mit einer gaußschen Funktion lässt sich das erreichen.
    sigma = 0.3
    mask = np.exp(-((xv - 0.5)**2 + (yv - 0.5)**2) / (2 * sigma**2))
    mask = tensorflow.convert_to_tensor(mask, dtype=tensorflow.float32)
    mask = tensorflow.expand_dims(mask, axis=-1)  # Anpassen der Dimension für Kanäle
    
    # Wenden Sie die Maske auf den Fehler an
    error = tensorflow.square(y_true - y_pred)
    weighted_error = error * mask
    return tensorflow.reduce_mean(weighted_error)

@deprecated("Denoising not optimized for this problem. Do not use!")
class LineDenoiser(Module):
    def __init__(self, debug=False, debug_folder="debug/debug_line_denoiser/"):
        super().__init__("line-denoiser")
        
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['strike-through-cleaner']
    
    def process(self, data: dict) -> list:
        model = load_model("models/denoise/model.keras", custom_objects={"weighted_mse": weighted_mse})
    
        sections: list = data.get('strike-through-cleaner', [])

        cleaned_images = []
        for idx, img in enumerate(sections):
            print(img.shape)
            o_h, o_w, _ = img.shape
            img_resized = cv2.resize(img, (384, 80))
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_resized = np.expand_dims(img_resized, axis=-1)
            img_resized = np.expand_dims(img_resized, axis=0)

            output = model.predict(cv2.bitwise_not(img_resized))
            output = np.squeeze(output)
            output = cv2.resize(output, (o_w, o_h))

            if self.debug:
                debug_path = os.path.join(self.debug_folder, f"debug_section_{idx}.png")
                cv2.imwrite(debug_path, output)
                print(f"[LineDenoiser] Debug-Bild gespeichert: {debug_path}")
            
            cleaned_images.append(output)
        
        return cleaned_images