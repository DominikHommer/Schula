import os
import cv2
import numpy as np
from .module_base import Module

class LinePrepareRecognizer(Module):
    def __init__(self, debug=False, debug_folder="debug/debug_lineprepared"):
        super().__init__("line-prepared")
        
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['line-cropper']
    
    def grayscale_to_blue(self, image_gray, text_color=(180, 0, 0)):
        threshold = 180
        color_img = np.ones((*image_gray.shape, 3), dtype=np.uint8) * 255
        text_mask = image_gray < threshold

        for c in range(3):
            color_img[:, :, c][text_mask] = text_color[c]

        return color_img

    def _cut_out_word_pairs(self, img):
        white_projection = np.sum(img == 255, axis=0)

        threshold = img.shape[0] - 2
        space_columns = np.where(white_projection >= threshold)[0]

        cut_positions = []
        min_space_width = 30
        last_cut = -min_space_width

        for idx in space_columns:
            if idx - last_cut >= min_space_width:
                cut_positions.append(idx)
                last_cut = idx

        segments = []
        start = 0
        for i in range(4, len(cut_positions), 4):
            end = cut_positions[i]
            segment = img[:, start:end, :]
            if segment.shape[1] > 200:
                segments.append(segment)
            start = end

        if start < img.shape[1] - 1:
            segments.append(img[:, start:, :])

        return segments

    def process(self, data: dict) -> list:
        images: list = data.get('line-cropper', [])

        cropped_images = []
        for idx, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, h=20)

            alpha = 1.5
            beta = 0.5
            contrast_enhanced = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)

            sharpen_kernel = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
            sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)
            sharpened = self.grayscale_to_blue(sharpened)

            if self.debug:
                debug_path = os.path.join(self.debug_folder, f"debug_section_{idx}.png")
                cv2.imwrite(debug_path, sharpened)
                print(f"[LinePrepareRecognizer] Debug-Bild gespeichert: {debug_path}")

            cuts = self._cut_out_word_pairs(sharpened)
            for x_idx, x in enumerate(cuts):
                debug_path = os.path.join(self.debug_folder, f"debug_section_cut_{idx}_{x_idx}.png")
                cv2.imwrite(debug_path, x)

            cropped_images.append(sharpened)

        return cropped_images