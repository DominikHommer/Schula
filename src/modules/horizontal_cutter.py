import os
from typing_extensions import deprecated
import cv2
import numpy as np
from .module_base import Module

@deprecated("Please use HorizontalCutterLineDetect")
class HorizontalCutter(Module):
    """
    Schneidet ein Bild horizontal in Abschnitte, indem graue Zeilen gefunden 
    und als Trenner verwendet werden.
    Bei aktiviertem Debug-Modus werden ein Debug-Bild mit eingezeichneten 
    Schnittpositionen erstellt.
    """
    def __init__(self, black_thresh=50, gray_min=100, gray_max=200, 
                 gray_tolerance=40, gray_threshold=50, cluster_gap=10, 
                 min_height=30, debug=False, debug_folder="debug/debug_horizontalcutter"):
        super().__init__("horizontal-cutter")

        self.black_thresh = black_thresh
        self.gray_min = gray_min
        self.gray_max = gray_max
        self.gray_tolerance = gray_tolerance
        self.gray_threshold = gray_threshold
        self.cluster_gap = cluster_gap
        self.min_height = min_height
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)
    
    def get_preconditions(self) -> list[str]:
        return ['input']

    def process(self, data: dict) -> list:
        if data.get('red-remover', None) is not None:
            image: np.ndarray = data['red-remover']
        else:
            image: np.ndarray = data['input']

        height, width, _ = image.shape
        gray_rows = []
        for y in range(height):
            row = image[y, :, :]
            max_rgb = np.max(row, axis=1)
            min_rgb = np.min(row, axis=1)
            mean_rgb = np.mean(row, axis=1)
            gray_mask = (max_rgb - min_rgb < self.gray_tolerance) & \
                        (mean_rgb >= self.gray_min) & (mean_rgb <= self.gray_max)
            if np.sum(gray_mask) >= self.gray_threshold:
                gray_rows.append(y)
        
        cut_positions = []
        if gray_rows:
            cluster = [gray_rows[0]]
            for i in range(1, len(gray_rows)):
                if gray_rows[i] - gray_rows[i - 1] <= self.cluster_gap:
                    cluster.append(gray_rows[i])
                else:
                    cut_positions.append(int(cluster[-1] + 5))
                    cluster = [gray_rows[i]]
            cut_positions.append(int(cluster[-1] + 5))
        
        cut_positions = [0] + cut_positions + [height]
        
        if self.debug:
            debug_img = image.copy()
            for pos in cut_positions:
                cv2.line(debug_img, (0, pos), (width, pos), (0, 0, 255), 1)
            debug_path = os.path.join(self.debug_folder, "debug_horizontalcutter.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"[HorizontalCutter] Debug-Bild mit Schnittpositionen gespeichert in: {debug_path}")
        
        sections = []
        for i in range(len(cut_positions) - 1):
            start = cut_positions[i]
            end = cut_positions[i + 1]
            section = image[start:end, :, :]
            if section.shape[0] > self.min_height:
                sections.append(section)
        return sections
