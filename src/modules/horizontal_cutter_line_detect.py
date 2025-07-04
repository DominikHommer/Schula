import os
import math
import statistics
import cv2
import numpy as np
from .module_base import Module

class HorizontalCutterLineDetect(Module):
    """
    Schneidet das Originalbild horizontal anhand erkannter Linien:
    1. Entfernt Blau für die Linien-Erkennung
    2. Erkennt horizontale Linien auf dem blau-bereinigten Bild
    3. Zeichnet Linien und schneidet am Originalbild
    Bei Debug aktiviert, wird ein Debug-Bild mit Schnittlinien gespeichert.
    """
    def __init__(
        self,
        combine_amount_of_lines: int = 1,
        angle_tolerance_deg: float = 0.2,
        cluster_gap: int = 60,
        y_offset: int = 10,
        blur_type: str = 'gaussian',
        blur_ksize: int = 1,
        blue_min: int = 0,
        dominance: int = 5,
        inpaint_radius: int = 5,
        min_height: int = 30,
        debug: bool = False,
        debug_folder: str = "debug/debug_horizontalCutterLineDetect"
    ):
        super().__init__("horizontal-cutter")
        if combine_amount_of_lines > 1:
            raise Exception("TrOCR only works with single line inputs :(")
        
        self.combine_amount_of_lines = combine_amount_of_lines
        self.angle_tol = math.radians(angle_tolerance_deg)
        self.cluster_gap = cluster_gap
        self.y_offset = y_offset
        self.blur_type = blur_type.lower()
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.blue_min = blue_min
        self.dominance = dominance
        self.inpaint_radius = inpaint_radius
        self.min_height = min_height
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['input']

    def _remove_gray(self, img: np.ndarray) -> np.ndarray:
        """
        Wir entfernen graue Schrift, falls existent
        """
        gray_inv = cv2.bitwise_not(img)
        _, writing_mask = cv2.threshold(gray_inv, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        writing_mask = cv2.morphologyEx(writing_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        gray = cv2.inpaint(img, writing_mask, 3, cv2.INPAINT_TELEA)

        if self.debug:
            dbg_path = os.path.join(self.debug_folder, "debug_horizontalCutterLineDetect_grayRemoved.jpg")
            cv2.imwrite(dbg_path, gray)

        return gray

    def _remove_blue(self, img: np.ndarray) -> np.ndarray:
        """
        Entfernt blau-ähnliche Pixel via Inpainting.
        """
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        blue_mask = (b > self.blue_min) & (b > g + self.dominance) & (b > r + self.dominance)
        img[blue_mask] = [255, 255, 255]
        return img
    
    def _cut_out(self, cut_positions: list[float], original: np.ndarray) -> np.ndarray:
        height, _, _ = original.shape
        sections = []

        i = 0
        while i < len(cut_positions) - self.combine_amount_of_lines:
            start = cut_positions[i]
            end = cut_positions[i + self.combine_amount_of_lines] + 15 # To get lower case g,y,etc...

            start = max(start, 0)
            end = min(end, height)

            sec = original[start:end, :, :]
            if sec.shape[0] > self.min_height:
                sections.append(sec)

            i += self.combine_amount_of_lines

        if len(cut_positions) % self.combine_amount_of_lines == 1 and i == len(cut_positions) - self.combine_amount_of_lines:
            start = cut_positions[i] - 15
            end = height
            sec = original[start:end, :, :]

            if sec.shape[0] > self.min_height:
                sections.append(sec)
        
        return sections

    def _get_segments(self, raw) -> list:
        if raw is None:
            print("[HorizontalCutterLineDetect] Keine Linien gefunden.")
            return []

        segments = []
        for seg in raw:
            x1, y1, x2, y2 = seg[0]
            angle = math.atan2(y2 - y1, x2 - x1)
            if abs(angle) <= self.angle_tol or abs(abs(angle) - math.pi) <= self.angle_tol:
                segments.append((x1, y1, x2, y2))
        
        if not segments or len(segments) < 20:
            print("[HorizontalCutterLineDetect] Keine oder zu wenige horizontalen Segmente gefunden.")
            return []
        
        return segments

    def _rotate_image(self, gray) -> np.ndarray:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        median_angle = np.median(angles)

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        _m = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(gray, _m, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        
        if self.debug:
            dbg_path = os.path.join(self.debug_folder, "debug_horizontalCutterLineDetect_rotated.jpg")
            cv2.imwrite(dbg_path, rotated)

        return rotated

    def process(self, data: dict) -> list:
        original: np.ndarray = data['red-remover']
        height, width, _ = original.shape

        temp = original.copy()
        temp = self._remove_blue(temp)

        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        gray = self._remove_gray(gray)

        gray = self._rotate_image(gray)

        if self.blur_type == 'median':
            gray = cv2.medianBlur(gray, self.blur_ksize)
        else:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), sigmaX=0)

        fld = cv2.ximgproc.createFastLineDetector(
            length_threshold=10,
            distance_threshold=1.4142,
            canny_th1=50,
            canny_th2=150,
            canny_aperture_size=3,
            do_merge=True
        )
        segments = self._get_segments(fld.detect(gray))
        if len(segments) == 0:
            return []
        
        mid_ys = sorted((y1 + y2) / 2 for x1, y1, x2, y2 in segments)
        clusters = [[mid_ys[0]]]
        for y in mid_ys[1:]:
            if y - clusters[-1][-1] <= self.cluster_gap:
                clusters[-1].append(y)
            else:
                clusters.append([y])

        heights = []
        cut_positions = [0]
        for i, cluster in enumerate(clusters):
            avg_y = sum(cluster) / len(cluster)
            y_cut = int(max(0, min(height - 1, avg_y + self.y_offset)))
            cut_positions.append(y_cut)
            heights.append(y_cut - cut_positions[i])
        cut_positions.append(height)

        median_cut_height = statistics.median(heights)

        cut_positions = [0]
        i = 0
        for cluster in clusters:
            avg_y = sum(cluster) / len(cluster)
            y_cut = int(max(0, min(height - 1, avg_y + self.y_offset)))

            cut_height = y_cut - cut_positions[i]
            if self.min_height > cut_height:
                continue

            # Cut based on median line height
            while cut_height >= median_cut_height * 1.75:
                new_y_cut = int(cut_positions[i] + median_cut_height)

                i += 1
                cut_positions.append(new_y_cut)

                cut_height = y_cut - cut_positions[i]

            i += 1
            cut_positions.append(y_cut)
            
        cut_positions.append(height)

        if self.debug:
            dbg = original.copy()
            for pos in cut_positions:
                cv2.line(dbg, (0, pos), (width, pos), (0, 0, 255), 1)
            dbg_path = os.path.join(self.debug_folder, "debug_horizontalCutterLineDetect.png")
            cv2.imwrite(dbg_path, dbg)
            print(f"[HorizontalCutterLineDetect] Debug-Bild gespeichert in: {dbg_path}")

        return self._cut_out(cut_positions, original)
