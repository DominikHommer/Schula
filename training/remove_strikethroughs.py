import os
import cv2
from ultralytics import YOLO

# ==== Einstellungen ====
model_path = '.../best.pt'  # Pfad zu gelernten Modell
input_folder = '...'  # Ordner mit Original-Bildern
output_folder = '...'  # Zielordner für bereinigte Bilder
confidence_threshold = 0.3  # Mindest-Confidence für Durchstreichungserkennung

# ==== Vorbereitung ====
os.makedirs(output_folder, exist_ok=True)  # Zielordner anlegen falls nicht vorhanden
model = YOLO(model_path)

# ==== Verarbeitung ====
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Bild laden
        img = cv2.imread(input_path)

        # Prediction
        results = model.predict(img, conf=confidence_threshold, verbose=False)

        # Alle Boxen auslesen und ausweißen
        for box in results[0].boxes.xyxy:  # (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = map(int, box)

            # Sicherheits-Check: Rand prüfen
            h, w, _ = img.shape
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Durchstreichung ausweißen
            img[y_min:y_max, x_min:x_max] = (255, 255, 255)

        # Bereinigtes Bild speichern
        cv2.imwrite(output_path, img)
        print(f"Bereinigt und gespeichert: {output_path}")

print("\n Alle Bilder erfolgreich bereinigt!")
