from ultralytics import YOLO

def predict():
    # Trainiertes Modell laden
    model = YOLO('/path/to/your/best.pt')

    # Prediction durchführen
    model.predict(
        source='/path/tp/testdir',  # Testdaten-Ordner
        save=True,            # Ergebnisse abspeichern
        imgsz=640,            # Bildgröße bei Prediction
        conf=0.25,            # Confidence Threshold (falls nötig anpassen)
        project='/path/to/ausgabeordner',  # Ausgabe-Ordner
        name='predict_schulaufgabe_testdatensatz',  # Name des Predict-Runs
        exist_ok=True         # Überschreiben erlauben falls Ordner existiert
    )

    print("\nPrediction abgeschlossen! Ergebnisse sind gespeichert unter /path/to/ausgabeordner/")

if __name__ == '__main__':
    predict()
