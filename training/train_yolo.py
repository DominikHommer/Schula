from ultralytics import YOLO

def train():
    # → ENTWEDER Standard-YOLO verwenden:
    # model = YOLO('yolov8m.pt')

    # → ODER ein bestehendes Modell weitertrainieren:
    model = YOLO('/path/to/your/best.pt')

    # Training starten
    model.train(
        data='/path/to/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        project='/path/to/runs', # Speicherort für Ergebnisse
        name='experiment_name',
        pretrained=True,
        optimizer='SGD',
        verbose=True,
        patience=20
    )

    print("\nTraining abgeschlossen!")

if __name__ == '__main__':
    train()
