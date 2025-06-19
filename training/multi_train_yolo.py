from ultralytics import YOLO

def multi_train():
    # Verschiedene Modelle und Settings, die du ausprobieren willst
    experiments = [
        {'model': 'yolov8m.pt', 'imgsz': 640, 'batch': 16},
        {'model': 'yolov8m.pt', 'imgsz': 1024, 'batch': 8},
        {'model': 'yolov8l.pt', 'imgsz': 640, 'batch': 8},
        {'model': 'yolov8l.pt', 'imgsz': 1024, 'batch': 4},
    ]

    for idx, exp in enumerate(experiments, start=1):
        print(f"\nStarte Experiment {idx}: Modell={exp['model']}, imgsz={exp['imgsz']}, batch={exp['batch']}\n")

        model = YOLO(exp['model'])

        model.train(
            data='/path/to/data.yaml',
            epochs=100,
            imgsz=exp['imgsz'],
            batch=exp['batch'],
            project='/path/to/runs',
            name=f"train_durchstreichung_exp{idx}",
            pretrained=True,
            optimizer='SGD',
            verbose=True,
            patience=20
        )

    print("\n Alle Experimente abgeschlossen!")

if __name__ == '__main__':
    multi_train()
