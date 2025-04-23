import cv2

class CVPipeline:
    """
    Stellt eine modulare Pipeline zusammen, in der verschiedene Verarbeitungsschritte
    (Klassen mit einer process()-Methode) sequentiell ausgeführt werden.
    """
    def __init__(self):
        self.stages = []
    
    def add_stage(self, stage):
        self.stages.append(stage)
    
    def run(self, input_data):
        data = input_data
        for stage in self.stages:
            data = stage.process(data)
        return data
    
    def run_and_save_text(self, image_path: str, output_txt: str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")
        
        data = image
        for stage in self.stages:
            data = stage.process(data)
        
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            with open(output_txt, "w", encoding="utf-8") as f:
                for text in data:
                    f.write(text + "\n")
            print(f"[CVPipeline] Erkannt Texte gespeichert in: {output_txt}")
        else:
            raise ValueError("Das Endergebnis der Pipeline entspricht nicht der erwarteten Textliste.")
        
    ## added ##
    def run_and_return_text(self, image_path: str):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")
        
        data = image
        for stage in self.stages:
            data = stage.process(data)
        
        return data