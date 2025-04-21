import os

from pipelines.cv_pipeline import CVPipeline
from modules.red_remover import RedRemover
from modules.horizontal_cutter import HorizontalCutter
from modules.line_cropper import LineCropper
from modules.text_recognizer import TextRecognizer


pipeline = CVPipeline()
pipeline.add_stage(RedRemover(debug=True))
pipeline.add_stage(HorizontalCutter(debug=True))
pipeline.add_stage(LineCropper(debug=True))
pipeline.add_stage(TextRecognizer(debug=True))

input_image_path = os.path.join("data", "input", "image.png")
output_text_file = os.path.join("data", "output", "recognized_texts.txt")

result = pipeline.run_and_save_text(input_image_path, output_text_file)

# Just a quick test
#new_pipeline = CVPipeline(input_data=result)
#new_pipeline.add_stage(TextRecognizer(debug=True))
#print(new_pipeline.run())
