import os

from pipelines.cv_pipeline import CVPipeline
from modules.red_remover import RedRemover
from modules.horizontal_cutter_line_detect import HorizontalCutterLineDetect
from modules.strikethrough_cleaner import StrikeThroughCleaner
from modules.line_cropper import LineCropper
from modules.line_prepare_recognizer import LinePrepareRecognizer
from modules.text_recognizer import TextRecognizer
from modules.text_corrector import TextCorrector

pipeline = CVPipeline()
pipeline.add_stage(RedRemover(debug=True))
pipeline.add_stage(HorizontalCutterLineDetect(debug=True))
pipeline.add_stage(StrikeThroughCleaner(debug=True))
## pipeline.add_stage(LineDenoiser(debug=True)) WARNING: Don't use currently. Model needs to be adapted for varying image sizes
pipeline.add_stage(LineCropper(debug=True))
pipeline.add_stage(LinePrepareRecognizer(debug=True))
pipeline.add_stage(TextRecognizer(debug=True))
pipeline.add_stage(TextCorrector(debug=True))

input_image_path = os.path.join("data", "input", "DA_1_Seite1.png")
input_image_path_2 = os.path.join("data", "input", "DA_2_Seite1.png")
input_image_path_3 = os.path.join("data", "input", "DA_3_Seite1.png")
input_image_path_4 = os.path.join("data", "input", "DA_4_Seite1.png")
output_text_file = os.path.join("data", "output", "recognized_texts.txt")

result = pipeline.run_and_save_text([input_image_path, input_image_path_2, input_image_path_3, input_image_path_4], output_text_file)
