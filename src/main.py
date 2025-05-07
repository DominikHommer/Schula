import os

from pipelines.cv_pipeline import CVPipeline
from modules.red_remover import RedRemover
from modules.horizontal_cutter import HorizontalCutter
from modules.horizontal_cutter_line_detect import HorizontalCutterLineDetect
from modules.strikethrough_cleaner import StrikeThroughCleaner
from modules.line_denoiser import LineDenoiser

from modules.line_cropper import LineCropper
from modules.text_recognizer import TextRecognizer


pipeline = CVPipeline()
pipeline.add_stage(RedRemover(debug=True))
pipeline.add_stage(HorizontalCutterLineDetect(debug=True))
pipeline.add_stage(StrikeThroughCleaner(debug=True))
# pipeline.add_stage(LineDenoiser(debug=True)) WARNING: Don't use currently. Model needs to be adapted for varying image sizes
pipeline.add_stage(LineCropper(debug=True))
pipeline.add_stage(TextRecognizer(debug=True))

input_image_path = os.path.join("data", "input", "image_pdf.pdf")
output_text_file = os.path.join("data", "output", "recognized_texts.txt")

result = pipeline.run_and_save_text(input_image_path, output_text_file)

# Just a quick test
#new_pipeline = CVPipeline(input_data=result)
#new_pipeline.add_stage(TextRecognizer(debug=True))
#print(new_pipeline.run())
