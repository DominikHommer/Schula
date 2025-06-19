import os
import streamlit as st

from modules.red_remover import RedRemover
from modules.horizontal_cutter_line_detect import HorizontalCutterLineDetect
from modules.strikethrough_cleaner import StrikeThroughCleaner
from modules.line_cropper import LineCropper
from modules.line_prepare_recognizer import LinePrepareRecognizer
from modules.text_recognizer import TextRecognizer
from modules.text_corrector import TextCorrector
from libs.file_helper import save_temp_file, normalize_paths

from .cv_pipeline import CVPipeline

from models.parser.student_text import StudentText

class StudentExamProcessorPipeline(CVPipeline):
    """
    Processing Pipeline mit vordefinierten Stages für die Schulaufgabe des Schülers.
    Sollte im Streamlit Kontext verwendet werden
    """
    def __init__(self, input_data: dict | None = None):
        super().__init__(input_data)

        self.add_stage(RedRemover(debug=False))
        self.add_stage(HorizontalCutterLineDetect(debug=False))
        self.add_stage(StrikeThroughCleaner(debug=False))
        self.add_stage(LineCropper(debug=False))
        self.add_stage(LinePrepareRecognizer(debug=True)) # sometimes good sometimes bad :/ 
        self.add_stage(TextRecognizer(debug=False))
        self.add_stage(TextCorrector(debug=False))

    def process_streamlit(self, uploaded_files, file_type):
        """
        Execute pipeline in streamlit context
        """
        attribute_processed = f"{file_type}_file_processed"

        paths = []
        for uploaded_file in uploaded_files:
            path = save_temp_file(uploaded_file, prefix=file_type)
            paths.append(path)

        paths = normalize_paths(paths)
        if len(paths) == 0:
            return
        
        if not st.session_state.get(attribute_processed, False):
            with st.spinner("Verarbeite Scan..."):
                _, full_text = self.run_and_save_text(paths)

                student_text = StudentText(
                    raw_text=full_text
                )

                try:
                    st.session_state[file_type + "_results"] = [student_text]
                    st.session_state[file_type + "_text"] = full_text
                    st.session_state[attribute_processed] = True
                    
                    st.success("Scan erfolgreich analysiert.")
                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung: {e}")
                finally:
                    try:
                        for path in paths:
                            os.remove(path)
                    except OSError as e:
                        st.warning(f"Konnte temporäre Datei nicht löschen, Fehler: {e}")

            st.rerun() # Rerun after processing to update UI correctly
