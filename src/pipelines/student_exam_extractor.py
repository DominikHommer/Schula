from modules.red_remover import RedRemover
from modules.horizontal_cutter_line_detect import HorizontalCutterLineDetect
from modules.strikethrough_cleaner import StrikeThroughCleaner
from modules.line_cropper import LineCropper
from modules.line_prepare_recognizer import LinePrepareRecognizer
from modules.text_recognizer import TextRecognizer
from modules.text_corrector import TextCorrector

from .cv_pipeline import CVPipeline

from libs.file_helper import save_temp_file

import os
import streamlit as st

class StudentExamProcessorPipeline(CVPipeline):
    """
    Processing Pipeline mit vordefinierten Stages für die Schulaufgabe des Schülers.
    Sollte im Streamlit Kontext verwendet werden
    """
    def __init__(self, input_data: dict = {}):
        super().__init__(input_data)

        self.add_stage(RedRemover(debug=False))
        self.add_stage(HorizontalCutterLineDetect(debug=False))
        self.add_stage(StrikeThroughCleaner(debug=False))
        self.add_stage(LineCropper(debug=False))
        self.add_stage(LinePrepareRecognizer(debug=True)) # sometimes good sometimes bad :/ 
        self.add_stage(TextRecognizer(debug=False))
        self.add_stage(TextCorrector(debug=False))

    def process_streamlit(self, uploaded_file, file_type):
        attribute_id = f"{file_type}_file_id"
        attribute_processed = f"{file_type}_file_processed"

        # Check if it's a new file or hasn't been processed
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state[attribute_id] != current_file_id:
            st.session_state[attribute_processed] = False # Mark as needing processing
            st.session_state[attribute_id] = current_file_id

        if not st.session_state.get(attribute_processed, False):
            with st.spinner("Verarbeite Scan..."):
                path = save_temp_file(uploaded_file, prefix=file_type)
                if path:
                    # --- Run CV Pipeline ---
                    extracted_text_list = self.run_and_save_text([path]) # Assuming returns list

                    # --- Join the text ---
                    if isinstance(extracted_text_list, list):
                        st.session_state[file_type+"_text"] = " ".join(extracted_text_list)
                    elif isinstance(extracted_text_list, str):
                         st.session_state[file_type+"_text"] = extracted_text_list # If it returns a string
                    else:
                         st.session_state[file_type+"_text"] = "" # Handle unexpected type
                         st.error("Fehler beim Extrahieren des Textes.")

                    st.session_state[file_type+"_file_processed"] = True
                    st.success("Scan verarbeitet.")
                    # Clean up temp file immediately if possible, or manage deletion later
                    try:
                        os.remove(path)
                    except OSError as e:
                        st.warning(f"Konnte temporäre Datei nicht löschen: {path}, Fehler: {e}")
                else:
                    st.error("Konnte Scan nicht speichern.")
            st.rerun() # Rerun after processing to update UI correctly
