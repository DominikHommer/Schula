from .llm_pipeline import LLMPipeline

from libs.file_helper import save_temp_file

import os
import streamlit as st

class PdfProcessorPipeline(LLMPipeline):
    """
    Png Processing Pipeline mit vordefinierten Stages.
    Sollte im Streamlit Kontext verwendet werden
    [WIP] -> Braucht multimodales LLM!
    """
    def __init__(self, input_data: dict = {}):
        super().__init__(input_data)

    def process_streamlit(self, uploaded_file, file_type):
        attribute_id = f"{file_type}_file_id"
        attribute_processed = f"{file_type}_file_processed"

        # Check if it's a new file or hasn't been processed
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state[attribute_id] != current_file_id:
            st.session_state[attribute_processed] = False # Mark as needing processing
            st.session_state[attribute_id] = current_file_id

        if not st.session_state.get(attribute_processed, False):
            with st.spinner("Verarbeite PDF..."):
                path = save_temp_file(uploaded_file, prefix=file_type)
                if path:
                    raise Exception('TODO: Add Lama PDF retriever here')

                    try:
                        os.remove(path)
                    except OSError as e:
                        st.warning(f"Konnte temporäre Datei nicht löschen: {path}, Fehler: {e}")
                else:
                    st.error("Konnte PDF nicht speichern.")
            st.rerun() # Rerun after processing to update UI correctly