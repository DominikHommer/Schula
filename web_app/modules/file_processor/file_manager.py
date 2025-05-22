import streamlit as st
import tempfile
import os

### --- Import CV Pipeline as Modules ---
from modules.cv_pipeline.src.modules.cv_pipeline import CVPipeline
from modules.cv_pipeline.src.modules.red_remover import RedRemover
from modules.cv_pipeline.src.modules.horizontal_cutter import HorizontalCutter
from modules.cv_pipeline.src.modules.line_cropper import LineCropper
from modules.cv_pipeline.src.modules.text_recognizer import TextRecognizer

def run_cv_pipeline(image_path):

    cv_pipeline = CVPipeline()  
    cv_pipeline.add_stage(RedRemover(debug=True))   
    cv_pipeline.add_stage(HorizontalCutter(debug=True)) 
    cv_pipeline.add_stage(LineCropper(debug=True))  
    cv_pipeline.add_stage(TextRecognizer(debug=True))  

    print(f"Running CV pipeline on: {image_path}")  

    extracted_text = cv_pipeline.run_and_return_text(image_path)    
    return extracted_text

def save_temp_file(uploaded_file, prefix="student"):
    if uploaded_file is None:
        return None

    # Create a named temporary file that persists for the session
    suffix = os.path.splitext(uploaded_file.name)[1]  # Get extension like .png
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix + "_")
    
    # Write the contents of the uploaded file to temp file
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    temp_file.close()

    return temp_file.name  # Return the full path to use elsewhere


### Modules ###

class pdf_processor:

    def __init__(self):
        pass

    def process(self, uploaded_file, file_type):
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
                    extracted_text_list = run_cv_pipeline(path) # Assuming returns list

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



class png_processor:
    
    def __init__(self):
        pass

    def process(self, uploaded_file, file_type):
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
                    
                    ### TODO: Add Lama PDF retriever here ###

                    try:
                        os.remove(path)
                    except OSError as e:
                        st.warning(f"Konnte temporäre Datei nicht löschen: {path}, Fehler: {e}")
                else:
                    st.error("Konnte PDF nicht speichern.")
            st.rerun() # Rerun after processing to update UI correctly