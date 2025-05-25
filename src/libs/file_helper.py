import streamlit as st
import tempfile
import os

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