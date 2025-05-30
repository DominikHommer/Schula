import tempfile
import os

def save_temp_file(uploaded_file, prefix: str ="student") -> str | None:
    """
    Saves filed as a temporary file
    Returns full file path
    """
    if uploaded_file is None:
        return None

    # Create a named temporary file that persists for the session
    suffix = os.path.splitext(uploaded_file.name)[1]  # Get extension like .png

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix + "_") as temp_file:
        with open(uploaded_file, "r", encoding='utf-8'):
            # Write the contents of the uploaded file to temp file
            temp_file.write(uploaded_file.read())
            temp_file.flush()
            temp_file.close()

    return temp_file.name  # Return the full path to use elsewhere
