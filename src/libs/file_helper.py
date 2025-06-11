import tempfile
import os
import fleep
from pdf2image import convert_from_path

def is_pdf(file_path) -> bool:
    with open(file_path, "rb") as file:
        info = fleep.get(file.read(128))

        return info.extension_matches("pdf")
    
    return False

def normalize_paths(paths):
    path_inputs = []
    for p_i, _input in enumerate(paths):
        if is_pdf(_input):
            images = convert_from_path(_input)
        
            for i, img in enumerate(images):
                path = os.path.join("data", "local", f"image_{p_i}_{i}.png")
                img.save(path)
                path_inputs.append(path)
        else:
            path_inputs.append(_input)

    return path_inputs

def save_temp_file(uploaded_file, prefix: str ="student") -> str | None:
    """
    Saves file as a temporary file
    Returns full file path
    """
    if uploaded_file is None:
        return None

    # Create a named temporary file that persists for the session
    suffix = os.path.splitext(uploaded_file.name)[1]  # Get extension like .png

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix + "_") as temp_file:
        # Write the contents of the uploaded file to temp file
        temp_file.write(uploaded_file.read())
        temp_file.flush()
        temp_file.close()

    return temp_file.name  # Return the full path to use elsewhere
