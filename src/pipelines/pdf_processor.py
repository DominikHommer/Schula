import json
import os
import sys
import streamlit as st

from .llm_pipeline import LLMPipeline
from libs.file_helper import save_temp_file, normalize_paths
from models.parser.assignment_sheet import AssignmentSheet  # aufgabenblatt
from models.parser.model_solution import ModelSolution  # musterlösung/erwartungshorizont
from models.parser.schulbuch_seite import SchulbuchSeite # Schulbuch not needed yet
from models.parser.student_text import StudentText 
from modules.structured_document_parser import StructuredDocumentParser

class PdfProcessorPipeline(LLMPipeline):
    """
    Png Processing Pipeline mit vordefinierten Stages.
    Sollte im Streamlit Kontext verwendet werden
    """
    def __init__(self, input_data: dict | None = None):
        super().__init__(input_data)

    def process_streamlit(self, uploaded_files, file_type):
        attribute_processed = f"{file_type}_file_processed"

        results = []

        paths = []
        for uploaded_file in uploaded_files:
            path = save_temp_file(uploaded_file, prefix=file_type)
            paths.append(path)

        paths = normalize_paths(paths)
        if len(paths) == 0:
            return

        if not st.session_state.get(attribute_processed, False):
            with st.spinner("Verarbeite Datei..."):
                progress_bar = st.progress(0.0)
                status = st.empty()

                def update_progress(current_page: int, total_pages: int):
                    progress_bar.progress(current_page / total_pages)
                    status.text(f"Verarbeite Seite {current_page} von {total_pages}")

                if file_type == "task": # Aufgabenstellung
                    schema = AssignmentSheet 
                    prompt = "Bitte analysiere das Aufgabenblatt und gib eine strukturierte JSON-Darstellung zur Aufgabenstellung zurück."
                elif file_type == "solution": # Musterlösung/Erwartungshorizont
                    schema = ModelSolution
                    prompt = "Bitte analysiere die Musterlösung und gib eine strukturierte JSON-Darstellung zurück."
                elif file_type == "schoolbook": # currently not really needed but maybe in the future!
                    schema = SchulbuchSeite
                    prompt = "Bitte transkribiere die Schulbuchseite und gib eine strukturierte JSON-Darstellung zurück."
                elif file_type == "student":
                    schema = StudentText 
                    prompt = """Bitte transkribiere den gesamten handgeschriebenen Text auf dieser Seite als einen einzigen, zusammenhängenden Block. 
                               Ignoriere dabei rote Schrift des Lehrers, sämtliche Korrekturen, Durchstreichungen oder sonstige Markierungen. 
                               Gib ausschließlich den reinen, unstrukturierten Text des Schülers zurück."""
                else:
                    st.error("Unkown Use Case")
                    return
                
                parser = StructuredDocumentParser(schema_model=schema, prompt=prompt, debug=False, callback = update_progress)
                
                try:
                    final_solution = parser.process({"paths": paths})
                    st.session_state[file_type + "_results"] = final_solution
                    st.session_state[file_type + "_text"] = json.dumps(
                        final_solution.model_dump(), indent=2, ensure_ascii=False
                    )
                    st.session_state[attribute_processed] = True
                    st.success("Dateien erfolgreich analysiert.")
                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung: {e}")
                finally:
                    try:
                        for path in paths:
                            os.remove(path)
                    except OSError as e:
                        st.warning(f"Konnte temporäre Datei nicht löschen, Fehler: {e}")
