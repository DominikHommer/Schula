import json
import os
import streamlit as st
from .llm_pipeline import LLMPipeline

from libs.file_helper import save_temp_file, normalize_paths
from models.parser.assignment_sheet import AssignmentSheet
from models.parser.model_solution import ModelSolution
from models.parser.schulbuch_seite import SchulbuchSeite
from models.parser.student_text import StudentText 
from modules.structured_document_parser import StructuredDocumentParser

class PdfProcessorPipeline(LLMPipeline):
    """
    Verarbeitet PDFs und implementiert eine zweistufige Logik für Schülerantworten,
    um hohe Textqualität und eine saubere Datenstruktur zu gewährleisten.
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
            with st.spinner("Verarbeite PDF..."):
                progress_bar = st.progress(0.0)
                status = st.empty()

                def update_progress(current_page: int, total_pages: int):
                    progress_bar.progress(current_page / total_pages)
                    status.text(f"Verarbeite Seite {current_page} von {total_pages}")

                schema = None
                prompt = ""

                if file_type == "task":
                    schema = AssignmentSheet 
                    prompt = "Bitte analysiere das Aufgabenblatt und gib eine strukturierte JSON-Darstellung zur Aufgabenstellung zurück."
                elif file_type == "solution":
                    schema = ModelSolution
                    prompt = "Bitte analysiere die Musterlösung und gib eine strukturierte JSON-Darstellung zurück."
                elif file_type == "student":
                    # Direkte Verwendung des einfachen Schemas mit passendem Prompt
                    schema = StudentText
                    prompt = """Bitte transkribiere den gesamten handgeschriebenen Text auf dieser Seite als einen einzigen, zusammenhängenden Block. 
                               Ignoriere dabei rote Schrift des Lehrers, sämtliche Korrekturen, Durchstreichungen oder sonstige Markierungen. 
                               Gib ausschließlich den reinen, unstrukturierten Text des Schülers zurück."""
                else:
                    st.error("Unbekannter Anwendungsfall")
                    return
                
                parser = StructuredDocumentParser(schema_model=schema, prompt=prompt, debug=False, callback=update_progress)

                try:
                    # Das Ergebnis vom Parser kommt direkt im richtigen Format (Liste von StudentText-Objekten).
                    # Eine komplexe Umwandlung ist nicht mehr nötig.
                    final_results = parser.process({"paths": paths})
                    
                    # Speichere das finale Ergebnis direkt in der Session.
                    st.session_state[file_type + "_results"] = final_results
                    st.session_state[file_type + "_text"] = json.dumps(
                        [r.model_dump() for r in final_results], indent=2, ensure_ascii=False
                    )
                    st.session_state[attribute_processed] = True
                    st.success("PDF erfolgreich analysiert.")

                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung: {e}")
                finally:
                    try:
                        for path in paths:
                            os.remove(path)
                    except OSError as e:
                        st.warning(f"Konnte temporäre Datei nicht löschen, Fehler: {e}")
