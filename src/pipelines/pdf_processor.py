from .llm_pipeline import LLMPipeline

from libs.file_helper import save_temp_file
from models.parser.assignment_sheet import AssignmentSheet  # aufgabenblatt
from models.parser.model_solution import ModelSolution  # musterlösung/erwartungshorizont
from models.parser.schulbuch_seite import SchulbuchSeite # Schulbuch not needed yet
from src.modules.structured_document_parser import StructuredDocumentParser

import json
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
                    if file_type == "task": # Aufgabenstellung
                        schema = AssignmentSheet 
                        prompt = "Bitte analysiere das Aufgabenblatt und gib eine strukturierte JSON-Darstellung zur Aufgabenstellung zurück."
                    elif file_type == "solution": # Musterlösung/Erwartungshorizont
                        schema = ModelSolution
                        prompt = "Bitte analysiere die Musterlösung und gib eine strukturierte JSON-Darstellung zurück."
                    elif file_type == "schoolbook": # currently not really needed but maybe in the future!
                        schema = SchulbuchSeite
                        prompt = "Bitte transkribiere die Schulbuchseite und gib eine strukturierte JSON-Darstellung zurück."
                    elif file_type == "student":
                        schema = ModelSolution
                        prompt = """Bitte transkribiere die Klausur dieses Schülers. Ignoriere hierfür die rote Schrift des Lehrers
                        und sämtliche so gekennzeichnete Verbesserungen, Durschstreichungen oder sonstige Markierungen. Für die Zuordnung der Aufgaben, 
                        achte auf Beschriftungen im Text wie Zahlen wie '1','2' (NICHT zweitens, erstens oder ähnliches!) etc. oder z.B. 'Aufgabe' mit einer anschließenden Zahl (diese kann auch mal vergessen werden, gehe dann chronologisch vor)
                        die gesondert über einen Textparagraphen stehen."""
                    else:
                        st.error("Unkown Use Case")
                        return
                    
                    parser = StructuredDocumentParser(schema_model=schema, prompt=prompt, debug=True, debug_output=f"debug_output_{file_type}.txt") # currently no stage added in constructor!

                    try:
                        results = parser.process({"pdf-path": path})
                        st.session_state[file_type + "_text"] = json.dumps(
                            [r.model_dump() for r in results], indent=2, ensure_ascii=False
                        )
                        st.session_state[attribute_processed] = True
                        st.success("PDF erfolgreich analysiert.")
                    except Exception as e:
                        st.error(f"Fehler bei der Verarbeitung: {e}")
                    finally:
                        try:
                            os.remove(path)
                        except OSError as e:
                            st.warning(f"Konnte temporäre Datei nicht löschen: {path}, Fehler: {e}")
                else:
                    st.error("Konnte PDF nicht speichern.")
            st.rerun()