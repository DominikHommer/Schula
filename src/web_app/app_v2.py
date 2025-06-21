import json
import pandas as pd
import streamlit as st

from libs.language_client import LanguageClient
from pipelines.pdf_processor import PdfProcessorPipeline
from pipelines.llm_extractor import LLMTextExtractorPipeline
from pipelines.student_exam_extractor import StudentExamProcessorPipeline
from models.parser.student_text import StudentText

llmClient = LanguageClient()
_studenExamProcessorPipeline = StudentExamProcessorPipeline()
_pdfProcessorPipeline = PdfProcessorPipeline()

PROGRESS_STEPS = {1: "🟢—⚪—⚪", 2: "⚪—🟢—⚪", 3: "⚪—⚪—🟢"}

def app_session_init():
    """
    Init app session states
    """

    files = ["student", "task", "solution", "extraction"]
    st.session_state.setdefault('step', 1)

    for file in files:
        st.session_state.setdefault(file + "_started", False)

        if file+"_file_processed" not in st.session_state:
            st.session_state[file+"_file_processed"] = False

        if file+"_text" not in st.session_state:
            st.session_state[file+"_text"] = ""

        if file+"_file_id" not in st.session_state:
            st.session_state[file+"_file_id"] = None # To track if the file changed


def show_progress():
    st.markdown(
        f"<div style='text-align:center; font-size:1.5rem;'>{PROGRESS_STEPS[st.session_state.step]}</div>",
        unsafe_allow_html=True,
    )

def run():
    st.set_page_config(page_title="Helferlein", layout="wide")
    app_session_init()
    # Persistenter Header
    st.markdown("<h1 style='text-align:center;'>Helferlein</h1>", unsafe_allow_html=True)
    st.divider()

    def _set_file_started(type: str):
        st.session_state[type + '_started'] = True

    _allowed_types = ["pdf", "jpg", "jpeg", "png"]
    # Schritt 1: Musterlösung hochladen
    if st.session_state.step == 1:
        st.subheader("Schritt 1: Musterlösung hochladen")
        if not st.session_state.solution_started:
            uploaded_solution_files = st.file_uploader(
                "Musterlösung hochladen",
                type = _allowed_types,
                key = "solution_uploader",
                accept_multiple_files = True,
            )

            st.info("Bitte lade ausschließlich die Musterlösung hoch und nicht die Aufgabenstellung aus der Klausur!")

            if st.button("Verarbeiten", type="primary", on_click = lambda: _set_file_started('solution'), disabled=not uploaded_solution_files):
                st.rerun()
            if uploaded_solution_files:
                st.session_state.solution_files = uploaded_solution_files
                
        else:
            
            uploaded_solution_files = st.session_state.solution_files
            try:
                _pdfProcessorPipeline.process_streamlit(uploaded_solution_files, "solution")
                st.session_state.step = 2

                st.rerun()
            except Exception as e:
                st.warning(f"Fehler bei convert_from_bytes : {e}")
        show_progress()

    # Schritt 2: Schulaufgabe hochladen
    elif st.session_state.step == 2:
        st.subheader("Schritt 2: Schulaufgabe hochladen")
        if not st.session_state.student_started:
            uploaded_student_files = st.file_uploader(
                "Klausur-Scan hochladen",
                type = _allowed_types,
                key = "student_uploader",
                accept_multiple_files = True,
                help = "Es kann nur eine Schulaufgabe aufeinmal verarbeitet werden",
            )

            st.button("Verarbeiten", type="primary", on_click = lambda: _set_file_started('student'), disabled=not uploaded_student_files)
                
            if uploaded_student_files:
                st.session_state.student_files = uploaded_student_files

        else:
            uploaded_student_files = st.session_state.student_files
            try:
                _pdfProcessorPipeline.process_streamlit(uploaded_student_files, "student")
                # _studenExamProcessorPipeline.process_streamlit(uploaded_student_files, "student")
                st.session_state.step = 3
                st.rerun()
            except Exception as e:
                st.warning(f"Fehler bei convert_from_bytes : {e}")
        show_progress()

    # Schritt 3: Vorschau & Extraktion
    elif st.session_state.step == 3:
        st.subheader("Schritt 3: Vorschau & Extraktion")
        cols = st.columns(2)
        # Vorschau Musterlösung
        with cols[0]:
            st.markdown("**Musterlösung**")
            try:
                data = json.loads(st.session_state.solution_text)
                if isinstance(data, dict):
                    data = [data]

                for _, block in enumerate(data, 1):
                    if block.get("assignment_title"):
                        st.write(f"**Titel:** {block['assignment_title']}")
                    if block.get("subject"):
                        st.write(f"**Fach:** {block['subject']}")

                    if block.get("solutions"):
                        for idx, solution in enumerate(block["solutions"], 1):
                            if solution.get('number', idx) is not None:
                                st.markdown(f"#### Aufgabe {solution.get('number', idx)}")
                            if solution.get("title"):
                                st.write(f"**Thema:** {solution['title']}")
                            if solution.get("solution_text"):
                                st.write(solution["solution_text"])
                            for sub in solution.get("subsolutions", []):
                                label = sub.get("label")
                                content = sub.get("solution", "")
                                if label:
                                    st.write(f"- **{label}** {content}")
                                else:
                                    st.write(f"- {content}")
            except Exception as e:
                st.warning(f"Fehler beim Anzeigen der Lösung: {e}")
                st.text_area("Extrahierter Lösungstext (roh)", st.session_state.solution_text, height=150, key="teacher_solution_area")
        # Vorschau Schulaufgabe
        with cols[1]:
            st.markdown("**Schulaufgabe**")
            
            if 'student_results' in st.session_state and st.session_state.student_results:
                try:
                    result: StudentText = st.session_state.student_results
                    annotated = [
                        f"{line.text}"
                        for idx, line in enumerate(result.lines, start=1)
                    ]
                    full_text = "\n".join(annotated)

                    st.markdown(full_text)
                    
                except Exception as e:
                    st.error(f"Fehler bei der Anzeige des Schülertextes: {e}")
                    st.json(st.session_state.student_text)
            else:
                st.warning("Kein Schülertext zur Anzeige vorhanden.")

        if st.button("Extrahieren", type="primary"):
                responses = LLMTextExtractorPipeline(llmClient).process_solutions(st.session_state.solution_results, st.session_state.student_results)
                st.session_state.extraction_started = True
                st.session_state.extraction_text = responses
                st.session_state.step = 4
                st.rerun()
            
    elif st.session_state.step == 4:
        st.subheader("Auswertung der Schülerantworten")
        st.markdown("---")

        # Buttons for resetting the states
        with st.sidebar:
            st.markdown("<h2 style='text-align:left; color:#FF4B4B;'>Resets (Achtung)</h2>", unsafe_allow_html=True)
            # student text
            if st.button(label="Schülerklausur", help="Setzt die extrahierte Schülerklausur zurück und ermöglicht das Hochladen neuer Dateien."):
                st.session_state.step = 2
                st.session_state.student_files = None
                st.session_state['student_started'] = False
                st.session_state.extraction_text = None
                st.rerun()
            # solution text
            if st.button(label="Schülerklausur und Musterlösung", help="Setzt die extrahierte Schülerklausur und Musterlösung zurück und ermöglicht das Hochladen neuer Dateien."):
                st.session_state.step = 1
                st.session_state.student_files = None
                st.session_state.solution_files = None
                st.session_state['student_started'] = False
                st.session_state['solution_started'] = False
                st.session_state.extraction_text = None
                st.rerun()
                    
        try:
            responses = st.session_state.extraction_text

            if not isinstance(responses, dict):
                st.warning("Die Antwort ist kein Dictionary.")
            else:
                for key in sorted(responses.keys(), key=lambda x: int(x)):
                    extraction_result = responses[key]

                    st.markdown(f"## Aufgabe {key}")

                    try:
                        table_data = []
                        for item in extraction_result.results:
                            for aspekt in item.Aspekt:
                                table_data.append({
                                    "Teilaufgabe": aspekt.Aspekt,
                                    "Musterlösung": aspekt.Beleg_Musterlösung,
                                    "Schüleraufsatz": aspekt.Beleg_Schüleraufsatz,
                                    "Anmerkungen": aspekt.Kommentar
                                })

                        if table_data:
                            df = pd.DataFrame(table_data)
                            st.dataframe(df, use_container_width=True, hide_index=True, row_height=100)
                        else:
                            st.info("Keine relevanten Aspekte für diese Aufgabe gefunden.")

                    except Exception as e:
                        st.error(f"Fehler beim Darstellen der Daten für Aufgabe {key}: {e}")
                        st.write(extraction_result)

                    st.markdown("---")

        except Exception as e:
            st.error(f"Fehler beim Anzeigen der Extraktion: {e}")
            
    else:
        st.subheader("Ungültiger Step")
