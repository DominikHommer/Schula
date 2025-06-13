import json
import streamlit as st
from pdf2image import convert_from_bytes
from langchain_core.messages import AIMessage, HumanMessage

from libs.language_client import LanguageClient
from pipelines.pdf_processor import PdfProcessorPipeline
from pipelines.llm_extractor import LLMTextExtractorPipeline
from pipelines.student_exam_extractor import StudentExamProcessorPipeline

CV_PIPELINE=False
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
    st.set_page_config(page_title="Helferlein", layout="centered")
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

            if st.button("Verarbeiten", type="primary", on_click = lambda: _set_file_started('solution')):
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

            st.button("Verarbeiten", type="primary", on_click = lambda: _set_file_started('student'))
                
            if uploaded_student_files:
                st.session_state.student_files = uploaded_student_files

        else:
            uploaded_student_files = st.session_state.student_files
            try:
                _pdfProcessorPipeline.process_streamlit(uploaded_student_files, "student")
                #_studenExamProcessorPipeline.process_streamlit(uploaded_solution_files, "student")
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
            
            try:
    
                data = json.loads(st.session_state.student_text)
                full_text = []

                
                if isinstance(data, dict) and 'raw_text' in data:
                    full_text.append(data['raw_text'])
                
                # Handle old, complex format (list of dicts) by iterating through it
                elif isinstance(data, list):
                    for block in data:
                        if isinstance(block, dict):
                            # Prioritize 'raw_text' as it's the most complete
                            if block.get("raw_text"):
                                full_text.append(block.get("raw_text"))
                            # Fallback to collecting text from solutions
                            elif block.get("solutions"):
                                for solution in block.get("solutions"):
                                    if solution.get("solution_text"):
                                        full_text.append(solution.get("solution_text"))
                                    if solution.get("subsolutions"):
                                        for sub in solution.get("subsolutions"):
                                            if sub.get("solution"):
                                                full_text.append(sub.get("solution"))
                
                if full_text:
                    st.write("**Gesamte Transkription der Schülerantwort:**")
                    # Display the collected text directly on the page
                    st.markdown("\n\n".join(full_text))
                else:
                    # If JSON is valid but no text could be extracted
                    st.warning("Dokument hat eine valide Struktur, aber es konnte kein Text extrahiert werden.")
                    st.code(st.session_state.student_text, language="json")

            except Exception:
                # If the text is not a valid JSON  display it directly.
                st.write("**Extrahierter Rohtext (keine JSON-Struktur erkannt):**")
                st.markdown(st.session_state.student_text if st.session_state.student_text else "Kein Text extrahiert.")

        if st.button("Extrahieren"):
            responses = LLMTextExtractorPipeline(llmClient).process_solutions(st.session_state.solution_results[0], st.session_state.student_results[0])
            st.session_state.extraction_started = True
            st.session_state.extraction_text = responses
            st.session_state.step = 4
            st.rerun()
            
    elif st.session_state.step == 4:
        st.subheader("Auswertung der Schülerantworten")
        st.markdown("---")

        try:
            responses: dict = st.session_state.extraction_text

            if not isinstance(responses, dict):
                st.warning("Die Antwort ist kein Dictionary.")
            else:
                for key in sorted(responses.keys(), key=lambda x: int(x)):
                    msg = responses[key]
                    if isinstance(msg, AIMessage):
                        st.markdown(f"### Aufgabe {key}")
                        st.markdown(msg.content)
                        st.markdown("---")
                    else:
                        st.write(f"Aufgabe {key}: Kein AIMessage-Objekt gefunden.")
                        
        except Exception as e:
            st.error(f"Fehler beim Anzeigen der Extraktion: {e}")

            
    else:
        st.subheader("Ungültiger Step")
