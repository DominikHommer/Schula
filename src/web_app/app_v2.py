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

    files = ["student", "task", "solution"]
    for f in files:
        st.session_state.setdefault(f + "_started", False)
        
    if "step" not in st.session_state:
        st.session_state.step = 1

    for file in files:
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

def _set_file_started(type: str):
    st.session_state[type + '_started'] = True
    st.rerun()

def run():
    st.set_page_config(page_title="Helferlein", layout="centered")
    app_session_init()

    # Persistenter Header
    st.markdown("<h1 style='text-align:center;'>Helferlein</h1>", unsafe_allow_html=True)
    st.divider()

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

            st.button("Verarbeiten", type="primary", on_click = lambda: _set_file_started('solution'))
            if uploaded_solution_files:
                st.session_state.solution_files = uploaded_solution_files
                
        else:
            uploaded_solution_files = st.session_state.solution_files
            try:
                _pdfProcessorPipeline.process_streamlit(uploaded_solution_files, "solution")
                st.session_state.step = 2
            except Exception as e:
                st.warning(f"Fehler bei convert_from_bytes : {e}")
            '''            
            if uploaded_solution_file and not st.session_state.solution_file_processed:
        # --- Process Solution File (only if changed and not processed) ---
            _pdfProcessorPipeline.process_streamlit(uploaded_solution_file, "solution")
            '''
        show_progress()

    # Schritt 2: Schulaufgabe hochladen
    elif st.session_state.step == 2:
        st.subheader("Schritt 2: Schulaufgabe hochladen")
        if not st.session_state.solution_started:
            uploaded_student_file = st.file_uploader("Schulaufgabe (PDF) hochladen", type=["pdf"], key="student_uploader")
            if uploaded_student_file:
                st.session_state.student_started = True
                st.session_state.student_file = uploaded_student_file
                st.rerun()
        else:
            uploaded_student_file = st.session_state.student_file
            try:
                _pdfProcessorPipeline.process_streamlit(uploaded_student_file, "student")
                #st.session_state.step = 2
            except Exception as e:
                st.warning(f"Fehler bei convert_from_bytes : {e}")
            '''            
            if uploaded_solution_file and not st.session_state.solution_file_processed:
        # --- Process Solution File (only if changed and not processed) ---
            _pdfProcessorPipeline.process_streamlit(uploaded_solution_file, "solution")
            '''
        show_progress()

    # Schritt 3: Vorschau & Extraktion
    else:
        st.subheader("Schritt 3: Vorschau & Extraktion")
        cols = st.columns(2)
        # Vorschau Musterlösung
        with cols[0]:
            st.markdown("**Musterlösung**")
            try:
                img = convert_from_bytes(st.session_state.solution_file.read(), first_page=1, last_page=1)[0]
                st.image(img, use_container_width=True)
            except:
                st.info("Keine Vorschau verfügbar.")
        # Vorschau Schulaufgabe
        with cols[1]:
            st.markdown("**Schulaufgabe**")
            try:
                img = convert_from_bytes(st.session_state.student_file.read(), first_page=1, last_page=1)[0]
                st.image(img, use_container_width=True)
            except:
                st.info("Keine Vorschau verfügbar.")

        if st.button("Extrahieren"):
            responses = llm_extractor.extract_all()
            st.markdown("---")
            for task_id, data in sorted(responses.items(), key=lambda x: int(x[0])):
                st.markdown(f"### 📝 Aufgabe {task_id}")
                if isinstance(data, dict):
                    for k, v in data.items():
                        st.markdown(f"**{k.replace('_', ' ').capitalize()}:** {v}")
                else:
                    st.write(data)
        show_progress()



