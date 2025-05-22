import streamlit as st
import sys
import tempfile
import os
import time

import ollama
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

### --- Import CV Pipeline as Modules ---
from modules.cv_pipeline.src.modules.cv_pipeline import CVPipeline
from modules.cv_pipeline.src.modules.red_remover import RedRemover
from modules.cv_pipeline.src.modules.horizontal_cutter import HorizontalCutter
from modules.cv_pipeline.src.modules.line_cropper import LineCropper
from modules.cv_pipeline.src.modules.text_recognizer import TextRecognizer

### --- Import LLM Pipeline as Modules ---
# Attention: conflicting dependencies (numpy version) for vector store
from modules.llm_pipeline.llm_manager import LlmManager, initialize_model
# from modules.llm_pipeline.vector_store import retriever

### --- Import File Processors ---
from modules.file_processor.file_manager import png_processor, pdf_processor





def app_session_init():
    # Initialize session state keys if they don't exist
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [AIMessage(content="Hi, lade die Dateien hoch und ich kann dir weiterhelfen :)")]

    files = ["student", "task", "solution"]

    for file in files:
        if file+"_file_processed" not in st.session_state:
            st.session_state[file+"_file_processed"] = False

    for file in files:
        if file+"_text" not in st.session_state:
            st.session_state[file+"_text"] = ""
    
    for file in files:
        if file+"_file_id" not in st.session_state:
            st.session_state[file+"_file_id"] = None # To track if the file changed

    # --- Display existing chat history ---
    # Check if chat_history exists and is iterable
    if "chat_history" in st.session_state and isinstance(st.session_state.chat_history, list):
         # IMPORTANT: Display history *before* potential updates in the main run() logic
         # Only display AI and Human messages here. System messages usually aren't shown.
        for msg in st.session_state.chat_history:
            if isinstance(msg, AIMessage):
                with st.chat_message("ai"):
                    st.write(msg.content)
            elif isinstance(msg, HumanMessage):
                 with st.chat_message("human"):
                    st.write(msg.content)
            # Do not display SystemMessage in the chat UI


def run():
    st.set_page_config(page_title="Helferlein")
    st.header("Lehrer :blue[Helferlein]")

    # --- Initialize session state ---
    # Call this early to ensure keys exist before widgets that might use them
    app_session_init()

    # --- Sidebar for File Upload and Reset ---
    with st.sidebar:
        st.subheader("Scans (PNG) und PDFs hochladen")
        uploaded_student_file = st.file_uploader("Klausur-Scan hochladen", type=["png"], key="student_uploader")
        uploaded_task_file = st.file_uploader("Aufgabenstellung hochladen", type=["pdf"], key="task_uploader")
        uploaded_solution_file = st.file_uploader("Musterlösung hochladen", type=["pdf"], key="solution_uploader")

        if st.button("Chat zurücksetzen (Musterlösung/Aufgabenstellung beibehalten)"):
            # Clear relevant session state parts
            st.session_state.chat_history = [AIMessage(content="Chat zurückgesetzt. Bitte lade Dateien hoch.")]
            st.session_state.student_file_processed = False
            st.session_state.student_text = ""
            st.session_state.student_file_id = None
            st.rerun()# Force rerun to clear chat display and show initial message

        if st.button("Gesamten Chat zurücksetzen", type="primary"):
            st.session_state.chat_history = [AIMessage(content="Chat zurückgesetzt. Bitte lade Dateien hoch.")]
            st.session_state.student_file_processed = False
            st.session_state.task_file_processed = False
            st.session_state.solution_file_processed = False
            st.session_state.student_text = ""
            st.session_state.task_text = ""
            st.session_state.solution_text = ""
            st.session_state.student_file_id = None
            st.session_state.task_file_id = None
            st.session_state.solution_file_id = None
            st.rerun()

    # --- Process Student File (only if changed and not processed) ---
    if uploaded_student_file is not None:
        # Call Png processor
        png_processor(uploaded_student_file, "student")

    # --- Process Task File (only if changed and not processed) ---
    if uploaded_task_file is not None:
        pdf_processor(uploaded_task_file, "task")
    
    # --- Process Solution File (only if changed and not processed) ---
    if uploaded_task_file is not None:
        pdf_processor(uploaded_solution_file,"solution")

    # --- Display Processed Text and Images ---
    st.subheader("Verarbeitete Dokumente")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.student_file_processed:
            st.image(uploaded_student_file, caption="Klausur-Scan", use_container_width=True)
            st.text_area("Extrahierter Klausurtext", st.session_state.student_text, height=150, key="student_text_area")
        else:
            st.info("Bitte Klausur-Scan hochladen.")

    with col2:
        if st.session_state.task_file_processed:
            st.image(uploaded_task_file, caption="Aufgabenstellung", use_container_width=True)
            st.text_area("Extrahierter Aufgabentext", st.session_state.task_text, height=150, key="teacher_text_area")
        else:
             st.info("Bitte Aufgabenstellung hochladen.")
    with col3:
        if st.session_state.solution_file_processed:
            st.image(uploaded_solution_file, caption="Musterlösung", use_container_width=True)
            st.text_area("Extrahierter Lösungstext", st.session_state.solution_text, height=150, key="teacher_text_area")
        else:
             st.info("Bitte Musterlösung hochladen.")


    st.divider() # Separator before chat

    # --- Chat Interface ---
    st.subheader("Chat")

    # Display chat history (app_session_init handles this now, called at the start)

    # Get user input
    prompt = st.chat_input("Schreibe eine Nachricht...")

    # Load LLM (consider caching if slow)
    # @st.cache_resource # Uncomment if model loading is slow
    # def cached_initialize_model():
    #    return initialize_model()
    # model = cached_initialize_model()

    model = initialize_model()
    chat_bot = LlmManager(model)

    # Handle chat logic
    if prompt:
        # Check if both files have been processed before allowing chat queries
        if st.session_state.student_file_processed and st.session_state.task_file_processed and st.session_state.solution_file_processed:
             # Pass the processed text from session state
            chat_bot.get_response(prompt, st.session_state.student_text, st.session_state.teacher_text) # TODO: Adjust for new files
            # After get_response updates session_state['chat_history'], rerun to display the new messages
            st.rerun()
        else:
            st.warning("Bitte laden Sie zuerst alle Dateien (Klausur, Aufgabenstelle, Musterlösung) hoch und warten Sie, bis sie verarbeitet wurden.")


if __name__ == "__main__":
    run()