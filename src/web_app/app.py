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

def app_session_init():
    """
    Init app session states
    """
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
    """
    Run streamlit app
    """
    st.set_page_config(page_title="Helferlein")
    st.header("Lehrer :blue[Helferlein]")

    # --- Initialize session state ---
    # Call this early to ensure keys exist before widgets that might use them
    app_session_init()

    # --- Sidebar for File Upload and Reset ---
    with st.sidebar:
        st.subheader("Scans (PNG) und PDFs hochladen")
        uploaded_student_file = st.file_uploader("Klausur-Scan hochladen", type=["pdf"], key="student_uploader")
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
    ###
    # TODO: Enable both jpg and pdf uploads, since pdf's get converted to jpg anyway
    ###
    if uploaded_student_file is not None:
        if CV_PIPELINE :
            _studenExamProcessorPipeline.process_streamlit(uploaded_student_file, "student")
        else:
            _pdfProcessorPipeline.process_streamlit(uploaded_student_file, "student")

    # --- Process Task File (only if changed and not processed) ---
    if uploaded_task_file is not None:
        _pdfProcessorPipeline.process_streamlit(uploaded_task_file, "task")
    
    # --- Process Solution File (only if changed and not processed) ---
    if uploaded_solution_file is not None:
        _pdfProcessorPipeline.process_streamlit(uploaded_solution_file, "solution")

    # --- Display Processed Text and Images ---
    st.subheader("Verarbeitete Dokumente")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.solution_file_processed:
            # Erste Seite als Vorschau anzeigen
            try:
                images = convert_from_bytes(uploaded_solution_file.read(), first_page=1, last_page=1)
                st.image(images[0], caption="Klausur", use_container_width=True)
            except Exception as e:
                st.warning(f"Fehler beim Anzeigen der Vorschau: {e}")

            # Strukturierte Darstellung der Musterlösung
            try:
                data = json.loads(st.session_state.student_text)
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
                st.text_area("Extrahierter Klausurtext (roh)", st.session_state.student_text, height=150, key="student_text_area")

        else:
            st.info("Bitte Klausur-Scan hochladen.")

    with col2:
        if st.session_state.task_file_processed:
            try:
                images = convert_from_bytes(uploaded_task_file.read(), first_page=1, last_page=1)
                st.image(images[0], caption="Aufgabenstellung", use_container_width=True)
            except Exception as e:
                st.warning(f"Fehler beim Anzeigen der Vorschau: {e}")

            try:
                data = json.loads(st.session_state.task_text)
                if isinstance(data, dict):
                    data = [data]

                for _, sheet in enumerate(data, 1):
                    st.write(f"**Titel:** {sheet.get('title', 'Kein Titel')}")
                    st.write(f"**Fach:** {sheet.get('subject', 'Unbekannt')}")
                    st.write(f"**Hinweise:** {sheet.get('instructions', '-')}")

                    if "tasks" in sheet and sheet["tasks"]:
                        st.markdown("#### Aufgaben:")
                        for task in sheet["tasks"]:
                            st.markdown(f"**Aufgabe {task.get('number', '?')}:** {task.get('instruction', '')}")
            except Exception as e:
                st.warning(f"Fehler beim Anzeigen der Aufgabenstruktur: {e}")
                st.text_area("Extrahierter Aufgabentext (roh)", st.session_state.task_text, height=150, key="task_text_area")

        else:
            st.info("Bitte Aufgabenstellung hochladen. (Optional)")


    with col3:
        if st.session_state.solution_file_processed:
            # Erste Seite als Vorschau anzeigen
            try:
                images = convert_from_bytes(uploaded_solution_file.read(), first_page=1, last_page=1)
                st.image(images[0], caption="Musterlösung", use_container_width=True)
            except Exception as e:
                st.warning(f"Fehler beim Anzeigen der Vorschau: {e}")

            # Strukturierte Darstellung der Musterlösung
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

        else:
            st.info("Bitte Musterlösung hochladen.")


    st.divider() # Separator before chat


    # Handle chat logic
    with st.container():
        if st.button(label="Magic", icon=":material/wand_stars:", use_container_width=True) and (st.session_state["student_file_processed"] and st.session_state["solution_file_processed"]):

            responses = LLMTextExtractorPipeline(llmClient)

            with st.chat_message("ai"):

                markdown_parts = []
                for task_id, response_data in sorted(responses.items(), key=lambda item: int(item[0])):
                    # Add a header and a horizontal rule for clear separation
                    markdown_parts.append(f"--- \n ### 📝 Aufgabe {task_id}")

                    if isinstance(response_data, dict):
                        for key, value in response_data.items():
                            # Format the key nicely (e.g., 'feedback' -> 'Feedback') and make it bold
                            formatted_key = key.replace("_", " ").capitalize()
                            markdown_parts.append(f"**{formatted_key}:** {value}")
                    else:
                        # Fallback if a response is just a simple string
                        markdown_parts.append(str(response_data))

                full_markdown = "\n\n".join(markdown_parts)
                st.markdown(full_markdown)
