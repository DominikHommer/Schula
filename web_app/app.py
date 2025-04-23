import streamlit as st
import sys
import tempfile
import os

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


def app_session_init():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [AIMessage("Hi, lade die Dateien hoch und ich kann dir weiterhelfen :)")]

    # Maybe add model selection back later
    # if "selected_model" not in st.session_state:
    #     st.session_state["selected_model"] = get_models()[0]

    # write chat history back
    chat_history = st.session_state["chat_history"]
    for history in chat_history:
        if isinstance(history, AIMessage):
            st.chat_message("ai").write(history.content)

        if isinstance(history, HumanMessage):
            st.chat_message("human").write(history.content)


### Maybe add model selection field back later
# def get_models():
#     models = ollama.list()
#     if not models:
#         print("No models found, please visit: https://ollama.dev/models")
#         sys.exit(1)

#     models_list = []
#     for model in models["models"]:
#         models_list.append(model["name"])

#     return models_list


def run_cv_pipeline(image_path):
    cv_pipeline = CVPipeline()
    cv_pipeline.add_stage(RedRemover(debug=True))
    cv_pipeline.add_stage(HorizontalCutter(debug=True))
    cv_pipeline.add_stage(LineCropper(debug=True))
    cv_pipeline.add_stage(TextRecognizer(debug=True))

    print(f"Running CV pipeline on: {image_path}")
    extracted_text = cv_pipeline.run_and_return_text(image_path)

    return extracted_text


def save_temp_file(uploaded_file, prefix="student"):
    if uploaded_file is None:
        return None

    # Create a named temporary file that persists for the session
    suffix = os.path.splitext(uploaded_file.name)[1]  # Get extension like .png
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix + "_")
    
    # Write the contents of the uploaded file to temp file
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    temp_file.close()

    return temp_file.name  # Return the full path to use elsewhere


def run():
    st.set_page_config(page_title="Helferlein")
    st.header("Lehrer :blue[Helferlein]")
    # st.selectbox("Select LLM:", get_models(), key="selected_model")

    ###
    st.subheader("Scans hochladen (PNG)")

    ### TODO: Handle multiple Files in Loop ###
    student_file = st.file_uploader("Klausur-Scans hochladen (PNG)", type=["png"], key="student_file", accept_multiple_files=False) # accept_multiple_files=True
    teacher_file = st.file_uploader("Musterlösung hochladen (PNG)", type=["png"], key="teacher_file", accept_multiple_files=False)

    # 
    student_path = save_temp_file(student_file, prefix="student")
    teacher_path = save_temp_file(teacher_file, prefix="teacher")

    st.write("Klausur gespeichert:", student_path)
    st.write("Musterlösung gespeichert:", teacher_path)
    

    if student_file:
        st.image(student_file, caption="Student's Answer", use_column_width=True)
        # Run your image-to-text pipeline here:
        student_text = run_cv_pipeline(student_path)
        st.session_state["student_text"] = student_text
        st.text_area("Extracted Student Text", student_text, height=150)

    if teacher_file:
        st.image(teacher_file, caption="Teacher's Solution", use_column_width=True)
        teacher_text = run_cv_pipeline(teacher_path)
        st.session_state["teacher_text"] = teacher_text
        st.text_area("Extracted Teacher Text", teacher_text, height=150)

        
    ### TODO ###

    student_text = st.session_state.get("student_text", "")
    teacher_text = st.session_state.get("teacher_text", "")

    # convert list of strings to just one string
    student_text = " ".join(student_text)
    teacher_text = " ".join(teacher_text)

    ###
    # Add something that indicated the files are still being processed

    ###

    app_session_init()

    prompt = st.chat_input("Schreibe einen prompt...")

    ### Load LLM Pipeline
    
    # Maybe add this back later
    # selected_model = st.session_state["selected_model"]
    # print("Selected model: ", selected_model)

    # initialize model
    model = initialize_model() # default deepseek:70b

    chat_bot = LlmManager(model)

    if prompt:
        chat_bot.get_response(prompt, student_text, teacher_text)


if __name__ == "__main__":
    run()






### --- Old Flask version --- ###


# import os
# import uuid # For unique session/file handling
# from flask import Flask, render_template, request, jsonify, session, redirect, url_for
# # --- SocketIO Imports ---
# from flask_socketio import SocketIO, emit, disconnect
# import eventlet # Or gevent
# # ------------------------
# import shutil # For deleting temporary directories
# import sys # To potentially modify path for module imports

# # --- Clean up and potentially adjust module paths ---
# # This assumes your app.py is in the root directory alongside the 'modules' folder.
# # If not, you might need to adjust sys.path more robustly.
# # MODULES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules'))
# # CV_PIPELINE_SRC = os.path.join(MODULES_ROOT, 'cv_pipeline', 'src', 'modules')
# # LLM_PIPELINE_SRC = os.path.join(MODULES_ROOT, 'llm_pipeline')

# # try:
# # --- Import Pipelines as Modules ---
# from modules.cv_pipeline.src.modules.cv_pipeline import CVPipeline
# from modules.cv_pipeline.src.modules.red_remover import RedRemover
# from modules.cv_pipeline.src.modules.horizontal_cutter import HorizontalCutter
# from modules.cv_pipeline.src.modules.line_cropper import LineCropper
# from modules.cv_pipeline.src.modules.text_recognizer import TextRecognizer

# # --- Exclude Vector store for now due to dependency conflicts --- # 
# from modules.llm_pipeline.llm_manager import LlmWebSocketManager, initialize_model
# # except ImportError as e:
# #     print(f"Error importing modules: {e}")
# #     print(f"Please ensure modules are installed and paths are correct.")
# #     print(f"MODULES_ROOT: {MODULES_ROOT}")
# #     print(f"CV_PIPELINE_SRC: {CV_PIPELINE_SRC}")
# #     print(f"LLM_PIPELINE_SRC: {LLM_PIPELINE_SRC}")
# #     sys.exit(1) # Exit if essential modules can't be imported


# # --- Configuration ---
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png'} # Add others if needed!!!

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SECRET_KEY'] = os.urandom(24) # SocketIO and Flask sessions need this

# # --- Initialize SocketIO ---
# # Use eventlet or gevent for production-ready async handling
# socketio = SocketIO(app, async_mode='eventlet') 
# # -------------------------

# # Ensure the upload folder exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # --- Initialize LLM ---
# print("Initializing base LLM model (client)...")
# try:
#     base_llm_model = initialize_model() # From llm_manager (e.g., returns Ollama client)
#     print("Base LLM model (client) initialized.")
#     # Instantiate the WebSocket manager, passing the Ollama client
#     llm_manager = LlmWebSocketManager(model=base_llm_model)
# except Exception as e:
#     print(f"FATAL: Could not initialize LLM model: {e}")
#     sys.exit(1)
# # --------------------

# # --- Initialize CV Pipeline (Example Instantiation - ADJUST AS NEEDED) ---
# # You might instantiate these once here if they are stateless, 
# # or inside run_cv_pipeline if they need specific config per call.
# try:
#     print("Initializing CV components...")
#     # This is a guess - adjust based on your CV component's __init__ methods
#     cv_pipeline = CVPipeline()
#     cv_pipeline.add_stage(RedRemover(debug=True))
#     cv_pipeline.add_stage(HorizontalCutter(debug=True))
#     cv_pipeline.add_stage(LineCropper(debug=True))
#     cv_pipeline.add_stage(TextRecognizer(debug=True))

#     print("CV components initialized.")
# except Exception as e:
#     print(f"Warning: Could not initialize CV components globally: {e}")
#     # Fallback: Will try to initialize in the function if global fails
#     cv_pipeline = None 
# # ------------------------------------------------------------------------

# def run_cv_pipeline(image_path):
#     """
#     Runs the configured CV pipeline on the given image path.
#     Adjust implementation based on your actual CVPipeline class.
#     """
#     global cv_pipeline # Use the global instance if available
#     try:
#         if cv_pipeline:
#             print(f"Running CV pipeline on: {image_path}")
#             extracted_text = cv_pipeline.run_and_return_text(image_path)

#             return extracted_text
#         else:
#             pass
#             # # Fallback: Instantiate locally if global init failed (less efficient)
#             # print("Instantiating CV pipeline locally for processing...")
#             # remover = RedRemover()
#             # cutter = HorizontalCutter()
#             # cropper = LineCropper()
#             # recognizer = TextRecognizer()
#             # pipeline = CVPipeline(remover, cutter, cropper, recognizer)
#             # extracted_text = pipeline.process_image(image_path) # ADJUST METHOD NAME
#             # return extracted_text
            
#     except Exception as e:
#         print(f"Error during CV pipeline execution for {image_path}: {e}")
#         # Log traceback here for debugging
#         # import traceback
#         # traceback.print_exc()
#         raise # Re-raise the exception to be caught by the route handler

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def get_session_dir():
#     """Gets or creates a unique directory for the current user session."""
#     # Use a Flask session ID if available, otherwise generate one for file path
#     if 'session_id' not in session:
#         session['session_id'] = str(uuid.uuid4())
#         # Initialize data structures in session
#         session['uploaded_files'] = {'tests': [], 'solution': None} 
#         session['extracted_texts'] = [] 
#         session['solution_text'] = None

#     session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['session_id'])
#     os.makedirs(session_dir, exist_ok=True)
#     return session_dir

# # === HTTP Routes ===

# @app.route('/')
# def index():
#     """Renders the main page."""
#     # Ensure Flask session exists
#     if 'session_id' not in session:
#          session['session_id'] = str(uuid.uuid4())
#          session['extracted_texts'] = []
#          session['solution_text'] = None
         
#     # Renders the HTML container. Chat history is handled by WebSocket state.
#     return render_template('index.html') # Make sure you have this template


# @app.route('/upload', methods=['POST'])
# def upload_files():
#     """Handles file uploads for tests and solution via HTTP."""
#     if 'session_id' not in session:
#          # Should generally not happen if '/' sets it, but good failsafe
#          session['session_id'] = str(uuid.uuid4())
#          session['extracted_texts'] = []
#          session['solution_text'] = None
         
#     session_dir = get_session_dir()
    
#     current_extracted_texts = [] # Store texts extracted in *this* request
#     current_solution_text = None
    
#     # --- Process Test Files ---
#     test_files = request.files.getlist('test_files') 
#     for file in test_files:
#         if file and file.filename and allowed_file(file.filename):
#             # Consider using secure_filename from werkzeug.utils
#             # from werkzeug.utils import secure_filename
#             # filename = secure_filename(file.filename)
#             filename = file.filename 
#             filepath = os.path.join(session_dir, filename)
#             file.save(filepath)
#             session.setdefault('uploaded_files', {'tests': [], 'solution': None})['tests'].append(filepath)

#             # --- Trigger CV Pipeline for Text Extraction ---
#             try:
#                 extracted_text = run_cv_pipeline(filepath) # Use the actual pipeline
#                 current_extracted_texts.append(extracted_text)
#                 print(f"Extracted text from {filename}") 
#             except Exception as e:
#                 print(f"Error extracting text from {filename}: {e}")
#                 # Optionally notify user about specific file failure
#                 return jsonify({"status": "error", "message": f"Failed to process {filename}: {e}"}), 500
                
#     # --- Process Solution File ---
#     solution_file = request.files.get('solution_file') 
#     if solution_file and solution_file.filename:
#         filename_base = "solution_" + solution_file.filename # Avoid name collision
#         filepath = os.path.join(session_dir, filename_base)
#         solution_file.save(filepath)
#         session.setdefault('uploaded_files', {'tests': [], 'solution': None})['solution'] = filepath 

#         # Decide how to handle the solution:
#         if allowed_file(solution_file.filename): # PNG
#             try:
#                 current_solution_text = run_cv_pipeline(filepath) # Use CV pipeline
#                 print(f"Extracted text from solution {filename_base}")
#             except Exception as e:
#                  print(f"Error extracting text from solution {filename_base}: {e}")
#                  return jsonify({"status": "error", "message": f"Failed to process solution {filename_base}: {e}"}), 500
#         elif solution_file.filename.rsplit('.', 1)[1].lower() in {'txt', 'md'}: # Text file
#             try:
#                 with open(filepath, 'r', encoding='utf-8') as f:
#                      current_solution_text = f.read()
#                 print(f"Loaded text from solution {filename_base}")
#             except Exception as e:
#                  print(f"Error reading solution text file {filename_base}: {e}")
#                  return jsonify({"status": "error", "message": f"Failed to read solution text file {filename_base}: {e}"}), 500
#         else:
#              print(f"Unsupported solution file type: {solution_file.filename}")
#              # Decide if this is an error or just ignore
#              # return jsonify({"status": "error", "message": f"Unsupported solution file type: {solution_file.filename}"}), 400


#     # --- Update Flask Session with extracted data ---
#     # Append new texts to existing list in session
#     session.setdefault('extracted_texts', []).extend(current_extracted_texts)
#     if current_solution_text:
#         session['solution_text'] = current_solution_text
    
#     session.modified = True 
    
#     print(f"Session {session['session_id']} updated with {len(current_extracted_texts)} new test texts and solution: {bool(current_solution_text)}")
    
#     # Inform client upload is done. Client JS should then connect/use WebSocket.
#     return jsonify({"status": "success", "message": "Files uploaded and processed. Ready to chat."})


# @app.route('/reset', methods=['POST']) # Use POST for actions that change state
# def reset_session():
#     """Clears Flask session data and temporary files."""
#     if 'session_id' in session:
#         session_id = session['session_id']
#         print(f"Resetting session: {session_id}")
#         session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
#         if os.path.exists(session_dir):
#             try:
#                 shutil.rmtree(session_dir) # Delete the session's upload directory
#                 print(f"Deleted session directory: {session_dir}")
#             except Exception as e:
#                 print(f"Error deleting directory {session_dir}: {e}")
#                 # Log error, but proceed with clearing session data
        
#         # Note: We don't explicitly clear LLM state from llm_manager here.
#         # We rely on the client disconnecting (triggering handle_disconnect)
#         # or the state being overwritten if the same client reconnects.
#         # A more complex mapping would be needed to clear specific websocket states
#         # based on a Flask session reset initiated via HTTP.

#     session.clear() 
#     print("Flask session cleared.")
#     # Redirect to the home page to reflect the reset state
#     return redirect(url_for('index'))

# # === SocketIO Event Handlers ===

# @socketio.on('connect')
# def handle_connect():
#     """Handles new client WebSocket connections."""
#     sid = request.sid
#     print(f'Client connected: {sid}')
    
#     # Try to retrieve context (uploaded data) from the Flask session.
#     # This assumes the user uploaded files via HTTP POST *before* connecting.
#     initial_context = {
#         "test_texts": session.get('extracted_texts', []), # Get latest from session
#         "solution_text": session.get('solution_text', None)
#     }
    
#     print(f"Initializing LLM conversation for {sid} with context: Test texts ({len(initial_context['test_texts'])} items), Solution ({bool(initial_context['solution_text'])})")
    
#     # Start managing state for this connection in the LLM manager
#     llm_manager.start_conversation(sid, initial_context)
    
#     # Send confirmation back to the client
#     emit('connection_ready', {'message': 'Connected. LLM state initialized.'})


# @socketio.on('disconnect')
# def handle_disconnect():
#     """Handles client WebSocket disconnections."""
#     sid = request.sid
#     print(f'Client disconnected: {sid}')
#     # Clean up the state for this specific connection in the LLM manager
#     llm_manager.end_conversation(sid)

# @socketio.on('chat_message')
# def handle_chat_message(data):
#     """Handles incoming chat messages via WebSocket."""
#     sid = request.sid
#     # Ensure data is a dictionary before accessing 'message'
#     if not isinstance(data, dict):
#         print(f"Received invalid data format from {sid}: {data}")
#         emit('error', {'message': 'Invalid message format received.'}, room=sid)
#         return
        
#     user_message = data.get('message')
    
#     if not user_message:
#         print(f"Received empty message from {sid}")
#         emit('error', {'message': 'No message content received.'}, room=sid)
#         return

#     print(f"Received message from {sid}: {user_message}")
    
#     # Get response using the state managed by the manager for this connection
#     try:
#         # The llm_manager uses the history and context stored for this 'sid'
#         llm_answer = llm_manager.get_llm_response(sid, user_message)
        
#         # Emit the response *back to the specific client*
#         print(f"Sending response to {sid}: {llm_answer[:100]}...") # Log snippet
#         emit('llm_response', {'answer': llm_answer}, room=sid) 
        
#     except Exception as e:
#         print(f"Error processing chat message for {sid}: {e}")
#         # Log traceback for detailed debugging
#         # import traceback
#         # traceback.print_exc()
#         emit('error', {'message': 'Failed to get LLM response from server.'}, room=sid)

# # --- Main execution ---
# if __name__ == '__main__':
#     print("Starting Flask-SocketIO server...")
#     # Use socketio.run, not app.run, and specify host/port
#     # Use host='0.0.0.0' to make it accessible on your network
#     # Set debug=False for production/stable testing
#     socketio.run(app, debug=True, host='0.0.0.0', port=5000)