import os
import uuid # For unique session/file handling
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import shutil # For deleting temporary directories

# --- Import Pipelines as Modules ---
# Assume you have functions like these, adjust imports as needed
from your_pipelines.png_processor import extract_text_from_png
from your_pipelines.llm_handler import get_llm_response, initialize_llm_session # Hypothetical functions


# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png'} # Add others if needed!!!

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24) # Important for session management

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_dir():
    """Gets or creates a unique directory for the current user session."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['uploaded_files'] = {'tests': [], 'solution': None} # Track files per session
        session['extracted_texts'] = [] # Store extracted text per session
        # Initialize LLM state for session if needed
        # session['llm_context'] = initialize_llm_session()

    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['session_id'])
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

# --- Routes ---

@app.route('/')
def index():
    """Renders the main page."""
    # Ensure session exists when loading the page
    get_session_dir()
    # Pass any necessary initial state to the template
    return render_template('index.html', chat_history=session.get('chat_history', []))

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handles file uploads for tests and solution."""
    session_dir = get_session_dir()
    
    uploaded_test_paths = []
    solution_path = None
    
    # --- Process Test Files ---
    test_files = request.files.getlist('test_files') # Input name="test_files"
    for file in test_files:
        if file and file.filename and allowed_file(file.filename):
            filename = file.filename # Use original name or secure_filename
            filepath = os.path.join(session_dir, filename)
            file.save(filepath)
            uploaded_test_paths.append(filepath)
            session['uploaded_files']['tests'].append(filepath) # Track file

            # --- Trigger Text Extraction ---
            try:
                extracted_text = extract_text_from_png(filepath)
                session['extracted_texts'].append(extracted_text)
                print(f"Extracted text from {filename}") # Logging
            except Exception as e:
                print(f"Error extracting text from {filename}: {e}")
                # Handle error appropriately (e.g., notify user)
                
    # --- Process Solution File ---
    solution_file = request.files.get('solution_file') # Input name="solution_file"
    if solution_file and solution_file.filename:
        # Decide how to handle the solution:
        # 1. If it's also a PNG needing OCR:
        if allowed_file(solution_file.filename):
            filename = "solution_" + solution_file.filename # Avoid name collision
            filepath = os.path.join(session_dir, filename)
            solution_file.save(filepath)
            session['uploaded_files']['solution'] = filepath # Track file
            try:
                solution_text = extract_text_from_png(filepath)
                session['solution_text'] = solution_text # Store extracted solution text
                print(f"Extracted text from solution {filename}")
            except Exception as e:
                 print(f"Error extracting text from solution {filename}: {e}")
        # 2. If it's a text file (e.g., .txt, .md):
        elif solution_file.filename.rsplit('.', 1)[1].lower() in {'txt', 'md'}:
            filename = "solution_" + solution_file.filename
            filepath = os.path.join(session_dir, filename)
            solution_file.save(filepath)
            session['uploaded_files']['solution'] = filepath # Track file
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                     session['solution_text'] = f.read()
                print(f"Loaded text from solution {filename}")
            except Exception as e:
                 print(f"Error reading solution text file {filename}: {e}")
        else:
             print(f"Unsupported solution file type: {solution_file.filename}")

    # Important: Update the session after modifications
    session.modified = True 
    
    # Redirect back to main page or return JSON status
    return jsonify({"status": "success", "message": "Files uploaded and processed."})
    # Or redirect: return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    """Handles incoming chat messages and interacts with the LLM."""
    if 'session_id' not in session:
        return jsonify({"error": "Session not found. Please upload files first."}), 400

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    # --- Prepare context for LLM ---
    # Retrieve data stored in the session
    extracted_texts = session.get('extracted_texts', [])
    solution_text = session.get('solution_text', None)
    # Retrieve or manage LLM conversation history/context if needed
    # llm_context = session.get('llm_context', {}) 
    chat_history = session.get('chat_history', [])


    # --- Call your LLM Pipeline ---
    try:
        # Pass necessary context: user message, extracted texts, solution,
        # vector DB access info (maybe configured globally or passed),
        # and potentially session-specific LLM state/history.
        llm_answer = get_llm_response(
            user_message=user_message,
            test_texts=extracted_texts,
            solution_text=solution_text,
            # vector_db_client=your_db_client, # How you access the DB
            # conversation_history=chat_history # Pass history if LLM needs it
        )
        
        # Update chat history in session
        chat_history.append({"user": user_message, "llm": llm_answer})
        session['chat_history'] = chat_history
        session.modified = True

        return jsonify({"answer": llm_answer})

    except Exception as e:
        print(f"Error getting LLM response: {e}")
        # Consider logging the full traceback
        return jsonify({"error": "Failed to get response from LLM."}), 500


@app.route('/reset', methods=['POST']) # Use POST for actions that change state
def reset_session():
    """Clears session data and temporary files."""
    if 'session_id' in session:
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['session_id'])
        if os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir) # Delete the session's upload directory
                print(f"Deleted session directory: {session_dir}")
            except Exception as e:
                print(f"Error deleting directory {session_dir}: {e}")
                # Log error, but proceed with clearing session data

    # Clear Flask session data
    session.clear() 
    # Optionally re-initialize minimal session state if needed immediately
    # get_session_dir() # Creates a new empty session

    # Redirect to the home page to reflect the reset state
    return redirect(url_for('index'))


# --- Main execution ---
if __name__ == '__main__':
    # Debug mode is helpful during development, but disable for production
    app.run(debug=True)