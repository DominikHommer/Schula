import ollama
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import MessagesPlaceholder

# --- Exclude Vector Store for now --- #
# from vector_store import retriever


DB_PATH = "vector_db_test"

def initialize_model(model="llama4:latest"):
    return ChatOllama(model=model)

class LlmManager:

    def __init__(self, model):
        self.model = model
        # No need for conversation management here if using Streamlit's session state

    # No need for select_prompt_template if we build messages directly

    def get_response(self, user_prompt, test_student, test_solution):

        chat_history = st.session_state.get("chat_history", []) # Ensure default is empty list

        # --- Construct the message list ---

        messages = []

        # Check if this is the first user interaction after the initial greeting
        is_first_interaction = len(chat_history) == 1 and isinstance(chat_history[0], AIMessage)

        if is_first_interaction:
            # System message defining the role and initial context
            system_content = f"""
            Du bist ein Deutschlehrer und sollst die Aufgaben einer Deutschklausur gemäß des Erwartungshorizontes bewerten.

            Hier ist die Klausur des Schülers:
            --- START SCHÜLERKLAUSUR ---
            {test_student}
            --- ENDE SCHÜLERKLAUSUR ---

            Hier die Musterlösung mit Erwartungshorizont:
            --- START MUSTERLÖSUNG ---
            {test_solution}
            --- ENDE MUSTERLÖSUNG ---

            Hier ist der Hintergrund aus dem im Unterricht verwendeten Deutschbuch:
            --- START DEUTSCHBUCH ---
            Placeholder Text for testing (Replace with actual retrieval later)
            --- ENDE DEUTSCHBUCH ---

            Du sollst hierbei wie folgt vorgehen:
            ---
            Define precise tasks here.
            ---
            Bewerte die Klausur nun basierend auf der ersten Frage des Lehrers.
            """
            # Used to tell the chat model how to behave and provide additional context. (langchain_doc)
            messages.append(SystemMessage(content=system_content))
            # Add user question
            messages.append(HumanMessage(content=user_prompt))

        else:
            # If a subsequent message, pass chat history and new user question
            if not any(isinstance(msg, SystemMessage) for msg in chat_history):
                 pass

            # Add existing history (excluding the initial AI greeting if desired, though usually kept)
            messages.extend(chat_history)
            # Add the new user question
            messages.append(HumanMessage(content=user_prompt))


        # --- Get response from LLM ---

        # Display user prompt in chat UI (not including the actual prompt passed to the model)
        st.chat_message("user").write(user_prompt)

        # Stream the response from the model using the list of messages
        model_output_stream = self.model.stream(messages)

        # Write the streaming response to the UI and capture the full response string
        with st.chat_message("ai"):
            ai_response_content = st.write_stream(model_output_stream) # This captures the full string

        # --- Update History ---
        # Append the actual user message
        st.session_state["chat_history"].append(HumanMessage(content=user_prompt))

        # Append the AI's response message - This should now work correctly
        # as ai_response_content is the string returned by st.write_stream
        st.session_state["chat_history"].append(AIMessage(content=ai_response_content))


# Helper function for vector store
def get_text_book_content(query):
    # retriever_invocation = query
    # text_book_content = retriever.invoke(retriever_invocation)
    return "Placeholder Text for testing"