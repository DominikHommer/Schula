# # imports for LLM 
# import ollama
# import streamlit as st
# from langchain_ollama import OllamaLLM, ChatOllama
# # For keeping the chat history and passing it along with the model
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate

# # from langchain.vectorstores.chroma import Chroma
# # --- Exclude Vector Store for now --- #
# # from vector_store import retriever


# DB_PATH = "vector_db_test"

# def initialize_model(model="deepseek-r1:70b"):
#         return ChatOllama(model=model)

# class LlmManager:

#     def __init__(self, model):
#         self.model = model

#     # def start_conversation(self, sid, initial_context=None):
#     #     with self._lock:
#     #         print(f"Starting WS conversation for sid: {sid}")
#     #         self.conversations[sid] = {"history": [], "context": initial_context}
#     #     return self.conversations[sid]

#     # def end_conversation(self, sid):
#     #     with self._lock:
#     #         if sid in self.conversations:
#     #             print(f"Ending conversation for sid: {sid}")
#     #             del self.conversations[sid]

#     # def get_conversation_state(self, sid):
#     #     with self._lock:
#     #          return self.conversations.get(sid)


#     def select_prompt_template(self, chat_history):

#         prompt_template = None

#         # if it's the first message, pass the text book data into the prompt
#         if len(chat_history) == 1 and isinstance(chat_history[0], AIMessage):
#             prompt_template = """ 
#             Du bist ein Deutschlehrer und sollst die Aufgaben einer Deutschklausur gemäß des Erwartungshorizontes bewerten.

#             Hier ist die Klausur des Schülers: {test_student}

#             Hier die Musterlösung mit Erwartungshozint: {test_solution}

#             Hier ist der Hintergründe aus dem im Unterricht verwendeten Deutschbuch: {text_book_content}

#             Hier ist die Frage des Lehrers: {question}

#             Du sollst hierbei wie folgt vorgehen:
#             ---
#             Define precise tasks here.
#             ---

#             """
            

#         else:
#             prompt_template = """
                                
#             Hier ist der bisherige Chatverlauf mit dir: {chat_history}

#             Hier ist eine weitere Frage des Lehrers: {question}

#             """

#         return prompt_template
              
         

#     def get_response(self, user_prompt, test_student, test_solution):

#         # --- Get prompt template --- #

#         # get current chat history
#         chat_history = st.session_state["chat_history"]
#         prompt_template = self.select_prompt_template(chat_history)

#         # get documents from vector store of the text books
#             # use the test_solution (and user prompt) for this
#         retriever_invocation = test_solution # + "|" + user_prompt
#         # exclude for now
#         # text_book_content = retriever.invoke(retriever_invocation)
#         text_book_content = "Placeholder Text for testing"

#         prompt_template = ChatPromptTemplate.from_template(prompt_template)

#         # fill in prompt
#         # if message is first message
#         if len(chat_history) == 1 and isinstance(chat_history[0], AIMessage):
#             prompt = prompt_template.format(
#                 test_student=test_student,
#                 test_solution=test_solution,
#                 question=user_prompt,
#                 test_book_content = text_book_content
#             )
#         else:
#             prompt = prompt_template.format(
#                 chat_history = chat_history,
#                 question = user_prompt
#             )


#         # write user prompt in chat field
#         st.chat_message("user").write(user_prompt)

#         # --- Chain model and prompt to get the LLM response --- #
#         # chain = prompt | self.model
#         model_output = self.model.stream(prompt)

#         # write the response and update history
#         with st.chat_message("ai"):
#             ai_message = st.write_stream(model_output)

#         # update history
#         ### This is incorrect ###
#         # the relevant texts are missing
#         st.session_state["chat_history"] += [HumanMessage(user_prompt)]

#         ### This produces error ### 
#         # this is currently wrong, we need to pass the response of the model
#         st.session_state["chat_history"] += [AIMessage(ai_message)]

#         # return model_output
import ollama
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import MessagesPlaceholder

# --- Exclude Vector Store for now --- #
# from vector_store import retriever


DB_PATH = "vector_db_test"

def initialize_model(model="deepseek-r1:70b"):
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

        # Display user prompt in chat UI
        st.chat_message("user").write(user_prompt)

        # Stream the response from the model using the list of messages
        # The ChatOllama model expects a list of BaseMessage objects
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

        # No need to return model_output_stream as it's consumed by st.write_stream

# Helper function for vector store
def get_text_book_content(query):
    # retriever_invocation = query
    # text_book_content = retriever.invoke(retriever_invocation)
    return "Placeholder Text for testing"