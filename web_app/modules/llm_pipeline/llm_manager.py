# imports for LLM 
import ollama
import streamlit as st
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# from langchain.vectorstores.chroma import Chroma
# --- Exclude Vector Store for now --- #
# from vector_store import retriever


DB_PATH = "vector_db_test"

def initialize_model(model="deepseek-r1:70b"):
        return ChatOllama(model=model)

class LlmManager:

    def __init__(self, model):
        self.model = model

    # def start_conversation(self, sid, initial_context=None):
    #     with self._lock:
    #         print(f"Starting WS conversation for sid: {sid}")
    #         self.conversations[sid] = {"history": [], "context": initial_context}
    #     return self.conversations[sid]

    # def end_conversation(self, sid):
    #     with self._lock:
    #         if sid in self.conversations:
    #             print(f"Ending conversation for sid: {sid}")
    #             del self.conversations[sid]

    # def get_conversation_state(self, sid):
    #     with self._lock:
    #          return self.conversations.get(sid)


    def select_prompt_template(self, chat_history):

        prompt_template = None

        # if it's the first message, pass the text book data into the prompt
        if not chat_history:
            prompt_template = """ 
            Du bist ein Deutschlehrer und sollst die Aufgaben einer Deutschklausur gemäß des Erwartungshorizontes bewerten.

            Hier ist die Klausur des Schülers: {test_student}

            Hier die Musterlösung mit Erwartungshozint: {test_solution}

            Hier ist der Hintergründe aus dem im Unterricht verwendeten Deutschbuch: {text_book_content}

            Hier ist die Frage des Lehrers: {question}

            Du sollst hierbei wie folgt vorgehen:
            ---
            Define precise tasks here.
            ---

            """
            

        else:
            prompt_template = """
                                
            Hier ist der bisherige Chatverlauf mit dir: {chat_history}

            Hier ist eine weitere Frage des Lehrers: {question}

            """

        return prompt_template
              
         

    def get_response(self, user_prompt, test_student, test_solution):

        # --- Get prompt template --- #
        prompt_template = self.select_prompt_template()

        # get documents from vector store of the text books
            # use the test_solution (and user prompt) for this
        retriever_invocation = test_solution # + "|" + user_prompt
        # exclude for now
        # text_book_content = retriever.invoke(retriever_invocation)
        text_book_content = "Placeholder Text for testing"

        prompt = ChatPromptTemplate.from_template(prompt_template)


        # get current chat history and update it (important)
        st.chat_message("user").write(prompt)
        chat_history = st.session_state["chat_history"]

        # update history
        st.session_state["chat_history"] += [HumanMessage(prompt)]


        # --- Chain model and prompt to get the LLM response --- #
        chain = prompt | self.model
        
        # if message is first message
        if len(chat_history) == 1 and isinstance(chat_history[0], AIMessage):
            model_output = chain.invoke({"question": user_prompt, "test_student":test_student, "test_solution":test_solution})
        else:
            model_output = chain.invoke({"chat_history": chat_history, "question":user_prompt})

        # write the response and update history
        with st.chat_message("ai"):
            ai_message = st.write_stream(model_output)

        st.session_state["chat_history"] += [AIMessage(ai_message)]
        
        return model_output

