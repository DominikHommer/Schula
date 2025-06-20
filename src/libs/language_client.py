from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
import os
import dotenv
dotenv.load_dotenv()

class LanguageClient():
    """
    Dient als einzige Schnittstelle für die Kommunikation zwischen LLM und Pipelines
    Todo:
        - Async, chunks and batches um Anfragen parallel zu bearbeiten
    """
    model: BaseChatModel
    
    # model = cached_initialize_model()
    # "deepseek-r1:70b"
    def __init__(self, model: str="gemma3:27b"):
        self.model = ChatOllama(model=model)

        # REMOVE IF GROQ NOT USED
        self.model = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key= os.environ.get("GROQ_API_KEY"),
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def use_structured_output(self, json_schema: dict):
        """
        Forces LLM to response in given json schema
        """
        self.model = self.model.with_structured_output(json_schema)

    def get_response(self, messages: list[BaseMessage], config: dict | None = None, schema: BaseModel | dict | None = None) -> dict:
        """
        Invokes the model. If a schema is provided, it creates a temporary chain
        with structured output for this specific call.
        """
        # Determine the model to use for this specific invocation
        chain = self.model
        if schema:
            # Create a new chain with structured output without modifying self.model
            chain = self.model.with_structured_output(schema)

        return chain.invoke(messages, config)
