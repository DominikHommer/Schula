from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel

class LanguageClient():
    """
    Dient als einzige Schnittstelle für die Kommunikation zwischen LLM und Pipelines
    Todo:
        - Async, chunks and batches um Anfragen parallel zu bearbeiten
    """
    model: BaseChatModel
    
    # model = cached_initialize_model()
    def __init__(self, model: str="deepseek-r1:70b"):
        self.model = ChatOllama(model=model)

    def use_structured_output(self, json_schema: dict):
        """
        Forces LLM to response in given json schema
        """
        self.model = self.model.with_structured_output(json_schema)

    def get_response(self, messages: list[BaseMessage], config: dict | None = None) -> dict:
        """
        Executes all messages and returns result
        """
        return self.model.invoke(messages, config)
