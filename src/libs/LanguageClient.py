from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
import streamlit as st

class LanguageClient():
    """
    Dient als einzige Schnittstelle für die Kommunikation zwischen LLM und Pipelines
    Todo:
        - Async, chunks and batches um Anfragen parallel zu bearbeiten
    """
    
    # model = cached_initialize_model()
    def __init__(self, model="deepseek-r1:70b"):
        self.model = ChatOllama(model=model)

    def use_structured_output(self, json_schema: dict):
        self.model = self.model.with_structured_output(json_schema)

    def get_response(self, messages: list[BaseMessage], config: dict = {}) -> dict:
        return self.model.invoke(messages)