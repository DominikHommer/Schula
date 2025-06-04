import typing
from langchain_core.messages import SystemMessage
from libs.language_client import LanguageClient
from .module_base import Module

class LLMModule(Module):
    """
    Base LLM Module
    """
    def __init__(self, module_key = None):
        super().__init__(module_key)

    def get_system_prompt(self) -> SystemMessage:
        """
        Returns system prompt of Module
        """
        raise Exception("Please define get_system_prompt")
    
    def get_structured_output(self):
        """
        Returns structured output for Module
        """
        raise Exception("Please define get_structured_output")
    
    def process(self, data: dict, llm: LanguageClient) -> typing.Any:
        """
        Executes current Module with LLM Client
        """
        raise Exception("Please define process")
