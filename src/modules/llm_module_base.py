import typing
from .module_base import Module
from langchain_core.messages import SystemMessage
from libs.language_client import LanguageClient

class LLMModule(Module):
    def __init__(self, module_key = None):
        super().__init__(module_key)

    def get_system_prompt(self) -> SystemMessage:
        raise Exception("Please define get_system_prompt")
    
    def get_structured_output(self):
        raise Exception("Please define get_structured_output")
    
    def process(self, data: dict, llm: LanguageClient) -> typing.Any:
        raise Exception("Please define process")
