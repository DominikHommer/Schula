from langchain_ollama.llms import OllamaLLM

class LanguageClient():
    def __init__(self, model):
        self.model = OllamaLLM(model=model)
