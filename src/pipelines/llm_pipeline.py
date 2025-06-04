from libs.language_client import LanguageClient
from .pipeline import Pipeline

class LLMPipeline(Pipeline):
    """
    Basic LLM Pipeline Interface
    """
    def __init__(self, llm_client: LanguageClient, input_data: dict | None = None):
        super().__init__(input_data)

        self.llm_client = llm_client

    def run(self, input_data = None):
        self.data['input'] = input_data
        for module in self.stages:
            super()._check_condition(module)

            self.data[module.module_key] = module.process(self.data, self.llm_client)

        # Output is result of last stage
        return self.data[self.stages[-1].module_key]
