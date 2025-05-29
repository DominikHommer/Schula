from .pipeline import Pipeline
from libs.LanguageClient import LanguageClient

class LLMPipeline(Pipeline):
    def __init__(self, llmClient: LanguageClient, input_data: dict = {}):
        super().__init__(input_data)

        self.llmClient = llmClient

    def run(self, input_data = None):
        self.data['input'] = input_data
        for module in self.stages:
            super()._check_condition(module)

            self.data[module.module_key] = module.process(self.data, self.llmClient)

        # Output is result of last stage
        return self.data[self.stages[-1].module_key]