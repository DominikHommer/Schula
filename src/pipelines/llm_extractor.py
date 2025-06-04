from modules.llm_text_extraction import LLMExtraction
from libs.language_client import LanguageClient

from .llm_pipeline import LLMPipeline

class LLMTextExtractorPipeline(LLMPipeline):
    """
    Pipeline zur Extraktion der Musterlösung aus dem Schülertext
    Sollte im Streamlit Kontext verwendet werden
    TODO: -> Input sollte Schülertext, splitted Musterlösung enthalten
          -> Extraktion sollte async parallel bearbeitet werden
    """
    def __init__(self, llmClient: LanguageClient, input_data: dict | None = None):
        super().__init__(llmClient, input_data)

        self.add_stage(LLMExtraction(debug=False))

    def process_streamlit(self):
        """
        Execute pipeline in streamlit context
        """
        return self.run()
