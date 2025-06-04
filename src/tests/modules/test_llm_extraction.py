# pylint: skip-file
import unittest
from unittest.mock import MagicMock

from modules.llm_text_extraction import LLMExtraction

class TestLLMExtraction(unittest.TestCase):
    def test_process_returns_extracted_essay(self):
        module = LLMExtraction(debug=False)

        mock_llm = MagicMock()
        mock_llm.use_structured_output = MagicMock()
        
        mock_result = {
            'extracted_essay': [
                "1. Identität in der Psychologie meint das Selbstverständnis ...",
                "2. Jeder Mensch entwickelt sich im Laufe des Lebens ...",
            ]
        }
        mock_llm.get_response.return_value = mock_result

        result = module.process({}, mock_llm)

        self.assertIsInstance(result, dict)
        self.assertIn('extracted_essay', result)
        self.assertIsInstance(result['extracted_essay'], list)
        self.assertGreater(len(result['extracted_essay']), 0)

        mock_llm.use_structured_output.assert_called_once()
        mock_llm.get_response.assert_called_once()


