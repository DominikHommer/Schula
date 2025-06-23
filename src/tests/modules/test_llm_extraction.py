# pylint: skip-file
import unittest
from unittest.mock import MagicMock

import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from libs.language_client import LanguageClient
from modules.llm_text_extraction import LLMExtraction
from models.parser.extraction_result import ExtractionResult

load_dotenv()
llmClient = LanguageClient()

class TestLLMExtraction(unittest.TestCase):
    def test_process_returns_extracted_essay(self):
        module = LLMExtraction(llmClient, debug=False)

        result = module.process({
            "student_text": "Tim zeigt im Fall Merkmale für die kritische Identität auf. Menschen mit einer Moratoriumsidentität verspüren keine Verpflichtung gegenüber den Werthaltungen ihrer Eltern und dafür eine hohe Experimentierfreudigkeit gegenüber neuen Werten. Diese Personen sind nicht sehr stabil, haben einen geringen Selbstwert und es fehlt ihnen an eindeutigen Werten.",
            "solution_text": "Beschreibung der kritischen Identität (Moratoriumsidentität): Ausprobieren neuer Werthaltungen und Einstellungen. Tim überlegt ob er nicht den Zweig der FOS wechseln sollte. Selbstachtung meint die gefühlsmäßig wertende Einstellung einer Person zu sich selbst und die Wertschätzunh die eine Person für sich selbst empfindet.",
        })

        self.assertIsInstance(result, ExtractionResult)
        self.assertIsNotNone(result.results)
        self.assertEqual(2, len(result.results))
        self.assertEqual(result.results[0].Aspekt[0].Aspekt, "Merkmale der Moratoriumsidentität")
