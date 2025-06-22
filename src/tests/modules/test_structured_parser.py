# pylint: skip-file
import unittest
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from modules.structured_document_parser import StructuredDocumentParser
from libs.language_client import LanguageClient
from libs.file_helper import normalize_paths
from models.parser.assignment_sheet import AssignmentSheet
from models.parser.model_solution import ModelSolution
from models.parser.schulbuch_seite import SchulbuchSeite

load_dotenv()
llmClient = LanguageClient()
class TestStructuredDocumentParser(unittest.TestCase):
    def test_assignment_sheet_parsing(self):
        parser = StructuredDocumentParser(
            schema_model=AssignmentSheet,
            prompt="Bitte analysiere das Aufgabenblatt und gib eine strukturierte JSON-Darstellung zur Aufgabenstellung zurück.",
            llm_client=llmClient,
            debug=False
        )
        
        with self.assertRaises(NotImplementedError):
            _ = parser.process({"paths": ["src/tests/fixtures/test_structured_parser_aufgabenstellung.pdf"]})

    def test_model_solution_parsing(self):
        parser = StructuredDocumentParser(
            schema_model=ModelSolution,
            prompt="Bitte analysiere die Musterlösung und gib eine strukturierte JSON-Darstellung zurück.",
            llm_client=llmClient,
            debug=False
        )

        path = os.path.abspath("tests/fixtures/test_structured_parser_muster.pdf")
        paths = normalize_paths([path])

        result = parser.process({"paths": paths})
        
        self.assertIsInstance(result, ModelSolution)
        self.assertIsNotNone(result.solutions)
        self.assertIsNotNone(result.solutions[0].subsolutions)
        self.assertIsNotNone(result.solutions[0].subsolutions[0].solution)

    def test_schulbuch_seite_parsing(self):
        parser = StructuredDocumentParser(
            schema_model=SchulbuchSeite,
            prompt="Bitte transkribiere die Schulbuchseite und gib eine strukturierte JSON-Darstellung zurück.",
            llm_client=llmClient,
            debug=False
        )

        path = os.path.abspath("tests/fixtures/schulbuch_test.pdf")
        paths = normalize_paths([path])

        with self.assertRaises(NotImplementedError):
            result = parser.process({"paths": paths})
        
        #self.assertIsInstance(result, SchulbuchSeite)
        #self.assertIsNotNone(result.raw_text)


