# pylint: skip-file
import unittest
import sys

sys.path.append('../../../models')

from modules.structured_document_parser import StructuredDocumentParser
from models.parser.assignment_sheet import AssignmentSheet
from models.parser.model_solution import ModelSolution
from models.parser.schulbuch_seite import SchulbuchSeite

class TestStructuredDocumentParser(unittest.TestCase):
    def test_assignment_sheet_parsing(self):
        parser = StructuredDocumentParser(
            schema_model=AssignmentSheet,
            prompt="Bitte analysiere das Aufgabenblatt und gib eine strukturierte JSON-Darstellung zur Aufgabenstellung zurück.",
            debug=True,
            debug_output="debug_assignment.txt"
        )
        result = parser.process({"pdf-path": "src/tests/fixtures/sample_assignment_sheet.pdf"})
        self.assertTrue(len(result) > 0)
        self.assertIsInstance(result[0], AssignmentSheet)

    def test_model_solution_parsing(self):
        parser = StructuredDocumentParser(
            schema_model=ModelSolution,
            prompt="Bitte analysiere die Musterlösung und gib eine strukturierte JSON-Darstellung zurück.",
            debug=True,
            debug_output="debug_model_solution.txt"
        )
        result = parser.process({"pdf-path": "src/tests/fixtures/sample_model_solution.pdf"})
        self.assertTrue(len(result) > 0)
        self.assertIsInstance(result[0], ModelSolution)

    def test_schulbuch_seite_parsing(self):
        parser = StructuredDocumentParser(
            schema_model=SchulbuchSeite,
            prompt="Bitte transkribiere die Schulbuchseite und gib eine strukturierte JSON-Darstellung zurück.",
            debug=True,
            debug_output="debug_schulbuch_seite.txt"
        )
        result = parser.process({"pdf-path": "src/tests/fixtures/schulbuch_test.pdf"})
        self.assertTrue(len(result) > 0)
        self.assertIsInstance(result[0], SchulbuchSeite)


