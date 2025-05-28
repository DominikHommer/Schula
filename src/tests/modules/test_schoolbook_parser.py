import unittest
import os
import json
from modules.schoolbook_parser import SchulbuchParser, SchulbuchSeite

TEST_PDF_PATH = os.path.join("tests", "fixtures", "schulbuch_test.pdf")
DEBUG_OUTPUT_PATH = os.path.join("tests", "fixtures", "test_output_schulbuch.txt")

class TestSchulbuchParser(unittest.TestCase):
    def setUp(self):
        self.parser = SchulbuchParser(debug=True, debug_output=DEBUG_OUTPUT_PATH)
        if not os.path.exists(TEST_PDF_PATH):
            self.skipTest(f"Test PDF {TEST_PDF_PATH} nicht gefunden")

    def test_process_and_validate_json(self):
        result = self.parser.process({"pdf-path": TEST_PDF_PATH})

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        for page in result:
            self.assertIsInstance(page, SchulbuchSeite)

            if page.title:
                self.assertIsInstance(page.title, str)

            if page.text_blocks:
                self.assertIsInstance(page.text_blocks, list)
                for block in page.text_blocks:
                    self.assertIsInstance(block.paragraphs, list)
                    for para in block.paragraphs:
                        self.assertIsInstance(para, str)

            if page.infographics:
                self.assertIsInstance(page.infographics, list)
                for info in page.infographics:
                    self.assertIsInstance(info.content, list)

            if page.raw_text:
                self.assertIsInstance(page.raw_text, str)

    def test_schema_matches_json_output(self):
        with open(DEBUG_OUTPUT_PATH, encoding="utf-8") as f:
            text = f.read()

        raw_jsons = text.split("=== Seite ")
        for chunk in raw_jsons[1:]: 
            json_start = chunk.find("{")
            if json_start != -1:
                try:
                    parsed = json.loads(chunk[json_start:])
                    page = SchulbuchSeite(**parsed)
                    self.assertIsInstance(page, SchulbuchSeite)
                except Exception as e:
                    self.fail(f"Fehler beim Parsen der Seite in SchulbuchSeite: {e}")

