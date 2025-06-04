# pylint: skip-file
import unittest
from unittest.mock import MagicMock

from modules.text_corrector import TextCorrector

class TestTextCorrector(unittest.TestCase):
    def test_process_corrects_text(self):
        corrector = TextCorrector(debug=False)

        corrector._setup = MagicMock()
        corrector.ner = MagicMock(return_value=[])
        corrector.checker = MagicMock()
        corrector.checker.candidates.return_value = {"selbstbewusst"}

        corrector.hunspell = MagicMock()
        corrector.hunspell.suggest.return_value = []

        corrector.symspell = MagicMock()
        corrector.symspell.lookup.return_value = []
        corrector.symspell.word_segmentation.return_value = MagicMock(corrected_string="", distance_sum=99)

        corrector.fill_mask = MagicMock(return_value=[{"token_str": "selbstbewusst"}])
        corrector.score_candidates_batch = MagicMock(return_value={"selbstbewusst": 0.9})

        data = {"text-recognizer": ["s3lbstbewust"]}

        result = corrector.process(data)

        self.assertIsInstance(result, list)
        self.assertIn("selbstbewusst", result)