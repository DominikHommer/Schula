from typing import Optional
from pydantic import BaseModel
from typing import Dict

class StudentExamAnswers(BaseModel):
    """
    ATTENTION: #TODO
    This is First possible Pydantic model for the student exam text.
    """
    answers: Dict[str, str] # e.g., {"1": "Answer for task 1", "2": "Answer for task 2"}