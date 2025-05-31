from typing import List, Optional
from pydantic import BaseModel, Field

class SubSolution(BaseModel):
    label: Optional[str] = Field(None, description="Bezeichnung der Teilaufgabe, z. B. 'a)'")
    solution: Optional[str] = Field(..., description="Lösung der Teilaufgabe, ggf. mit Rechenweg oder Begründung")

class TaskSolution(BaseModel):
    number: Optional[int] = Field(None, description="Nummer der Aufgabe")
    title: Optional[str] = Field(None, description="Titel oder Thema der Aufgabe (optional)")
    solution_text: Optional[str] = Field(None, description="Allgemeine Lösung oder Einleitung für diese Aufgabe")
    subsolutions: List[SubSolution] = Field(default_factory=list, description="Teil-Lösungen der Unteraufgaben")

class ModelSolution(BaseModel):
    assignment_title: Optional[str] = Field(None, description="Titel des zugehörigen Aufgabenblatts")
    subject: Optional[str] = Field(None, description="Fach der Lösung, z. B. 'Deutsch', 'Mathe'")
    solutions: List[TaskSolution] = Field(..., description="Lösungen zu allen Aufgaben")
    raw_text: Optional[str] = Field(None, description="Kompletter erkannter Text der Musterlösung, falls OCR-basiert")
