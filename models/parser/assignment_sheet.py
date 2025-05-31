from typing import List, Optional
from pydantic import BaseModel, Field

class SubTask(BaseModel):
    label: Optional[str] = Field(None, description="Bezeichnung des Unterpunkts, z. B. 'a)', '1.'")
    instruction: Optional[str] = Field(..., description="Konkret zu bearbeitende Teilaufgabe")

class Task(BaseModel):
    number: Optional[int] = Field(None, description="Nummer der Aufgabe (z. B. 1, 2, 3)")
    title: Optional[str] = Field(None, description="Kurzer Titel oder Thema der Aufgabe, falls vorhanden")
    instruction: Optional[str] = Field(..., description="Hauptanweisung der Aufgabe")
    subtasks: List[SubTask] = Field(default_factory=list, description="Liste der Teilaufgaben")
    material_hint: Optional[str] = Field(None, description="Hinweis auf zu verwendendes Material (z. B. Text, Bild)")

class AssignmentSheet(BaseModel):
    title: Optional[str] = Field(None, description="Titel des Aufgabenblatts, z. B. 'Schulaufgabe Deutsch 10. Klasse'")
    subject: Optional[str] = Field(None, description="Fach, z. B. 'Deutsch'")
    grade: Optional[str] = Field(None, description="Jahrgangsstufe, z. B. '10. Klasse'")
    instructions: Optional[str] = Field(None, description="Allgemeine Bearbeitungshinweise oder Zeitvorgaben")
    tasks: List[Task] = Field(..., description="Liste aller Aufgaben auf dem Aufgabenblatt")
    raw_text: Optional[str] = Field(None, description="Vollständig extrahierter Fließtext von der Seite")
