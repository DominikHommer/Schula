from pydantic import BaseModel, Field
from typing import List

class SubAspect(BaseModel):
    Aspekt: str = Field(..., description="Der Aspekt der Teilaufgabe, z.B. eine geforderte Definition.")
    Beleg_Schüleraufsatz: str = Field(..., description="Der Beleg für den in der Teilaufgabe der Muterlösung geforderten Aspekt aus dem Schüleraufsatz.")
    Beleg_Musterlösung: str = Field(..., description="Der Teil aus der Musterlösung, wo die Anforderungen für den Aspekt aufgeschlüsselt sind.")
    Kommentar: str = Field(..., description="Weitere Anmerkungen zur Bearbeitung des Schülers. Z.B. wurde der Aspekt nur teils bearbeitet oder in einer anderen Aufgabe durch den Schüler bearbeitet.")

class ExtractedSolutionItem(BaseModel):
    Teilaufgabe: str = Field(None, description="Teilaufgabe wie z.B. Aufgabe 1 a)")
    Aspekt: List[SubAspect] = Field(..., description="Ein Teilaspekt der Teilaufgabe. Beispielsweise werden unter Aufgabe 1 a) verschiedene Defintionen gefordert, diese sollen einzeln behandelt werden.")

class ExtractionResult(BaseModel):
    results: List[ExtractedSolutionItem]