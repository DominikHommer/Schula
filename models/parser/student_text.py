from typing import List
from pydantic import BaseModel, Field

class Line(BaseModel):
    """
    Eine einzelne Zeile der Transkription.
    """
    text: str = Field(
        ...,
        description="Der exakte Text dieser Zeile"
    )

class StudentText(BaseModel):
    """
    Modell, das die Transkription als Liste von Zeilen (mit Nummern) enthält.
    """
    lines: List[Line] = Field(
        ...,
        description="Liste aller transkribierten Zeilen."
    )
