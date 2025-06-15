from pydantic import BaseModel, Field

class StudentText(BaseModel):
    """
    Ein einfaches Datenmodell, das die gesamte unstrukturierte Transkription
    einer Schülerantwort als Rohtext enthält.
    """
    
    raw_text: str = Field(
        ...,
        description="Die vollständige und exakte Transkription des gesamten Textes aus der Schülerantwort."
    )
