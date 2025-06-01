from typing import Optional
from pydantic import BaseModel, Field

class Infographic(BaseModel):
    title: Optional[str] = None
    content: list[str] = Field(default_factory=list)

class TextBlock(BaseModel):
    heading: Optional[str] = None
    paragraphs: list[str] = Field(default_factory=list)

class SchulbuchSeite(BaseModel):
    page_number: Optional[int] = None
    title: Optional[str] = None
    text_blocks: list[TextBlock] = Field(default_factory=list)
    infographics: list[Infographic] = Field(default_factory=list)
    materials_links: list[str] = Field(default_factory=list)
    raw_text: Optional[str] = None