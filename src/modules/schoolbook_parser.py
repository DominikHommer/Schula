import os
import time
import json
import base64
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pdf2image import convert_from_path
from pydantic import BaseModel, Field

from .module_base import Module

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

class SchulbuchParser(Module):
    def __init__(self, debug=False, debug_output="output_schulbuch.txt"):
        super().__init__("schulbuch-parser")
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.schema_json = SchulbuchSeite.model_json_schema()
        self.debug = debug
        self.output_path = debug_output

    def get_preconditions(self) -> list[str]:
        return ['pdf-path']  # oder ein Bild als 'image' je nach Integration

    def process(self, data: dict) -> list[SchulbuchSeite]:
        pdf_path: str = data.get("pdf-path")
        if not pdf_path or not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF nicht gefunden: {pdf_path}")
        
        pages = convert_from_path(pdf_path, dpi=300)
        results = []

        if self.debug:
            f_out = open(self.output_path, "w", encoding="utf-8")

        for i, page in enumerate(pages):
            print(f"[SchulbuchParser] Verarbeite Seite {i+1}...")
            image_path = f"temp_page_{i+1:02d}.jpg"
            page.save(image_path, "JPEG")

            with open(image_path, "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{b64}"

            messages = [
                {"role": "system", "content": self._build_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Bitte transkribiere die Schulbuchseite und gib eine strukturierte Darstellung in JSON zurück."},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ]

            data = None
            for attempt in range(1, 6):
                try:
                    completion = self.client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=messages,
                        temperature=0.3,
                        max_completion_tokens=3000,
                        top_p=1,
                        stream=False,
                        response_format={"type": "json_object"},
                    )
                    raw = completion.choices[0].message.content
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    break
                except Exception as e:
                    print(f"[SchulbuchParser] Fehler bei Seite {i+1} (Versuch {attempt}): {e}")
                    if attempt < 5:
                        time.sleep(attempt * 10)
                    else:
                        print(f"[SchulbuchParser] Max. Versuche erreicht – Seite {i+1} wird übersprungen.")

            if data:
                parsed = SchulbuchSeite(**data)
                results.append(parsed)

                if self.debug:
                    f_out.write(f"\n\n=== Seite {i+1} ===\n\n")
                    f_out.write(json.dumps(parsed.dict(), indent=2, ensure_ascii=False))

            os.remove(image_path)

        if self.debug:
            f_out.close()
            print(f"[SchulbuchParser] Debug-Ausgabe gespeichert unter: {self.output_path}")
        return results

    def _build_prompt(self) -> str:
        return f"""
Du bekommst ein Bild von einer Seite aus einem Schulbuch. Deine Aufgabe ist es, den erkennbaren Text so strukturiert wie möglich zu digitalisieren – ohne Interpretation, Zusammenfassung oder inhaltliche Ergänzung.

Extrahiere z. B. Titel, Textblöcke mit Überschriften, Infografiken (mit Titeln und Listeninhalten), Materialien oder Weblinks. Falls möglich, gib auch den vollständigen Fließtext als `raw_text` an.

Halte dich streng an folgendes JSON-Schema:

{self.schema_json}

Behalte die Sprache Deutsch bei. Gib ausschließlich die tatsächlich auf der Seite enthaltenen Informationen wieder.
"""

