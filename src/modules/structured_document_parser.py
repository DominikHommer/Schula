import os
import time
import json
import base64
from pathlib import Path
from typing import Type
from dotenv import load_dotenv
from groq import Groq
from pdf2image import convert_from_path
from pydantic import BaseModel

from .module_base import Module

class StructuredDocumentParser(Module):
    def __init__(self, schema_model: Type[BaseModel], prompt: str, debug=False, debug_output="output.txt"):
        super().__init__("structured-document-parser")
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.schema_model = schema_model
        self.schema_json = schema_model.model_json_schema()
        self.prompt_text = prompt
        self.debug = debug
        self.output_path = debug_output

    def process(self, data: dict) -> list[BaseModel]:
        pdf_path: str = data.get("pdf-path")
        if not pdf_path or not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF nicht gefunden: {pdf_path}")

        pages = convert_from_path(pdf_path, dpi=300)
        results = []

        if self.debug:
            f_out = open(self.output_path, "w", encoding="utf-8")

        for i, page in enumerate(pages):
            print(f"[Parser] Verarbeite Seite {i+1}...")
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
                        {"type": "text", "text": "Bitte extrahiere strukturierte Informationen im angegebenen Format."},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ]

            parsed = None
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
                    parsed_data = json.loads(raw) if isinstance(raw, str) else raw
                    parsed = self.schema_model(**parsed_data)
                    break
                except Exception as e:
                    print(f"[Parser] Fehler bei Seite {i+1} (Versuch {attempt}): {e}")
                    if attempt < 5:
                        time.sleep(attempt * 10)

            if parsed:
                results.append(parsed)
                if self.debug:
                    f_out.write(f"\n\n=== Seite {i+1} ===\n\n")
                    f_out.write(json.dumps(parsed.model_dump(), indent=2, ensure_ascii=False))

            os.remove(image_path)

        if self.debug:
            f_out.close()
        return results

    def _build_prompt(self) -> str:
        return f"""
                {self.prompt_text}

                Halte dich streng an folgendes JSON-Schema:

                {self.schema_json}

                Gib ausschließlich Informationen zurück, die im Bild enthalten sind.
                """
