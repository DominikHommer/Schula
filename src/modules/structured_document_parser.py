import time
import json
import base64
from typing import Type
from pydantic import BaseModel
from models.parser.model_solution import PageExtraction, ModelSolution, TaskSolution
from models.parser.student_text import StudentText
from typing import List, Optional
from libs.language_client import LanguageClient

class StructuredDocumentParser:
    # --- Handles different LLM providers ---
    def __init__(self, schema_model: Type[BaseModel], prompt: str, llm_client: LanguageClient,debug: bool = False, callback=None):
        self.schema_model = schema_model
        self.system_prompt_template = prompt
        self.debug = debug
        self.callback = callback
        self.output_path = "debug_output.txt"
        self.client = llm_client

    # --- 1. The Main Dispatcher Method (Unchanged) ---
    def process(self, data: dict) -> BaseModel:
        paths: list[str] = data.get("paths")
        if not paths:
            raise ValueError("No paths provided to process.")

        if self.schema_model == StudentText:
            print("[Parser] Dispatching to: Transcription Mode")
            return self._process_transcription(paths)
        elif self.schema_model == ModelSolution:
            print("[Parser] Dispatching to: Structured Solution Mode")
            return self._process_structured_solution(paths)
        else:
            raise NotImplementedError(
                f"Processing for schema '{self.schema_model.__name__}' is not implemented."
            )

    # --- 2. Internal Method for Simple Transcription (Ollama option added) ---

    def _process_transcription(self, paths: list[str]) -> StudentText:
        """
        Transcribe each image in `paths` into a list of lines.
        Returns a StudentText model containing all lines from all pages.
        """
        # Wir sammeln hier Dikt-Einträge für Pydantic
        all_lines_data: list[dict] = []

        system_prompt = (
            "You are an expert OCR and transcription engine. "
            "Extract all text from the provided image **line by line**. "
            "Respond with a JSON object conforming to the `StudentText` schema, "
            "which has exactly one key: `lines`. "
            "`lines` is a list of objects, each with a single field:\n"
            "  - `text` (string): the exact text of that line\n"
            "Ignore any red teacher markup, corrections, strikethroughs or other annotations."
        )

        for i, path in enumerate(paths):
            if self.callback:
                self.callback(i + 1, len(paths))
            print(f"[Parser] Transcribing page {i+1}/{len(paths)}...")

            # Bild laden & Base64
            with open(path, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]}
                ]

                response = self.client.get_response(
                    messages=messages,
                    schema=StudentText,
                    temperature=0.2
                )
                for item in response.lines:
                    all_lines_data.append({"text": item.text})

            except Exception as e:
                print(f"[Parser] WARNUNG: Seite {i+1} konnte nicht transkribiert werden: {e}")
                continue

        # 3) Am Ende erstellen wir das Pydantic-Objekt aus den dicts
        return StudentText(lines=all_lines_data)




    # --- 3. Internal Method for Structured Solution Extraction (Ollama option added) ---
    def _process_structured_solution(self, paths: list[str]) -> ModelSolution:
        page_results: List[PageExtraction] = []
        context_for_next_page: Optional[str] = None
        
        for i, path in enumerate(paths):
            if self.callback: self.callback(i + 1, len(paths))
            print(f"[Parser] Verarbeite Seite {i+1}/{len(paths)}...")
            system_prompt = self._build_prompt_for_page(context_for_next_page)
            with open(path, "rb") as img_file: b64 = base64.b64encode(img_file.read()).decode("utf-8")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Bitte extrahiere die Aufgaben..."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}
            ]

            parsed_page: Optional[PageExtraction] = None
            for attempt in range(1, 4):
                try:
                    # --- GROQ API CALL ---
                    response = self.client.get_response(
                        messages=messages,
                        schema=PageExtraction,
                        temperature=0.2
                    )
                    parsed_page = response
                    break
                except Exception as e:
                    print(f"[Parser] Fehler bei Seite {i+1} (Versuch {attempt}): {e}")
                    if attempt < 3: time.sleep(attempt * 5)

            if parsed_page:
                page_results.append(parsed_page)
                if parsed_page.tasks: context_for_next_page = parsed_page.tasks[-1].model_dump_json(indent=2)
            else:
                print(f"[Parser] WARNUNG: Seite {i+1} konnte nicht verarbeitet werden. Überspringe.")
        
        print("[Parser] Alle Seiten verarbeitet. Führe Ergebnisse zusammen...")
        final_result = self._merge_results(page_results)
        print("[Parser] Zusammenführung abgeschlossen.")
        return final_result

    # --- 4. Helper Methods for Structured Solution Extraction (Unchanged) ---
    def _build_prompt_for_page(self, context_from_previous_page: Optional[str]) -> str:
        # ... (This method is correct and remains unchanged) ...
        schema_json = json.dumps(PageExtraction.model_json_schema(), indent=2)
        prompt = (f"{self.system_prompt_template}\n\n...--- REQUIRED JSON SCHEMA ---\n{schema_json}\n--- END OF SCHEMA ---\n")
        if context_from_previous_page:
            prompt += (f"\n--- CONTEXT FROM PREVIOUS PAGE ---\n...\n")
        else:
            prompt += "\nThis is the first page..."
        return prompt

    def _merge_results(self, page_results: List[PageExtraction]) -> ModelSolution:
        # ... (This method is correct and remains unchanged) ...
        if not page_results: return ModelSolution(solutions=[])
        final_tasks: List[TaskSolution] = []
        for page_data in page_results:
            if not page_data.tasks: continue
            if page_data.is_first_task_a_continuation and final_tasks:
                last_task = final_tasks[-1]
                continuation_task = page_data.tasks[0]
                if continuation_task.solution_text: last_task.solution_text = (last_task.solution_text or "") + "\n" + continuation_task.solution_text
                if continuation_task.subsolutions: last_task.subsolutions.extend(continuation_task.subsolutions)
                final_tasks.extend(page_data.tasks[1:])
            else:
                final_tasks.extend(page_data.tasks)
        final_solution = ModelSolution(solutions=final_tasks)
        return final_solution
