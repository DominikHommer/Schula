import os
import instructor
from groq import Groq
from pydantic import BaseModel

class LanguageClient:
    """
    Verwaltet die Kommunikation mit dem Groq-Modell via Instructor für strukturierte Antworten.
    """
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not found in environment variables")

        # Initialisiere Groq + Instructor Client mit JSON-Modus
        self.client = instructor.from_groq(
            Groq(api_key=api_key),
            mode=instructor.Mode.JSON
        )

    def get_response(self, messages: list[dict], schema: type[BaseModel], temperature : float, seed : int = 42) -> BaseModel:
        """
        Führt die Modellabfrage mit Instructor und strukturierter Antwortvalidierung durch.
        """
        return self.client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            response_model=schema,
            temperature=temperature,
            seed=seed
        )
