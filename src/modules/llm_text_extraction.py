import os
import json
from pydantic import ValidationError
from models.parser.extraction_result import ExtractionResult
from .llm_module_base import LLMModule
from libs.language_client import LanguageClient

# instructor + Groq
import instructor
from groq import Groq

class LLMExtraction(LLMModule):
    """
    LLM-Modul zur Extraktion der Aspekt-Bewertung aus Schüleraufsätzen.
    Jetzt über Instructor API mit Groq.
    """

    def __init__(self, language_client: LanguageClient, debug=False, debug_folder="debug/debug_llm_extraction"):
        super().__init__("llm_extraction")
        self.schema_json = ExtractionResult.model_json_schema()
        self.language_client = language_client
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

        # Initialisiere Groq + Instructor Client
        self.client = instructor.from_groq(
            Groq(api_key=os.environ.get("GROQ_API_KEY")),
            mode=instructor.Mode.JSON
        )

    def get_preconditions(self) -> list[str]:
        return []

    def get_system_prompt(self) -> str:
        return f"""
        Du bist ein sehr präzises Assistenz-LLM, das darauf spezialisiert ist, Musterlösungen einer Klausuraufgabe genaustens zu analysieren.
        Deine Aufgabe ist die entschprechenden Teilaufgaben und Anforderungen im Schülertext zu suchen und wiederzugeben ob diese erfolgreich bearbeitet wurden oder nicht.
        Hierfür steht dir immer die jeweilige Aufgabe samt Teilaufgaben aus der Musterlösung zur Verfügung. Außerdem der gesamte Aufsatz des Schülers.

        **Wichtige Anweisungen für die Zuordnung:**
        1.  **Ausgabe Sprache**: Antworte auf Deutsch.
        2.  **Fehlerhafter Aufsatz**: Beachte, dass der Aufsatz durch eine Handschrifterkennung transkripiert wurde, weshalb dieser erhebliche Rechtschreib-, Grammatik- und Logikfehler aufweisen kann, welche nicht auf den Schüler zurückzuführen sind. Versuche dies bei der Zuordnung zu berücksichtigen.
        3.  **Zuordung zur Aufgabe**: Da dir die gesamte Klausur zur Verfügung gestellt wird merke es an, dass die Lösung der Aufgabe in einer anderen Aufgabe erwähnt wurde, falls es dir ersichtlich scheint.
        4.  **Klare Antwort**: Antworte nur mit für die dir gestellte Aufgabe relevanten Information, KEINE abschließenden Worte oder sonstige Anmerkungen, die irrelevant sind.
        5.  **Struktur Aufsatz**: Teilaspekte zu einer Aufgabe können mit größeren Abständen im Aufsatz genannt werden. Falls diese klar zu einer Aufgabe zugeordnet werden können.
        6.  **Zuordnung anhand Aspekte in Musterlösung**: Suche lediglich nach Aspekten, die in der Musterlösung der Aufgabe aufgeführt werden, und versuche nicht um jeden Preis Zuordnungen zu finden. Gib diese nur wieder, wenn sie wirklich im Schülertext behandelt wurden.
        7. **Zeilenmarkierung beibehalten und referenzieren**: Entferne oder verändere **niemals** Ausdrücke im Format `"[number]"`. Diese kennzeichnen die Originalzeilennummern im Schülertext. Wenn du Textstellen analysierst und zuordnest, **füge die zugehörigen Zeilennummern in genau dieser Form am Ende deiner jeweiligen Aussage hinzu** (z. B. `[3]`, `[7]`, `[3][4]`).

        --- Antwortstruktur ---
        Halte dich bei der Antwort streng an folgendes json-schema:
        {self.schema_json}
        """

    def process(self, data: dict, _) -> ExtractionResult:
        """
        Führt die Extraktion via Instructor-Groq durch und validiert gegen das ExtractionResult-Modell.
        """
        few_shot_student = """
Bitte analysiere den folgenden Schüleraufsatz im Vergleich zur Musterlösung und gib für jede Teilaufgabe an, welche Aspekte vom Schüler korrekt wiedergegeben wurden. 
Ordne alle Aspekte klar der jeweiligen Teilaufgabe zu und gib nur relevante Informationen wieder.  Achte darauf die Nachfolgenden Zeilennummern mit anzuhängen!
Halte dich dabei strikt an die im JSON-Schema geforderte Struktur. 

--- AUFSATZ START ---
Die Identität ist das Selbstverständnis eines Menschen als Individuum [1].
Tim glaubt, dass er nichts wert ist und sich niemand für ihn interessiert [2].
Er hat das Gefühl, seine Eltern wollen etwas anderes aus ihm machen [3].
--- AUFSATZ ENDE ---

--- MUSTERLÖSUNG START ---
Definition von Identität: Selbstverständnis eines Menschen als unverwechselbare Person.
Realselbst: Bild, das eine Person von sich selbst hat.
Ideal-Selbst: Wie man gerne wäre / wie andere einen gerne hätten.
Diskrepanz: Wenn Realselbst und Ideal-Selbst nicht übereinstimmen.
--- MUSTERLÖSUNG ENDE ---
        """

        few_shot_response = {
            "results": [
                {
                    "Teilaufgabe": "1a",
                    "Aspekt": [
                        {
                            "Aspekt": "Definition von Identität",
                            "Beleg_Schüleraufsatz": "Die Identität ist das Selbstverständnis eines Menschen als Individuum [1]",
                            "Beleg_Musterlösung": "Selbstverständnis eines Menschen als unverwechselbare Person.",
                            "Kommentar": "Der Schüler hat die Definition korrekt übernommen."
                        }
                    ]
                },
                {
                    "Teilaufgabe": "1b",
                    "Aspekt": [
                        {
                            "Aspekt": "Realselbst",
                            "Beleg_Schüleraufsatz": "Tim glaubt, dass er nichts wert ist und sich niemand für ihn interessiert [2]",
                            "Beleg_Musterlösung": "Bild, das eine Person von sich selbst hat.",
                            "Kommentar": "Realselbst korrekt beschrieben."
                        },
                        {
                            "Aspekt": "Ideal-Selbst",
                            "Beleg_Schüleraufsatz": "Er hat das Gefühl, seine Eltern wollen etwas anderes aus ihm machen [3]",
                            "Beleg_Musterlösung": "Wie man gerne wäre / wie andere einen gerne hätten.",
                            "Kommentar": "Ideal-Selbst richtig erkannt."
                        }
                    ]
                }
            ]
        }

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": few_shot_student.strip()},
            {"role": "assistant", "content": json.dumps(few_shot_response, indent=2, ensure_ascii=False)},
            {
                "role": "user",
                "content": f"""
                Bitte analysiere den folgenden Schüleraufsatz im Vergleich zur Musterlösung und gib für jede Teilaufgabe an, welche Aspekte vom Schüler korrekt wiedergegeben wurden. 
                Ordne alle Aspekte klar der jeweiligen Teilaufgabe zu und gib nur relevante Informationen wieder. Achte darauf die Nachfolgenden Zeilennummern mit anzuhängen!
                Halte dich dabei strikt an die im JSON-Schema geforderte Struktur. 

                --- AUFSATZ START ---
                {data["student_text"]}
                --- AUFSATZ ENDE ---

                --- MUSTERLÖSUNG START ---
                {data["solution_text"]}
                --- MUSTERLÖSUNG ENDE ---
                                """.strip()
                            }
                        ]

        try:
            return self.language_client.get_response(
                messages=messages,
                schema=ExtractionResult,
                temperature=0.0,
                seed = 42
            )

        except ValidationError as ve:
            print("JSON konnte nicht validiert werden:", ve)
            if self.debug:
                with open(os.path.join(self.debug_folder, "validation_error.json"), "w", encoding="utf-8") as f:
                    f.write(ve.json(indent=2))
            raise ve

        except Exception as e:
            print("Fehler beim Verarbeiten der LLM-Antwort:", str(e))
            if self.debug:
                with open(os.path.join(self.debug_folder, "llm_response_failed.txt"), "w", encoding="utf-8") as f:
                    f.write(str(e))
            raise e

