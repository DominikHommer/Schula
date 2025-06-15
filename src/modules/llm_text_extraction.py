import os
from langchain_core.messages import SystemMessage

from libs.language_client import LanguageClient
from .llm_module_base import LLMModule

class LLMExtraction(LLMModule):
    """
    """
    def __init__(self, debug=False, debug_folder="debug/debug_llm_extraction"):
        super().__init__("llm_extraction")

        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return []
    
    def get_system_prompt(self, student_text, solution_text) -> SystemMessage:
        return SystemMessage(content=f"""
        Du bist ein sehr präzises Assistenz-LLM, das darauf spezialisiert ist, Musterlösungen einer Klausuraufgabe genaustens zu analysieren.
        Deine Aufgabe ist die entschprechenden Teilaufgaben und Anforderungen im Schülertext zu suchen und wiederzugeben ob diese erfolgreich bearbeitet wurden oder nicht.
        Hierfür steht dir immer die jeweilige Aufgabe samt Teilaufgaben aus der Musterlösung zur Verfügung. Außerdem die entsprechende Aufgabe aus der Klausur des Schülers.
        Wenn es bei der Transkripierung zu Problemen kam, wird dir die gesamte Klausur zur Verfügung gestellt, versuche dies ebenfalls bei deiner Zuordnung zu berücksichtigen.

                             
        Hier ist der Aufsatz des Schülers:
        --- AUFSATZ START ---
        {student_text}
        --- AUFSATZ ENDE ---

        Hier sind die Musterlösung:
        --- MUSTERLÖSUNG START---
        {solution_text}
        --- MUSTERLÖSUNG ENDE---


        **Wichtige Anweisungen für die Zuordnung jedes Satzes:**
        1.  **Ausgabe Sprache**: Antworte auf Deutsch.
        2.  **Fehlerhafter Aufsatz**: Beachte, dass der Aufsatz durch eine Handschrifterkennung transkripiert wurde, weshalb dieser erhebliche Rechtschreib-, Grammatik- und Logikfehler aufweisen kann, welche nicht auf den Schüler zurückzuführen sind. Versuche dies bei der Zuordnung zu berücksichtigen.
        3.  **Zuordung zur Aufgabe**: Da dir wohlmöglich die gesamte Klausur zur Verfügung gestellt wird merke es an, dass die Lösung der Aufgabe in einer anderen Aufgabe erwähnt wurde, falls es dir ersichtlich scheint.
        4.  **Klare Antwort**: Antworte nur mit für die dir gestellte Aufgabe relevanten Information, KEINE Abschließenden Worte oder sonstige Anmerkunge die unrelevant sind.   """)
    
    def get_structured_output():
        pass

    def process(self, data: dict, llm: LanguageClient) -> list:
        ## TODO: use a structured output here
        #llm.use_structured_output(self.get_structured_output())

        return llm.get_response([
            self.get_system_prompt(data["student_text"], data["solution_text"])
        ])
