import os
from langchain_core.messages import SystemMessage

from libs.language_client import LanguageClient
from .llm_module_base import LLMModule

class LLMExtraction(LLMModule):
    """
    TODO: DECLARE
    """
    def __init__(self, debug=False, debug_folder="debug/debug_llm_extraction"):
        super().__init__("llm_extraction")

        # THIS IS JUST EXAMPLE DATA TODO: FIX ASAP!
        self.student_essay = """ 
1. Identität in der Psychologie meint das Selbstverständnis eines Menschen als einzigartige und unverwechselbare Person, sowohl in ihrer eigenen Betrachtung, als auch in der ihrer Umwelt. Sie beinhaltet <vor> Vorstellungen darüber, für wen man sich hält, wer man gerne sein möchte, wer man werden möchte, was andere von einen denken und wie andere einen gerne hätten, Im Inhalt des Fallbeispiels “Tim” wird deutlich gemacht, dass Tims Zukunftspläne sich stark von den Wünschen seiner Eltern unterscheiden. Er fühlt sich bezüglich<e> der Berufswahl oft unsicher (Z.4) und möchte keinen Rat von seinen Eltern annehmen (Z.7). Da sich seine eigenen Wert- und Normvorstellungen stark von denen seiner Eltern unterscheiden und der sich mit Ihnen nicht identifizieren kann, spricht man bei Tim von der <diffusen:kritischen> Identität. Die kritische Identität, auch Moratoriumstidentität genannt, ist nur eine von vier möglichen Identitäten. Laut Marcia erfolgt in dieser eine starke Reflexion der als Kind übernommenen Werthaltungen, jedoch besteht nur ein niedrige Selbstwertgefühl und Unentschlossenheit. Tim denkt, er sein in seiner Klasse ein “Looser” und keiner könne <in:ihn> <Le> Leiden (siehe Anlage), außerdem hat er Schwierigkeiten, enge Freundschaften zu schließen (Z.9). Er hinterfragt zwar die Vorstellungen seiner Eltern (Z.11). besitzt jedoch keine Experimentierfreudigkeit oder Verpflichtung gegenüber seinen eigenen Wünsche.
2. <Jeder Mensch: Die Persönlichkeit des Menschen> entwickelt sich im Laufe des Lebens. Carl Rogers erklärt mit seiner personenzentrierten Theorie die Entwicklung der Persönlichkeit und welche psychischen Probleme dabei entstehen können. Die Grundannahmen dieser Theorie beinhalten die Aktualisierungstendenz, welche das angeborene und <B> bestehende Bestreben des Menschen nach Selbstbestimmung und Unabhängigkeit meint, <di> das organisimische Bewerten, bei dem Wünsche und Ziele anschließend gefördert oder wieder eingeschränkt werden können und zuletzt das Selbstkonzept, welches im Folgenden genauer verdeutlicht wird. Das Selbstkonzept besteht aus dem Real-Selbst und dem Ideal-Selbst. Während das Real-Selbst das tatsächliche Bild von einem bedeutet, beschreibt das Ideal-Selbst die <Werte und Normen> Person, die man gerne sein möchte, Das Real- und das Ideal-Selbst sollten sich nur so wenig wie möglich unterscheiden, da es sonst zu einer Selbstinkonsistenz führen kann. Selbstkonsistenz meint die Unstimmigkeit der zwei <”selbst”> Selbsts. Tim hat <Vorstellungen: Interessen> und Wünsche (Z.3), welche sein <Real:Ideal>-Selbst deutlich definieren. Durch sein Umfeld wird ihm jedoch en. was ganz anderes vermittelt (siehe Anlage), wodurch sich sein Ideal-Selbst stark vom Real-Selbst unterscheidet. Dadurch könnte sich bei Tim ein <starke> starres Selbstkonzept entwickelt haben. Dies meint das<s> starke Misstrauen gegenüber seinen Bezugspersonen, wie zum Beispiel der Gedanke, dass keiner einen Künstler mögen würde (siehe Anlage) und die fehlende Kommunikation mit seinen Eltern (Z.11). Tim ist sein Talent und der damit verbundene Zukunftswunsch bewusst, jedoch setzt er sich aufgrund der verschiedenen Faktoren nicht mit seinen Zweifeln auseinander.
"""
        self.task1 = "Verdeutlichen Sie Tims Identitässtatus auf der Grundlage des Identitätsmodells nach Marcia"
        self.task2 = "Die Bildung des Selbstkonzepts steht in eingem Zusammenhang mit dem Verhalten von Bezugspersonen. Erklären Sie mithilfe relevanter Annahmen der personenzentrierten Theorie nach Rogers, wie sich ein starres Selbstkonzept von Tim entwickelt haben könnte. (Die Grundannahmen der personenzentrierten Theorie nach Rogers sowie die Inkongurentz brauchen nicht verdeutlicht werden)"
        
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return []
    
    def get_system_prompt(self) -> SystemMessage:
        return SystemMessage(content=f"""
Du bist ein sehr präzises Assistenz-LLM, das darauf spezialisiert ist, Schüleraufsätze Satz für Satz zu analysieren.
Deine Aufgabe ist es, den gegebenen Aufsatz sorgfältig zu lesen. Für JEDEN EINZELNEN SATZ des Aufsatzes musst du entscheiden, zu welcher der beiden genannten Aufgabenstellungen er primär gehört. Der Originaltext des Schülers darf dabei unter KEINEN Umständen verändert werden.

Hier ist der Aufsatz des Schülers:
--- AUFSATZ START ---
{self.student_essay}
--- AUFSATZ ENDE ---

Hier sind die beiden Aufgabenstellungen:
1. Aufgabenstellung: "{self.task1}"
2. Aufgabenstellung: "{self.task2}"

Bitte analysiere den Aufsatz Satz für Satz und gib den Text so aus, dass JEDER SATZ des Aufsatzes
einer der beiden Aufgabenstellungen zugeordnet ist.

**Wichtige Anweisungen für die Zuordnung jedes Satzes:**
1.  **Satzweise Zuordnung:** Betrachte jeden Satz des Aufsatzes einzeln. Entscheide für jeden einzelnen Satz, zu welcher der beiden Aufgaben er am besten passt.
2.  **Vollständige Zuordnung:** Jeder einzelne Satz des Aufsatzes MUSS einer der beiden Aufgaben zugeordnet werden. Es darf keinen Satz geben, der nicht zugeordnet wird oder als "Nicht zuzuordnen" klassifiziert wird.
3.  **Beste Passung pro Satz:** Ordne jeden Satz der Aufgabe zu, zu der er thematisch die stärkste Verbindung aufweist oder für die er den relevanteren Beitrag leistet.
4.  **Umgang mit Überschneidungen auf Satzebene:** Wenn ein Satz inhaltlich Aspekte beider Aufgaben berührt, entscheide dich für die Aufgabe, die den Hauptfokus dieses spezifischen Satzes am besten widerspiegelt, und ordne ihn dieser primär zu.
5.  **Strikte Textintegrität:** Der Text des Schülers muss für die Zuordnung exakt so wiedergegeben werden, wie er im Originalaufsatz steht. Es dürfen absolut keine Wörter hinzugefügt, entfernt, zusammengefasst, interpretiert oder umformuliert werden. **Die Sätze sollen in der Ausgabe NICHT nummeriert werden.** Die Zuordnung erfolgt ausschließlich durch die korrekte Gruppierung der originalen Sätze unter den jeweiligen Aufgabenüberschriften.
6.  **Beibehaltung der Reihenfolge:** Die Reihenfolge der Aufgaben im Ergebnis muss der gegebenen Reihenfolge (Aufgabe 1, dann Aufgabe 2) entsprechen. Die Sätze, die einer Aufgabe zugeordnet werden, sollten in der Reihenfolge erscheinen, in der sie im Originalaufsatz vorkommen.
7.  **Reines Ausgabeformat:** Antworte bitte ausschließlich mit dem Ergebnis der Zuordnung im oben detailliert beschriebenen Format, **inklusive der Leerzeile nach jeder Aufgabenbeschreibung und vor dem zugeordneten Schülertext**. Gib keine einleitenden Sätze, keine abschließenden Bemerkungen, keine Kommentare, keine Erklärungen und **keine Nummerierungen der Sätze** deinerseits aus. Nur die strukturierte Zuordnung des unformatierten Textes.
        """)
    
    # See here: https://python.langchain.com/docs/how_to/structured_output/#typeddict-or-json-schema
    def get_structured_output(self) -> dict:
        return {
            'title': 'Extracted Tasks',
            'description': 'Tasks extracted from given essay',
            'type': 'object',
            'properties': {
                'extracted_essay': {
                    'type': 'array',
                    'description': 'Complete content of essay for a task',
                    'items': {
                        'type': 'string',
                        'description': 'Content from essay for task'
                    }
                }
            },
            'required': ['extracted_essay'],
        }
    
    def process(self, data: dict, llm: LanguageClient) -> list:
        llm.use_structured_output(self.get_structured_output())

        result = llm.get_response([self.get_system_prompt()])

        # FIXME: STructured output currently is not correctly declared
        return result