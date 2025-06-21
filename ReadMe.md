# Schula

**Schula** ist eine Anwendung zur Analyse und Bewertung von Schülerantworten basierend auf einer Musterlösung (Erwartungshorizont). Es verwendet Language Models (LLMs), um Textsegmente von Schülerantworten den entsprechenden Punkten der Musterlösung zuzuordnen.

## Features

* Pipeline-basierter Aufbau mit klar definierten Modulen
* Integration von LLMs für semantisches Matching
* Nutzung von Tools wie OCR, Zeilendetektion, Strikethrough-Erkennung
* Web-GUI über Streamlit

## Ausführung

* Starte die Webanwendung mit:

```bash
streamlit run src/main_app.py
```

* Alternativ kann eine Pipeline direkt gestartet werden:

```bash
python3 src/main.py
```

## Beispiel-Anwendung

```bash
streamlit run src/main_app.py
```

Die Anwendung lässt sich lokal starten und bietet eine visuelle Benutzeroberfläche zur Interaktion mit dem Dokumenten-Parser und dem LLM-Modul.

## Installation

Das Projekt kann mit dem folgenden Befehl installiert werden:

```bash
git clone https://github.com/DominikHommer/Schula
cd schula
pip install -r requirements.txt
```

> **Hinweis:** Einige Pakete (z. B. `ultralytics`, `tensorflow`) müssen zusätzlich manuell installiert werden. Siehe Abschnitt [Anforderungen](#anforderungen).

## Projektstruktur

```
schula/
├── src/
│   ├── main_app.py               # Einstiegspunkt für die Streamlit-App
│   ├── app.py                    # Alternativer App-Einstieg (veraltet oder Headless)
│   ├── app_v2.py                 # Experimentelle Streamlit-Version
│   ├── libs/                     # LLM-Clients, Architektur, Hilfsfunktionen
│   │   ├── language_client.py
│   │   ├── file_helper.py
│   │   ├── Architecture.md       # (ehemalig, Inhalte nun in README integriert)
│   ├── modules/                  # Verarbeitungsmodule und zentrale Komponenten
│   │   ├── red_remover.py
│   │   ├── text_recognizer.py
│   │   ├── strikethrough_cleaner.py
│   │   ├── horizontal_cutter.py
│   │   ├── horizontal_cutter_line_detect.py
│   │   ├── line_cropper.py
│   │   ├── line_denoiser.py
│   │   ├── line_prepare_recognizer.py
│   │   ├── structured_document_parser.py  # CV-Pipeline Einstiegspunkt
│   │   ├── llm_text_extraction.py         # LLM-Zuordnungsschicht
│   │   ├── module_base.py
│   │   ├── llm_module_base.py
│   ├── pipelines/                # Konkrete Verarbeitungspipelines
│   │   ├── pipeline.py
│   │   ├── llm_pipeline.py
│   │   ├── llm_extractor.py
│   │   ├── pdf_processor.py
│   │   ├── student_exam_extractor.py
│   │   ├── cv_pipeline.py
│   ├── tests/                    # Unit- und Integrationstests

```

## Architektur

Die Architektur von **Schula** folgt einem modularen, gekapselten Aufbau, der stark auf einen klar definierten LLM-Lifecycle setzt. Jede Einheit der Verarbeitung (ob CV oder LLM) kapselt ihre eigenen Schritte und kommuniziert nur über definierte Schnittstellen:

* **CV-Pipeline**: Verantwortlich für alle bildverarbeitenden Schritte – von der PDF-Zerlegung über Segmentierung bis zur OCR. Ergebnis sind strukturierte Texteinheiten.
* **CV-Pipeline**: Verantwortlich für alle bildverarbeitenden Schritte – von der PDF-Zerlegung über Segmentierung bis zur OCR. Ergebnis sind strukturierte Texteinheiten.
* **LLMClient** (in `libs/`): Vermittelt zwischen LLMs und Modulen

  * Initialisiert LLMs
  * Führt Tool-/Agent-Schleifen aus
  * Gibt ein Result zurück
* **LLMPipeline** (in `pipelines/`): Organisiert die Zusammenarbeit

  * Erstellt und verwaltet LLMClient
  * Übergibt ihn an Module
* **Module (in ****************************`modules/`****************************)**:

  * Nutzen LLMClient
  * Definieren ihre Prompts und Tools
  * Führen ihren eigenen Lifecycle aus (Init → Loop → Result)

### 1. Modul-Basisklassen (module\_base.py & llm\_module\_base.py)

* `module_base.py`: Definiert eine abstrakte Basisklasse für alle Verarbeitungsschritte (OCR, Bildvorverarbeitung etc.) mit einer einheitlichen `run()`-Schnittstelle.
* `llm_module_base.py`: Erweitert die Basisklasse um LLM-spezifische Funktionalität wie Prompt-Erstellung, Model-Interaktion und strukturierte Ausgabe via Pydantic.

### 2. LLMClient (src/libs/language\_client.py)

* Verwaltet die gesamte Kommunikation mit externen Language Models (z. B. OpenAI, Groq, HuggingFace).
* Jedes Modul, das LLMs verwendet, startet einen eigenen LLM-Lifecycle:

  * Initialisierung
  * Tool-/Agent-Loop mit interaktiver Verarbeitung
  * Ergebnis-Rückgabe

### 3. LLMPipeline (src/pipelines/llm\_pipeline.py)

* Zentraler Koordinator für LLM-basierte Module.
* Ruft relevante Extraktoren auf (z. B. `llm_extractor`, `student_exam_extractor`).
* Nutzt `language_client` zur Abwicklung der Prompt-Kommunikation.

### 4. CV-Pipeline (modules/structured\_document\_parser.py + pipelines/pdf\_processor.py + pipelines/cv\_pipeline.py)

* Diese Pipeline übernimmt alle bildbasierten Vorverarbeitungsschritte.
* Typische Schritte:

  * Laden und Rendern von PDF-Seiten als Bilder
  * Entfernen von roten Markierungen (`red_remover.py`)
  * Entfernen von Durchstreichungen (`strikethrough_cleaner.py`)
  * Horizontales Schneiden in Zeilen (`horizontal_cutter.py`, `horizontal_cutter_line_detect.py`)
  * Zuschneiden und Entstören von Zeilenbildern (`line_cropper.py`, `line_denoiser.py`)
  * OCR mit `text_recognizer.py`
  * Gruppierung und Strukturierung der Textergebnisse
* Ergebnis: Strukturierte, bereinigte Texteinheiten zur Übergabe an die LLM-Pipeline

### 5. Weitere Pipelines (src/pipelines)

* `llm_extractor.py`: Führt LLM-Analyse durch und parst strukturierte Ergebnisse.
* `student_exam_extractor.py`: Extrahiert relevante Abschnitte aus Schülerantworten.

### 6. Module (src/modules)

* Alle OCR-, Bildverarbeitungs- und Preprocessing-Module erben von `ModuleBase`.
* LLM-gestützte Module erben von `LLMModuleBase` und definieren eigene Prompts sowie erwartete Antwortformate.
* Der modulare Aufbau erlaubt einfache Erweiterbarkeit und Wiederverwendung.

## Typischer Ablauf

1. **PDF-Upload**: Ein Dokument mit Musterlösung und Schülerantwort wird geladen.
2. **Preprocessing**: Entfernen von Linien, Durchstreichungen, Farbinformationen.
3. **OCR**: Erkennung der Textinhalte aus den Segmenten.
4. **Segmentierung**: Zerlegung in Textblöcke (z. B. Aufgaben, Lösungen, Randnotizen).
5. **LLM Matching**: Über Pydantic-Modelle und Prompts werden die Textsegmente den Punkten aus dem Erwartungshorizont zugeordnet.
6. **Darstellung**: Ergebnisse und Zuweisungen werden über die Streamlit-Oberfläche visualisiert.

## Anforderungen

* Python < 3.13 wird empfohlen
* Zusätzliche manuelle Installation nötig:

  * `ultralytics==8.3.115`
  * `tensorflow==2.19.0`

## Linting & Tests

### Linting

* Ausführen von `pylint src/` zur statischen Codeanalyse
* Dokumentation: [https://docs.pylint.org/](https://docs.pylint.org/)
* Optional: Typprüfung mit `mypy src/`
* Dokumentation: [https://mypy.readthedocs.io/en/latest/getting\_started.html](https://mypy.readthedocs.io/en/latest/getting_started.html)

### Unit Tests

* Jedes Modul sollte durch Unit-Tests abgedeckt sein
* Pipelines sollten sinnvoll getestet werden
* Tests befinden sich im Ordner `tests`, Mockdaten unter `tests/fixtures`
* Tests ausführen mit:

```bash
cd src
python -m unittest
```

## Lizenz

MIT License
