from modules.llm_text_extraction import LLMExtraction
from libs.language_client import LanguageClient

from .llm_pipeline import LLMPipeline

from typing import Dict
from models.parser.model_solution import ModelSolution
from models.parser.student_text import StudentText
import streamlit as st

def update_progress(current_page: int, total_pages: int, progress_bar, status):
    progress_bar.progress(current_page / total_pages)
    status.text(f"Verarbeite Seite {current_page} von {total_pages}")

def _combine_task_solution_text(task_model: ModelSolution) -> str:
    """Helper function to convert a single TaskSolution object into a single text block."""
    parts = []
    # Add task number and title for context
    if task_model.number is not None:
        title_part = f" ({task_model.title})" if task_model.title else ""
        parts.append(f"Task {task_model.number}{title_part}:")

    if task_model.solution_text:
        parts.append(task_model.solution_text)
    
    for sub_solution in task_model.subsolutions:
        if sub_solution.solution:
            text_to_add = sub_solution.solution
            if sub_solution.label:  # Optionally prepend label
                text_to_add = f"{sub_solution.label} {text_to_add}"
            parts.append(text_to_add)
            
    return "\n".join(parts) if parts else "No text provided for this task."

class LLMTextExtractorPipeline(LLMPipeline):
    """
    Pipeline zur Extraktion der Musterlösung aus dem Schülertext
    Sollte im Streamlit Kontext verwendet werden
    TODO: -> Input sollte Schülertext, splitted Musterlösung enthalten
          -> Extraktion sollte async parallel bearbeitet werden
    """
    def __init__(self, llmClient: LanguageClient, input_data: dict | None = None):
        super().__init__(llmClient, input_data)

        self.add_stage(LLMExtraction(debug=False))

    def process_solutions(self, model_solution: ModelSolution, student_solution: StudentText) -> Dict[str, any]:
        """
        Processes model and student solutions to generate LLM prompts.
        """
        progress_bar = st.progress(0.0)
        status = st.empty()

        raw_student_text = student_solution.raw_text

        llm_responses = {}
        print(f"Processing Assignment: {model_solution.assignment_title or 'N/A'}")
        print(f"Subject: {model_solution.subject or 'N/A'}\n")

        # Filter tasks that have a number to iterate over
        model_tasks = [task for task in model_solution.solutions if task.number is not None]
        total_tasks = len(model_tasks)

        # Loop through the valid tasks
        with st.status("Starte Aufgabenverarbeitung...", expanded=True) as status:
            for j, model_task in enumerate(model_tasks):
                task_id_str = str(model_task.number)
                combined_model_solution = _combine_task_solution_text(model_task)
                data = {"student_text": raw_student_text, "solution_text": combined_model_solution}

                # Progress update
                progress = (j + 1) / len(model_tasks)
                progress_bar.progress(progress)
                status.update(label=f"Verarbeite Aufgabe: {j+1} von {total_tasks}")

                llm_responses[task_id_str] = self.run(data)
                print(f"Generated prompt for Task {task_id_str} (using full student text).")

        st.success("Alle Aufgaben verarbeitet!")

        return llm_responses