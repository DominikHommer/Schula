from modules.llm_text_extraction import LLMExtraction
from libs.language_client import LanguageClient

from .llm_pipeline import LLMPipeline

from typing import Dict, Union
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

    def process_solutions(self, model_solution: ModelSolution, student_solution: Union[ModelSolution, StudentText]) -> Dict[str, any]:
        """
        Processes model and student solutions to generate LLM prompts.

        Handles two cases:
        1. Task counts match: Compares each task one-to-one.
        2. Task counts differ: Compares each model solution task against the entire student submission.
        """
        llm_responses = {}
        print(f"Processing Assignment: {model_solution.assignment_title or 'N/A'}")
        print(f"Subject: {model_solution.subject or 'N/A'}\n")

        # Case for the structuring of the student solution
        if isinstance(student_solution, ModelSolution):
            # --- SCENARIO 1: Task counts match, process one-to-one ---
            if len(model_solution.solutions) == len(student_solution.solutions):
                print("Task counts match. Processing tasks one-to-one.\n")
            
                for model_task, student_task in zip(model_solution.solutions, student_solution.solutions):
                    if model_task.number is None:
                        print(f"Skipping task with title '{model_task.title or 'Untitled'}' as it has no number.\n")
                        continue

                    task_id_str = str(model_task.number)
                    
                    # Get combined text for both model and student task
                    combined_model_solution = _combine_task_solution_text(model_task)
                    combined_student_answer = _combine_task_solution_text(student_task)

                    data = {"student_text": combined_student_answer, "solution_text": combined_model_solution}
                    llm_responses[task_id_str] = self.run(data)
                    print(f"Generated prompt for Task {task_id_str}.")

            # --- SCENARIO 2: Task counts differ, process each model task against the entire student submission ---
            else:
                print(f"Task counts differ (Model: {len(model_solution.solutions)}, Student: {len(student_solution.solutions)}). Using full student text for each prompt.\n")
                
                # First, combine the ENTIRE student submission into a single string.
                # This is done once to be efficient.
                all_student_tasks_text = []
                for student_task in student_solution.solutions:
                    all_student_tasks_text.append(_combine_task_solution_text(student_task))
                
                # Join with a clear separator to distinguish between tasks for the LLM
                entire_student_submission_text = "\n\n---\n\n".join(all_student_tasks_text)

                # Now, loop through the MODEL solution and create a prompt for each task
                for model_task in model_solution.solutions:
                    if model_task.number is None:
                        print(f"Skipping task with title '{model_task.title or 'Untitled'}' as it has no number.\n")
                        continue
                    
                    task_id_str = str(model_task.number)
                    
                    # Get the solution text for the current model task
                    combined_model_solution = _combine_task_solution_text(model_task)

                    # The student text is always the entire submission
                    data = {"student_text": entire_student_submission_text, "solution_text": combined_model_solution}
                    llm_responses[task_id_str] = self.run(data)
                    print(f"Generated prompt for Task {task_id_str} (using full student text).")

        elif isinstance(student_solution, StudentText):
            progress_bar = st.progress(0.0)
            status = st.empty()

            model_tasks = [task for task in model_solution.solutions if task.number is not None]
            total_tasks = len(model_tasks)
            entire_student_submission_text = student_solution.raw_text

            # Now, loop through the MODEL solution and create a prompt for each task
            for i, model_task in enumerate(model_solution.solutions):
                if model_task.number is None:
                    print(f"Skipping task with title '{model_task.title or 'Untitled'}' as it has no number.\n")
                    continue
                
                task_id_str = str(model_task.number)
                
                # Get the solution text for the current model task
                combined_model_solution = _combine_task_solution_text(model_task)

                # The student text is always the entire submission
                data = {"student_text": entire_student_submission_text, "solution_text": combined_model_solution}
                with st.spinner(f"Bearbeite Aufgabe {task_id_str}..."):
                     llm_responses[task_id_str] = self.run(data)

                # Progress update
                progress = (i + 1) / total_tasks
                progress_bar.progress(progress)
                status.text(f"Verarbeite Aufgaben: {i + 1} von {total_tasks}")

                print(f"Generated prompt for Task {task_id_str} (using full student text).")

        else:
            print("Error: Unkown Struture for student solution.")

        return llm_responses
