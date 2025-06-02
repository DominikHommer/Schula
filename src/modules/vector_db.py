import typing
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from .module_base import Module

from models.parser.assignment_sheet import AssignmentSheet
from models.parser.model_solution import ModelSolution
from models.parser.schulbuch_seite import SchulbuchSeite

class VectorDBModule(Module):
    def __init__(self, module_key="vector_store", index_path="faiss_index"):
        super().__init__(module_key)
        self.index_path = index_path
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = self._load_index_if_exists()

    def _load_index_if_exists(self):
        if os.path.exists(self.index_path):
            return FAISS.load_local(
                self.index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True  # only when trusty source
            )
        return None


    def get_preconditions(self) -> list[str]:
        return ["structured"]

    def process(self, data: dict):
        file_type = data.get("file_type")
        structured_blocks = data.get("structured", [])

        documents = []

        for block in structured_blocks:
            try:
                if file_type == "assignment":
                    obj = AssignmentSheet(**block)
                    content = self._format_assignment(obj)
                elif file_type == "solution":
                    obj = ModelSolution(**block)
                    content = self._format_solution(obj)
                elif file_type == "book_page":
                    obj = SchulbuchSeite(**block)
                    content = self._format_book_page(obj)
                else:
                    return {"status": "error", "message": f"Unbekannter file_type: {file_type}"}

                documents.append(Document(page_content=content, metadata={"file_type": file_type}))
            except Exception as e:
                return {"status": "error", "message": f"Fehler beim Parsen: {e}"}

        if not documents:
            return {"status": "warning", "message": "Keine Dokumente zum Indexieren gefunden."}

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        else:
            self.vectorstore.add_documents(documents)

        self.vectorstore.save_local(self.index_path)

        return {"status": "ok", "num_docs": len(documents)}

    def search(self, query: str, k: int = 3):
        if not self.vectorstore:
            raise Exception("Vectorstore ist leer oder nicht initialisiert.")
        return self.vectorstore.similarity_search(query, k=k)

    def _format_assignment(self, assignment: AssignmentSheet) -> str:
        lines = [f"Titel: {assignment.title}", f"Fach: {assignment.subject}", f"Stufe: {assignment.grade}"]
        if assignment.instructions:
            lines.append(f"Hinweise: {assignment.instructions}")
        for task in assignment.tasks:
            lines.append(f"Aufgabe {task.number}: {task.title} - {task.instruction}")
            for sub in task.subtasks:
                lines.append(f"{sub.label} {sub.instruction}")
        if assignment.raw_text:
            lines.append(f"Rohtext: {assignment.raw_text}")
        return "\n".join([l for l in lines if l])

    def _format_solution(self, solution: ModelSolution) -> str:
        lines = [f"Titel: {solution.assignment_title}", f"Fach: {solution.subject}"]
        for sol in solution.solutions:
            lines.append(f"Aufgabe {sol.number}: {sol.title} - {sol.solution_text}")
            for sub in sol.subsolutions:
                lines.append(f"{sub.label} {sub.solution}")
        if solution.raw_text:
            lines.append(f"Rohtext: {solution.raw_text}")
        return "\n".join([l for l in lines if l])

    def _format_book_page(self, page: SchulbuchSeite) -> str:
        lines = [f"Seite: {page.page_number}", f"Titel: {page.title}"]
        for block in page.text_blocks:
            lines.append(f"Überschrift: {block.heading}")
            lines.extend(block.paragraphs)
        for info in page.infographics:
            lines.append(f"Infografik: {info.title}")
            lines.extend(info.content)
        if page.raw_text:
            lines.append(f"Rohtext: {page.raw_text}")
        return "\n".join([l for l in lines if l])
