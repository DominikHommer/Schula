### imports ###
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma

import ollama

from vector import retriever

DB_PATH = "vector_db_test"
### ------- ###


model = OllamaLLM(model="deepseek-r1:70b")

# define template for answering questions

template = """ 
Du bist ein Deutschlehrer und sollst die Aufgaben einer Deutschklausur gemäß des Erwartungshorizontes bewerten.

Hier ist der Erwartungshorizont und Hintergründe aus dem im Unterricht verwendeten Deutschbuch: {task}

Hier ist die Frage des Lehrers: {question}

Du sollst hierbei wie folgt vorgehen:
---
Define precise tasks here.
---

"""

# db = Chroma(persist_directory=DB_PATH, embedding_function=)

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    # get knowledge from the school textbook for example
    # task = db.similarity_search_with_score(question,  k=5)
    task = retriever.invoke(question)
    result = chain.invoke({"task":task, "question":question})
    print(result)