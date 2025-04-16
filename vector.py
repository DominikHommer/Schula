from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader  # for pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas

DATA_DIR = "data"
DB_PATH = "vector_db_test"

### functions ###
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len
    )
    return text_splitter.split_documents(documents)


def add_to_db(chunks: list[Document]):
    # load embeddings model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # define vector store
    vector_store = Chroma(
        collection_name="test_collection",
        persist_directory = DB_PATH,    # this avoids regenerating the chroma db every time you run the LLM
        embedding_function = embeddings
    )

    chunks_with_ids = get_chunk_ids(chunks)

    # already existing ids
    existing_items = vector_store.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        # vector_store.persist()
    else:
        print("No new documents to add")

    # experiment with different approaches
    retriever = vector_store.as_retriever(search_kwargs={"k":20})

    return retriever


def get_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

    # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

### --------- ###


### main 

# load pdfs from directory
loader = PyPDFDirectoryLoader(DATA_DIR)
documents = loader.load()
chunks = split_documents(documents)
retriever = add_to_db(chunks)




# if add_documents:

#     # load pdfs from directory
#     loader = PyPDFDirectoryLoader("directory")
#     documents = loader.load()

#     # chunck the text book into manageable pieces
#     chunks = split_documents(documents)

#     last_page_id = None
#     current_chunk_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         current_page_id = f"{source}:{page}"
#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0

#     # Calculate the chunk ID.
#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id

#         # Add it to the page meta-data.
#         chunk.metadata["id"] = chunk_id



# retriever = vector_store.as_retriever(search_kwargs={"k":20})