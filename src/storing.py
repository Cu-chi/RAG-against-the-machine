from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")


def create_vector_store(chunks: list[Document], db: str) -> Chroma:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,
        persist_directory=db,
    )
    return vector_store


def search_vector_store(query: str, db: str, k: int = 5) -> list[Document]:
    vector_store = Chroma(
        persist_directory=db,
        embedding_function=embeddings
    )

    results = vector_store.similarity_search(query, k=k)
    return results
