import bm25s
from typing import List
from langchain_core.documents import Document
from pathlib import Path


def store_chunks(chunks: List[Document],
                 output: str = "data/processed/") -> None:
    corpus = [chunk.page_content for chunk in chunks]
    metadata_corpus = [{"page_content": chunk.page_content,
                        "metadata": chunk.metadata} for chunk in chunks]
    corpus_tokens = bm25s.tokenize(corpus)

    retriever = bm25s.BM25(corpus=metadata_corpus)
    retriever.index(corpus_tokens)

    Path(output).mkdir(parents=True, exist_ok=True)
    retriever.save(f"{output}/bm25_model")


def load_retriever(model_path: str = "data/processed") -> bm25s.BM25:
    return bm25s.BM25.load(f"{model_path}/bm25_model", load_corpus=True)


def search_query(query: str, retriever: bm25s.BM25,
                 k: int = 5) -> list:
    query_tokens = bm25s.tokenize(query)
    documents = retriever.retrieve(query_tokens, k=k,
                                   return_as="documents")

    return documents
