import bm25s
from typing import List
from langchain_core.documents import Document
from pathlib import Path


def store_chunks(chunks: List[Document], output: str = "data/processed/"):
    corpus = [chunk.page_content for chunk in chunks]
    metadata_corpus = [dict(chunk) for chunk in chunks]
    corpus_tokens = bm25s.tokenize(corpus)

    retriever = bm25s.BM25(corpus=metadata_corpus)
    retriever.index(corpus_tokens)

    Path(output).mkdir(parents=True, exist_ok=True)
    retriever.save(f"{output}/bm25_model", corpus=corpus)


def search_query(query: str, k: int = 5,
                 model_path: str = "data/processed/"):
    retriever = bm25s.BM25.load(f"{model_path}/bm25_model", load_corpus=True)
    query_tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(query_tokens, k=k)

    return results, scores
