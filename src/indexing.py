"""Module with function to index the data."""
import bm25s
from typing import List
from langchain_core.documents import Document
from pathlib import Path


def store_chunks(chunks: List[Document],
                 output: str = "data/processed/") -> None:
    """Index and store the model to specified output.

    Args:
        chunks (List[Document]): chunks to index
        output (str, optional): output dir. Defaults to "data/processed/".
    """
    corpus = [chunk.page_content for chunk in chunks]
    metadata_corpus = [{"page_content": chunk.page_content,
                        "metadata": chunk.metadata} for chunk in chunks]
    corpus_tokens = bm25s.tokenize(corpus)

    retriever = bm25s.BM25(corpus=metadata_corpus)
    print("Indexing curpus...")
    retriever.index(corpus_tokens, leave_progress=True)

    Path(output).mkdir(parents=True, exist_ok=True)
    print(f"Saving bm25 model to {output}/bm25_model")
    retriever.save(f"{output}/bm25_model")
