import bm25s


def load_retriever(model_path: str = "data/processed") -> bm25s.BM25:
    return bm25s.BM25.load(f"{model_path}/bm25_model", load_corpus=True)


def search_query(query: str, retriever: bm25s.BM25,
                 k: int = 5) -> list:
    query_tokens = bm25s.tokenize(query)
    documents = retriever.retrieve(query_tokens, k=k,
                                   return_as="documents")

    docs: list = documents[0].tolist()
    return docs
