"""Module with functions for the retriever."""
import bm25s


def load_retriever(model_path: str = "data/processed") -> bm25s.BM25:
    """Load the retriever.

    Args:
        model_path (str, optional): path to BM25 model.
        Defaults to "data/processed".

    Returns:
        bm25s.BM25: the loaded model
    """
    return bm25s.BM25.load(f"{model_path}/bm25_model", load_corpus=True)


def search_query(query: str, retriever: bm25s.BM25,
                 k: int = 5) -> list:
    """Search a query in the indexed documents.

    Args:
        query (str): the query
        retriever (bm25s.BM25): the BM25 model
        k (int, optional): The max documents to return. Defaults to 5.

    Returns:
        list: the documents that have a link with the query
    """
    query_tokens = bm25s.tokenize(query)
    documents = retriever.retrieve(query_tokens, k=k,
                                   return_as="documents")

    docs: list = documents[0].tolist()
    return docs
