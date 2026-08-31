"""Module exposing the RAG pipeline over a local FastAPI HTTP API."""
from fastapi import FastAPI, Query
from src.models import StudentSearchResults, StudentSearchResultsAndAnswer
from src.retriever import load_retriever
from src.context import ContextBuilder
from src.generation import LLMGenerator
from src.models import MinimalSearchResults, MinimalSource, MinimalAnswer
import uuid

app = FastAPI(
    title="RAG-against-the-machine API",
    description="Local HTTP API to query the BM25 index and"
    " generate answers with Qwen.",
    version="1.0"
)

print("Starting API and loading models...")
retriever = load_retriever()
context_builder = ContextBuilder()
generator = LLMGenerator()


@app.get("/search", response_model=StudentSearchResults)
def api_search(
    query: str = Query(..., description="The query to search"),
    k: int = Query(5, ge=1, le=50, description="Number of results")
) -> StudentSearchResults:
    """Endpoint HTTP to search in the local index.

    Args:
        query (str, optional): The query to search.
        k (int, optional): Number of results. Defaults to 5.

    Returns:
        StudentSearchResults: The StudentSearchResults model
    """
    from src.retriever import search_query

    if not query or query.isspace():
        return StudentSearchResults(
            search_results=[
                MinimalSearchResults(question_id=str(uuid.uuid4()),
                                     question=query,
                                     retrieved_sources=[])
            ],
            k=k
        )

    documents = search_query(query, retriever, k)
    minimal_sources = []
    for doc in documents:
        if isinstance(doc, dict):
            meta = doc.get("metadata", {})
            if isinstance(meta, dict):
                minimal_sources.append(MinimalSource(
                    file_path=meta.get("source", ""),
                    first_character_index=meta.get("first_character_index", 0),
                    last_character_index=meta.get("last_character_index", 0)
                ))

    search_result = MinimalSearchResults(
        question_id=str(uuid.uuid4()),
        question=query,
        retrieved_sources=minimal_sources
    )
    return StudentSearchResults(search_results=[search_result], k=k)


@app.get("/answer", response_model=StudentSearchResultsAndAnswer)
def api_answer(
    query: str = Query(..., description="The query to answer"),
    k: int = Query(5, ge=1, le=50, description="Number of context sources")
) -> StudentSearchResultsAndAnswer:
    """Endpoint HTTP to answer a question with Qwen.

    Args:
        query (str, optional): The query to answer.
        k (int, optional): Number of context sources. Defaults to 5.

    Returns:
        StudentSearchResultsAndAnswer: The StudentSearchResultsAndAnswer model
    """
    search_res = api_search(query, k).search_results[0]
    context = context_builder.format_context(search_res)
    answer_text = generator.generate_answer(query, context)

    minimal_answer = MinimalAnswer(
        answer=answer_text,
        question_id=search_res.question_id,
        question=query,
        retrieved_sources=search_res.retrieved_sources
    )

    return StudentSearchResultsAndAnswer(
        search_results=[minimal_answer],
        k=k
    )
