"""Pydantic models for the project."""
import uuid
from pydantic import BaseModel, Field
from typing import List


class MinimalSource(BaseModel):
    """Model that represents a minimal source of information."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Model that represents an unanswered question."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Model that represents an answered question."""

    sources: List


class RagDataset(BaseModel):
    """Model that represents a dataset of RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Model that represents the search result."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Model that represents the search result and an answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Model that represents search results."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Model that represents search results with answers."""

    search_results: List[MinimalAnswer]
    k: int
