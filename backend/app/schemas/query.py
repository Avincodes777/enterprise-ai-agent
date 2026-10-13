"""Pydantic schemas for RAG query request and response."""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

from app.rag.answer_generator import RAGResponse


class QueryRequest(BaseModel):
    """Request model for RAG knowledge query endpoint."""

    query: str = Field(..., description="User question or search query")
    top_k: Optional[int] = Field(
        default=None, description="Maximum number of context chunks to retrieve"
    )
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum similarity score threshold (0.0 to 1.0)"
    )
    collection_name: Optional[str] = Field(
        default=None, description="Target vector store collection name"
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: Any) -> str:
        """Validate query is a non-empty, non-whitespace string without silent coercion."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("query cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, value: Any) -> Optional[int]:
        """Validate top_k is a positive integer when provided without silent coercion."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("top_k must be a positive integer")
        return value

    @field_validator("score_threshold", mode="before")
    @classmethod
    def validate_score_threshold(cls, value: Any) -> Optional[float]:
        """Validate score_threshold is between 0.0 and 1.0 when provided without silent coercion."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score_threshold must be a numeric value between 0.0 and 1.0")
        val = float(value)
        if val < 0.0 or val > 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        return val

    @field_validator("collection_name", mode="before")
    @classmethod
    def validate_collection_name(cls, value: Any) -> Optional[str]:
        """Validate collection_name is non-empty and non-whitespace when provided without silent coercion."""
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("collection_name cannot be empty or whitespace-only")
        return value.strip()


# Reuse RAGResponse as the canonical response schema
QueryResponse = RAGResponse
