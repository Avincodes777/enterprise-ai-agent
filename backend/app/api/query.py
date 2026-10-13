"""API router for RAG query endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_rag_answer_generator_dependency
from app.rag.answer_generator import (
    RAGAnswerGenerationError,
    RAGAnswerGenerator,
    RAGResponse,
)
from app.rag.context_builder import RAGContextError
from app.schemas.query import QueryRequest
from app.services.embedding import EmbeddingError
from app.services.llm import LLMError
from app.services.retriever import RetrieverError
from app.services.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


@router.post(
    "/query",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Query enterprise knowledge base",
    description="Execute grounded retrieval-augmented answer generation for a given query.",
)
def query_knowledge_base(
    request: QueryRequest,
    answer_generator: RAGAnswerGenerator = Depends(get_rag_answer_generator_dependency),
) -> RAGResponse:
    """
    Execute RAG answer generation for the incoming user query.

    Args:
        request: Validated query request parameters.
        answer_generator: Injected RAGAnswerGenerator service.

    Returns:
        Structured RAGResponse containing answer, sources, and context.
    """
    try:
        response = answer_generator.answer(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            collection_name=request.collection_name,
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        cause = getattr(exc, "__cause__", None)
        exc_str = str(exc).lower()

        # Check if error is related to LLM provider
        if (
            isinstance(exc, LLMError)
            or isinstance(cause, LLMError)
            or "llm" in exc_str
        ):
            logger.error("LLM failure during query processing: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Language model provider error encountered while generating answer.",
            )

        # Check if error is related to retrieval / vector store / embedding
        if (
            isinstance(exc, (RetrieverError, VectorStoreError, EmbeddingError, RAGContextError))
            or isinstance(cause, (RetrieverError, VectorStoreError, EmbeddingError, RAGContextError))
            or "retriev" in exc_str
            or "context" in exc_str
            or "vector" in exc_str
        ):
            logger.error("Retrieval failure during query processing: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Knowledge retrieval service is currently unavailable.",
            )

        # General / unexpected operational failures
        logger.error("Unexpected error during query processing: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the query.",
        )
