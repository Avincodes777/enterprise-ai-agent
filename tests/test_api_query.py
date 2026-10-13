"""Focused API tests for the FastAPI RAG query endpoint and application lifecycle."""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_rag_answer_generator_dependency,
    get_shared_rag_answer_generator,
    reset_rag_answer_generator_cache,
)
from app.main import app
from app.rag.answer_generator import (
    DEFAULT_EMPTY_RETRIEVAL_ANSWER,
    RAGAnswerGenerationError,
    RAGResponse,
)
from app.rag.context_builder import RAGContextError
from app.schemas.query import QueryRequest
from app.services.llm import LLMError, LLMProviderError
from app.services.retriever import RetrieverError
from app.services.vector_store import VectorSearchResult, VectorStoreError


class StubRAGAnswerGenerator:
    """Configurable stub for RAGAnswerGenerator dependency injection."""

    def __init__(self, response: RAGResponse = None, exception: Exception = None):
        self.response = response
        self.exception = exception
        self.call_count = 0
        self.last_query = None
        self.last_top_k = None
        self.last_score_threshold = None
        self.last_collection_name = None

    def answer(
        self,
        query: str,
        top_k=None,
        score_threshold=None,
        collection_name=None,
    ) -> RAGResponse:
        self.call_count += 1
        self.last_query = query
        self.last_top_k = top_k
        self.last_score_threshold = score_threshold
        self.last_collection_name = collection_name
        if self.exception:
            raise self.exception
        return self.response


class TestAPIQueryEndpoint(unittest.TestCase):
    """Test suite covering the /api/query endpoint, validation, errors, and lifecycle."""

    def setUp(self):
        self.client = TestClient(app)
        self.sample_sources = [
            VectorSearchResult(
                chunk_id="chunk_sec_01",
                document_name="security_policy.pdf",
                page_number=3,
                chunk_index=1,
                text="Employees must use MFA for external access.",
                score=0.915,
                metadata={"category": "compliance", "version": "2.0"},
            )
        ]
        self.default_response = RAGResponse(
            query="What is the MFA policy?",
            answer="All employees must use MFA for external access.",
            sources=self.sample_sources,
            context_text="[SOURCE 1]\nDocument: security_policy.pdf\nPage: 3\nChunk: 1\nScore: 0.915\nContent:\nEmployees must use MFA for external access.",
        )
        self.stub = StubRAGAnswerGenerator(response=self.default_response)
        app.dependency_overrides[get_rag_answer_generator_dependency] = lambda: self.stub

    def tearDown(self):
        app.dependency_overrides.clear()
        reset_rag_answer_generator_cache()

    # -------------------------------------------------------------------------
    # 1. Valid Query & Structured Response
    # -------------------------------------------------------------------------
    def test_valid_query_returns_structured_response(self):
        """1. Verify valid query returns HTTP 200 with structured JSON response."""
        response = self.client.post("/api/query", json={"query": "What is the MFA policy?"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("query", data)
        self.assertIn("answer", data)
        self.assertIn("sources", data)
        self.assertIn("context_text", data)
        self.assertEqual(data["query"], "What is the MFA policy?")

    # -------------------------------------------------------------------------
    # 2. Answer and Source References Serialization
    # -------------------------------------------------------------------------
    def test_answer_and_source_references_serialized_correctly(self):
        """2. Verify answer and all source provenance fields are serialized accurately."""
        response = self.client.post("/api/query", json={"query": "What is the MFA policy?"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "All employees must use MFA for external access.")
        self.assertEqual(len(data["sources"]), 1)

        source = data["sources"][0]
        self.assertEqual(source["chunk_id"], "chunk_sec_01")
        self.assertEqual(source["document_name"], "security_policy.pdf")
        self.assertEqual(source["page_number"], 3)
        self.assertEqual(source["chunk_index"], 1)
        self.assertEqual(source["text"], "Employees must use MFA for external access.")
        self.assertAlmostEqual(source["score"], 0.915, places=3)
        self.assertEqual(source["metadata"], {"category": "compliance", "version": "2.0"})
        self.assertIn("[SOURCE 1]", data["context_text"])

    # -------------------------------------------------------------------------
    # 3. top_k Parameter Propagation
    # -------------------------------------------------------------------------
    def test_top_k_propagates_to_answer_generator(self):
        """3. Verify top_k parameter propagates to RAGAnswerGenerator."""
        response = self.client.post(
            "/api/query",
            json={"query": "What is the MFA policy?", "top_k": 7},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stub.last_top_k, 7)

    # -------------------------------------------------------------------------
    # 4. score_threshold Parameter Propagation
    # -------------------------------------------------------------------------
    def test_score_threshold_propagates_to_answer_generator(self):
        """4. Verify score_threshold parameter propagates to RAGAnswerGenerator."""
        response = self.client.post(
            "/api/query",
            json={"query": "What is the MFA policy?", "score_threshold": 0.85},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stub.last_score_threshold, 0.85)

    # -------------------------------------------------------------------------
    # 5. collection_name Parameter Propagation
    # -------------------------------------------------------------------------
    def test_collection_name_propagates_to_answer_generator(self):
        """5. Verify collection_name parameter propagates to RAGAnswerGenerator."""
        response = self.client.post(
            "/api/query",
            json={"query": "What is the MFA policy?", "collection_name": "custom_hr_docs"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stub.last_collection_name, "custom_hr_docs")

    # -------------------------------------------------------------------------
    # 6. Empty Query Rejection
    # -------------------------------------------------------------------------
    def test_empty_query_returns_http_422(self):
        """6. Verify empty query string returns HTTP 422 unprocessable entity."""
        response = self.client.post("/api/query", json={"query": ""})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stub.call_count, 0)

    def test_missing_or_non_string_query_returns_http_422(self):
        """Verify missing, null, or non-string query returns HTTP 422."""
        invalid_payloads = [
            {},
            {"query": None},
            {"query": 12345},
            {"query": True},
            {"query": ["hello"]},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/api/query", json=payload)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 7. Whitespace-Only Query Rejection
    # -------------------------------------------------------------------------
    def test_whitespace_only_query_returns_http_422(self):
        """7. Verify whitespace-only query string returns HTTP 422."""
        for ws in ["   ", "\t\t", " \n \t "]:
            with self.subTest(whitespace=ws):
                response = self.client.post("/api/query", json={"query": ws})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 8. Invalid top_k Rejection
    # -------------------------------------------------------------------------
    def test_invalid_top_k_returns_http_422(self):
        """8. Verify non-positive or non-integer top_k values return HTTP 422."""
        invalid_values = [0, -1, -10, "five", True]
        for val in invalid_values:
            with self.subTest(top_k=val):
                response = self.client.post(
                    "/api/query",
                    json={"query": "Valid query", "top_k": val},
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 9. Invalid score_threshold Rejection
    # -------------------------------------------------------------------------
    def test_invalid_score_threshold_returns_http_422(self):
        """9. Verify score_threshold outside [0.0, 1.0] or non-numeric returns HTTP 422."""
        invalid_values = [-0.1, 1.1, 2.0, "high", True]
        for val in invalid_values:
            with self.subTest(score_threshold=val):
                response = self.client.post(
                    "/api/query",
                    json={"query": "Valid query", "score_threshold": val},
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # Additional: Invalid collection_name Rejection
    # -------------------------------------------------------------------------
    def test_invalid_collection_name_returns_http_422(self):
        """Verify empty or whitespace-only collection_name returns HTTP 422."""
        for val in ["", "   ", "\t"]:
            with self.subTest(collection_name=val):
                response = self.client.post(
                    "/api/query",
                    json={"query": "Valid query", "collection_name": val},
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 10. Empty Retrieval Returns Safe Answer Response
    # -------------------------------------------------------------------------
    def test_empty_retrieval_returns_safe_answer_response(self):
        """10. Verify empty retrieval yields HTTP 200 with safe response and no sources."""
        empty_response = RAGResponse(
            query="Unknown subject",
            answer=DEFAULT_EMPTY_RETRIEVAL_ANSWER,
            sources=[],
            context_text="",
        )
        self.stub.response = empty_response

        response = self.client.post("/api/query", json={"query": "Unknown subject"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], DEFAULT_EMPTY_RETRIEVAL_ANSWER)
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["context_text"], "")

    # -------------------------------------------------------------------------
    # 11. Retrieval Failures Return Error Response
    # -------------------------------------------------------------------------
    def test_retrieval_failure_returns_appropriate_error_response(self):
        """11. Verify retrieval/vector-store failures map to HTTP 503 without leaking details."""
        retrieval_errors = [
            RAGAnswerGenerationError("Context retrieval failed: Qdrant connection refused"),
            RetrieverError("Failed to connect to vector database"),
            VectorStoreError("Search timeout"),
        ]
        for err in retrieval_errors:
            with self.subTest(error=type(err).__name__):
                self.stub.exception = err
                response = self.client.post("/api/query", json={"query": "Any query"})

                self.assertEqual(response.status_code, 503)
                data = response.json()
                self.assertIn("detail", data)
                self.assertEqual(
                    data["detail"],
                    "Knowledge retrieval service is currently unavailable.",
                )
                # Verify internal error details or hostnames are not leaked
                self.assertNotIn("Qdrant", data["detail"])
                self.assertNotIn("refused", data["detail"])

    # -------------------------------------------------------------------------
    # 12. LLM Provider Failures Return Error Response
    # -------------------------------------------------------------------------
    def test_llm_provider_failure_returns_appropriate_error_response(self):
        """12. Verify LLM provider failures map to HTTP 502 without leaking details."""
        llm_errors = [
            RAGAnswerGenerationError("LLM generation failed: OpenAI API rate limit exceeded"),
            LLMProviderError("Upstream provider returned HTTP 500"),
            LLMError("Token limit exceeded"),
        ]
        for err in llm_errors:
            with self.subTest(error=type(err).__name__):
                self.stub.exception = err
                response = self.client.post("/api/query", json={"query": "Any query"})

                self.assertEqual(response.status_code, 502)
                data = response.json()
                self.assertIn("detail", data)
                self.assertEqual(
                    data["detail"],
                    "Language model provider error encountered while generating answer.",
                )
                # Verify sensitive credentials or raw exception messages are not leaked
                self.assertNotIn("OpenAI", data["detail"])
                self.assertNotIn("rate limit", data["detail"])

    def test_unexpected_internal_error_returns_http_500(self):
        """Verify unhandled internal errors return HTTP 500 without leaking stack traces."""
        self.stub.exception = RuntimeError("Fatal memory crash or unexpected internal fault")
        response = self.client.post("/api/query", json={"query": "Any query"})

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertEqual(
            data["detail"],
            "An unexpected error occurred while processing the query.",
        )
        self.assertNotIn("Fatal memory crash", data["detail"])

    # -------------------------------------------------------------------------
    # 13. Existing /health Endpoint Remains Functional
    # -------------------------------------------------------------------------
    def test_existing_health_endpoint_remains_functional(self):
        """13. Verify GET /health endpoint still functions correctly."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    # -------------------------------------------------------------------------
    # 14. Dependency Overrides Work
    # -------------------------------------------------------------------------
    def test_dependency_overrides_work(self):
        """14. Verify FastAPI dependency injection override mechanism functions properly."""
        custom_response = RAGResponse(
            query="Test query",
            answer="Custom overridden answer",
            sources=[],
            context_text="",
        )
        custom_stub = StubRAGAnswerGenerator(response=custom_response)

        app.dependency_overrides[get_rag_answer_generator_dependency] = lambda: custom_stub
        response = self.client.post("/api/query", json={"query": "Test query"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Custom overridden answer")
        self.assertEqual(custom_stub.call_count, 1)

    # -------------------------------------------------------------------------
    # 15. Unit Tests Independence From Docker and External APIs
    # -------------------------------------------------------------------------
    def test_tests_remain_independent_of_docker_and_external_apis(self):
        """15. Verify that API routes and validation can be tested with mock/stub components."""
        # Validate that QueryRequest model directly enforces constraints without external dependencies
        req = QueryRequest(query="Test", top_k=3, score_threshold=0.5, collection_name="docs")
        self.assertEqual(req.query, "Test")
        self.assertEqual(req.top_k, 3)
        self.assertEqual(req.score_threshold, 0.5)
        self.assertEqual(req.collection_name, "docs")

    def test_shared_rag_answer_generator_lifecycle_caches_instance(self):
        """Verify get_shared_rag_answer_generator reuses instance across calls without recreation."""
        from unittest.mock import patch
        with patch("app.api.dependencies.get_rag_answer_generator") as mock_factory:
            mock_factory.return_value = MagicMock()
            reset_rag_answer_generator_cache()

            gen1 = get_shared_rag_answer_generator()
            gen2 = get_shared_rag_answer_generator()

            self.assertIs(gen1, gen2)
            self.assertEqual(mock_factory.call_count, 1)
            reset_rag_answer_generator_cache()


if __name__ == "__main__":
    unittest.main()
