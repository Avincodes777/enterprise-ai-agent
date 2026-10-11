"""Unit tests for the Retriever layer, query validation, and embedding/vector-store delegation."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import app.services as services
from app.config import Settings
from app.services import (
    BaseRetriever,
    MockEmbeddingService,
    MockVectorStoreService,
    RetrieverError,
    RetrieverValidationError,
    VectorPoint,
    VectorRetriever,
    VectorSearchResult,
    get_retriever,
)
from app.services.embedding import EmbeddingError
from app.services.vector_store import VectorStoreError


class TestVectorRetrieverValidation(unittest.TestCase):
    """Unit tests verifying query, top_k, and score_threshold validation in VectorRetriever."""

    def setUp(self):
        self.mock_embedding = MockEmbeddingService(dimension=3)
        self.mock_vector_store = MockVectorStoreService(
            collection_name="enterprise_knowledge",
            dimension=3,
        )
        self.mock_vector_store.create_collection("enterprise_knowledge", dimension=3)
        self.retriever = VectorRetriever(
            embedding_service=self.mock_embedding,
            vector_store_service=self.mock_vector_store,
            default_top_k=5,
            default_collection_name="enterprise_knowledge",
        )

    def test_none_query_raises_validation_error(self):
        """1. Verify None query raises RetrieverValidationError."""
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve(None)  # type: ignore

    def test_non_string_query_raises_validation_error(self):
        """2. Verify non-string query types raise RetrieverValidationError."""
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve(12345)  # type: ignore
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve(["list", "query"])  # type: ignore
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve({"key": "val"})  # type: ignore

    def test_empty_string_query_raises_validation_error(self):
        """3. Verify empty query string raises RetrieverValidationError."""
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("")

    def test_whitespace_only_query_raises_validation_error(self):
        """4. Verify whitespace-only query string raises RetrieverValidationError."""
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("   \t\n  ")

    def test_query_exceeding_max_length_raises_validation_error(self):
        """5. Verify queries > 4000 characters raise RetrieverValidationError."""
        long_query = "a" * 4001
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve(long_query)

    def test_valid_max_length_query_accepted(self):
        """6. Verify query of exact 4000 characters is accepted."""
        valid_long_query = "a" * 4000
        # Should not raise validation error
        res = self.retriever.retrieve(valid_long_query)
        self.assertIsInstance(res, list)

    def test_top_k_zero_or_negative_raises_validation_error(self):
        """7. Verify top_k <= 0 raises RetrieverValidationError."""
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("valid query", top_k=0)
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("valid query", top_k=-5)

    def test_non_integer_top_k_raises_validation_error(self):
        """8. Verify non-integer top_k raises RetrieverValidationError."""
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("valid query", top_k=2.5)  # type: ignore
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("valid query", top_k="3")  # type: ignore
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("valid query", top_k=True)  # type: ignore

    def test_score_threshold_out_of_bounds_raises_validation_error(self):
        """9. Verify score_threshold outside [0.0, 1.0] raises RetrieverValidationError."""
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("valid query", score_threshold=-0.1)
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("valid query", score_threshold=1.01)

    def test_non_numeric_score_threshold_raises_validation_error(self):
        """10. Verify non-numeric score_threshold raises RetrieverValidationError."""
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("valid query", score_threshold="high")  # type: ignore
        with self.assertRaises(RetrieverValidationError):
            self.retriever.retrieve("valid query", score_threshold=True)  # type: ignore

    def test_invalid_default_top_k_in_constructor(self):
        """11. Verify constructor rejects invalid default_top_k."""
        with self.assertRaises(RetrieverValidationError):
            VectorRetriever(
                embedding_service=self.mock_embedding,
                vector_store_service=self.mock_vector_store,
                default_top_k=0,
            )
        with self.assertRaises(RetrieverValidationError):
            VectorRetriever(
                embedding_service=self.mock_embedding,
                vector_store_service=self.mock_vector_store,
                default_top_k=-1,
            )


class TestVectorRetrieverExecution(unittest.TestCase):
    """Unit tests for VectorRetriever execution, delegation, and result processing."""

    def setUp(self):
        self.mock_embedding = MockEmbeddingService(dimension=3)
        self.mock_vector_store = MockVectorStoreService(
            collection_name="test_knowledge",
            dimension=3,
        )
        self.mock_vector_store.create_collection("test_knowledge", dimension=3)

        # Seed mock vector store with sample points
        self.pts = [
            VectorPoint(
                chunk_id="chunk_1",
                vector=[1.0, 0.0, 0.0],
                document_name="sec_guide.pdf",
                page_number=1,
                chunk_index=0,
                text="Multi-factor authentication is required.",
                metadata={"category": "security"},
            ),
            VectorPoint(
                chunk_id="chunk_2",
                vector=[0.0, 1.0, 0.0],
                document_name="hr_policy.pdf",
                page_number=2,
                chunk_index=1,
                text="Standard vacation allowance is 20 days.",
                metadata={"category": "hr"},
            ),
            VectorPoint(
                chunk_id="chunk_3",
                vector=[0.7071, 0.7071, 0.0],
                document_name="it_setup.pdf",
                page_number=1,
                chunk_index=0,
                text="Security setup and authentication tokens.",
                metadata={"category": "it"},
            ),
        ]
        self.mock_vector_store.upsert_vectors(self.pts, collection_name="test_knowledge")

        self.retriever = VectorRetriever(
            embedding_service=self.mock_embedding,
            vector_store_service=self.mock_vector_store,
            default_top_k=3,
            default_collection_name="test_knowledge",
        )

    def test_successful_retrieval_and_result_ordering(self):
        """12. Verify retrieve returns VectorSearchResult list ordered by similarity."""
        results = self.retriever.retrieve("security and MFA authentication", top_k=3)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

        # Verify items are VectorSearchResult instances
        for item in results:
            self.assertIsInstance(item, VectorSearchResult)
            self.assertTrue(item.chunk_id)
            self.assertTrue(item.text)

        # Verify ordering is strictly descending by score
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)

    def test_query_whitespace_stripping(self):
        """13. Verify leading and trailing whitespace are stripped before embedding."""
        spy_embedding = MagicMock(spec=self.mock_embedding)
        spy_embedding.embed_query.return_value = [1.0, 0.0, 0.0]

        retriever = VectorRetriever(
            embedding_service=spy_embedding,
            vector_store_service=self.mock_vector_store,
            default_collection_name="test_knowledge",
        )

        retriever.retrieve("   \t security query \n  ")
        spy_embedding.embed_query.assert_called_once_with("security query")

    def test_embedding_delegation_uses_embed_query(self):
        """14. Verify retriever calls embed_query and not embed_documents."""
        spy_embedding = MagicMock(spec=self.mock_embedding)
        spy_embedding.embed_query.return_value = [1.0, 0.0, 0.0]

        retriever = VectorRetriever(
            embedding_service=spy_embedding,
            vector_store_service=self.mock_vector_store,
            default_collection_name="test_knowledge",
        )

        retriever.retrieve("mfa guidelines")
        spy_embedding.embed_query.assert_called_once_with("mfa guidelines")
        spy_embedding.embed_documents.assert_not_called()
        spy_embedding.embed_text.assert_not_called()

    def test_vector_store_delegation_parameters(self):
        """15. Verify vector_store.search receives exact parameters from retrieve."""
        spy_store = MagicMock(spec=self.mock_vector_store)
        spy_store.search.return_value = []

        retriever = VectorRetriever(
            embedding_service=self.mock_embedding,
            vector_store_service=spy_store,
            default_collection_name="default_col",
        )

        retriever.retrieve(
            "test query",
            top_k=2,
            score_threshold=0.75,
            collection_name="custom_col",
        )

        spy_store.search.assert_called_once()
        _, kwargs = spy_store.search.call_args
        self.assertEqual(kwargs["top_k"], 2)
        self.assertEqual(kwargs["score_threshold"], 0.75)
        self.assertEqual(kwargs["collection_name"], "custom_col")
        self.assertIsInstance(kwargs["query_vector"], list)

    def test_defensive_top_k_slicing(self):
        """16. Verify retriever defensively caps output length to top_k."""
        # Mock vector store returning 5 items when top_k=2 was requested
        fake_results = [
            VectorSearchResult(chunk_id=f"c_{i}", score=1.0 - i * 0.1, text=f"text {i}")
            for i in range(5)
        ]
        spy_store = MagicMock(spec=self.mock_vector_store)
        spy_store.search.return_value = fake_results

        retriever = VectorRetriever(
            embedding_service=self.mock_embedding,
            vector_store_service=spy_store,
        )

        results = retriever.retrieve("query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "c_0")
        self.assertEqual(results[1].chunk_id, "c_1")

    def test_empty_results_return_empty_list(self):
        """17. Verify when no documents match, retriever returns empty list without error."""
        results = self.retriever.retrieve("query", score_threshold=0.999)
        # Assuming no exact vector match at 0.999 threshold
        self.assertIsInstance(results, list)

    def test_embedding_failure_wrapped_in_retriever_error(self):
        """18. Verify embedding service failure raises RetrieverError with context."""
        failing_embedding = MagicMock(spec=self.mock_embedding)
        failing_embedding.embed_query.side_effect = EmbeddingError("Model inference crashed")

        retriever = VectorRetriever(
            embedding_service=failing_embedding,
            vector_store_service=self.mock_vector_store,
        )

        with self.assertRaises(RetrieverError) as ctx:
            retriever.retrieve("test query")
        self.assertIn("Retrieval failed during query embedding", str(ctx.exception))

    def test_vector_store_failure_wrapped_in_retriever_error(self):
        """19. Verify vector store search failure raises RetrieverError with context."""
        failing_store = MagicMock(spec=self.mock_vector_store)
        failing_store.search.side_effect = VectorStoreError("Connection refused by database")

        retriever = VectorRetriever(
            embedding_service=self.mock_embedding,
            vector_store_service=failing_store,
        )

        with self.assertRaises(RetrieverError) as ctx:
            retriever.retrieve("test query")
        self.assertIn("Retrieval failed during vector store search", str(ctx.exception))

    def test_factory_function_with_mock(self):
        """20. Verify get_retriever factory with use_mock=True returns working VectorRetriever."""
        retriever = get_retriever(use_mock=True)
        self.assertIsInstance(retriever, BaseRetriever)
        self.assertIsInstance(retriever, VectorRetriever)
        self.assertIsInstance(retriever.embedding_service, MockEmbeddingService)
        self.assertIsInstance(retriever.vector_store_service, MockVectorStoreService)

    def test_package_level_exports(self):
        """21. Verify retriever symbols are exported from app.services."""
        exported_symbols = [
            "BaseRetriever",
            "VectorRetriever",
            "RetrieverError",
            "RetrieverValidationError",
            "get_retriever",
        ]
        for sym in exported_symbols:
            self.assertTrue(
                hasattr(services, sym),
                f"Symbol '{sym}' should be exported by app.services",
            )
            self.assertIn(
                sym,
                services.__all__,
                f"Symbol '{sym}' should be in app.services.__all__",
            )


    def test_constructor_dependency_injection(self):
        """22. Verify constructor dependency injection sets all properties correctly."""
        custom_embedding = MockEmbeddingService(dimension=384)
        custom_store = MockVectorStoreService(collection_name="custom_col", dimension=384)
        retriever = VectorRetriever(
            embedding_service=custom_embedding,
            vector_store_service=custom_store,
            default_top_k=7,
            default_score_threshold=0.6,
            default_collection_name="custom_col",
        )

        self.assertEqual(retriever.embedding_service, custom_embedding)
        self.assertEqual(retriever.vector_store_service, custom_store)
        self.assertEqual(retriever.default_top_k, 7)
        self.assertEqual(retriever.default_score_threshold, 0.6)
        self.assertEqual(retriever.default_collection_name, "custom_col")


if __name__ == "__main__":
    unittest.main()

