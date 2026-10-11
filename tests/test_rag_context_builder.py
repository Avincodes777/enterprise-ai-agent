"""Unit tests for RAGContext and RAGContextBuilder."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import app.rag as rag_pkg
from app.rag import (
    RAGContext,
    RAGContextBuilder,
    RAGContextError,
    get_rag_context_builder,
)
from app.services import (
    BaseRetriever,
    RetrieverError,
    RetrieverValidationError,
    VectorSearchResult,
)


class StubRetriever(BaseRetriever):
    """Simple stub retriever returning configured results."""

    def __init__(self, results=None):
        self.results = results or []
        self.last_query = None
        self.last_top_k = None
        self.last_score_threshold = None
        self.last_collection_name = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold=None,
        collection_name=None,
    ):
        self.last_query = query
        self.last_top_k = top_k
        self.last_score_threshold = score_threshold
        self.last_collection_name = collection_name
        return self.results


class TestRAGContextModel(unittest.TestCase):
    """Tests verifying RAGContext model behavior and derived properties."""

    def test_rag_context_derived_properties_with_results(self):
        """1. Verify has_results is True and total_results matches when results present."""
        sample_results = [
            VectorSearchResult(
                chunk_id="chunk_1",
                document_name="guide.pdf",
                page_number=1,
                chunk_index=0,
                text="Some text",
                score=0.9,
            )
        ]
        context = RAGContext(
            query="test query",
            results=sample_results,
            context_text="Formatted text",
        )
        self.assertTrue(context.has_results)
        self.assertEqual(context.total_results, 1)
        self.assertEqual(context.query, "test query")
        self.assertEqual(context.context_text, "Formatted text")

    def test_rag_context_derived_properties_when_empty(self):
        """2. Verify has_results is False and total_results is 0 when empty."""
        context = RAGContext(
            query="empty query",
            results=[],
            context_text="",
        )
        self.assertFalse(context.has_results)
        self.assertEqual(context.total_results, 0)
        self.assertEqual(context.context_text, "")


class TestRAGContextBuilder(unittest.TestCase):
    """Tests for RAGContextBuilder orchestration, formatting, and parameter propagation."""

    def setUp(self):
        self.sample_results = [
            VectorSearchResult(
                chunk_id="sec_p1_c0",
                document_name="security_policy.pdf",
                page_number=1,
                chunk_index=0,
                text="Multi-factor authentication (MFA) is strictly mandatory.",
                score=0.892,
                metadata={"topic": "auth", "department": "infosec"},
            ),
            VectorSearchResult(
                chunk_id="hr_p2_c1",
                document_name="hr_policy.pdf",
                page_number=2,
                chunk_index=1,
                text="Employees receive 20 annual leave days.",
                score=0.745,
                metadata={"category": "benefits"},
            ),
        ]
        self.stub_retriever = StubRetriever(results=self.sample_results)
        self.builder = RAGContextBuilder(
            retriever=self.stub_retriever,
            default_top_k=4,
            default_score_threshold=0.5,
            default_collection_name="enterprise_docs",
        )

    def test_successful_context_creation(self):
        """3. Verify build_context produces structured RAGContext."""
        context = self.builder.build_context("How does MFA work?")
        self.assertIsInstance(context, RAGContext)
        self.assertEqual(context.query, "How does MFA work?")
        self.assertEqual(len(context.results), 2)
        self.assertTrue(context.has_results)

    def test_query_propagation_to_retriever(self):
        """4. Verify query is passed to retriever unchanged."""
        self.builder.build_context("Specific Search Query")
        self.assertEqual(self.stub_retriever.last_query, "Specific Search Query")

    def test_parameter_defaults_used_when_omitted(self):
        """5. Verify builder defaults are propagated when optional args are None."""
        self.builder.build_context("query")
        self.assertEqual(self.stub_retriever.last_top_k, 4)
        self.assertEqual(self.stub_retriever.last_score_threshold, 0.5)
        self.assertEqual(self.stub_retriever.last_collection_name, "enterprise_docs")

    def test_explicit_parameters_override_defaults(self):
        """6. Verify explicit top_k, score_threshold, and collection_name override defaults."""
        self.builder.build_context(
            "query",
            top_k=10,
            score_threshold=0.8,
            collection_name="custom_collection",
        )
        self.assertEqual(self.stub_retriever.last_top_k, 10)
        self.assertEqual(self.stub_retriever.last_score_threshold, 0.8)
        self.assertEqual(self.stub_retriever.last_collection_name, "custom_collection")

    def test_result_order_preserved(self):
        """7. Verify order of results in context matches retriever output exactly."""
        context = self.builder.build_context("query")
        self.assertEqual(context.results[0].chunk_id, "sec_p1_c0")
        self.assertEqual(context.results[1].chunk_id, "hr_p2_c1")

    def test_context_text_formatting_contents(self):
        """8. Verify all required document provenance and text elements appear in context_text."""
        context = self.builder.build_context("query")
        text = context.context_text

        # Headers and indices
        self.assertIn("[SOURCE 1]", text)
        self.assertIn("[SOURCE 2]", text)

        # Source 1 verification
        self.assertIn("Document: security_policy.pdf", text)
        self.assertIn("Page: 1", text)
        self.assertIn("Chunk: 0", text)
        self.assertIn("Score: 0.892", text)
        self.assertIn("Multi-factor authentication (MFA) is strictly mandatory.", text)

        # Source 2 verification
        self.assertIn("Document: hr_policy.pdf", text)
        self.assertIn("Page: 2", text)
        self.assertIn("Chunk: 1", text)
        self.assertIn("Score: 0.745", text)
        self.assertIn("Employees receive 20 annual leave days.", text)

    def test_metadata_formatting_and_ordering(self):
        """9. Verify metadata keys are deterministically sorted and formatted."""
        context = self.builder.build_context("query")
        text = context.context_text

        # Source 1 metadata has 'department' and 'topic'
        self.assertIn("department=infosec", text)
        self.assertIn("topic=auth", text)
        # Verify alphabetical ordering of metadata keys
        dept_pos = text.find("department=infosec")
        topic_pos = text.find("topic=auth")
        self.assertTrue(dept_pos < topic_pos, "Keys should be sorted alphabetically")

    def test_empty_metadata_omits_metadata_header(self):
        """10. Verify metadata line is omitted when chunk metadata is empty."""
        no_meta_results = [
            VectorSearchResult(
                chunk_id="no_meta_chunk",
                document_name="plain.pdf",
                page_number=1,
                chunk_index=0,
                text="Plain content without extra metadata",
                score=0.9,
                metadata={},
            )
        ]
        retriever = StubRetriever(results=no_meta_results)
        builder = RAGContextBuilder(retriever=retriever)
        context = builder.build_context("query")

        self.assertIn("[SOURCE 1]", context.context_text)
        self.assertNotIn("Metadata:", context.context_text)
        self.assertIn("Content:", context.context_text)

    def test_empty_retrieval_produces_empty_context_text(self):
        """11. Verify empty retrieval returns valid RAGContext with context_text=''."""
        empty_retriever = StubRetriever(results=[])
        builder = RAGContextBuilder(retriever=empty_retriever)
        context = builder.build_context("no match query")

        self.assertEqual(context.results, [])
        self.assertEqual(context.context_text, "")
        self.assertFalse(context.has_results)
        self.assertEqual(context.total_results, 0)

    def test_retriever_errors_wrapped_in_rag_context_error(self):
        """12. Verify retriever exceptions are wrapped in RAGContextError and not swallowed."""
        failing_retriever = MagicMock(spec=BaseRetriever)
        failing_retriever.retrieve.side_effect = RetrieverError("Database timeout")

        builder = RAGContextBuilder(retriever=failing_retriever)
        with self.assertRaises(RAGContextError) as ctx:
            builder.build_context("query")
        self.assertIn("Failed to build RAG context", str(ctx.exception))
        self.assertIn("Database timeout", str(ctx.exception))

    def test_constructor_validation_rejects_invalid_inputs(self):
        """13. Verify constructor rejects invalid retriever or invalid defaults."""
        with self.assertRaises(TypeError):
            RAGContextBuilder(retriever="not_a_retriever")  # type: ignore

        with self.assertRaises(ValueError):
            RAGContextBuilder(retriever=self.stub_retriever, default_top_k=0)

        with self.assertRaises(ValueError):
            RAGContextBuilder(retriever=self.stub_retriever, default_top_k=-5)

        with self.assertRaises(ValueError):
            RAGContextBuilder(retriever=self.stub_retriever, default_score_threshold=1.5)

        with self.assertRaises(ValueError):
            RAGContextBuilder(retriever=self.stub_retriever, default_score_threshold=-0.1)

    def test_deterministic_context_text_generation(self):
        """14. Verify identical inputs produce identical context_text output."""
        ctx1 = self.builder.build_context("query")
        ctx2 = self.builder.build_context("query")
        self.assertEqual(ctx1.context_text, ctx2.context_text)

    def test_factory_function(self):
        """15. Verify get_rag_context_builder factory function."""
        builder = get_rag_context_builder(use_mock=True)
        self.assertIsInstance(builder, RAGContextBuilder)
        self.assertIsInstance(builder.retriever, BaseRetriever)

    def test_package_level_exports(self):
        """16. Verify RAG package exports."""
        expected_symbols = [
            "RAGContext",
            "RAGContextBuilder",
            "RAGContextError",
            "get_rag_context_builder",
        ]
        for sym in expected_symbols:
            self.assertTrue(hasattr(rag_pkg, sym), f"Symbol '{sym}' should be in app.rag")
            self.assertIn(sym, rag_pkg.__all__, f"Symbol '{sym}' should be in app.rag.__all__")


if __name__ == "__main__":
    unittest.main()
