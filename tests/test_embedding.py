import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.config import Settings
from app.services.chunker import DocumentChunk, chunk_text
from app.services.embedding import (
    BaseEmbeddingService,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingResult,
    FastEmbedEmbeddingService,
    MockEmbeddingService,
    get_embedding_service,
)


class TestBaseEmbeddingInterface(unittest.TestCase):
    """Tests verifying the abstract BaseEmbeddingService contract."""

    def test_cannot_instantiate_abstract_base_class(self):
        """Verify BaseEmbeddingService cannot be directly instantiated."""
        with self.assertRaises(TypeError):
            BaseEmbeddingService()  # type: ignore

    def test_embedding_result_model(self):
        """Verify EmbeddingResult Pydantic schema."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="test-model",
            dimension=3,
            total_tokens=10,
        )
        self.assertEqual(result.dimension, 3)
        self.assertEqual(result.model, "test-model")
        self.assertEqual(len(result.embeddings), 1)
        self.assertEqual(result.total_tokens, 10)


class TestMockEmbeddingService(unittest.TestCase):
    """Deterministic, zero-network unit tests for MockEmbeddingService."""

    def setUp(self):
        self.mock_service = MockEmbeddingService(dimension=384)

    def test_mock_properties(self):
        """Verify model_name and dimension properties."""
        self.assertEqual(self.mock_service.dimension, 384)
        self.assertEqual(self.mock_service.model_name, "mock-embedding-model")

    def test_custom_dimension(self):
        """Verify custom configured dimension."""
        custom_mock = MockEmbeddingService(dimension=128)
        self.assertEqual(custom_mock.dimension, 128)
        vec = custom_mock.embed_text("test")
        self.assertEqual(len(vec), 128)

    def test_single_text_embedding(self):
        """Verify single text embedding produces exact float vector."""
        vec = self.mock_service.embed_text("Enterprise AI Agent")
        self.assertEqual(len(vec), 384)
        self.assertTrue(all(isinstance(x, float) for x in vec))

    def test_deterministic_output(self):
        """Verify same input text produces identical vectors across multiple calls."""
        text = "Multi-source knowledge retrieval pipeline"
        vec1 = self.mock_service.embed_text(text)
        vec2 = self.mock_service.embed_text(text)
        self.assertEqual(vec1, vec2)

    def test_different_texts_produce_different_vectors(self):
        """Verify distinct inputs produce distinct vectors."""
        vec1 = self.mock_service.embed_text("Policy document page 1")
        vec2 = self.mock_service.embed_text("Security guidelines page 2")
        self.assertNotEqual(vec1, vec2)

    def test_batch_embedding(self):
        """Verify batch document embedding returns matching list of vectors."""
        texts = ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]
        vectors = self.mock_service.embed_documents(texts)
        self.assertEqual(len(vectors), 3)
        for vec in vectors:
            self.assertEqual(len(vec), 384)

    def test_empty_batch_returns_empty_list(self):
        """Verify embedding an empty list returns an empty list."""
        self.assertEqual(self.mock_service.embed_documents([]), [])

    def test_query_embedding(self):
        """Verify query embedding produces deterministic vector."""
        query = "What is the policy on MFA?"
        vec = self.mock_service.embed_query(query)
        self.assertEqual(len(vec), 384)

    def test_empty_and_whitespace_input_validation(self):
        """Verify None, empty string, and whitespace raise EmbeddingInputError."""
        with self.assertRaises(EmbeddingInputError):
            self.mock_service.embed_text("")

        with self.assertRaises(EmbeddingInputError):
            self.mock_service.embed_text("   \n\t  ")

        with self.assertRaises(EmbeddingInputError):
            self.mock_service.embed_query("")

        with self.assertRaises(EmbeddingInputError):
            self.mock_service.embed_documents(["Valid text", ""])


class TestFastEmbedEmbeddingService(unittest.TestCase):
    """Integration unit tests for local FastEmbedEmbeddingService."""

    @classmethod
    def setUpClass(cls):
        # Instantiate FastEmbed local service with default settings
        cls.settings = Settings(
            EMBEDDING_PROVIDER="fastembed",
            EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
            EMBEDDING_DIMENSION=384,
            _env_file=None,
        )
        cls.service = FastEmbedEmbeddingService(settings=cls.settings)

    def test_fastembed_properties(self):
        """Verify FastEmbed model_name and dimension."""
        self.assertEqual(self.service.dimension, 384)
        self.assertEqual(self.service.model_name, "BAAI/bge-small-en-v1.5")

    def test_fastembed_single_text_embedding(self):
        """Verify FastEmbed generates exact 384-dimensional float vector."""
        text = "Enterprise knowledge base system architecture."
        vector = self.service.embed_text(text)
        self.assertEqual(len(vector), 384)
        self.assertTrue(all(isinstance(x, float) for x in vector))

    def test_fastembed_batch_documents_embedding(self):
        """Verify FastEmbed embeds batches of documents."""
        docs = [
            "Section 1: General provisions and scope.",
            "Section 2: Multi-factor authentication guidelines.",
        ]
        vectors = self.service.embed_documents(docs)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 384)
        self.assertEqual(len(vectors[1]), 384)

    def test_fastembed_empty_batch_returns_empty_list(self):
        """Verify FastEmbed handles empty document sequence."""
        self.assertEqual(self.service.embed_documents([]), [])

    def test_fastembed_query_embedding(self):
        """Verify FastEmbed generates query embedding vector."""
        query = "How do I configure my MFA token?"
        q_vector = self.service.embed_query(query)
        self.assertEqual(len(q_vector), 384)
        self.assertTrue(all(isinstance(x, float) for x in q_vector))

    def test_fastembed_empty_input_validation(self):
        """Verify FastEmbed rejects empty and whitespace inputs."""
        with self.assertRaises(EmbeddingInputError):
            self.service.embed_text("")

        with self.assertRaises(EmbeddingInputError):
            self.service.embed_text("   \t\n ")

        with self.assertRaises(EmbeddingInputError):
            self.service.embed_query("")

        with self.assertRaises(EmbeddingInputError):
            self.service.embed_documents(["Valid passage", "   "])

    def test_dimension_mismatch_detection(self):
        """Verify service raises EmbeddingDimensionMismatchError if dimension expectation is violated."""
        mismatched_service = FastEmbedEmbeddingService(
            model_name="BAAI/bge-small-en-v1.5",
            expected_dimension=512,  # Model actually outputs 384
        )
        with self.assertRaises(EmbeddingDimensionMismatchError):
            mismatched_service.embed_text("This text will trigger a dimension error.")


class TestDocumentChunkCompatibility(unittest.TestCase):
    """Tests verifying DocumentChunk output from chunker directly passes into EmbeddingService."""

    def test_document_chunk_text_embedding_compatibility(self):
        """Verify DocumentChunk.text works seamlessly with embed_text and embed_documents."""
        raw_text = (
            "Enterprise Policy Standard 2026. "
            "All employees must complete annual security awareness training. "
            "Data encryption at rest is strictly enforced."
        )
        # 1. Generate real DocumentChunks using existing TextChunker
        chunks: list[DocumentChunk] = chunk_text(
            raw_text, chunk_size=60, overlap=10, doc_id="policy.pdf"
        )
        self.assertTrue(len(chunks) >= 2)

        # 2. Test with MockEmbeddingService
        mock_service = MockEmbeddingService(dimension=384)
        single_vec = mock_service.embed_text(chunks[0].text)
        self.assertEqual(len(single_vec), 384)

        batch_texts = [chunk.text for chunk in chunks]
        batch_vectors = mock_service.embed_documents(batch_texts)
        self.assertEqual(len(batch_vectors), len(chunks))
        for vec in batch_vectors:
            self.assertEqual(len(vec), 384)

        # 3. Test with FastEmbedEmbeddingService
        fastembed_service = FastEmbedEmbeddingService()
        fe_batch_vectors = fastembed_service.embed_documents(batch_texts)
        self.assertEqual(len(fe_batch_vectors), len(chunks))
        for vec in fe_batch_vectors:
            self.assertEqual(len(vec), 384)


class TestEmbeddingFactory(unittest.TestCase):
    """Tests for get_embedding_service factory function."""

    def test_get_mock_embedding_service(self):
        """Verify factory returns MockEmbeddingService when requested."""
        service = get_embedding_service(use_mock=True)
        self.assertIsInstance(service, MockEmbeddingService)
        self.assertEqual(service.dimension, 384)

    def test_get_fastembed_service_by_default(self):
        """Verify default settings return FastEmbedEmbeddingService."""
        service = get_embedding_service()
        self.assertIsInstance(service, FastEmbedEmbeddingService)
        self.assertEqual(service.dimension, 384)

    def test_unimplemented_openai_provider_raises_error(self):
        """Verify openai provider raises NotImplementedError until implemented."""
        openai_settings = Settings(
            EMBEDDING_PROVIDER="openai",
            EMBEDDING_MODEL="text-embedding-3-small",
            _env_file=None,
        )
        with self.assertRaises(NotImplementedError):
            get_embedding_service(settings=openai_settings)


if __name__ == "__main__":
    unittest.main()
