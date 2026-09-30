"""Live integration tests for Qdrant vector database operations.

These tests execute against a real running Qdrant instance (e.g. in Docker Compose).
They are separated from the default unit test suite to maintain Docker-independence.
"""

import os
import sys
import unittest
from pathlib import Path

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.config import get_settings
from app.services import (
    FastEmbedEmbeddingService,
    QdrantVectorStoreService,
    VectorPoint,
    VectorSearchResult,
)


def is_live_qdrant_available() -> bool:
    """Check if a real Qdrant instance is reachable on the configured URL."""
    try:
        settings = get_settings()
        service = QdrantVectorStoreService(
            url=settings.qdrant_url or "http://localhost:6333",
            timeout=2.0,
        )
        return service.health_check()
    except Exception:
        return False


class TestQdrantLiveIntegration(unittest.TestCase):
    """End-to-end integration tests using real FastEmbed and a real Qdrant container."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.qdrant_url = cls.settings.qdrant_url or "http://localhost:6333"
        cls.test_collection_name = f"test_live_integration_{cls.settings.qdrant_collection_name}"
        cls.embedding_service = FastEmbedEmbeddingService(settings=cls.settings)
        cls.vector_store = QdrantVectorStoreService(
            url=cls.qdrant_url,
            collection_name=cls.test_collection_name,
            api_key=cls.settings.qdrant_api_key,
            timeout=cls.settings.qdrant_timeout_seconds,
            dimension=cls.settings.embedding_dimension,
            settings=cls.settings,
        )

    def setUp(self):
        if not self.vector_store.health_check():
            self.skipTest(f"Live Qdrant instance is not reachable at {self.qdrant_url}")

    def tearDown(self):
        # Clean up test collection after each test run to ensure isolation
        try:
            if self.vector_store.collection_exists(self.test_collection_name):
                self.vector_store.delete_collection(self.test_collection_name)
        except Exception:
            pass

    def test_live_qdrant_end_to_end_pipeline(self):
        """
        Verify real end-to-end workflow against live Qdrant container:
        1. Create real collection with COSINE metric and configured dimension
        2. Generate real FastEmbed dense vectors
        3. Upsert VectorPoint records into Qdrant
        4. Verify point count
        5. Search via cosine similarity with query embedding
        6. Verify top result precision and metadata preservation
        7. Verify idempotent upsert (count does not double)
        8. Delete collection and verify absence
        """
        # Step 1: Create real collection
        self.vector_store.create_collection(
            collection_name=self.test_collection_name,
            dimension=self.settings.embedding_dimension,
            recreate_if_exists=True,
        )
        self.assertTrue(self.vector_store.collection_exists(self.test_collection_name))

        # Step 2: Generate real FastEmbed document embeddings
        sample_texts = [
            "Enterprise AI agents use retrieval to ground answers in internal documents.",
            "Multi-factor authentication is mandatory for enterprise security.",
            "The platform stores document provenance including page and chunk metadata.",
        ]
        embeddings = self.embedding_service.embed_documents(sample_texts)
        self.assertEqual(len(embeddings), 3)
        for vec in embeddings:
            self.assertEqual(len(vec), self.settings.embedding_dimension)

        # Step 3: Construct VectorPoint instances
        points = [
            VectorPoint(
                chunk_id="doc_policy_p1_c0",
                vector=embeddings[0],
                document_name="architecture.pdf",
                page_number=1,
                chunk_index=0,
                text=sample_texts[0],
                metadata={"topic": "rag", "department": "engineering"},
            ),
            VectorPoint(
                chunk_id="doc_policy_p2_c0",
                vector=embeddings[1],
                document_name="security_policy.pdf",
                page_number=2,
                chunk_index=0,
                text=sample_texts[1],
                metadata={"topic": "security", "department": "infosec"},
            ),
            VectorPoint(
                chunk_id="doc_policy_p3_c0",
                vector=embeddings[2],
                document_name="provenance_guide.pdf",
                page_number=3,
                chunk_index=0,
                text=sample_texts[2],
                metadata={"topic": "provenance", "department": "data"},
            ),
        ]

        # Step 4: Upsert vectors into real Qdrant
        upserted_count = self.vector_store.upsert_vectors(
            points,
            collection_name=self.test_collection_name,
        )
        self.assertEqual(upserted_count, 3)

        # Step 5: Verify point count in Qdrant
        count = self.vector_store.count(self.test_collection_name)
        self.assertEqual(count, 3)

        # Step 6: Embed query and search
        query_text = "enterprise security and multi-factor authentication"
        query_vector = self.embedding_service.embed_query(query_text)
        self.assertEqual(len(query_vector), self.settings.embedding_dimension)

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=3,
            collection_name=self.test_collection_name,
        )

        # Step 7: Verify search results & metadata preservation
        self.assertEqual(len(results), 3)
        top_res = results[0]
        self.assertIsInstance(top_res, VectorSearchResult)

        # Top result must be the Security Policy / MFA document
        self.assertEqual(top_res.chunk_id, "doc_policy_p2_c0")
        self.assertEqual(top_res.document_name, "security_policy.pdf")
        self.assertEqual(top_res.page_number, 2)
        self.assertEqual(top_res.chunk_index, 0)
        self.assertEqual(top_res.text, "Multi-factor authentication is mandatory for enterprise security.")
        self.assertEqual(top_res.metadata["topic"], "security")
        self.assertEqual(top_res.metadata["department"], "infosec")
        self.assertTrue(top_res.score > 0.0)

        # Verify score ordering (descending)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)

        # Step 8: Test Idempotency (re-upserting identical points must not double point count)
        re_upserted_count = self.vector_store.upsert_vectors(
            points,
            collection_name=self.test_collection_name,
        )
        self.assertEqual(re_upserted_count, 3)
        count_after_re_upsert = self.vector_store.count(self.test_collection_name)
        self.assertEqual(count_after_re_upsert, 3)

        # Step 9: Delete collection and verify removal
        delete_success = self.vector_store.delete_collection(self.test_collection_name)
        self.assertTrue(delete_success)
        self.assertFalse(self.vector_store.collection_exists(self.test_collection_name))


if __name__ == "__main__":
    unittest.main()
