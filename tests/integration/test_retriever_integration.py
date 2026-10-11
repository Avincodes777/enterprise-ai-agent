"""Live integration tests for the Retriever layer using real FastEmbed and Qdrant.

These tests execute the full end-to-end retrieval flow:
User Query -> Real FastEmbed -> Real Qdrant -> VectorSearchResult

Separated from the default unit test suite to keep standard unit testing Docker-independent.
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
    VectorRetriever,
    VectorSearchResult,
)


class TestRetrieverLiveIntegration(unittest.TestCase):
    """Live end-to-end integration tests for VectorRetriever with real FastEmbed and Qdrant."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.qdrant_url = cls.settings.qdrant_url or "http://localhost:6333"
        cls.test_collection_name = f"test_live_retriever_{cls.settings.qdrant_collection_name}"

        cls.embedding_service = FastEmbedEmbeddingService(settings=cls.settings)
        cls.vector_store_service = QdrantVectorStoreService(
            url=cls.qdrant_url,
            collection_name=cls.test_collection_name,
            api_key=cls.settings.qdrant_api_key,
            timeout=cls.settings.qdrant_timeout_seconds,
            dimension=cls.settings.embedding_dimension,
            settings=cls.settings,
        )

        cls.retriever = VectorRetriever(
            embedding_service=cls.embedding_service,
            vector_store_service=cls.vector_store_service,
            default_top_k=3,
            default_collection_name=cls.test_collection_name,
            settings=cls.settings,
        )

    def setUp(self):
        if not self.vector_store_service.health_check():
            self.skipTest(f"Live Qdrant instance is not reachable at {self.qdrant_url}")

        # Setup test collection
        self.vector_store_service.create_collection(
            collection_name=self.test_collection_name,
            dimension=self.settings.embedding_dimension,
            recreate_if_exists=True,
        )

        # Seed distinguishable realistic corporate knowledge vectors
        self.sample_docs = [
            {
                "chunk_id": "sec_auth_p1_c0",
                "text": "Enterprise multi-factor authentication (MFA) and zero-trust identity verification are strictly enforced across all corporate systems.",
                "document_name": "security_policy.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "metadata": {"topic": "authentication_security", "department": "infosec"},
            },
            {
                "chunk_id": "hr_leave_p3_c0",
                "text": "Full-time employees are entitled to 20 days of paid annual leave plus 10 public holidays per calendar year.",
                "document_name": "hr_policy.pdf",
                "page_number": 3,
                "chunk_index": 0,
                "metadata": {"topic": "employee_leave", "department": "human_resources"},
            },
            {
                "chunk_id": "db_backup_p5_c0",
                "text": "Production databases undergo automated full snapshot backups every 24 hours with point-in-time recovery retained for 30 days.",
                "document_name": "disaster_recovery.pdf",
                "page_number": 5,
                "chunk_index": 0,
                "metadata": {"topic": "database_backup", "department": "infrastructure"},
            },
        ]

        texts = [doc["text"] for doc in self.sample_docs]
        embeddings = self.embedding_service.embed_documents(texts)

        points = [
            VectorPoint(
                chunk_id=doc["chunk_id"],
                vector=emb,
                document_name=doc["document_name"],
                page_number=doc["page_number"],
                chunk_index=doc["chunk_index"],
                text=doc["text"],
                metadata=doc["metadata"],
            )
            for doc, emb in zip(self.sample_docs, embeddings)
        ]

        self.vector_store_service.upsert_vectors(points, collection_name=self.test_collection_name)

    def tearDown(self):
        # Clean up temporary collection to prevent stale data
        try:
            if self.vector_store_service.collection_exists(self.test_collection_name):
                self.vector_store_service.delete_collection(self.test_collection_name)
        except Exception:
            pass

    def test_live_retrieval_semantic_relevance_and_ranking(self):
        """
        Verify live end-to-end retrieval correctly identifies and ranks relevant chunks:
        1. Authentication query -> ranks security_policy.pdf #1
        2. Employee leave query -> ranks hr_policy.pdf #1
        3. Database backup query -> ranks disaster_recovery.pdf #1
        """
        # Query 1: Authentication and Security
        res_sec = self.retriever.retrieve(
            "How does the company handle authentication and security?",
            top_k=3,
        )
        self.assertEqual(len(res_sec), 3)
        self.assertIsInstance(res_sec[0], VectorSearchResult)
        self.assertEqual(res_sec[0].chunk_id, "sec_auth_p1_c0")
        self.assertEqual(res_sec[0].document_name, "security_policy.pdf")
        self.assertEqual(res_sec[0].page_number, 1)
        self.assertEqual(res_sec[0].chunk_index, 0)
        self.assertEqual(res_sec[0].metadata["topic"], "authentication_security")
        self.assertIn("multi-factor authentication", res_sec[0].text)
        self.assertTrue(res_sec[0].score > res_sec[1].score)

        # Query 2: Employee Leave Policy
        res_hr = self.retriever.retrieve(
            "How many days of employee leave are allowed?",
            top_k=3,
        )
        self.assertEqual(len(res_hr), 3)
        self.assertEqual(res_hr[0].chunk_id, "hr_leave_p3_c0")
        self.assertEqual(res_hr[0].document_name, "hr_policy.pdf")
        self.assertEqual(res_hr[0].page_number, 3)
        self.assertEqual(res_hr[0].chunk_index, 0)
        self.assertEqual(res_hr[0].metadata["topic"], "employee_leave")
        self.assertIn("20 days of paid annual leave", res_hr[0].text)
        self.assertTrue(res_hr[0].score > res_hr[1].score)

        # Query 3: Database Backup Policy
        res_db = self.retriever.retrieve(
            "What is the database backup policy?",
            top_k=3,
        )
        self.assertEqual(len(res_db), 3)
        self.assertEqual(res_db[0].chunk_id, "db_backup_p5_c0")
        self.assertEqual(res_db[0].document_name, "disaster_recovery.pdf")
        self.assertEqual(res_db[0].page_number, 5)
        self.assertEqual(res_db[0].chunk_index, 0)
        self.assertEqual(res_db[0].metadata["topic"], "database_backup")
        self.assertIn("snapshot backups every 24 hours", res_db[0].text)
        self.assertTrue(res_db[0].score > res_db[1].score)

    def test_live_retrieval_top_k_and_score_threshold(self):
        """Verify top_k limiting and score_threshold filtering over live Qdrant."""
        # Test top_k=1 limiting
        res_top1 = self.retriever.retrieve(
            "authentication and security guidelines",
            top_k=1,
        )
        self.assertEqual(len(res_top1), 1)
        self.assertEqual(res_top1[0].chunk_id, "sec_auth_p1_c0")

        # Test score_threshold filtering (realistic threshold retains match, high threshold excludes all)
        res_filtered = self.retriever.retrieve(
            "authentication and security guidelines",
            top_k=3,
            score_threshold=0.50,
        )
        self.assertTrue(len(res_filtered) >= 1)
        self.assertEqual(res_filtered[0].chunk_id, "sec_auth_p1_c0")
        for r in res_filtered:
            self.assertGreaterEqual(r.score, 0.50)

        # High threshold excludes all
        res_none = self.retriever.retrieve(
            "authentication and security guidelines",
            top_k=3,
            score_threshold=0.95,
        )
        self.assertEqual(res_none, [])

    def test_live_retrieval_collection_override(self):
        """Verify collection_name parameter propagation works dynamically."""
        alt_collection = f"{self.test_collection_name}_alt"
        try:
            self.vector_store_service.create_collection(
                collection_name=alt_collection,
                dimension=self.settings.embedding_dimension,
            )
            # Retrieve from empty alternate collection should return [] without error
            res_alt = self.retriever.retrieve(
                "any query",
                collection_name=alt_collection,
            )
            self.assertEqual(res_alt, [])
        finally:
            if self.vector_store_service.collection_exists(alt_collection):
                self.vector_store_service.delete_collection(alt_collection)


if __name__ == "__main__":
    unittest.main()
