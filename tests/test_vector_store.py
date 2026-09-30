"""Unit tests for the Vector Store service layer, point ID generation, and mock/Qdrant services."""

import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.config import Settings
from app.services.vector_store import (
    BaseVectorStoreService,
    MockVectorStoreService,
    QdrantVectorStoreService,
    VectorPoint,
    VectorSearchResult,
    VectorStoreCollectionNotFoundError,
    VectorStoreError,
    VectorStoreValidationError,
    chunk_id_to_point_id,
    get_vector_store_service,
)


class TestPointIdGeneration(unittest.TestCase):
    """Test deterministic UUIDv5 point ID generation."""

    def test_chunk_id_to_point_id_deterministic(self):
        """1. Verify same chunk_id always yields identical UUID string."""
        chunk_id = "finance_doc_p1_c0"
        id1 = chunk_id_to_point_id(chunk_id)
        id2 = chunk_id_to_point_id(chunk_id)
        self.assertEqual(id1, id2)

    def test_different_chunk_ids_yield_different_point_ids(self):
        """2. Verify distinct chunk_ids yield distinct UUID strings."""
        id1 = chunk_id_to_point_id("doc1_p1_c0")
        id2 = chunk_id_to_point_id("doc1_p1_c1")
        self.assertNotEqual(id1, id2)

    def test_valid_uuid_format(self):
        """3. Verify generated point ID is a valid RFC 4122 UUIDv5."""
        point_id = chunk_id_to_point_id("sample_chunk_id")
        parsed = uuid.UUID(point_id)
        self.assertEqual(parsed.version, 5)

    def test_invalid_chunk_id_raises_validation_error(self):
        """4. Verify empty or non-string chunk_id raises VectorStoreValidationError."""
        with self.assertRaises(VectorStoreValidationError):
            chunk_id_to_point_id("")

        with self.assertRaises(VectorStoreValidationError):
            chunk_id_to_point_id("   ")

        with self.assertRaises(VectorStoreValidationError):
            chunk_id_to_point_id(None)  # type: ignore


class TestMockVectorStoreService(unittest.TestCase):
    """Unit tests for the in-memory MockVectorStoreService."""

    def setUp(self):
        self.mock_store = MockVectorStoreService(
            collection_name="test_collection",
            dimension=3,
        )

    def test_health_check(self):
        """5. Verify mock health check behavior."""
        self.assertTrue(self.mock_store.health_check())
        self.mock_store.set_healthy(False)
        self.assertFalse(self.mock_store.health_check())
        self.mock_store.set_healthy(True)
        self.assertTrue(self.mock_store.health_check())

    def test_collection_lifecycle(self):
        """6. Verify collection creation, existence check, and deletion."""
        self.assertFalse(self.mock_store.collection_exists("test_collection"))
        self.assertTrue(self.mock_store.create_collection("test_collection", dimension=3))
        self.assertTrue(self.mock_store.collection_exists("test_collection"))
        self.assertTrue(self.mock_store.delete_collection("test_collection"))
        self.assertFalse(self.mock_store.collection_exists("test_collection"))
        self.assertFalse(self.mock_store.delete_collection("non_existent_collection"))

    def test_duplicate_collection_creation(self):
        """7. Verify recreate_if_exists flag behavior."""
        self.mock_store.create_collection("test_collection", dimension=3)
        pts = [
            VectorPoint(
                chunk_id="chunk_1",
                vector=[1.0, 0.0, 0.0],
                document_name="test.pdf",
                page_number=1,
                chunk_index=0,
                text="sample point",
            )
        ]
        self.mock_store.upsert_vectors(pts, collection_name="test_collection")
        self.assertEqual(self.mock_store.count("test_collection"), 1)

        # recreate_if_exists=False preserves points
        self.mock_store.create_collection("test_collection", dimension=3, recreate_if_exists=False)
        self.assertEqual(self.mock_store.count("test_collection"), 1)

        # recreate_if_exists=True wipes points
        self.mock_store.create_collection("test_collection", dimension=3, recreate_if_exists=True)
        self.assertEqual(self.mock_store.count("test_collection"), 0)

    def test_invalid_dimension_raises_error(self):
        """8. Verify collection dimension <= 0 raises validation error."""
        with self.assertRaises(VectorStoreValidationError):
            self.mock_store.create_collection("test_col", dimension=0)
        with self.assertRaises(VectorStoreValidationError):
            self.mock_store.create_collection("test_col", dimension=-5)

    def test_upsert_and_count(self):
        """9. Verify upserting points and counting total points."""
        self.mock_store.create_collection("test_collection", dimension=3)
        pts = [
            VectorPoint(
                chunk_id="chunk_1",
                vector=[1.0, 0.0, 0.0],
                document_name="doc.pdf",
                page_number=1,
                chunk_index=0,
                text="First chunk",
                metadata={"tag": "a"},
            ),
            VectorPoint(
                chunk_id="chunk_2",
                vector=[0.0, 1.0, 0.0],
                document_name="doc.pdf",
                page_number=1,
                chunk_index=1,
                text="Second chunk",
                metadata={"tag": "b"},
            ),
        ]
        count = self.mock_store.upsert_vectors(pts, collection_name="test_collection")
        self.assertEqual(count, 2)
        self.assertEqual(self.mock_store.count("test_collection"), 2)

    def test_upsert_empty_or_mismatched_dimension_raises_error(self):
        """10. Verify upsert validation rejects empty list or dimension mismatch."""
        self.mock_store.create_collection("test_collection", dimension=3)

        with self.assertRaises(VectorStoreValidationError):
            self.mock_store.upsert_vectors([], collection_name="test_collection")

        bad_dim_pt = [
            VectorPoint(
                chunk_id="bad_chunk",
                vector=[1.0, 0.0],  # 2 dimensions instead of 3
                text="bad",
            )
        ]
        with self.assertRaises(VectorStoreValidationError):
            self.mock_store.upsert_vectors(bad_dim_pt, collection_name="test_collection")

    def test_search_cosine_ranking_and_top_k(self):
        """11. Verify cosine similarity search, ordering, and top_k limiting."""
        self.mock_store.create_collection("test_collection", dimension=3)
        pts = [
            VectorPoint(
                chunk_id="chunk_x",
                vector=[1.0, 0.0, 0.0],
                document_name="doc.pdf",
                page_number=1,
                chunk_index=0,
                text="Point X along axis 0",
            ),
            VectorPoint(
                chunk_id="chunk_y",
                vector=[0.0, 1.0, 0.0],
                document_name="doc.pdf",
                page_number=1,
                chunk_index=1,
                text="Point Y along axis 1",
            ),
            VectorPoint(
                chunk_id="chunk_diag",
                vector=[0.7071, 0.7071, 0.0],
                document_name="doc.pdf",
                page_number=1,
                chunk_index=2,
                text="Point Diag between 0 and 1",
            ),
        ]
        self.mock_store.upsert_vectors(pts, collection_name="test_collection")

        # Search query exactly along axis 0 [1, 0, 0]
        results = self.mock_store.search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=2,
            collection_name="test_collection",
        )

        self.assertEqual(len(results), 2)
        # First result should be chunk_x with score ~1.0
        self.assertEqual(results[0].chunk_id, "chunk_x")
        self.assertAlmostEqual(results[0].score, 1.0, places=3)
        # Second result should be chunk_diag with score ~0.7071
        self.assertEqual(results[1].chunk_id, "chunk_diag")
        self.assertAlmostEqual(results[1].score, 0.7071, places=3)

    def test_search_score_threshold(self):
        """12. Verify score_threshold filters out lower scoring candidates."""
        self.mock_store.create_collection("test_collection", dimension=3)
        pts = [
            VectorPoint(chunk_id="c1", vector=[1.0, 0.0, 0.0], text="c1"),
            VectorPoint(chunk_id="c2", vector=[0.0, 1.0, 0.0], text="c2"),
        ]
        self.mock_store.upsert_vectors(pts, collection_name="test_collection")

        # Threshold 0.5 should exclude orthogonal vector [0, 1, 0] (score = 0.0)
        results = self.mock_store.search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
            score_threshold=0.5,
            collection_name="test_collection",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c1")

    def test_search_invalid_inputs_raise_error(self):
        """13. Verify search rejects empty query vectors or invalid top_k."""
        self.mock_store.create_collection("test_collection", dimension=3)

        with self.assertRaises(VectorStoreValidationError):
            self.mock_store.search([], collection_name="test_collection")

        with self.assertRaises(VectorStoreValidationError):
            self.mock_store.search([1.0, 0.0], collection_name="test_collection")

        with self.assertRaises(VectorStoreValidationError):
            self.mock_store.search([1.0, 0.0, 0.0], top_k=0, collection_name="test_collection")

        with self.assertRaises(VectorStoreCollectionNotFoundError):
            self.mock_store.search([1.0, 0.0, 0.0], collection_name="non_existent")


class TestQdrantVectorStoreService(unittest.TestCase):
    """Unit tests for QdrantVectorStoreService with mocked client."""

    def test_service_initialization_with_settings(self):
        """14. Verify QdrantVectorStoreService reads settings properly without network calls."""
        custom_settings = Settings(
            QDRANT_URL="http://qdrant.internal:6333",
            QDRANT_COLLECTION_NAME="custom_docs",
            QDRANT_API_KEY="test-secret-key",
            QDRANT_TIMEOUT_SECONDS=15.0,
            EMBEDDING_DIMENSION=384,
            _env_file=None,
        )
        service = QdrantVectorStoreService(settings=custom_settings)
        self.assertEqual(service.url, "http://qdrant.internal:6333")
        self.assertEqual(service.collection_name, "custom_docs")
        self.assertEqual(service.dimension, 384)

    def test_qdrant_health_check_returns_boolean(self):
        """15. Verify health_check returns True on success and False on exception."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = []
        service = QdrantVectorStoreService(client=mock_client)
        self.assertTrue(service.health_check())

        mock_client.get_collections.side_effect = RuntimeError("Connection refused")
        self.assertFalse(service.health_check())

    def test_qdrant_create_collection_cosine_config(self):
        """16. Verify create_collection configures cosine distance and dimension."""
        from qdrant_client import models

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        service = QdrantVectorStoreService(client=mock_client, dimension=384)
        result = service.create_collection("enterprise_knowledge", dimension=384)

        self.assertTrue(result)
        mock_client.create_collection.assert_called_once()
        _, kwargs = mock_client.create_collection.call_args
        self.assertEqual(kwargs["collection_name"], "enterprise_knowledge")
        vector_params = kwargs["vectors_config"]
        self.assertIsInstance(vector_params, models.VectorParams)
        self.assertEqual(vector_params.size, 384)
        self.assertEqual(vector_params.distance, models.Distance.COSINE)

    def test_qdrant_upsert_vectors_translation(self):
        """17. Verify upsert_vectors creates PointStruct with deterministic UUID point IDs."""
        mock_client = MagicMock()
        service = QdrantVectorStoreService(client=mock_client, dimension=3)

        points = [
            VectorPoint(
                chunk_id="doc_p1_c0",
                vector=[0.1, 0.2, 0.3],
                document_name="guide.pdf",
                page_number=1,
                chunk_index=0,
                text="Guide content",
                metadata={"dept": "engineering"},
            )
        ]

        count = service.upsert_vectors(points, collection_name="test_col")
        self.assertEqual(count, 1)

        mock_client.upsert.assert_called_once()
        _, kwargs = mock_client.upsert.call_args
        self.assertEqual(kwargs["collection_name"], "test_col")
        submitted_points = kwargs["points"]
        self.assertEqual(len(submitted_points), 1)

        pt = submitted_points[0]
        expected_uuid = chunk_id_to_point_id("doc_p1_c0")
        self.assertEqual(pt.id, expected_uuid)
        self.assertEqual(pt.vector, [0.1, 0.2, 0.3])
        self.assertEqual(pt.payload["chunk_id"], "doc_p1_c0")
        self.assertEqual(pt.payload["document_name"], "guide.pdf")
        self.assertEqual(pt.payload["page_number"], 1)
        self.assertEqual(pt.payload["chunk_index"], 0)
        self.assertEqual(pt.payload["text"], "Guide content")
        self.assertEqual(pt.payload["metadata"], {"dept": "engineering"})

    def test_qdrant_search_and_result_mapping(self):
        """18. Verify search invokes query_points and maps scored points into VectorSearchResult."""
        from types import SimpleNamespace

        mock_client = MagicMock()
        fake_scored_point = SimpleNamespace(
            id="760ff957-bb8d-5bdf-aab4-6f9e505f6f1b",
            score=0.954,
            payload={
                "chunk_id": "doc_p1_c0",
                "document_name": "annual_report.pdf",
                "page_number": 2,
                "chunk_index": 3,
                "text": "Financial summary highlights",
                "metadata": {"year": 2026},
            },
        )
        mock_response = SimpleNamespace(points=[fake_scored_point])
        mock_client.query_points.return_value = mock_response

        service = QdrantVectorStoreService(client=mock_client, dimension=3)
        results = service.search(
            query_vector=[0.1, 0.2, 0.3],
            top_k=5,
            score_threshold=0.7,
            collection_name="test_col",
        )

        mock_client.query_points.assert_called_once_with(
            collection_name="test_col",
            query=[0.1, 0.2, 0.3],
            limit=5,
            score_threshold=0.7,
        )

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertIsInstance(res, VectorSearchResult)
        self.assertEqual(res.chunk_id, "doc_p1_c0")
        self.assertEqual(res.document_name, "annual_report.pdf")
        self.assertEqual(res.page_number, 2)
        self.assertEqual(res.chunk_index, 3)
        self.assertEqual(res.text, "Financial summary highlights")
        self.assertAlmostEqual(res.score, 0.954, places=3)
        self.assertEqual(res.metadata, {"year": 2026})

    def test_qdrant_count_and_delete(self):
        """20. Verify QdrantVectorStoreService count and delete_collection behavior."""
        from types import SimpleNamespace

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.count.return_value = SimpleNamespace(count=42)

        service = QdrantVectorStoreService(client=mock_client)
        self.assertEqual(service.count("test_col"), 42)
        mock_client.count.assert_called_once_with(collection_name="test_col")

        self.assertTrue(service.delete_collection("test_col"))
        mock_client.delete_collection.assert_called_once_with(collection_name="test_col")

    def test_qdrant_count_non_existent_collection_raises_error(self):
        """21. Verify count on non-existent collection raises VectorStoreCollectionNotFoundError."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        service = QdrantVectorStoreService(client=mock_client)
        with self.assertRaises(VectorStoreCollectionNotFoundError):
            service.count("missing_col")

    def test_package_level_exports(self):
        """23. Verify all vector store symbols are exported from app.services."""
        import app.services as services

        exported_symbols = [
            "VectorStoreError",
            "VectorStoreConnectionError",
            "VectorStoreValidationError",
            "VectorStoreCollectionNotFoundError",
            "VectorPoint",
            "VectorSearchResult",
            "BaseVectorStoreService",
            "QdrantVectorStoreService",
            "MockVectorStoreService",
            "get_vector_store_service",
            "chunk_id_to_point_id",
        ]

        for sym in exported_symbols:
            self.assertTrue(
                hasattr(services, sym),
                f"Symbol '{sym}' should be exported by app.services",
            )
            self.assertIn(
                sym,
                services.__all__,
                f"Symbol '{sym}' should be listed in app.services.__all__",
            )


if __name__ == "__main__":
    unittest.main()


