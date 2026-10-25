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

    # -------------------------------------------------------------------------
    # 24. Document ID Provenance in Qdrant Payload
    # -------------------------------------------------------------------------
    def test_qdrant_upsert_and_search_preserves_document_id(self):
        """24. Verify Qdrant payload stores document_id and search maps it to VectorSearchResult."""
        from types import SimpleNamespace

        mock_client = MagicMock()
        service = QdrantVectorStoreService(client=mock_client, dimension=3)

        point = VectorPoint(
            chunk_id="doc_provenance_chunk_1",
            document_id="doc_test_uuid_456",
            vector=[0.5, 0.5, 0.0],
            document_name="spec.pdf",
            text="Specification text",
        )
        service.upsert_vectors([point], collection_name="provenance_col")

        mock_client.upsert.assert_called_once()
        submitted_pt = mock_client.upsert.call_args[1]["points"][0]
        self.assertEqual(submitted_pt.payload["document_id"], "doc_test_uuid_456")
        self.assertEqual(submitted_pt.payload["chunk_id"], "doc_provenance_chunk_1")

        # Test search mapping
        mock_scored = SimpleNamespace(
            id=submitted_pt.id,
            score=0.99,
            payload={
                "chunk_id": "doc_provenance_chunk_1",
                "document_id": "doc_test_uuid_456",
                "document_name": "spec.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "text": "Specification text",
                "metadata": {},
            },
        )
        mock_client.query_points.return_value = SimpleNamespace(points=[mock_scored])

        results = service.search([0.5, 0.5, 0.0], top_k=1, collection_name="provenance_col")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document_id, "doc_test_uuid_456")

    # -------------------------------------------------------------------------
    # 25. MockVectorStoreService Preserves Document ID
    # -------------------------------------------------------------------------
    def test_mock_vector_store_preserves_document_id(self):
        """25. Verify MockVectorStoreService stores and returns document_id in search results."""
        mock_service = MockVectorStoreService(collection_name="mock_prov_col", dimension=3)
        mock_service.create_collection("mock_prov_col")

        point = VectorPoint(
            chunk_id="mock_chunk_1",
            document_id="doc_mock_identity_789",
            vector=[1.0, 0.0, 0.0],
            document_name="mock_doc.pdf",
            text="Mock document text",
        )
        mock_service.upsert_vectors([point], collection_name="mock_prov_col")

        results = mock_service.search([1.0, 0.0, 0.0], top_k=1, collection_name="mock_prov_col")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document_id, "doc_mock_identity_789")

    # -------------------------------------------------------------------------
    # 26. Legacy Points Without Document ID Handled Gracefully
    # -------------------------------------------------------------------------
    def test_legacy_points_without_document_id_handled_gracefully(self):
        """26. Verify points with missing or None document_id deserialize cleanly with document_id=None."""
        mock_service = MockVectorStoreService(collection_name="legacy_col", dimension=3)
        mock_service.create_collection("legacy_col")

        legacy_point = VectorPoint(
            chunk_id="legacy_chunk_1",
            vector=[0.0, 1.0, 0.0],
            document_name="legacy_doc.pdf",
            text="Legacy point without document_id",
        )
        mock_service.upsert_vectors([legacy_point], collection_name="legacy_col")

        results = mock_service.search([0.0, 1.0, 0.0], top_k=1, collection_name="legacy_col")
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].document_id)


class TestMockVectorStoreDeletion(unittest.TestCase):
    """Unit tests covering document-scoped deletion on MockVectorStoreService."""

    def setUp(self):
        self.store = MockVectorStoreService(collection_name="del_test_col", dimension=3)
        self.store.create_collection("del_test_col")

        # Ingest 2 points for doc_A, 2 points for doc_B, and 1 legacy point (document_id=None)
        self.points = [
            VectorPoint(
                chunk_id="chunk_a1",
                document_id="doc_alpha",
                vector=[1.0, 0.0, 0.0],
                document_name="alpha.pdf",
                text="Alpha 1",
            ),
            VectorPoint(
                chunk_id="chunk_a2",
                document_id="doc_alpha",
                vector=[0.9, 0.1, 0.0],
                document_name="alpha.pdf",
                text="Alpha 2",
            ),
            VectorPoint(
                chunk_id="chunk_b1",
                document_id="doc_beta",
                vector=[0.0, 1.0, 0.0],
                document_name="beta.pdf",
                text="Beta 1",
            ),
            VectorPoint(
                chunk_id="chunk_b2",
                document_id="doc_beta",
                vector=[0.0, 0.9, 0.1],
                document_name="beta.pdf",
                text="Beta 2",
            ),
            VectorPoint(
                chunk_id="chunk_legacy",
                document_id=None,
                vector=[0.0, 0.0, 1.0],
                document_name="legacy.pdf",
                text="Legacy point without document_id",
            ),
        ]
        self.store.upsert_vectors(self.points, collection_name="del_test_col")
        self.assertEqual(self.store.count("del_test_col"), 5)

    def test_delete_by_document_id_removes_only_target_document(self):
        """27. Verify mock deletion removes points for specified document_id only."""
        deleted_count = self.store.delete_by_document_id("doc_alpha", collection_name="del_test_col")
        self.assertEqual(deleted_count, 2)
        self.assertEqual(self.store.count("del_test_col"), 3)

        # Verify doc_beta points remain searchable
        res_beta = self.store.search([0.0, 1.0, 0.0], top_k=5, collection_name="del_test_col")
        beta_docs = [r for r in res_beta if r.document_id == "doc_beta"]
        self.assertEqual(len(beta_docs), 2)

        # Verify no doc_alpha points exist
        alpha_docs = [r for r in res_beta if r.document_id == "doc_alpha"]
        self.assertEqual(len(alpha_docs), 0)

    def test_legacy_points_without_document_id_remain_untouched(self):
        """28. Verify legacy points with None document_id are never removed during deletion."""
        self.store.delete_by_document_id("doc_alpha", collection_name="del_test_col")
        self.store.delete_by_document_id("doc_beta", collection_name="del_test_col")

        # Only the legacy point should remain
        self.assertEqual(self.store.count("del_test_col"), 1)
        res = self.store.search([0.0, 0.0, 1.0], top_k=5, collection_name="del_test_col")
        self.assertEqual(len(res), 1)
        self.assertIsNone(res[0].document_id)
        self.assertEqual(res[0].chunk_id, "chunk_legacy")

    def test_nonexistent_document_id_returns_zero_without_error(self):
        """29. Verify deletion of unknown document ID returns 0 and touches nothing."""
        deleted_count = self.store.delete_by_document_id("doc_nonexistent", collection_name="del_test_col")
        self.assertEqual(deleted_count, 0)
        self.assertEqual(self.store.count("del_test_col"), 5)

    def test_empty_or_whitespace_document_id_rejected(self):
        """30. Verify invalid document_id raises VectorStoreValidationError."""
        for invalid_id in ["", "   ", "\t\n", None]:
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaises(VectorStoreValidationError):
                    self.store.delete_by_document_id(invalid_id, collection_name="del_test_col")  # type: ignore

    def test_delete_from_nonexistent_collection_raises_error(self):
        """31. Verify deletion against missing collection raises VectorStoreCollectionNotFoundError."""
        with self.assertRaises(VectorStoreCollectionNotFoundError):
            self.store.delete_by_document_id("doc_alpha", collection_name="nonexistent_collection")

    def test_configured_collection_name_is_respected(self):
        """32. Verify default configured collection name is used when collection_name is None."""
        # Insert a point into default collection
        self.store.create_collection("default_col")
        self.store._default_collection_name = "default_col"
        self.store.upsert_vectors(
            [VectorPoint(chunk_id="chunk_def", document_id="doc_def", vector=[1.0, 0.0, 0.0])],
            collection_name="default_col",
        )
        deleted = self.store.delete_by_document_id("doc_def")
        self.assertEqual(deleted, 1)
        self.assertEqual(self.store.count("default_col"), 0)

    def test_mock_create_payload_index(self):
        """33. Verify mock payload index registration."""
        self.assertTrue(self.store.create_payload_index("document_id", collection_name="del_test_col"))
        self.assertIn("document_id", self.store._collections["del_test_col"]["indices"])

        with self.assertRaises(VectorStoreValidationError):
            self.store.create_payload_index("", collection_name="del_test_col")

        with self.assertRaises(VectorStoreCollectionNotFoundError):
            self.store.create_payload_index("document_id", collection_name="missing_col")


class TestQdrantVectorStoreDeletion(unittest.TestCase):
    """Unit tests covering document-scoped deletion adapter for QdrantVectorStoreService."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.service = QdrantVectorStoreService(
            client=self.mock_client,
            collection_name="qdrant_test_col",
            dimension=3,
        )

    def test_qdrant_delete_builds_exact_filter_and_returns_count(self):
        """34. Verify Qdrant adapter builds correct MatchValue filter and returns exact point count."""
        from types import SimpleNamespace
        from qdrant_client import models

        self.mock_client.collection_exists.return_value = True
        self.mock_client.count.return_value = SimpleNamespace(count=4)
        self.mock_client.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )

        deleted = self.service.delete_by_document_id(
            document_id="doc_target_999",
            collection_name="qdrant_test_col",
        )

        self.assertEqual(deleted, 4)

        # Verify count called with exact filter
        self.mock_client.count.assert_called_once()
        count_kwargs = self.mock_client.count.call_args[1]
        self.assertEqual(count_kwargs["collection_name"], "qdrant_test_col")
        self.assertTrue(count_kwargs["exact"])
        count_filter = count_kwargs["count_filter"]
        self.assertIsInstance(count_filter, models.Filter)
        self.assertEqual(count_filter.must[0].key, "document_id")
        self.assertEqual(count_filter.must[0].match.value, "doc_target_999")

        # Verify delete called with FilterSelector
        self.mock_client.delete.assert_called_once()
        del_kwargs = self.mock_client.delete.call_args[1]
        self.assertEqual(del_kwargs["collection_name"], "qdrant_test_col")
        self.assertTrue(del_kwargs["wait"])
        points_selector = del_kwargs["points_selector"]
        self.assertIsInstance(points_selector, models.FilterSelector)
        self.assertEqual(points_selector.filter.must[0].key, "document_id")
        self.assertEqual(points_selector.filter.must[0].match.value, "doc_target_999")

    def test_qdrant_delete_nonexistent_document_id_returns_zero(self):
        """35. Verify Qdrant returns 0 when no points match document_id."""
        from types import SimpleNamespace
        from qdrant_client import models

        self.mock_client.collection_exists.return_value = True
        self.mock_client.count.return_value = SimpleNamespace(count=0)
        self.mock_client.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )

        deleted = self.service.delete_by_document_id(
            document_id="doc_not_present",
            collection_name="qdrant_test_col",
        )
        self.assertEqual(deleted, 0)

    def test_qdrant_delete_validation_errors(self):
        """36. Verify invalid document_id raises VectorStoreValidationError."""
        for invalid in ["", "   ", "\n", None]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(VectorStoreValidationError):
                    self.service.delete_by_document_id(invalid, collection_name="qdrant_test_col")  # type: ignore

    def test_qdrant_delete_nonexistent_collection_raises_error(self):
        """37. Verify deletion against missing collection raises VectorStoreCollectionNotFoundError."""
        self.mock_client.collection_exists.return_value = False
        with self.assertRaises(VectorStoreCollectionNotFoundError):
            self.service.delete_by_document_id("doc_1", collection_name="missing_col")

    def test_qdrant_delete_failure_wrapped_in_vector_store_error(self):
        """38. Verify Qdrant delete call failure is wrapped in VectorStoreError."""
        from types import SimpleNamespace

        self.mock_client.collection_exists.return_value = True
        self.mock_client.count.return_value = SimpleNamespace(count=3)
        self.mock_client.delete.side_effect = RuntimeError("Qdrant gRPC connection lost")

        with self.assertRaises(VectorStoreError) as ctx:
            self.service.delete_by_document_id("doc_fail", collection_name="qdrant_test_col")

        self.assertIn("Failed to delete points for document 'doc_fail'", str(ctx.exception))

    def test_qdrant_delete_wait_timeout_raises_vector_store_error(self):
        """39. Verify wait_timeout update status raises VectorStoreError."""
        from types import SimpleNamespace
        from qdrant_client import models

        self.mock_client.collection_exists.return_value = True
        self.mock_client.count.return_value = SimpleNamespace(count=3)
        self.mock_client.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.WAIT_TIMEOUT
        )

        with self.assertRaises(VectorStoreError) as ctx:
            self.service.delete_by_document_id("doc_timeout", collection_name="qdrant_test_col")

        self.assertIn("timed out", str(ctx.exception))

    def test_qdrant_create_payload_index_success(self):
        """40. Verify Qdrant create_payload_index calls client with KEYWORD schema."""
        from qdrant_client import models

        self.mock_client.collection_exists.return_value = True
        result = self.service.create_payload_index(
            field_name="document_id",
            collection_name="qdrant_test_col",
        )
        self.assertTrue(result)

        self.mock_client.create_payload_index.assert_called_once_with(
            collection_name="qdrant_test_col",
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def test_qdrant_create_payload_index_failure_wrapped(self):
        """41. Verify Qdrant create_payload_index wraps exceptions in VectorStoreError."""
        self.mock_client.collection_exists.return_value = True
        self.mock_client.create_payload_index.side_effect = RuntimeError("Schema error")

        with self.assertRaises(VectorStoreError):
            self.service.create_payload_index(
                field_name="document_id",
                collection_name="qdrant_test_col",
            )


class TestMockVectorStoreCountByDocumentId(unittest.TestCase):
    """Unit tests for MockVectorStoreService count_by_document_id."""

    def setUp(self):
        self.store = MockVectorStoreService(collection_name="test_col", dimension=3)
        self.store.create_collection("test_col", dimension=3)
        points = [
            VectorPoint(chunk_id="c1", document_id="doc_alpha", vector=[1.0, 0.0, 0.0]),
            VectorPoint(chunk_id="c2", document_id="doc_alpha", vector=[0.0, 1.0, 0.0]),
            VectorPoint(chunk_id="c3", document_id="doc_beta", vector=[0.0, 0.0, 1.0]),
            VectorPoint(chunk_id="c4", vector=[0.5, 0.5, 0.0]),  # legacy, no document_id
        ]
        self.store.upsert_vectors(points, collection_name="test_col")

    def test_mock_count_exact_document_id(self):
        """42. Verify exact document ID returns correct count."""
        self.assertEqual(self.store.count_by_document_id("doc_alpha", collection_name="test_col"), 2)
        self.assertEqual(self.store.count_by_document_id("doc_beta", collection_name="test_col"), 1)

    def test_mock_count_different_document_id_does_not_contribute(self):
        """43. Verify different document ID does not contribute to the count."""
        count_alpha = self.store.count_by_document_id("doc_alpha", collection_name="test_col")
        count_total = self.store.count("test_col")
        self.assertEqual(count_alpha, 2)
        self.assertEqual(count_total, 4)

    def test_mock_count_nonexistent_document_id_returns_zero(self):
        """44. Verify nonexistent document ID returns zero."""
        self.assertEqual(self.store.count_by_document_id("doc_missing", collection_name="test_col"), 0)

    def test_mock_count_missing_collection_raises_error(self):
        """45. Verify missing collection raises VectorStoreCollectionNotFoundError."""
        with self.assertRaises(VectorStoreCollectionNotFoundError):
            self.store.count_by_document_id("doc_alpha", collection_name="nonexistent_col")

    def test_mock_count_invalid_document_id_raises_validation_error(self):
        """46. Verify invalid document IDs raise VectorStoreValidationError."""
        for invalid_id in ["", "   ", "\t\n", None]:
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaises(VectorStoreValidationError):
                    self.store.count_by_document_id(invalid_id, collection_name="test_col")  # type: ignore[arg-type]

    def test_mock_count_respects_default_collection(self):
        """47. Verify default collection name is used when collection_name is None."""
        self.assertEqual(self.store.count_by_document_id("doc_alpha"), 2)


class TestQdrantVectorStoreCountByDocumentId(unittest.TestCase):
    """Unit tests covering exact vector counting for QdrantVectorStoreService."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.service = QdrantVectorStoreService(
            client=self.mock_client,
            collection_name="qdrant_test_col",
            dimension=3,
        )

    def test_qdrant_count_exact_filter_and_returns_count(self):
        """48. Verify Qdrant adapter builds exact filter, passes exact=True, returns count, and does not delete."""
        from types import SimpleNamespace
        from qdrant_client import models

        self.mock_client.collection_exists.return_value = True
        self.mock_client.count.return_value = SimpleNamespace(count=5)

        count = self.service.count_by_document_id(
            document_id="doc_target_123",
            collection_name="qdrant_test_col",
        )

        self.assertEqual(count, 5)

        # Verify exact filter construction
        self.mock_client.count.assert_called_once()
        count_kwargs = self.mock_client.count.call_args[1]
        self.assertEqual(count_kwargs["collection_name"], "qdrant_test_col")
        self.assertTrue(count_kwargs["exact"])
        count_filter = count_kwargs["count_filter"]
        self.assertIsInstance(count_filter, models.Filter)
        self.assertEqual(count_filter.must[0].key, "document_id")
        self.assertEqual(count_filter.must[0].match.value, "doc_target_123")

        # Crucial safety guarantee: delete must NEVER be called
        self.mock_client.delete.assert_not_called()

    def test_qdrant_count_nonexistent_document_id_returns_zero(self):
        """49. Verify Qdrant returns 0 when no points match document_id."""
        from types import SimpleNamespace

        self.mock_client.collection_exists.return_value = True
        self.mock_client.count.return_value = SimpleNamespace(count=0)

        count = self.service.count_by_document_id(
            document_id="doc_absent",
            collection_name="qdrant_test_col",
        )
        self.assertEqual(count, 0)
        self.mock_client.delete.assert_not_called()

    def test_qdrant_count_missing_collection_raises_error(self):
        """50. Verify missing collection raises VectorStoreCollectionNotFoundError."""
        self.mock_client.collection_exists.return_value = False
        with self.assertRaises(VectorStoreCollectionNotFoundError):
            self.service.count_by_document_id("doc_1", collection_name="missing_col")

    def test_qdrant_count_transient_failure_wrapped_in_vector_store_error(self):
        """51. Verify Qdrant communication/transient errors are wrapped in VectorStoreError."""
        self.mock_client.collection_exists.return_value = True
        self.mock_client.count.side_effect = RuntimeError("Qdrant connection dropped")

        with self.assertRaises(VectorStoreError) as ctx:
            self.service.count_by_document_id("doc_err", collection_name="qdrant_test_col")

        self.assertIn("Failed to count points for document 'doc_err'", str(ctx.exception))

    def test_qdrant_count_invalid_document_id_raises_validation_error(self):
        """52. Verify invalid document IDs raise VectorStoreValidationError."""
        for invalid in ["", "   ", "\n", None]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(VectorStoreValidationError):
                    self.service.count_by_document_id(invalid, collection_name="qdrant_test_col")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()


