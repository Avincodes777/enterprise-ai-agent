"""Live integration tests for safe document deletion with registry coordination.

Exercises DELETE /api/documents/{document_id} against a live running Qdrant container
and isolated SQLite registry, verifying exact document vector deletion, legacy point safety,
non-target document preservation, and collection persistence.
"""

from pathlib import Path
import sys
import tempfile
import unittest
import uuid

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_document_registry_dependency,
    get_vector_store_dependency,
    reset_document_registry_cache,
    reset_vector_store_cache,
)
from app.config import get_settings
from app.database import (
    DocumentRegistryRepository,
    DocumentStatus,
)
from app.main import app
from app.services import (
    FastEmbedEmbeddingService,
    QdrantVectorStoreService,
    VectorPoint,
)


class TestLiveDocumentDeletionAPI(unittest.TestCase):
    """Live integration tests for document deletion API coordinated with Qdrant and SQLite."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.qdrant_url = cls.settings.qdrant_url or "http://localhost:6333"
        # Unique test-owned collection name to ensure zero impact on production collection
        cls.test_collection_name = f"test_live_del_{uuid.uuid4().hex[:12]}"
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

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_live_del_registry.db"
        self.repo = DocumentRegistryRepository(db_path=str(self.db_path))

        # Create isolated test collection
        self.vector_store.create_collection(
            collection_name=self.test_collection_name,
            dimension=self.settings.embedding_dimension,
            recreate_if_exists=True,
        )

        # Configure API dependency overrides
        app.dependency_overrides[get_document_registry_dependency] = lambda: self.repo
        app.dependency_overrides[get_vector_store_dependency] = lambda: self.vector_store
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        reset_document_registry_cache()
        reset_vector_store_cache()

        # Clean up only the test-owned collection
        try:
            if self.vector_store.collection_exists(self.test_collection_name):
                self.vector_store.delete_collection(self.test_collection_name)
        except Exception:
            pass

        self.temp_dir.cleanup()

    def test_live_document_deletion_api_workflow(self):
        """
        Verify live API deletion end-to-end:
        1. Register multiple documents in SQLite (eligible doc_A, eligible doc_B, legacy doc_C).
        2. Upsert synthetic vector points with distinct document_ids and one legacy point without document_id.
        3. Reject deletion of legacy doc_C with HTTP 409, preserving all vectors.
        4. Delete eligible doc_A via DELETE /api/documents/{doc_A_id}.
        5. Verify doc_A vectors are completely removed from Qdrant.
        6. Verify doc_B vectors remain present and searchable.
        7. Verify legacy point without document_id remains intact.
        8. Verify doc_A SQLite record is removed; doc_B and doc_C SQLite records remain.
        9. Verify test collection remains intact (never deleted).
        """
        # Step 1: Register records in isolated SQLite registry
        rec_a = self.repo.create_record(
            "alpha_specs.pdf",
            self.test_collection_name,
            provenance_eligible=True,
        )
        self.repo.update_status_succeeded(rec_a.document_id, pages_processed=2, chunks_created=2, chunks_stored=2)

        rec_b = self.repo.create_record(
            "beta_security.pdf",
            self.test_collection_name,
            provenance_eligible=True,
        )
        self.repo.update_status_succeeded(rec_b.document_id, pages_processed=2, chunks_created=2, chunks_stored=2)

        rec_legacy = self.repo.create_record(
            "legacy_manual.pdf",
            self.test_collection_name,
            provenance_eligible=False,
        )
        self.repo.update_status_succeeded(rec_legacy.document_id, pages_processed=1, chunks_created=1, chunks_stored=1)

        # Step 2: Generate real embeddings and upsert points
        texts = [
            "Alpha architecture microservices and container configuration.",
            "Alpha deployment guidelines on kubernetes clusters.",
            "Beta access control and multi-factor authorization policy.",
            "Beta zero trust networking rules.",
            "Unattributed legacy chunk without document provenance identifier.",
        ]
        vectors = self.embedding_service.embed_documents(texts)

        points = [
            # Doc A points
            VectorPoint(
                chunk_id="chunk_a1",
                document_id=rec_a.document_id,
                vector=vectors[0],
                document_name="alpha_specs.pdf",
                page_number=1,
                chunk_index=0,
                text=texts[0],
            ),
            VectorPoint(
                chunk_id="chunk_a2",
                document_id=rec_a.document_id,
                vector=vectors[1],
                document_name="alpha_specs.pdf",
                page_number=2,
                chunk_index=1,
                text=texts[1],
            ),
            # Doc B points
            VectorPoint(
                chunk_id="chunk_b1",
                document_id=rec_b.document_id,
                vector=vectors[2],
                document_name="beta_security.pdf",
                page_number=1,
                chunk_index=0,
                text=texts[2],
            ),
            VectorPoint(
                chunk_id="chunk_b2",
                document_id=rec_b.document_id,
                vector=vectors[3],
                document_name="beta_security.pdf",
                page_number=2,
                chunk_index=1,
                text=texts[3],
            ),
            # Legacy point (no document_id)
            VectorPoint(
                chunk_id="chunk_legacy",
                document_id=None,
                vector=vectors[4],
                document_name="legacy_manual.pdf",
                page_number=1,
                chunk_index=0,
                text=texts[4],
            ),
        ]

        upserted = self.vector_store.upsert_vectors(points, collection_name=self.test_collection_name)
        self.assertEqual(upserted, 5)
        self.assertEqual(self.vector_store.count(self.test_collection_name), 5)

        # Step 3: Deletion of legacy document must be rejected with 409
        legacy_res = self.client.delete(f"/api/documents/{rec_legacy.document_id}")
        self.assertEqual(legacy_res.status_code, 409)
        self.assertIn("legacy unverified provenance", legacy_res.json()["detail"])
        # All 5 points remain in Qdrant
        self.assertEqual(self.vector_store.count(self.test_collection_name), 5)
        # SQLite record still exists
        self.assertIsNotNone(self.repo.get_record(rec_legacy.document_id))

        # Step 4: Delete eligible Doc A via API
        del_res = self.client.delete(f"/api/documents/{rec_a.document_id}")
        self.assertEqual(del_res.status_code, 200)
        data = del_res.json()
        self.assertEqual(data["document_id"], rec_a.document_id)
        self.assertTrue(data["deleted"])
        self.assertEqual(data["vectors_deleted"], 2)

        # Step 5: Verify Doc A SQLite record is retained as durable tombstone with status DELETED
        rec_a_tombstone = self.repo.get_record(rec_a.document_id)
        self.assertIsNotNone(rec_a_tombstone)
        self.assertEqual(rec_a_tombstone.status, DocumentStatus.DELETED)
        self.assertEqual(rec_a_tombstone.chunks_stored, 0)

        # Step 6: Verify Qdrant point counts & search results
        self.assertEqual(self.vector_store.count(self.test_collection_name), 3)

        # Search for Beta content
        query_beta = self.embedding_service.embed_query("multi-factor authorization policy")
        results_beta = self.vector_store.search(
            query_vector=query_beta,
            top_k=5,
            collection_name=self.test_collection_name,
        )
        # Verify Doc B chunks are found
        doc_b_found = [r for r in results_beta if r.document_id == rec_b.document_id]
        self.assertEqual(len(doc_b_found), 2)

        # Verify Doc A chunks are completely absent
        doc_a_found = [r for r in results_beta if r.document_id == rec_a.document_id]
        self.assertEqual(len(doc_a_found), 0)

        # Step 7: Verify legacy point without document_id remains intact
        legacy_found = [r for r in results_beta if r.chunk_id == "chunk_legacy"]
        self.assertEqual(len(legacy_found), 1)
        self.assertIsNone(legacy_found[0].document_id)

        # Step 8: Verify other documents remain in SQLite
        self.assertIsNotNone(self.repo.get_record(rec_b.document_id))
        self.assertIsNotNone(self.repo.get_record(rec_legacy.document_id))

        # Step 9: Verify collection was not deleted
        self.assertTrue(self.vector_store.collection_exists(self.test_collection_name))


if __name__ == "__main__":
    unittest.main()
