"""Persistent SQLite repository for tracking uploaded document ingestion lifecycle."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import logging
import math
from pathlib import Path
import sqlite3
from typing import Generator, List, Optional
import uuid

from app.config import Settings, get_settings
from app.database.models import DocumentRecord, DocumentStatus

logger = logging.getLogger(__name__)

# Default duration after which an in-flight deletion claim is deemed abandoned and eligible for recovery
DEFAULT_DELETION_LEASE_SECONDS: float = 300.0

# Default duration after which an in-flight ingestion is deemed abandoned and eligible for recovery
DEFAULT_INGESTION_LEASE_SECONDS: float = 900.0

# Default duration after which an in-flight reconciliation claim is deemed abandoned and eligible for recovery
DEFAULT_RECONCILIATION_LEASE_SECONDS: float = 60.0

# Default minimum age (seconds) for failed/deletion_failed records before reconciliation candidate discovery
DEFAULT_RECONCILIATION_MIN_AGE_SECONDS: int = 120

# Default maximum number of reconciliation candidates to discover in a single preview query
DEFAULT_RECONCILIATION_LIMIT: int = 50



# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class DocumentRegistryError(Exception):
    """Base exception for document registry repository operations."""
    pass


class DocumentNotFoundError(DocumentRegistryError):
    """Raised when a requested document ID does not exist in the registry."""
    pass


class DocumentStateConflictError(DocumentRegistryError):
    """Raised when a status transition is rejected because the document is not in the expected state."""
    pass


# -----------------------------------------------------------------------------
# Repository Implementation
# -----------------------------------------------------------------------------
class DocumentRegistryRepository:
    """
    Isolated SQLite repository managing document registration, status tracking,
    and provenance retrieval.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        connection: Optional[sqlite3.Connection] = None,
        settings: Optional[Settings] = None,
    ):
        curr_settings = settings or get_settings()
        self._db_path = db_path or curr_settings.sqlite_db_path
        self._connection = connection
        self._initialized = False

    @property
    def db_path(self) -> str:
        """Configured database filesystem path or URI."""
        return self._db_path

    def _ensure_dir_exists(self) -> None:
        """Create parent directory of database file if writing to local filesystem."""
        if self._connection is not None or self._db_path == ":memory:":
            return
        if self._db_path.startswith("file:"):
            return
        try:
            db_file = Path(self._db_path)
            if db_file.parent and not db_file.parent.exists():
                db_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create database directory for '%s': %s", self._db_path, exc)
            raise DocumentRegistryError(
                f"Failed to create database directory: {exc}"
            ) from exc

    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        """Initialize SQLite schema idempotently if not already created."""
        if self._initialized:
            return

        schema_sql = """
        CREATE TABLE IF NOT EXISTS document_registry (
            document_id TEXT PRIMARY KEY,
            document_name TEXT NOT NULL,
            status TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            pages_processed INTEGER,
            chunks_created INTEGER,
            chunks_stored INTEGER,
            error_code TEXT,
            error_message TEXT,
            provenance_eligible INTEGER NOT NULL DEFAULT 0,
            claim_token TEXT,
            previous_status TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_doc_registry_created_at
            ON document_registry(created_at DESC, document_id DESC);
        """
        try:
            conn.executescript(schema_sql)

            # Idempotent migration for existing tables created before new columns
            cursor = conn.execute("PRAGMA table_info(document_registry);")
            columns = [row["name"] for row in cursor.fetchall()]
            if "provenance_eligible" not in columns:
                conn.execute(
                    "ALTER TABLE document_registry ADD COLUMN provenance_eligible INTEGER NOT NULL DEFAULT 0;"
                )
            if "claim_token" not in columns:
                conn.execute(
                    "ALTER TABLE document_registry ADD COLUMN claim_token TEXT;"
                )
            if "previous_status" not in columns:
                conn.execute(
                    "ALTER TABLE document_registry ADD COLUMN previous_status TEXT;"
                )

            self._initialized = True
        except Exception as exc:
            logger.error("Failed to initialize document registry tables: %s", exc)
            raise DocumentRegistryError(
                f"Failed to initialize registry database schema: {exc}"
            ) from exc

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Provide a managed SQLite connection, ensuring proper commit, rollback,
        and timely connection closure.
        """
        self._ensure_dir_exists()

        if self._connection is not None:
            self._connection.row_factory = sqlite3.Row
            self._ensure_tables(self._connection)
            yield self._connection
            return

        conn = None
        try:
            conn = sqlite3.connect(self._db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            self._ensure_tables(conn)
            yield conn
            conn.commit()
        except DocumentRegistryError:
            if conn:
                conn.rollback()
            raise
        except Exception as exc:
            if conn:
                conn.rollback()
            logger.error("SQLite operation error: %s", exc, exc_info=True)
            raise DocumentRegistryError(f"Database operation failed: {exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        """Map a sqlite3.Row to a validated DocumentRecord model."""
        keys = row.keys()
        return DocumentRecord(
            document_id=row["document_id"],
            document_name=row["document_name"],
            status=DocumentStatus(row["status"]),
            collection_name=row["collection_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            pages_processed=row["pages_processed"],
            chunks_created=row["chunks_created"],
            chunks_stored=row["chunks_stored"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            provenance_eligible=bool(row["provenance_eligible"]) if "provenance_eligible" in keys else False,
            claim_token=row["claim_token"] if "claim_token" in keys else None,
            previous_status=row["previous_status"] if "previous_status" in keys else None,
        )

    def create_record(
        self,
        document_name: str,
        collection_name: str,
        document_id: Optional[str] = None,
        provenance_eligible: bool = False,
    ) -> DocumentRecord:
        """
        Create a new document registry record with 'processing' status.

        Args:
            document_name: Original uploaded filename.
            collection_name: Target vector store collection.
            document_id: Optional custom unique document identifier.
            provenance_eligible: Whether vector points are verified to carry document provenance.

        Returns:
            The created DocumentRecord.
        """
        if not document_name or not document_name.strip():
            raise DocumentRegistryError("document_name cannot be empty or whitespace-only")
        if not collection_name or not collection_name.strip():
            raise DocumentRegistryError("collection_name cannot be empty or whitespace-only")

        doc_id = document_id.strip() if document_id and document_id.strip() else f"doc_{uuid.uuid4().hex[:16]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        prov_val = 1 if provenance_eligible else 0

        insert_sql = """
        INSERT INTO document_registry (
            document_id, document_name, status, collection_name,
            created_at, updated_at, provenance_eligible
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self._get_connection() as conn:
            conn.execute(
                insert_sql,
                (
                    doc_id,
                    document_name.strip(),
                    DocumentStatus.PROCESSING.value,
                    collection_name.strip(),
                    now_iso,
                    now_iso,
                    prov_val,
                ),
            )


        return self.get_record(doc_id)  # type: ignore[return-value]

    def update_status_succeeded(
        self,
        document_id: str,
        pages_processed: int,
        chunks_created: int,
        chunks_stored: int,
        collection_name: Optional[str] = None,
    ) -> DocumentRecord:
        """
        Update a document record to 'succeeded' with processing metrics.

        Args:
            document_id: Target document identifier.
            pages_processed: Number of pages parsed.
            chunks_created: Total text chunks created.
            chunks_stored: Total vector points upserted.
            collection_name: Optional updated vector collection name.

        Returns:
            The updated DocumentRecord.

        Raises:
            DocumentNotFoundError: If the document_id does not exist.
            DocumentStateConflictError: If the document is not in 'processing' status.
        """
        if not document_id or not document_id.strip():
            raise DocumentRegistryError("document_id cannot be empty")

        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            if collection_name is not None and collection_name.strip():
                update_sql = """
                UPDATE document_registry
                SET status = ?, pages_processed = ?, chunks_created = ?, chunks_stored = ?,
                    collection_name = ?, error_code = NULL, error_message = NULL, updated_at = ?
                WHERE document_id = ? AND status = ?;
                """
                cursor = conn.execute(
                    update_sql,
                    (
                        DocumentStatus.SUCCEEDED.value,
                        pages_processed,
                        chunks_created,
                        chunks_stored,
                        collection_name.strip(),
                        now_iso,
                        document_id.strip(),
                        DocumentStatus.PROCESSING.value,
                    ),
                )
            else:
                update_sql = """
                UPDATE document_registry
                SET status = ?, pages_processed = ?, chunks_created = ?, chunks_stored = ?,
                    error_code = NULL, error_message = NULL, updated_at = ?
                WHERE document_id = ? AND status = ?;
                """
                cursor = conn.execute(
                    update_sql,
                    (
                        DocumentStatus.SUCCEEDED.value,
                        pages_processed,
                        chunks_created,
                        chunks_stored,
                        now_iso,
                        document_id.strip(),
                        DocumentStatus.PROCESSING.value,
                    ),
                )

            if cursor.rowcount == 0:
                select_cursor = conn.execute(
                    "SELECT status FROM document_registry WHERE document_id = ?;",
                    (document_id.strip(),),
                )
                existing = select_cursor.fetchone()
                if existing is None:
                    raise DocumentNotFoundError(f"Document with ID '{document_id}' not found in registry.")
                current_status = existing["status"]
                raise DocumentStateConflictError(
                    f"Cannot transition document '{document_id}' to 'succeeded': "
                    f"current status is '{current_status}', expected 'processing'."
                )

        return self.get_record(document_id.strip())  # type: ignore[return-value]

    def update_status_failed(
        self,
        document_id: str,
        error_code: str,
        error_message: str,
    ) -> DocumentRecord:
        """
        Update a document record to 'failed' with safe failure classification.
        Can only transition from 'processing' status to prevent late workers from
        overwriting newer registry states.

        Args:
            document_id: Target document identifier.
            error_code: Safe categorized error code.
            error_message: Sanitized error description.

        Returns:
            The updated DocumentRecord.

        Raises:
            DocumentNotFoundError: If the document_id does not exist.
            DocumentStateConflictError: If the document is not in 'processing' status.
        """
        if not document_id or not document_id.strip():
            raise DocumentRegistryError("document_id cannot be empty")

        now_iso = datetime.now(timezone.utc).isoformat()
        clean_code = (error_code or "UNKNOWN_ERROR").strip()
        clean_message = (error_message or "An unclassified error occurred.").strip()

        update_sql = """
        UPDATE document_registry
        SET status = ?, error_code = ?, error_message = ?, updated_at = ?
        WHERE document_id = ? AND status = ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                update_sql,
                (
                    DocumentStatus.FAILED.value,
                    clean_code,
                    clean_message,
                    now_iso,
                    document_id.strip(),
                    DocumentStatus.PROCESSING.value,
                ),
            )
            if cursor.rowcount == 0:
                select_cursor = conn.execute(
                    "SELECT status FROM document_registry WHERE document_id = ?;",
                    (document_id.strip(),),
                )
                existing = select_cursor.fetchone()
                if existing is None:
                    raise DocumentNotFoundError(f"Document with ID '{document_id}' not found in registry.")
                current_status = existing["status"]
                raise DocumentStateConflictError(
                    f"Cannot transition document '{document_id}' to 'failed': "
                    f"current status is '{current_status}', expected 'processing'."
                )

        return self.get_record(document_id.strip())  # type: ignore[return-value]

    def get_record(self, document_id: str) -> Optional[DocumentRecord]:
        """
        Retrieve a single document record by unique document_id.

        Args:
            document_id: Identifier of the document to retrieve.

        Returns:
            DocumentRecord if found, None otherwise.
        """
        if not document_id or not document_id.strip():
            return None

        select_sql = """
        SELECT * FROM document_registry WHERE document_id = ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(select_sql, (document_id.strip(),))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list_records(self, limit: int = 50, offset: int = 0) -> List[DocumentRecord]:
        """
        Retrieve a paginated list of document records ordered newest first.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of DocumentRecord objects ordered by created_at DESC, document_id DESC.
        """
        if limit <= 0:
            raise DocumentRegistryError("limit must be greater than 0")
        if offset < 0:
            raise DocumentRegistryError("offset must be non-negative")

        select_sql = """
        SELECT * FROM document_registry
        ORDER BY created_at DESC, document_id DESC
        LIMIT ? OFFSET ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(select_sql, (limit, offset))
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    def count_records(self) -> int:
        """
        Return the total number of registered documents.

        Returns:
            Total integer count.
        """
        count_sql = "SELECT COUNT(*) AS total FROM document_registry;"
        with self._get_connection() as conn:
            cursor = conn.execute(count_sql)
            row = cursor.fetchone()
            return int(row["total"]) if row else 0

    def list_reconciliation_candidates(
        self,
        min_age_seconds: int = DEFAULT_RECONCILIATION_MIN_AGE_SECONDS,
        limit: int = DEFAULT_RECONCILIATION_LIMIT,
    ) -> List[DocumentRecord]:
        """
        Discover eligible reconciliation candidate records for read-only preview.

        Candidate eligibility criteria:
        1. status IN ('failed', 'deletion_failed')
        2. provenance_eligible == 1 (verified document-to-vector provenance)
        3. updated_at is older than min_age_seconds grace period (datetime(updated_at) < datetime(cutoff))
        4. ordered oldest first with deterministic tie-breaker (updated_at ASC, document_id ASC)
        5. capped at limit records

        This method is strictly read-only and performs zero writes or state transitions.

        Args:
            min_age_seconds: Non-negative number of seconds defining the grace period.
                Records updated more recently than this grace period are excluded.
            limit: Positive integer limit on the maximum number of candidates to return.

        Returns:
            List of eligible DocumentRecord candidates ordered oldest first.

        Raises:
            DocumentRegistryError: If min_age_seconds or limit are invalid or out of range.
        """
        if isinstance(min_age_seconds, bool) or not isinstance(min_age_seconds, (int, float)) or min_age_seconds < 0:
            raise DocumentRegistryError("min_age_seconds must be a non-negative number")
        if math.isnan(min_age_seconds) or math.isinf(min_age_seconds):
            raise DocumentRegistryError("min_age_seconds must be a finite number")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise DocumentRegistryError("limit must be a positive integer greater than 0")

        now_utc = datetime.now(timezone.utc)
        cutoff_utc = now_utc - timedelta(seconds=float(min_age_seconds))
        cutoff_iso = cutoff_utc.isoformat()

        candidate_sql = """
        SELECT * FROM document_registry
        WHERE (
            status IN ('failed', 'deletion_failed')
            OR status = 'reconciling'
          )
          AND provenance_eligible = 1
          AND datetime(updated_at) < datetime(?)
        ORDER BY updated_at ASC, document_id ASC
        LIMIT ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(candidate_sql, (cutoff_iso, limit))
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    def claim_for_deletion(
        self,
        document_id: str,
        claim_timeout_seconds: float = DEFAULT_DELETION_LEASE_SECONDS,
    ) -> Optional[DocumentRecord]:
        """
        Atomically transition an eligible document record to 'deleting' status.

        Only records that satisfy provenance_eligible = 1 and either:
        1. Have status in ('succeeded', 'failed', 'deletion_failed'), OR
        2. Have status = 'deleting' with updated_at older than claim_timeout_seconds
           (safely reclaiming an abandoned or crashed deletion attempt).

        Concurrency & Lease Limitations:
        - Relies on UTC ISO-8601 timestamps and SQLite's datetime() UTC normalization.
        - Active deletions taking longer than claim_timeout_seconds could theoretically
          be reclaimed; claim_timeout_seconds should be significantly larger than normal
          deletion latency (which is typically < 2 seconds).
        - SQLite atomic write lock guarantees that between concurrent callers, at most
          one caller can successfully transition the record into the new lease window.
        - Processing documents, unverified legacy documents (provenance_eligible = 0),
          and non-existent records are never claimed.

        Args:
            document_id: Target document identifier.
            claim_timeout_seconds: Number of seconds after which a claim in 'deleting'
                status is deemed abandoned and can be reclaimed (default: 300.0).

        Returns:
            The claimed DocumentRecord if successfully transitioned, or None otherwise.
        """
        if not document_id or not document_id.strip():
            return None

        clean_id = document_id.strip()
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        cutoff_utc = now_utc - timedelta(seconds=max(0.0, claim_timeout_seconds))
        cutoff_iso = cutoff_utc.isoformat()
        claim_token = f"del_{uuid.uuid4().hex}"

        claim_sql = """
        UPDATE document_registry
        SET status = ?, claim_token = ?, updated_at = ?
        WHERE document_id = ?
          AND provenance_eligible = 1
          AND (
            status IN (?, ?, ?)
            OR (status = ? AND datetime(updated_at) < datetime(?))
          );
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                claim_sql,
                (
                    DocumentStatus.DELETING.value,
                    claim_token,
                    now_iso,
                    clean_id,
                    DocumentStatus.SUCCEEDED.value,
                    DocumentStatus.FAILED.value,
                    DocumentStatus.DELETION_FAILED.value,
                    DocumentStatus.DELETING.value,
                    cutoff_iso,
                ),
            )
            if cursor.rowcount == 0:
                return None

        return self.get_record(clean_id)

    def claim_abandoned_ingestion(
        self,
        document_id: str,
        lease_seconds: float = DEFAULT_INGESTION_LEASE_SECONDS,
    ) -> Optional[DocumentRecord]:
        """
        Atomically transition an abandoned 'processing' record to 'failed'.

        A record is eligible to be claimed if and only if:
        1. document_id matches.
        2. provenance_eligible == 1 (verified document-to-vector provenance).
        3. status == 'processing'.
        4. updated_at is older than lease_seconds (datetime(updated_at) < datetime(cutoff)).

        Concurrency & Atomicity:
        - Executed as a single atomic conditional SQL UPDATE statement.
        - Under concurrent claim attempts on the same document, exactly one caller
          can successfully update the row and transition it out of 'processing'.
        - Subsequent callers will find rowcount == 0 and return None.

        Args:
            document_id: Target document identifier.
            lease_seconds: Positive duration in seconds after which a processing record
                is deemed abandoned (default: DEFAULT_INGESTION_LEASE_SECONDS).

        Returns:
            The claimed DocumentRecord in 'failed' status if successfully transitioned,
            or None if the document is not eligible (not found, fresh, ineligible provenance,
            or not in 'processing' status).

        Raises:
            DocumentRegistryError: If document_id is invalid or lease_seconds is not positive.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise DocumentRegistryError("document_id must be a non-empty string")
        if lease_seconds is None or not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise DocumentRegistryError("lease_seconds must be a positive number greater than 0")

        clean_id = document_id.strip()
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        cutoff_utc = now_utc - timedelta(seconds=float(lease_seconds))
        cutoff_iso = cutoff_utc.isoformat()

        claim_sql = """
        UPDATE document_registry
        SET status = ?, error_code = ?, error_message = ?, updated_at = ?
        WHERE document_id = ?
          AND provenance_eligible = 1
          AND status = ?
          AND datetime(updated_at) < datetime(?);
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                claim_sql,
                (
                    DocumentStatus.FAILED.value,
                    "INGESTION_ABANDONED",
                    "Document ingestion timed out and was reclaimed as abandoned.",
                    now_iso,
                    clean_id,
                    DocumentStatus.PROCESSING.value,
                    cutoff_iso,
                ),
            )
            if cursor.rowcount == 0:
                return None

        return self.get_record(clean_id)

    def heartbeat_ingestion(self, document_id: str) -> DocumentRecord:
        """
        Renew an in-flight ingestion's activity timestamp while in 'processing' status.

        Args:
            document_id: Unique persistent identifier of the document.

        Returns:
            The updated DocumentRecord with fresh updated_at timestamp.

        Raises:
            DocumentRegistryError: If document_id is invalid or empty.
            DocumentNotFoundError: If the document does not exist in registry.
            DocumentStateConflictError: If the document is not in 'processing' status.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise DocumentRegistryError("document_id must be a non-empty string")

        clean_id = document_id.strip()
        now_iso = datetime.now(timezone.utc).isoformat()

        heartbeat_sql = """
        UPDATE document_registry
        SET updated_at = ?
        WHERE document_id = ?
          AND status = ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                heartbeat_sql,
                (
                    now_iso,
                    clean_id,
                    DocumentStatus.PROCESSING.value,
                ),
            )
            if cursor.rowcount == 0:
                select_cursor = conn.execute(
                    "SELECT status FROM document_registry WHERE document_id = ?;",
                    (clean_id,),
                )
                existing = select_cursor.fetchone()
                if existing is None:
                    raise DocumentNotFoundError(f"Document with ID '{clean_id}' not found in registry.")
                current_status = existing["status"]
                raise DocumentStateConflictError(
                    f"Cannot renew heartbeat for document '{clean_id}': "
                    f"current status is '{current_status}', expected 'processing'."
                )

        return self.get_record(clean_id)  # type: ignore[return-value]

    def update_status_deletion_failed(
        self,
        document_id: str,
        error_code: str,
        error_message: str,
        claim_token: Optional[str] = None,
    ) -> Optional[DocumentRecord]:
        """
        Update a document record to 'deletion_failed' with safe failure classification.
        Can only transition from 'deleting' status to prevent overwriting newer registry states.
        If claim_token is supplied, verifies that the active claim matches before updating.

        Args:
            document_id: Target document identifier.
            error_code: Categorized error code.
            error_message: Sanitized error description.
            claim_token: Optional generation attempt token to verify ownership.

        Returns:
            The updated DocumentRecord.

        Raises:
            DocumentRegistryError: If document_id is invalid.
            DocumentNotFoundError: If the document does not exist.
            DocumentStateConflictError: If the document is not in 'deleting' status or token mismatches.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise DocumentRegistryError("document_id must be a non-empty string")

        clean_id = document_id.strip()
        now_iso = datetime.now(timezone.utc).isoformat()
        clean_code = (error_code or "DELETION_ERROR").strip()
        clean_message = (error_message or "An error occurred while deleting the document.").strip()

        if claim_token is not None:
            clean_token = claim_token.strip()
            update_sql = """
            UPDATE document_registry
            SET status = ?, error_code = ?, error_message = ?, claim_token = NULL, updated_at = ?
            WHERE document_id = ? AND status = ? AND claim_token = ?;
            """
            params = (
                DocumentStatus.DELETION_FAILED.value,
                clean_code,
                clean_message,
                now_iso,
                clean_id,
                DocumentStatus.DELETING.value,
                clean_token,
            )
        else:
            update_sql = """
            UPDATE document_registry
            SET status = ?, error_code = ?, error_message = ?, claim_token = NULL, updated_at = ?
            WHERE document_id = ? AND status = ?;
            """
            params = (
                DocumentStatus.DELETION_FAILED.value,
                clean_code,
                clean_message,
                now_iso,
                clean_id,
                DocumentStatus.DELETING.value,
            )

        with self._get_connection() as conn:
            cursor = conn.execute(update_sql, params)
            if cursor.rowcount == 0:
                select_cursor = conn.execute(
                    "SELECT status, claim_token FROM document_registry WHERE document_id = ?;",
                    (clean_id,),
                )
                existing = select_cursor.fetchone()
                if existing is None:
                    raise DocumentNotFoundError(f"Document with ID '{clean_id}' not found in registry.")
                current_status = existing["status"]
                stored_token = existing["claim_token"]
                if claim_token is not None and stored_token != claim_token.strip():
                    raise DocumentStateConflictError(
                        f"Cannot transition document '{clean_id}' to 'deletion_failed': "
                        f"claim token mismatch for document '{clean_id}' (stale claim attempt)."
                    )
                raise DocumentStateConflictError(
                    f"Cannot transition document '{clean_id}' to 'deletion_failed': "
                    f"current status is '{current_status}', expected 'deleting'."
                )

        return self.get_record(clean_id)

    def finalize_deletion(
        self,
        document_id: str,
        claim_token: str,
    ) -> DocumentRecord:
        """
        Finalize a user-requested deletion by transitioning from 'deleting' to 'deleted' (durable tombstone).

        Verifies that the record is currently in 'deleting' status and still owned by the specific claim_token.
        Clears chunks_stored to 0 and clears claim_token, but preserves document_id, document_name,
        collection_name, created_at, provenance_eligible, and error metadata.

        Args:
            document_id: Unique persistent identifier of the document.
            claim_token: Generation attempt token issued during claim_for_deletion.

        Returns:
            The finalized DocumentRecord with status='deleted'.

        Raises:
            DocumentRegistryError: If document_id or claim_token are invalid strings.
            DocumentNotFoundError: If the document does not exist in registry.
            DocumentStateConflictError: If the document is not in 'deleting' status or the claim_token
                does not match (e.g. claim expired and was re-claimed by another worker).
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise DocumentRegistryError("document_id must be a non-empty string")
        if not claim_token or not isinstance(claim_token, str) or not claim_token.strip():
            raise DocumentRegistryError("claim_token must be a non-empty string")

        clean_id = document_id.strip()
        clean_token = claim_token.strip()
        now_iso = datetime.now(timezone.utc).isoformat()

        finalize_sql = """
        UPDATE document_registry
        SET status = ?,
            chunks_stored = 0,
            claim_token = NULL,
            updated_at = ?
        WHERE document_id = ?
          AND status = ?
          AND claim_token = ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                finalize_sql,
                (
                    DocumentStatus.DELETED.value,
                    now_iso,
                    clean_id,
                    DocumentStatus.DELETING.value,
                    clean_token,
                ),
            )
            if cursor.rowcount == 0:
                select_cursor = conn.execute(
                    "SELECT status, claim_token FROM document_registry WHERE document_id = ?;",
                    (clean_id,),
                )
                existing = select_cursor.fetchone()
                if existing is None:
                    raise DocumentNotFoundError(f"Document with ID '{clean_id}' not found in registry.")
                current_status = existing["status"]
                stored_token = existing["claim_token"]
                raise DocumentStateConflictError(
                    f"Cannot finalize deletion for document '{clean_id}': "
                    f"current status is '{current_status}', claim token matching={stored_token == clean_token} "
                    f"(expected status 'deleting' and matching claim token)."
                )

        return self.get_record(clean_id)  # type: ignore[return-value]

    def delete_record(
        self,
        document_id: str,
        expected_status: DocumentStatus = DocumentStatus.DELETING,
    ) -> bool:
        """
        Permanently remove a document registry record only if it is in the expected deletion status.

        Args:
            document_id: Target document identifier.
            expected_status: Required status guard (defaults to DELETING).

        Returns:
            True if deleted, False if no matching record found in expected status.
        """
        if not document_id or not document_id.strip():
            return False

        delete_sql = """
        DELETE FROM document_registry
        WHERE document_id = ? AND status = ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                delete_sql,
                (document_id.strip(), expected_status.value),
            )
            return cursor.rowcount > 0

    def claim_for_reconciliation(
        self,
        document_id: str,
        lease_seconds: float = DEFAULT_RECONCILIATION_LEASE_SECONDS,
    ) -> Optional[DocumentRecord]:
        """
        Atomically transition an eligible document record to 'reconciling' status with an unpredictable ownership token.

        A record is eligible to be claimed if and only if:
        1. document_id matches.
        2. provenance_eligible == 1 (verified document-to-vector provenance).
        3. status IN ('failed', 'deletion_failed') OR (status = 'reconciling' AND datetime(updated_at) < datetime(cutoff)).

        Concurrency & Atomicity:
        - Executed as a single atomic conditional SQL UPDATE statement.
        - Exactly one concurrent claimant obtains ownership of a particular claim generation.
        - Generates an unpredictable ownership token persisted with the claim.

        Args:
            document_id: Target document identifier.
            lease_seconds: Positive duration in seconds defining the reconciliation lease window (default: DEFAULT_RECONCILIATION_LEASE_SECONDS).

        Returns:
            The claimed DocumentRecord with status='reconciling' and fresh claim_token if claimed, or None.

        Raises:
            DocumentRegistryError: If document_id is invalid or lease_seconds is not positive or finite.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise DocumentRegistryError("document_id must be a non-empty string")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise DocumentRegistryError("lease_seconds must be a positive number greater than 0")
        if math.isnan(lease_seconds) or math.isinf(lease_seconds):
            raise DocumentRegistryError("lease_seconds must be a finite positive number")

        clean_id = document_id.strip()
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        cutoff_utc = now_utc - timedelta(seconds=float(lease_seconds))
        cutoff_iso = cutoff_utc.isoformat()
        claim_token = f"rec_{uuid.uuid4().hex}"

        claim_sql = """
        UPDATE document_registry
        SET status = ?,
            claim_token = ?,
            previous_status = CASE
                WHEN status = ? THEN previous_status
                ELSE status
            END,
            updated_at = ?
        WHERE document_id = ?
          AND provenance_eligible = 1
          AND (
            status IN (?, ?)
            OR (status = ? AND datetime(updated_at) < datetime(?))
          );
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                claim_sql,
                (
                    DocumentStatus.RECONCILING.value,
                    claim_token,
                    DocumentStatus.RECONCILING.value,
                    now_iso,
                    clean_id,
                    DocumentStatus.FAILED.value,
                    DocumentStatus.DELETION_FAILED.value,
                    DocumentStatus.RECONCILING.value,
                    cutoff_iso,
                ),
            )
            if cursor.rowcount == 0:
                return None

        return self.get_record(clean_id)

    def verify_reconciliation_claim(
        self,
        document_id: str,
        claim_token: str,
        lease_seconds: float = DEFAULT_RECONCILIATION_LEASE_SECONDS,
    ) -> bool:
        """
        Verify that the given claim_token is the active, unexpired owner of the reconciling document.

        Args:
            document_id: Target document identifier.
            claim_token: Claim ownership token to verify.
            lease_seconds: Positive duration in seconds defining the lease window.

        Returns:
            True if the record is currently in 'reconciling' status with matching claim_token
            and an active (unexpired) lease window. False otherwise.

        Raises:
            DocumentRegistryError: If arguments fail validation.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise DocumentRegistryError("document_id must be a non-empty string")
        if not claim_token or not isinstance(claim_token, str) or not claim_token.strip():
            raise DocumentRegistryError("claim_token must be a non-empty string")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise DocumentRegistryError("lease_seconds must be a positive number greater than 0")
        if math.isnan(lease_seconds) or math.isinf(lease_seconds):
            raise DocumentRegistryError("lease_seconds must be a finite positive number")

        clean_id = document_id.strip()
        clean_token = claim_token.strip()
        now_utc = datetime.now(timezone.utc)
        cutoff_utc = now_utc - timedelta(seconds=float(lease_seconds))
        cutoff_iso = cutoff_utc.isoformat()

        check_sql = """
        SELECT 1 FROM document_registry
        WHERE document_id = ?
          AND status = ?
          AND claim_token = ?
          AND datetime(updated_at) >= datetime(?);
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                check_sql,
                (
                    clean_id,
                    DocumentStatus.RECONCILING.value,
                    clean_token,
                    cutoff_iso,
                ),
            )
            return cursor.fetchone() is not None

    def finalize_reconciliation(
        self,
        document_id: str,
        claim_token: str,
        chunks_stored: int = 0,
    ) -> DocumentRecord:
        """
        Finalize a successful reconciliation attempt, verifying ownership atomically.

        Requirements:
        - Atomic conditional verification of document_id, status = 'reconciling', and claim_token.
        - Preserves failed vs deletion_failed outcomes based on the pre-claim status.
        - For failed ingestion, returns the record to 'failed', clears claim_token, updates chunks_stored and updated_at.
        - For deletion_failed, returns the record to 'deletion_failed' with updated chunks_stored (does not permanently delete row in this step).
        - Raises DocumentStateConflictError if claim_token does not match or status is not 'reconciling'.

        Args:
            document_id: Target document identifier.
            claim_token: Expected claim ownership token.
            chunks_stored: Updated integer count of vectors stored (defaults to 0).

        Returns:
            The finalized DocumentRecord.

        Raises:
            DocumentRegistryError: If arguments fail validation.
            DocumentNotFoundError: If the document does not exist.
            DocumentStateConflictError: If claim token does not match or document is not in 'reconciling' status.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise DocumentRegistryError("document_id must be a non-empty string")
        if not claim_token or not isinstance(claim_token, str) or not claim_token.strip():
            raise DocumentRegistryError("claim_token must be a non-empty string")
        if isinstance(chunks_stored, bool) or not isinstance(chunks_stored, int) or chunks_stored < 0:
            raise DocumentRegistryError("chunks_stored must be a non-negative integer")

        clean_id = document_id.strip()
        clean_token = claim_token.strip()
        now_iso = datetime.now(timezone.utc).isoformat()

        finalize_sql = """
        UPDATE document_registry
        SET status = COALESCE(previous_status, ?),
            chunks_stored = ?,
            claim_token = NULL,
            updated_at = ?
        WHERE document_id = ?
          AND status = ?
          AND claim_token = ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                finalize_sql,
                (
                    DocumentStatus.FAILED.value,
                    chunks_stored,
                    now_iso,
                    clean_id,
                    DocumentStatus.RECONCILING.value,
                    clean_token,
                ),
            )
            if cursor.rowcount == 0:
                existing = self.get_record(clean_id)
                if existing is None:
                    raise DocumentNotFoundError(f"Document with ID '{clean_id}' not found in registry.")
                raise DocumentStateConflictError(
                    f"Cannot finalize reconciliation for document '{clean_id}': "
                    f"claim token mismatch or state conflict (current status: '{existing.status.value}')."
                )

        return self.get_record(clean_id)  # type: ignore[return-value]

    def release_reconciliation_claim(
        self,
        document_id: str,
        claim_token: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> DocumentRecord:
        """
        Release a reconciliation claim after a recoverable failure, restoring retryable status.

        Requirements:
        - Atomic conditional verification of document_id, status = 'reconciling', and claim_token.
        - Preserves failed vs deletion_failed outcomes based on the pre-claim status.
        - Clears claim_token and optionally updates error_code and error_message.
        - Raises DocumentStateConflictError if claim_token does not match or status is not 'reconciling'.

        Args:
            document_id: Target document identifier.
            claim_token: Expected claim ownership token.
            error_code: Optional updated failure classification code.
            error_message: Optional updated sanitized failure message.

        Returns:
            The released DocumentRecord in retryable status.

        Raises:
            DocumentRegistryError: If arguments fail validation.
            DocumentNotFoundError: If the document does not exist.
            DocumentStateConflictError: If claim token does not match or document is not in 'reconciling' status.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            raise DocumentRegistryError("document_id must be a non-empty string")
        if not claim_token or not isinstance(claim_token, str) or not claim_token.strip():
            raise DocumentRegistryError("claim_token must be a non-empty string")

        clean_id = document_id.strip()
        clean_token = claim_token.strip()
        now_iso = datetime.now(timezone.utc).isoformat()

        clean_code = error_code.strip() if error_code and isinstance(error_code, str) and error_code.strip() else None
        clean_msg = error_message.strip() if error_message and isinstance(error_message, str) and error_message.strip() else None

        release_sql = """
        UPDATE document_registry
        SET status = COALESCE(previous_status, ?),
            error_code = COALESCE(?, error_code),
            error_message = COALESCE(?, error_message),
            claim_token = NULL,
            updated_at = ?
        WHERE document_id = ?
          AND status = ?
          AND claim_token = ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                release_sql,
                (
                    DocumentStatus.FAILED.value,
                    clean_code,
                    clean_msg,
                    now_iso,
                    clean_id,
                    DocumentStatus.RECONCILING.value,
                    clean_token,
                ),
            )
            if cursor.rowcount == 0:
                existing = self.get_record(clean_id)
                if existing is None:
                    raise DocumentNotFoundError(f"Document with ID '{clean_id}' not found in registry.")
                raise DocumentStateConflictError(
                    f"Cannot release reconciliation claim for document '{clean_id}': "
                    f"claim token mismatch or state conflict (current status: '{existing.status.value}')."
                )

        return self.get_record(clean_id)  # type: ignore[return-value]

