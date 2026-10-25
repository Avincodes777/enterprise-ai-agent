"""Database and persistence package for Enterprise AI Agent."""

from app.database.models import DocumentRecord, DocumentStatus
from app.database.registry import (
    DEFAULT_DELETION_LEASE_SECONDS,
    DEFAULT_INGESTION_LEASE_SECONDS,
    DEFAULT_RECONCILIATION_LEASE_SECONDS,
    DEFAULT_RECONCILIATION_LIMIT,
    DEFAULT_RECONCILIATION_MIN_AGE_SECONDS,
    DocumentNotFoundError,
    DocumentRegistryError,
    DocumentRegistryRepository,
    DocumentStateConflictError,
)

__all__ = [
    "DocumentRecord",
    "DocumentStatus",
    "DocumentRegistryError",
    "DocumentNotFoundError",
    "DocumentStateConflictError",
    "DocumentRegistryRepository",
    "DEFAULT_DELETION_LEASE_SECONDS",
    "DEFAULT_INGESTION_LEASE_SECONDS",
    "DEFAULT_RECONCILIATION_LEASE_SECONDS",
    "DEFAULT_RECONCILIATION_MIN_AGE_SECONDS",
    "DEFAULT_RECONCILIATION_LIMIT",
]
