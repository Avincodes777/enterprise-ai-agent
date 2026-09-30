import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.config import Settings, get_settings


class TestConfig(unittest.TestCase):
    """Unit tests for centralized configuration and validation."""

    def test_default_app_env(self):
        """1. Verify default APP_ENV is 'development'."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.app_env, "development")

    def test_default_embedding_provider(self):
        """2. Verify default EMBEDDING_PROVIDER is 'fastembed'."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.embedding_provider, "fastembed")

    def test_default_embedding_model(self):
        """3. Verify default EMBEDDING_MODEL is 'BAAI/bge-small-en-v1.5'."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.embedding_model, "BAAI/bge-small-en-v1.5")

    def test_default_embedding_dimension(self):
        """4. Verify default EMBEDDING_DIMENSION is 384."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.embedding_dimension, 384)

    def test_default_embedding_batch_size(self):
        """5. Verify default EMBEDDING_BATCH_SIZE is 64."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.embedding_batch_size, 64)

    def test_default_embedding_timeout(self):
        """6. Verify default EMBEDDING_TIMEOUT_SECONDS is 30.0."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.embedding_timeout_seconds, 30.0)

    def test_environment_variable_override(self):
        """7. Verify environment variables correctly override defaults."""
        custom_env = {
            "APP_ENV": "staging",
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIMENSION": "1536",
            "EMBEDDING_BATCH_SIZE": "128",
            "EMBEDDING_TIMEOUT_SECONDS": "45.5",
            "OPENAI_API_KEY": "sk-test-key-12345",
        }
        with patch.dict(os.environ, custom_env, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.app_env, "staging")
            self.assertEqual(settings.embedding_provider, "openai")
            self.assertEqual(settings.embedding_model, "text-embedding-3-small")
            self.assertEqual(settings.embedding_dimension, 1536)
            self.assertEqual(settings.embedding_batch_size, 128)
            self.assertEqual(settings.embedding_timeout_seconds, 45.5)
            self.assertEqual(settings.openai_api_key, "sk-test-key-12345")

    def test_invalid_provider_raises_validation_error(self):
        """8. Verify unsupported provider raises validation error."""
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "unsupported_provider"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

    def test_invalid_dimension_raises_validation_error(self):
        """9. Verify non-positive EMBEDDING_DIMENSION raises validation error."""
        with patch.dict(os.environ, {"EMBEDDING_DIMENSION": "0"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

        with patch.dict(os.environ, {"EMBEDDING_DIMENSION": "-100"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

    def test_invalid_batch_size_raises_validation_error(self):
        """10. Verify non-positive EMBEDDING_BATCH_SIZE raises validation error."""
        with patch.dict(os.environ, {"EMBEDDING_BATCH_SIZE": "0"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

        with patch.dict(os.environ, {"EMBEDDING_BATCH_SIZE": "-5"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

    def test_invalid_timeout_raises_validation_error(self):
        """11. Verify non-positive EMBEDDING_TIMEOUT_SECONDS raises validation error."""
        with patch.dict(os.environ, {"EMBEDDING_TIMEOUT_SECONDS": "0.0"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

        with patch.dict(os.environ, {"EMBEDDING_TIMEOUT_SECONDS": "-1.5"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

    def test_configuration_loads_without_openai_api_key(self):
        """12. Verify configuration successfully instantiates when OPENAI_API_KEY is absent."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertIsNone(settings.openai_api_key)
            self.assertEqual(settings.embedding_provider, "fastembed")

    def test_configuration_does_not_attempt_network_requests(self):
        """13. Verify loading settings performs zero socket/network operations."""
        import socket

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(socket, "socket", side_effect=RuntimeError("Network access forbidden")):
                settings = Settings(_env_file=None)
                self.assertIsNotNone(settings)
                self.assertEqual(settings.embedding_provider, "fastembed")


    def test_default_qdrant_settings(self):
        """14. Verify default Qdrant configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.qdrant_collection_name, "enterprise_knowledge")
            self.assertIsNone(settings.qdrant_api_key)
            self.assertEqual(settings.qdrant_timeout_seconds, 10.0)
            self.assertIsNone(settings.qdrant_url)

    def test_qdrant_environment_variable_override(self):
        """15. Verify Qdrant environment variables override defaults."""
        custom_env = {
            "QDRANT_URL": "http://qdrant.internal:6333",
            "QDRANT_COLLECTION_NAME": "custom_knowledge_base",
            "QDRANT_API_KEY": "secret-qdrant-key",
            "QDRANT_TIMEOUT_SECONDS": "25.5",
        }
        with patch.dict(os.environ, custom_env, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.qdrant_url, "http://qdrant.internal:6333")
            self.assertEqual(settings.qdrant_collection_name, "custom_knowledge_base")
            self.assertEqual(settings.qdrant_api_key, "secret-qdrant-key")
            self.assertEqual(settings.qdrant_timeout_seconds, 25.5)

    def test_invalid_qdrant_collection_name_raises_validation_error(self):
        """16. Verify empty or whitespace-only QDRANT_COLLECTION_NAME raises validation error."""
        with patch.dict(os.environ, {"QDRANT_COLLECTION_NAME": ""}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

        with patch.dict(os.environ, {"QDRANT_COLLECTION_NAME": "   \t\n  "}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

    def test_invalid_qdrant_timeout_raises_validation_error(self):
        """17. Verify non-positive QDRANT_TIMEOUT_SECONDS raises validation error."""
        with patch.dict(os.environ, {"QDRANT_TIMEOUT_SECONDS": "0.0"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

        with patch.dict(os.environ, {"QDRANT_TIMEOUT_SECONDS": "-10.0"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)


if __name__ == "__main__":
    unittest.main()

