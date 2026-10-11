"""Unit tests for provider-independent LLM service layer and configuration."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.config import Settings, get_settings
from app.services.llm import (
    BaseLLMService,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMResponseError,
    LLMValidationError,
    MockLLMService,
    OpenAIChatService,
    get_llm_service,
    validate_generate_inputs,
)


class TestLLMInputValidation(unittest.TestCase):
    """Test suite for strict generation input validation."""

    def test_empty_or_whitespace_prompt_rejected(self):
        """1. Verify empty or whitespace-only prompt raises LLMValidationError."""
        for invalid_prompt in ["", "   ", "\n\t  "]:
            with self.assertRaises(LLMValidationError):
                validate_generate_inputs(prompt=invalid_prompt)

    def test_non_string_or_boolean_prompt_rejected(self):
        """2. Verify non-string or boolean prompt raises LLMValidationError."""
        for invalid_prompt in [None, 123, True, False, ["hello"], {"p": "v"}]:
            with self.assertRaises(LLMValidationError):
                validate_generate_inputs(prompt=invalid_prompt)  # type: ignore

    def test_valid_prompt_accepted(self):
        """3. Verify non-empty string prompt succeeds validation."""
        try:
            validate_generate_inputs(prompt="Hello, world!")
        except LLMValidationError:
            self.fail("validate_generate_inputs raised LLMValidationError unexpectedly on valid prompt")

    def test_system_prompt_none_accepted(self):
        """4. Verify system_prompt=None is accepted."""
        try:
            validate_generate_inputs(prompt="Hello", system_prompt=None)
        except LLMValidationError:
            self.fail("validate_generate_inputs raised LLMValidationError unexpectedly on system_prompt=None")

    def test_empty_or_whitespace_system_prompt_rejected(self):
        """5. Verify empty or whitespace-only system prompt raises LLMValidationError."""
        for invalid_sys in ["", "   ", "\t\n"]:
            with self.assertRaises(LLMValidationError):
                validate_generate_inputs(prompt="Hello", system_prompt=invalid_sys)

    def test_non_string_or_boolean_system_prompt_rejected(self):
        """6. Verify non-string or boolean system prompt raises LLMValidationError."""
        for invalid_sys in [123, True, False, ["sys"]]:
            with self.assertRaises(LLMValidationError):
                validate_generate_inputs(prompt="Hello", system_prompt=invalid_sys)  # type: ignore

    def test_valid_system_prompt_accepted(self):
        """7. Verify non-empty string system prompt succeeds validation."""
        try:
            validate_generate_inputs(prompt="Hello", system_prompt="You are a helpful assistant.")
        except LLMValidationError:
            self.fail("validate_generate_inputs raised LLMValidationError unexpectedly on valid system_prompt")

    def test_temperature_out_of_bounds_rejected(self):
        """8. Verify temperature < 0.0 or > 2.0 raises LLMValidationError."""
        for invalid_temp in [-0.1, -1.0, 2.01, 3.0]:
            with self.assertRaises(LLMValidationError):
                validate_generate_inputs(prompt="Hello", temperature=invalid_temp)

    def test_temperature_non_numeric_or_boolean_rejected(self):
        """9. Verify non-numeric or boolean temperature raises LLMValidationError."""
        for invalid_temp in [True, False, "0.5", [0.5], {"temp": 0.5}]:
            with self.assertRaises(LLMValidationError):
                validate_generate_inputs(prompt="Hello", temperature=invalid_temp)  # type: ignore

    def test_valid_temperature_accepted(self):
        """10. Verify valid numeric temperatures between 0.0 and 2.0 (and None for default) succeed."""
        for valid_temp in [None, 0.0, 0, 0.7, 1.0, 1, 2.0, 2]:
            try:
                validate_generate_inputs(prompt="Hello", temperature=valid_temp)
            except LLMValidationError:
                self.fail(f"validate_generate_inputs raised unexpectedly on valid temperature {valid_temp}")

    def test_max_tokens_non_positive_rejected(self):
        """11. Verify max_tokens <= 0 raises LLMValidationError."""
        for invalid_tokens in [0, -1, -500]:
            with self.assertRaises(LLMValidationError):
                validate_generate_inputs(prompt="Hello", max_tokens=invalid_tokens)

    def test_max_tokens_non_integer_or_boolean_rejected(self):
        """12. Verify float, boolean, or non-int max_tokens raises LLMValidationError."""
        for invalid_tokens in [True, False, 100.5, "100", [100]]:
            with self.assertRaises(LLMValidationError):
                validate_generate_inputs(prompt="Hello", max_tokens=invalid_tokens)  # type: ignore

    def test_valid_max_tokens_accepted(self):
        """13. Verify None and positive integer max_tokens succeed."""
        for valid_tokens in [None, 1, 100, 4096]:
            try:
                validate_generate_inputs(prompt="Hello", max_tokens=valid_tokens)
            except LLMValidationError:
                self.fail(f"validate_generate_inputs raised unexpectedly on valid max_tokens {valid_tokens}")


class TestMockLLMService(unittest.TestCase):
    """Test suite for MockLLMService behavior and deterministic outputs."""

    def setUp(self):
        self.service = MockLLMService(model_name="mock-test-model")

    def test_provider_name_and_model_name(self):
        """14. Verify provider_name is 'mock' and model_name matches configuration."""
        self.assertEqual(self.service.provider_name, "mock")
        self.assertEqual(self.service.model_name, "mock-test-model")

    def test_deterministic_output_without_system_prompt(self):
        """15. Verify identical prompts produce deterministic output."""
        out1 = self.service.generate("What is zero trust architecture?")
        out2 = self.service.generate("What is zero trust architecture?")
        self.assertEqual(out1, out2)
        self.assertIn("mock-test-model", out1)
        self.assertIn("What is zero trust architecture?", out1)

    def test_deterministic_output_with_system_prompt(self):
        """16. Verify system prompt is deterministically included in response."""
        out = self.service.generate(
            prompt="Analyze this invoice.",
            system_prompt="You are a financial analyst.",
            temperature=0.5,
            max_tokens=256,
        )
        self.assertIn("(System: You are a financial analyst.)", out)
        self.assertIn("Analyze this invoice.", out)
        self.assertEqual(self.service.last_prompt, "Analyze this invoice.")
        self.assertEqual(self.service.last_system_prompt, "You are a financial analyst.")
        self.assertEqual(self.service.last_temperature, 0.5)
        self.assertEqual(self.service.last_max_tokens, 256)
        self.assertEqual(self.service.call_count, 1)

    def test_mock_rejects_invalid_inputs(self):
        """17. Verify MockLLMService enforces input validation."""
        with self.assertRaises(LLMValidationError):
            self.service.generate("")
        with self.assertRaises(LLMValidationError):
            self.service.generate("Valid prompt", temperature=-0.5)

    def test_mock_uses_configured_defaults_when_omitted(self):
        """17a. Verify configured defaults are used when temperature and max_tokens are omitted."""
        service = MockLLMService(
            model_name="mock-model",
            temperature=0.8,
            max_tokens=500,
        )
        service.generate("Test prompt")
        self.assertEqual(service.last_temperature, 0.8)
        self.assertEqual(service.last_max_tokens, 500)

    def test_mock_preserves_explicit_zero_temperature(self):
        """17b. Verify explicit temperature 0.0 overrides a non-zero configured default."""
        service = MockLLMService(
            model_name="mock-model",
            temperature=0.8,
            max_tokens=500,
        )
        service.generate("Test prompt", temperature=0.0)
        self.assertEqual(service.last_temperature, 0.0)
        self.assertEqual(service.last_max_tokens, 500)

    def test_mock_explicit_values_override_defaults(self):
        """17c. Verify explicit per-request values override configured defaults."""
        service = MockLLMService(
            model_name="mock-model",
            temperature=0.8,
            max_tokens=500,
        )
        service.generate("Test prompt", temperature=1.5, max_tokens=120)
        self.assertEqual(service.last_temperature, 1.5)
        self.assertEqual(service.last_max_tokens, 120)


class TestOpenAIChatService(unittest.TestCase):
    """Test suite for OpenAIChatService with mocked SDK."""

    def test_lazy_initialization_no_client_at_construction(self):
        """18. Verify constructor does not create client or make network calls."""
        service = OpenAIChatService(api_key="sk-fake-key", model_name="gpt-4o-mini")
        self.assertIsNone(service._client)
        self.assertEqual(service.provider_name, "openai")
        self.assertEqual(service.model_name, "gpt-4o-mini")

    def test_missing_api_key_raises_configuration_error(self):
        """19. Verify generating with missing API key raises LLMConfigurationError."""
        service = OpenAIChatService(api_key=None, settings=Settings(_env_file=None))
        with self.assertRaises(LLMConfigurationError) as ctx:
            service.generate("Hello world")
        self.assertIn("API key is required", str(ctx.exception))

    def test_successful_response_normalization(self):
        """20. Verify response normalization from OpenAI ChatCompletion object."""
        mock_client = MagicMock()
        mock_choice = SimpleNamespace(
            message=SimpleNamespace(content="Enterprise AI response text")
        )
        mock_response = SimpleNamespace(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response

        service = OpenAIChatService(
            api_key="sk-test",
            model_name="gpt-4o-mini",
            client=mock_client,
        )

        result = service.generate(
            prompt="Explain MFA.",
            system_prompt="Security expert persona.",
            temperature=0.2,
            max_tokens=100,
        )

        self.assertEqual(result, "Enterprise AI response text")
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertEqual(
            kwargs["messages"],
            [
                {"role": "system", "content": "Security expert persona."},
                {"role": "user", "content": "Explain MFA."},
            ],
        )

    def test_configured_defaults_used_in_api_call_when_omitted(self):
        """20a. Verify configured temperature and max_tokens are used when omitted in generate()."""
        mock_client = MagicMock()
        mock_choice = SimpleNamespace(message=SimpleNamespace(content="Default response"))
        mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[mock_choice])

        service = OpenAIChatService(
            api_key="sk-test",
            temperature=0.75,
            max_tokens=256,
            client=mock_client,
        )

        service.generate("Hello world")
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 0.75)
        self.assertEqual(kwargs["max_tokens"], 256)

    def test_explicit_zero_temperature_preserved_in_api_call(self):
        """20b. Verify explicit temperature 0.0 overrides non-zero configured default in api call."""
        mock_client = MagicMock()
        mock_choice = SimpleNamespace(message=SimpleNamespace(content="Zero temp response"))
        mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[mock_choice])

        service = OpenAIChatService(
            api_key="sk-test",
            temperature=0.75,
            max_tokens=256,
            client=mock_client,
        )

        service.generate("Hello world", temperature=0.0)
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["max_tokens"], 256)

    def test_explicit_overrides_precedence_in_api_call(self):
        """20c. Verify explicit temperature and max_tokens take precedence over defaults in api call."""
        mock_client = MagicMock()
        mock_choice = SimpleNamespace(message=SimpleNamespace(content="Override response"))
        mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[mock_choice])

        service = OpenAIChatService(
            api_key="sk-test",
            temperature=0.75,
            max_tokens=256,
            client=mock_client,
        )

        service.generate("Hello world", temperature=1.2, max_tokens=64)
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["temperature"], 1.2)
        self.assertEqual(kwargs["max_tokens"], 64)

    def test_empty_choices_raises_response_error(self):
        """21. Verify response with empty choices raises LLMResponseError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[])

        service = OpenAIChatService(api_key="sk-test", client=mock_client)
        with self.assertRaises(LLMResponseError):
            service.generate("Hello")

    def test_missing_or_non_string_content_raises_response_error(self):
        """22. Verify choice message with None or non-string content raises LLMResponseError."""
        mock_client = MagicMock()
        mock_choice = SimpleNamespace(message=SimpleNamespace(content=None))
        mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[mock_choice])

        service = OpenAIChatService(api_key="sk-test", client=mock_client)
        with self.assertRaises(LLMResponseError):
            service.generate("Hello")

        # Non-string content
        mock_choice.message.content = {"raw": "dict"}
        with self.assertRaises(LLMResponseError):
            service.generate("Hello")

    def test_provider_exception_wrapped_in_llm_provider_error(self):
        """23. Verify upstream exceptions are wrapped in LLMProviderError with cause chaining."""
        mock_client = MagicMock()
        original_exc = RuntimeError("Connection timed out to OpenAI API")
        mock_client.chat.completions.create.side_effect = original_exc

        service = OpenAIChatService(api_key="sk-test", client=mock_client)
        with self.assertRaises(LLMProviderError) as ctx:
            service.generate("Hello")

        self.assertIn("OpenAI completion request failed", str(ctx.exception))
        self.assertIs(ctx.exception.__cause__, original_exc)


class TestLLMFactoryAndConfiguration(unittest.TestCase):
    """Test suite for factory function and LLM configuration."""

    def test_default_factory_returns_mock_service(self):
        """24. Verify default get_llm_service returns MockLLMService."""
        settings = Settings(_env_file=None)
        service = get_llm_service(settings=settings)
        self.assertIsInstance(service, MockLLMService)
        self.assertEqual(service.provider_name, "mock")

    def test_explicit_use_mock_flag(self):
        """25. Verify use_mock=True forces MockLLMService even if settings configured for openai."""
        settings = Settings(
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test",
            _env_file=None,
        )
        service = get_llm_service(settings=settings, use_mock=True)
        self.assertIsInstance(service, MockLLMService)
        self.assertEqual(service.provider_name, "mock")

    def test_factory_returns_openai_service_when_configured(self):
        """26. Verify get_llm_service returns OpenAIChatService when provider is 'openai'."""
        settings = Settings(
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test-key",
            LLM_MODEL="gpt-4o",
            LLM_TEMPERATURE=0.3,
            LLM_MAX_TOKENS=512,
            _env_file=None,
        )
        service = get_llm_service(settings=settings)
        self.assertIsInstance(service, OpenAIChatService)
        self.assertEqual(service.provider_name, "openai")
        self.assertEqual(service.model_name, "gpt-4o")

    def test_config_validation_invalid_provider_rejected(self):
        """27. Verify invalid LLM_PROVIDER raises ValueError."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

    def test_config_validation_empty_model_rejected(self):
        """28. Verify empty LLM_MODEL raises ValueError."""
        with patch.dict(os.environ, {"LLM_MODEL": "   "}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

    def test_config_validation_temperature_bounds(self):
        """29. Verify LLM_TEMPERATURE outside [0.0, 2.0] raises ValueError."""
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "2.5"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "-0.1"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

    def test_config_validation_max_tokens_positive(self):
        """30. Verify non-positive LLM_MAX_TOKENS raises ValueError."""
        with patch.dict(os.environ, {"LLM_MAX_TOKENS": "0"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)
        with patch.dict(os.environ, {"LLM_MAX_TOKENS": "-10"}, clear=True):
            with self.assertRaises(ValueError):
                Settings(_env_file=None)

    def test_existing_embedding_and_qdrant_settings_unaffected(self):
        """31. Verify existing embedding and vector store defaults remain intact."""
        settings = Settings(_env_file=None)
        self.assertEqual(settings.embedding_provider, "fastembed")
        self.assertEqual(settings.embedding_model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(settings.embedding_dimension, 384)
        self.assertEqual(settings.qdrant_collection_name, "enterprise_knowledge")
        self.assertEqual(settings.llm_provider, "mock")
        self.assertEqual(settings.llm_model, "gpt-4o-mini")
        self.assertEqual(settings.llm_temperature, 0.0)
        self.assertIsNone(settings.llm_max_tokens)


if __name__ == "__main__":
    unittest.main()
