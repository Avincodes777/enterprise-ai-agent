"""Provider-independent LLM service layer for text generation and reasoning."""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class LLMError(Exception):
    """Base exception for all LLM service operations."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when an LLM provider is misconfigured or missing credentials."""
    pass


class LLMProviderError(LLMError):
    """Raised when an upstream LLM API call fails at runtime."""
    pass


class LLMResponseError(LLMError):
    """Raised when an upstream LLM returns an invalid, empty, or unparseable response."""
    pass


class LLMValidationError(LLMError, ValueError):
    """Raised when prompt, system_prompt, temperature, or token limits fail input validation."""
    pass


# -----------------------------------------------------------------------------
# Validation Helper
# -----------------------------------------------------------------------------
def validate_generate_inputs(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> None:
    """
    Validate input arguments for text generation according to enterprise standards.

    Raises:
        LLMValidationError: If any input fails validation criteria.
    """
    # 1. Prompt validation
    if isinstance(prompt, bool) or not isinstance(prompt, str):
        raise LLMValidationError(
            f"Prompt must be a string, got {type(prompt).__name__}."
        )
    if not prompt.strip():
        raise LLMValidationError("Prompt cannot be empty or whitespace-only.")

    # 2. System prompt validation
    if system_prompt is not None:
        if isinstance(system_prompt, bool) or not isinstance(system_prompt, str):
            raise LLMValidationError(
                f"system_prompt must be None or a string, got {type(system_prompt).__name__}."
            )
        if not system_prompt.strip():
            raise LLMValidationError(
                "system_prompt cannot be empty or whitespace-only when provided."
            )

    # 3. Temperature validation
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise LLMValidationError(
                f"temperature must be a numeric value, got {type(temperature).__name__}."
            )
        if temperature < 0.0 or temperature > 2.0:
            raise LLMValidationError(
                f"temperature must be between 0.0 and 2.0, got {temperature}."
            )

    # 4. max_tokens validation
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise LLMValidationError(
                f"max_tokens must be a positive integer, got {max_tokens}."
            )


# -----------------------------------------------------------------------------
# Abstract Base Interface
# -----------------------------------------------------------------------------
class BaseLLMService(ABC):
    """Abstract interface defining the provider-independent LLM service contract."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name identifier."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate completion text from a prompt.

        Args:
            prompt: User prompt string (non-empty).
            system_prompt: Optional system prompt string.
            temperature: Optional generation temperature (0.0 to 2.0, defaults to configured setting).
            max_tokens: Optional maximum completion tokens (defaults to configured setting).

        Returns:
            Generated response as a plain string.

        Raises:
            LLMValidationError: If inputs fail validation.
            LLMConfigurationError: If credentials or provider setup are invalid.
            LLMProviderError: If external upstream provider fails.
            LLMResponseError: If response from provider is missing or invalid.
        """
        pass


# -----------------------------------------------------------------------------
# Mock LLM Service Implementation
# -----------------------------------------------------------------------------
class MockLLMService(BaseLLMService):
    """
    Deterministic mock LLM service making zero external or network calls.

    Provides deterministic, reproducible string outputs for test suites.
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        settings: Optional[Settings] = None,
    ):
        self._model_name = model_name
        self._settings = settings or get_settings()
        self._default_temperature = (
            temperature
            if temperature is not None
            else getattr(self._settings, "llm_temperature", 0.0)
        )
        self._default_max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(self._settings, "llm_max_tokens", None)
        )
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None
        self.last_temperature: Optional[float] = None
        self.last_max_tokens: Optional[int] = None
        self.call_count: int = 0
        self.history: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Validate inputs
        validate_generate_inputs(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        effective_temp = (
            float(temperature) if temperature is not None else float(self._default_temperature)
        )
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._default_max_tokens
        )

        clean_prompt = prompt.strip()
        clean_system = system_prompt.strip() if system_prompt is not None else None

        # Record call for inspection in tests
        self.last_prompt = clean_prompt
        self.last_system_prompt = clean_system
        self.last_temperature = effective_temp
        self.last_max_tokens = effective_max_tokens
        self.call_count += 1
        self.history.append({
            "prompt": clean_prompt,
            "system_prompt": clean_system,
            "temperature": effective_temp,
            "max_tokens": effective_max_tokens,
        })

        if clean_system:
            return (
                f"[Mock LLM Output | model={self._model_name}] "
                f"(System: {clean_system}) {clean_prompt}"
            )
        return f"[Mock LLM Output | model={self._model_name}] {clean_prompt}"


# -----------------------------------------------------------------------------
# OpenAI Chat Service Implementation
# -----------------------------------------------------------------------------
class OpenAIChatService(BaseLLMService):
    """
    Provider implementation backed by the official OpenAI Chat Completions SDK.

    Follows lazy client initialization: no network calls or client creation occur
    until the first generate() call is executed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        settings: Optional[Settings] = None,
        client: Optional[Any] = None,
    ):
        self._settings = settings or get_settings()
        self._api_key = api_key if api_key is not None else self._settings.openai_api_key
        self._model_name = (
            model_name
            if model_name is not None
            else getattr(self._settings, "llm_model", "gpt-4o-mini")
        )
        self._default_temperature = (
            temperature
            if temperature is not None
            else getattr(self._settings, "llm_temperature", 0.0)
        )
        self._default_max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(self._settings, "llm_max_tokens", None)
        )
        # Client is lazily created or injected for testing
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self) -> Any:
        """Lazily initialize and return the OpenAI client."""
        if self._client is not None:
            return self._client

        effective_key = self._api_key
        if not effective_key or not effective_key.strip():
            raise LLMConfigurationError(
                "OpenAI API key is required when using OpenAIChatService. "
                "Set the OPENAI_API_KEY environment variable or pass api_key in constructor."
            )

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=effective_key.strip())
            return self._client
        except Exception as e:
            raise LLMConfigurationError(f"Failed to instantiate OpenAI client: {e}") from e

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Validate inputs
        validate_generate_inputs(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        effective_temp = (
            float(temperature) if temperature is not None else float(self._default_temperature)
        )
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._default_max_tokens
        )

        client = self._get_client()

        # Build messages payload
        messages: List[Dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        # Build request parameters
        params: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": effective_temp,
        }
        if effective_max_tokens is not None:
            params["max_tokens"] = effective_max_tokens

        try:
            response = client.chat.completions.create(**params)
        except Exception as e:
            raise LLMProviderError(f"OpenAI completion request failed: {e}") from e

        # Normalize and validate response text
        if not response or not getattr(response, "choices", None):
            raise LLMResponseError(
                "OpenAI returned an empty or malformed response with no choices."
            )

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise LLMResponseError("OpenAI response choice missing message object.")

        content = getattr(message, "content", None)
        if content is None or not isinstance(content, str):
            raise LLMResponseError(
                "OpenAI response message contained missing or non-string content."
            )

        return content


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------
def get_llm_service(
    settings: Optional[Settings] = None,
    use_mock: bool = False,
) -> BaseLLMService:
    """
    Factory function retrieving the configured LLM service instance.

    Args:
        settings: Optional Settings instance (defaults to get_settings()).
        use_mock: If True, always returns MockLLMService regardless of settings.

    Returns:
        An instance of BaseLLMService.

    Raises:
        LLMConfigurationError: If provider configuration is invalid or unsupported.
    """
    current_settings = settings or get_settings()

    if use_mock:
        return MockLLMService(
            model_name="mock-model",
            temperature=current_settings.llm_temperature,
            max_tokens=current_settings.llm_max_tokens,
            settings=current_settings,
        )

    provider = current_settings.llm_provider.lower()
    if provider == "mock":
        return MockLLMService(
            model_name=current_settings.llm_model,
            temperature=current_settings.llm_temperature,
            max_tokens=current_settings.llm_max_tokens,
            settings=current_settings,
        )
    elif provider == "openai":
        return OpenAIChatService(
            api_key=current_settings.openai_api_key,
            model_name=current_settings.llm_model,
            temperature=current_settings.llm_temperature,
            max_tokens=current_settings.llm_max_tokens,
            settings=current_settings,
        )
    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: '{provider}'. Supported providers are: 'mock', 'openai'."
        )
