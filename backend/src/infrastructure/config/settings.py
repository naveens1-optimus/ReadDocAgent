"""Environment-to-settings loader.

Bridges the environment (infrastructure concern) and the pure configuration
models in :mod:`domain.schema.settings` (domain concern). This is the only
module that knows which environment-variable name backs which setting.

Resolution happens exactly once per process and is cached, so importing
:func:`get_settings` from anywhere is cheap and always yields the same
instance:

    from infrastructure.config.settings import get_settings

    settings = get_settings()
    endpoint = settings.document_intelligence.endpoint

All required variables are validated up front, so a misconfigured deployment
fails at startup with a message naming the offending variable -- rather than
part-way through a document run, after Azure calls have already been billed.
"""

from __future__ import annotations

from functools import lru_cache

from domain.schema.settings import (
    AppSettings,
    AzureOpenAISettings,
    BlobStorageSettings,
    CosmosDbSettings,
    DocumentIntelligenceSettings,
    LangSmithSettings,
    RetryPolicySettings,
    Settings,
)
from infrastructure.utilities.load_env import (
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    load_env,
    load_required_env,
)
from infrastructure.utilities.logging_config import get_logger

__all__ = ["get_settings", "reset_settings_cache", "build_settings"]

logger = get_logger(__name__)


def _build_app_settings() -> AppSettings:
    """Assemble general application settings."""
    return AppSettings(
        environment=get_env("APP_ENV", "local"),
        log_level=(get_env("LOG_LEVEL", "INFO") or "INFO").upper(),
        log_format=(get_env("LOG_FORMAT", "text") or "text").lower(),
        confidence_threshold=get_env_float("CONFIDENCE_THRESHOLD", 0.80),
        arithmetic_tolerance=get_env_float("ARITHMETIC_TOLERANCE", 0.01),
        max_upload_size_mb=get_env_int("MAX_UPLOAD_SIZE_MB", 20),
    )


def _build_document_intelligence_settings() -> DocumentIntelligenceSettings:
    """Assemble Azure Document Intelligence settings (key-based auth)."""
    return DocumentIntelligenceSettings(
        endpoint=load_required_env("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
        api_key=load_required_env("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
        api_version=get_env("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2024-11-30"),
    )


def _build_azure_openai_settings() -> AzureOpenAISettings:
    """Assemble Azure OpenAI settings (key-based auth).

    ``AZURE_OPENAI_VISION_DEPLOYMENT`` falls back to the chat deployment,
    since a single vision-capable deployment can serve both roles and
    requiring the value twice is needless friction.
    """
    chat_deployment = load_required_env("AZURE_OPENAI_CHAT_DEPLOYMENT")
    return AzureOpenAISettings(
        endpoint=load_required_env("AZURE_OPENAI_ENDPOINT"),
        api_key=load_required_env("AZURE_OPENAI_KEY"),
        api_version=get_env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        chat_deployment=chat_deployment,
        vision_deployment=get_env("AZURE_OPENAI_VISION_DEPLOYMENT", chat_deployment),
    )


def _build_blob_storage_settings() -> BlobStorageSettings:
    """Assemble Azure Blob Storage settings (account-key connection string)."""
    return BlobStorageSettings(
        connection_string=load_required_env("AZURE_STORAGE_CONNECTION_STRING"),
        input_container=get_env("AZURE_STORAGE_INPUT_CONTAINER", "idp-input"),
        output_container=get_env("AZURE_STORAGE_OUTPUT_CONTAINER", "idp-output"),
    )


def _build_cosmos_db_settings() -> CosmosDbSettings:
    """Assemble Cosmos DB settings for the checkpointer (key-based auth)."""
    return CosmosDbSettings(
        endpoint=load_required_env("AZURE_COSMOS_ENDPOINT"),
        key=load_required_env("AZURE_COSMOS_KEY"),
        database_name=get_env("AZURE_COSMOS_DATABASE", "idp_langgraph"),
        container_name=get_env("AZURE_COSMOS_CONTAINER", "checkpoints"),
    )


def _build_langsmith_settings() -> LangSmithSettings:
    """Assemble LangSmith tracing settings.

    The API key is only required when tracing is enabled; that rule is
    enforced by ``LangSmithSettings`` itself.
    """
    return LangSmithSettings(
        tracing_enabled=get_env_bool("LANGSMITH_TRACING", False),
        api_key=get_env("LANGSMITH_API_KEY"),
        project=get_env("LANGSMITH_PROJECT", "doc-int-agent-system"),
        endpoint=get_env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    )


def _build_retry_policy_settings() -> RetryPolicySettings:
    """Assemble the shared Azure retry/backoff policy."""
    return RetryPolicySettings(
        max_attempts=get_env_int("AZURE_MAX_ATTEMPTS", 4),
        base_delay_seconds=get_env_float("AZURE_RETRY_BASE_DELAY_SECONDS", 1.0),
        max_delay_seconds=get_env_float("AZURE_RETRY_MAX_DELAY_SECONDS", 30.0),
    )


def build_settings(*, load_dotenv_file: bool = True) -> Settings:
    """Build a :class:`Settings` instance from the current environment.

    Uncached, so tests can construct settings against a patched environment
    without disturbing the process-wide cache used by :func:`get_settings`.

    Args:
        load_dotenv_file: Load ``backend/src/.env`` first. Tests that set
            variables directly should pass ``False`` so a developer's local
            ``.env`` cannot influence the result.

    Returns:
        A fully validated settings aggregate.

    Raises:
        MissingEnvironmentVariableError: A required variable is absent.
        pydantic.ValidationError: A value is present but invalid.
    """
    if load_dotenv_file:
        load_env()

    settings = Settings(
        app=_build_app_settings(),
        document_intelligence=_build_document_intelligence_settings(),
        azure_openai=_build_azure_openai_settings(),
        blob_storage=_build_blob_storage_settings(),
        cosmos_db=_build_cosmos_db_settings(),
        langsmith=_build_langsmith_settings(),
        retry=_build_retry_policy_settings(),
    )
    logger.debug("Settings resolved successfully")
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton, building it on first use.

    Cached via ``lru_cache`` so that configuration is resolved -- and
    validated -- exactly once per process.
    """
    return build_settings()


def reset_settings_cache() -> None:
    """Clear the settings cache.

    Intended for tests that need to re-resolve configuration after changing
    environment variables.
    """
    get_settings.cache_clear()
