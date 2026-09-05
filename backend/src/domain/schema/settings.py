"""Configuration value objects -- one Pydantic model per Azure service.

These are *pure* models: they perform validation only and never touch
``os.environ``, files or the network. Populating them from the environment is
the job of :mod:`infrastructure.config.settings`, which keeps the domain layer
free of infrastructure concerns and makes every model trivially constructible
in a unit test.

Every credential is typed :class:`~pydantic.SecretStr`, so printing a settings
object -- as the startup banner and the demo scripts do -- renders keys as
``**********`` instead of leaking them into logs or a terminal transcript.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

__all__ = [
    "AppEnvironment",
    "LogFormat",
    "LogLevel",
    "DocumentIntelligenceSettings",
    "AzureOpenAISettings",
    "BlobStorageSettings",
    "CosmosDbSettings",
    "LangSmithSettings",
    "RetryPolicySettings",
    "AppSettings",
    "Settings",
]

AppEnvironment = Literal["local", "development", "staging", "production"]
LogFormat = Literal["text", "json"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

#: A ratio constrained to the inclusive 0.0-1.0 range, as Azure Document
#: Intelligence confidence scores are.
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class _StrictModel(BaseModel):
    """Base for settings models: immutable, and rejects unknown fields.

    ``extra="forbid"`` turns a typo in a settings key into an immediate,
    explicit error instead of a silently ignored value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


def _normalise_endpoint(value: str) -> str:
    """Validate an Azure endpoint URL and strip any trailing slash.

    The Azure SDKs join paths onto the endpoint, so a trailing slash produces
    double-slashed URLs that some services reject. Normalising once here means
    ``.env`` may contain either form.
    """
    candidate = value.strip()
    if not candidate:
        raise ValueError("endpoint must not be empty")
    if not candidate.startswith(("http://", "https://")):
        raise ValueError(f"endpoint must be an absolute http(s) URL, got {candidate!r}")
    return candidate.rstrip("/")


class DocumentIntelligenceSettings(_StrictModel):
    """Azure AI Document Intelligence connection settings (Agent 2).

    Authenticated with ``AzureKeyCredential``; see
    :mod:`infrastructure.services.azure_document_intelligence_service`.
    """

    endpoint: str = Field(
        description=(
            "Resource endpoint, e.g. https://<name>.cognitiveservices.azure.com"
        )
    )
    api_key: SecretStr = Field(description="Resource KEY 1 or KEY 2.")
    api_version: str = Field(
        default="2024-11-30",
        description="Service API version; 2024-11-30 is GA for SDK 1.0.x.",
    )

    _normalise = field_validator("endpoint")(_normalise_endpoint)


class AzureOpenAISettings(_StrictModel):
    """Azure OpenAI connection settings (Agents 1 and 3).

    Two deployment names are tracked because document classification needs a
    vision-capable model while enrichment and summarisation only need text.
    They may point at the same deployment.
    """

    endpoint: str = Field(
        description="Resource endpoint, e.g. https://<name>.openai.azure.com"
    )
    api_key: SecretStr = Field(description="Resource KEY 1 or KEY 2.")
    api_version: str = Field(default="2024-10-21")
    chat_deployment: str = Field(
        description="Deployment for text tasks: structuring, enrichment, summary."
    )
    vision_deployment: str = Field(
        description="Vision-capable deployment used to classify documents."
    )

    _normalise = field_validator("endpoint")(_normalise_endpoint)

    @field_validator("chat_deployment", "vision_deployment")
    @classmethod
    def _require_deployment_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("deployment name must not be empty")
        return value.strip()


class BlobStorageSettings(_StrictModel):
    """Azure Blob Storage settings (Agent 4).

    Uses a connection string, which embeds the account key -- hence
    ``SecretStr``.
    """

    connection_string: SecretStr = Field(
        description="Storage account connection string (contains the account key)."
    )
    input_container: str = Field(
        default="idp-input", description="Container for uploaded source documents."
    )
    output_container: str = Field(
        default="idp-output",
        description="Container for processed output and processing reports.",
    )

    @field_validator("input_container", "output_container")
    @classmethod
    def _validate_container_name(cls, value: str) -> str:
        """Enforce Azure container naming rules.

        Azure rejects invalid names only at call time, which would surface as
        a confusing failure inside Agent 4 after all the expensive extraction
        work is already done. Validating at startup fails fast instead.
        """
        name = value.strip().lower()
        if not 3 <= len(name) <= 63:
            raise ValueError(
                f"container name must be 3-63 characters, got {len(name)}: {name!r}"
            )
        if not name[0].isalnum() or not name[-1].isalnum():
            raise ValueError(f"container name must start and end alphanumeric: {name!r}")
        if "--" in name:
            raise ValueError(
                f"container name must not contain consecutive hyphens: {name!r}"
            )
        if not all(char.isalnum() or char == "-" for char in name):
            raise ValueError(
                f"container name may only contain lowercase letters, digits and "
                f"hyphens: {name!r}"
            )
        return name


class CosmosDbSettings(_StrictModel):
    """Azure Cosmos DB settings for the LangGraph checkpointer.

    The checkpointer is what makes the human-in-the-loop pause durable: a run
    pauses on one request and resumes on a later one, so the saved state has
    to outlive the process that created it.

    The database and container are created on first use, so only the account
    needs to exist beforehand.
    """

    endpoint: str = Field(
        description="Account endpoint, e.g. https://<name>.documents.azure.com:443/"
    )
    key: SecretStr = Field(description="Account primary or secondary key.")
    database_name: str = Field(default="idp_langgraph")
    container_name: str = Field(default="checkpoints")

    #: Validated business entities live in their own container, apart from
    #: LangGraph's checkpoint plumbing.
    entity_container: str = Field(default="entities")

    _normalise = field_validator("endpoint")(_normalise_endpoint)

    @field_validator("database_name", "container_name", "entity_container")
    @classmethod
    def _require_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Cosmos DB database and container names must not be empty")
        return value.strip()


class LangSmithSettings(_StrictModel):
    """LangSmith tracing settings.

    Tracing is opt-in. When disabled the pipeline runs normally and all
    tracing calls no-op, so the project is usable without a LangSmith account.
    """

    tracing_enabled: bool = Field(default=False)
    api_key: SecretStr | None = Field(default=None)
    project: str = Field(default="doc-int-agent-system")
    endpoint: str = Field(default="https://api.smith.langchain.com")

    _normalise = field_validator("endpoint")(_normalise_endpoint)

    @model_validator(mode="after")
    def _require_key_when_enabled(self) -> LangSmithSettings:
        """An API key is mandatory once tracing is switched on.

        Without this check, enabling tracing with no key silently produces no
        traces -- the worst outcome, because the pipeline looks instrumented
        but nothing is actually recorded.
        """
        if self.tracing_enabled and self.api_key is None:
            raise ValueError(
                "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set. "
                "Either supply the key or set LANGSMITH_TRACING=false."
            )
        return self


class RetryPolicySettings(_StrictModel):
    """Exponential-backoff policy shared by every Azure adapter."""

    max_attempts: int = Field(
        default=4, ge=1, le=10, description="Total attempts, including the first."
    )
    base_delay_seconds: float = Field(
        default=1.0,
        gt=0.0,
        description="Base delay; grows as base * 2**(attempt-1), plus jitter.",
    )
    max_delay_seconds: float = Field(
        default=30.0, gt=0.0, description="Ceiling on any single backoff delay."
    )

    @model_validator(mode="after")
    def _check_delay_ordering(self) -> RetryPolicySettings:
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                f"max_delay_seconds ({self.max_delay_seconds}) must be >= "
                f"base_delay_seconds ({self.base_delay_seconds})"
            )
        return self


class AppSettings(_StrictModel):
    """Application behaviour not tied to a specific Azure service."""

    environment: AppEnvironment = Field(default="local")
    log_level: LogLevel = Field(default="INFO")
    log_format: LogFormat = Field(default="text")

    confidence_threshold: Confidence = Field(
        default=0.80,
        description=(
            "Fields below this Azure confidence score are flagged by Agent 3 "
            "and trigger the human-in-the-loop checkpoint."
        ),
    )
    arithmetic_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        description="Absolute tolerance when cross-validating money totals.",
    )
    max_upload_size_mb: int = Field(default=20, gt=0, le=500)

    @property
    def max_upload_size_bytes(self) -> int:
        """Upload ceiling in bytes, for direct comparison against payloads."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Whether this is a production deployment."""
        return self.environment == "production"


class Settings(_StrictModel):
    """Aggregate root: the whole application configuration.

    Resolved once at startup by ``infrastructure.config.settings.get_settings``
    and handed to the DI container, so no component reads the environment for
    itself.
    """

    app: AppSettings
    document_intelligence: DocumentIntelligenceSettings
    azure_openai: AzureOpenAISettings
    blob_storage: BlobStorageSettings
    cosmos_db: CosmosDbSettings
    langsmith: LangSmithSettings
    retry: RetryPolicySettings

    def summary(self) -> str:
        """Return a credential-free, human-readable configuration summary.

        Safe to log at startup: it deliberately reports only endpoints,
        deployment names and thresholds -- never key material.
        """
        tracing_state = "on" if self.langsmith.tracing_enabled else "off"
        return "\n".join(
            (
                f"environment          : {self.app.environment}",
                f"log level / format   : {self.app.log_level} / {self.app.log_format}",
                f"doc intelligence     : {self.document_intelligence.endpoint} "
                f"(api {self.document_intelligence.api_version})",
                f"azure openai         : {self.azure_openai.endpoint} "
                f"(api {self.azure_openai.api_version})",
                f"  chat deployment    : {self.azure_openai.chat_deployment}",
                f"  vision deployment  : {self.azure_openai.vision_deployment}",
                f"blob containers      : {self.blob_storage.input_container} -> "
                f"{self.blob_storage.output_container}",
                f"langsmith tracing    : {tracing_state} "
                f"(project {self.langsmith.project})",
                f"confidence threshold : {self.app.confidence_threshold}",
                f"arithmetic tolerance : {self.app.arithmetic_tolerance}",
                f"retry                : {self.retry.max_attempts} attempts, base "
                f"{self.retry.base_delay_seconds}s, max "
                f"{self.retry.max_delay_seconds}s",
                f"checkpointer         : cosmos {self.cosmos_db.endpoint} "
                f"({self.cosmos_db.database_name}/{self.cosmos_db.container_name})",
                f"entity store         : {self.cosmos_db.database_name}/"
                f"{self.cosmos_db.entity_container}",
            )
        )
