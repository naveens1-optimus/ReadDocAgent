"""Azure Blob Storage adapter (connection-string / account-key auth)."""

from __future__ import annotations

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings
from langsmith import traceable

from application.interface.blob_storage_service import IBlobStorageService
from domain.schema.settings import BlobStorageSettings, RetryPolicySettings
from infrastructure.utilities.logging_config import get_logger
from infrastructure.utilities.retry import azure_retry

__all__ = ["AzureBlobStorageService"]

logger = get_logger(__name__)


class AzureBlobStorageService(IBlobStorageService):
    """Stores documents in Azure Blob Storage."""

    def __init__(
        self, settings: BlobStorageSettings, retry_policy: RetryPolicySettings
    ) -> None:
        self._client = BlobServiceClient.from_connection_string(
            settings.connection_string.get_secret_value()
        )
        self._upload_with_retry = azure_retry(retry_policy)(self._upload_once)
        # Containers already confirmed to exist, so we only pay for the check
        # once per container per process.
        self._known_containers: set[str] = set()

    # Plain SDK calls are invisible to LangSmith, unlike LangChain
    # components. @traceable adds a span; it no-ops when tracing is off.
    @traceable(name="blob_upload", run_type="tool")
    def upload(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> str:
        """Upload bytes and return the blob URL."""
        self._ensure_container(container)
        url = self._upload_with_retry(container, blob_name, data, content_type)
        logger.info(
            "Uploaded %s bytes to %s/%s", len(data), container, blob_name,
            extra={"container": container, "blob_name": blob_name},
        )
        return url

    def _upload_once(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: str | None,
    ) -> str:
        blob = self._client.get_blob_client(container=container, blob=blob_name)
        blob.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
            if content_type
            else None,
        )
        return blob.url

    def _ensure_container(self, container: str) -> None:
        """Create the container on first use."""
        if container in self._known_containers:
            return
        try:
            self._client.create_container(container)
            logger.info("Created blob container %r", container)
        except ResourceExistsError:
            pass  # already there, which is the normal case
        self._known_containers.add(container)
