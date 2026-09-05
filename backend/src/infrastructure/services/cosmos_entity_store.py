"""Cosmos DB adapter for validated entities (key-based auth)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from langsmith import traceable

from application.interface.entity_store import IEntityStore
from domain.schema.settings import CosmosDbSettings, RetryPolicySettings
from infrastructure.utilities.logging_config import get_logger
from infrastructure.utilities.retry import azure_retry

__all__ = ["CosmosEntityStore"]

logger = get_logger(__name__)


class CosmosEntityStore(IEntityStore):
    """Stores validated entities in their own Cosmos container.

    Separate from the checkpointer's container: that one holds LangGraph's
    framework state, this one holds the business data the pipeline produced.
    Partitioned on ``document_id``, which is also the item id, so a lookup is
    a direct point read rather than a query.
    """

    def __init__(
        self, settings: CosmosDbSettings, retry_policy: RetryPolicySettings
    ) -> None:
        try:
            client = CosmosClient(
                settings.endpoint, settings.key.get_secret_value()
            )
            database = client.create_database_if_not_exists(settings.database_name)
            self._container = database.create_container_if_not_exists(
                id=settings.entity_container,
                partition_key=PartitionKey(path="/document_id"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not open the Cosmos entity container "
                f"{settings.database_name}/{settings.entity_container} at "
                f"{settings.endpoint}. Check AZURE_COSMOS_ENDPOINT and "
                f"AZURE_COSMOS_KEY. Cause: {exc}"
            ) from exc

        self._save_with_retry = azure_retry(retry_policy)(self._save_once)
        self._get_with_retry = azure_retry(retry_policy)(self._get_once)

    @traceable(name="entity_save", run_type="tool")
    def save(
        self, document_id: str, document_type: str, entity: dict[str, Any]
    ) -> str:
        """Upsert the entity for a document."""
        item = {
            "id": document_id,
            "document_id": document_id,
            "document_type": document_type,
            "entity": entity,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_with_retry(item)
        logger.info(
            "Saved %s entity for %s", document_type, document_id,
            extra={"document_id": document_id, "document_type": document_type},
        )
        return document_id

    def get(self, document_id: str) -> dict[str, Any] | None:
        """Read back a stored entity."""
        return self._get_with_retry(document_id)

    def _save_once(self, item: dict[str, Any]) -> None:
        self._container.upsert_item(item)

    def _get_once(self, document_id: str) -> dict[str, Any] | None:
        try:
            return self._container.read_item(
                item=document_id, partition_key=document_id
            )
        except CosmosResourceNotFoundError:
            return None
