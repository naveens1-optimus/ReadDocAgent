"""Builds the services, agents, graph and handlers, and registers them.

The one place that wires interfaces to implementations, so swapping an
implementation means changing a single line here.
"""

from __future__ import annotations

from application.handler.process_document_handler import ProcessDocumentHandler
from application.interface.blob_storage_service import IBlobStorageService
from application.interface.document_analysis_service import IDocumentAnalysisService
from application.interface.entity_store import IEntityStore
from application.interface.language_model_service import ILanguageModelService
from domain.schema.settings import Settings
from infrastructure.agents.data_extraction_agent import DataExtractionAgent
from infrastructure.agents.validation_agent import ValidationAgent
from infrastructure.agents.document_classification_agent import (
    DocumentClassificationAgent,
)
from infrastructure.di_container import DIContainer
from infrastructure.services.azure_blob_storage_service import AzureBlobStorageService
from infrastructure.services.azure_document_intelligence_service import (
    AzureDocumentIntelligenceService,
)
from infrastructure.services.azure_openai_service import AzureOpenAIService
from infrastructure.services.cosmos_entity_store import CosmosEntityStore
from infrastructure.utilities.langsmith_setup import configure_langsmith
from infrastructure.utilities.logging_config import get_logger
from infrastructure.workflows.document_processing_workflow import (
    DocumentProcessingWorkflow,
    build_checkpointer,
)

__all__ = ["register_services"]

logger = get_logger(__name__)


def register_services(container: DIContainer, settings: Settings) -> None:
    """Create everything the API needs and register it in the container."""
    # Tracing first, so the graph and Azure OpenAI clients built below are
    # instrumented from the moment they exist.
    configure_langsmith(settings.langsmith)

    storage = AzureBlobStorageService(settings.blob_storage, settings.retry)
    language_model = AzureOpenAIService(settings.azure_openai, settings.retry)
    analysis = AzureDocumentIntelligenceService(
        settings.document_intelligence, settings.retry
    )

    entities = CosmosEntityStore(settings.cosmos_db, settings.retry)

    container.register(IBlobStorageService, storage)
    container.register(IEntityStore, entities)
    container.register(ILanguageModelService, language_model)
    container.register(IDocumentAnalysisService, analysis)

    classifier = DocumentClassificationAgent(language_model, analysis)
    container.register(DocumentClassificationAgent, classifier)

    extractor = DataExtractionAgent(analysis, language_model)
    container.register(DataExtractionAgent, extractor)

    validator = ValidationAgent(language_model)
    container.register(ValidationAgent, validator)

    workflow = DocumentProcessingWorkflow(
        classifier=classifier,
        extractor=extractor,
        validator=validator,
        storage=storage,
        entities=entities,
        settings=settings,
        checkpointer=build_checkpointer(settings.cosmos_db),
    )
    container.register(DocumentProcessingWorkflow, workflow)

    container.register(
        ProcessDocumentHandler,
        ProcessDocumentHandler(workflow, storage, settings),
    )

    logger.info("Registered services, agents, graph and handlers")
