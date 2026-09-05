"""Tests for the FastAPI application factory, middleware and error handling."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.app import REQUEST_ID_HEADER, create_app
from version import __version__


@pytest.fixture
def client(valid_env: dict[str, str]) -> Iterator[TestClient]:
    """A client whose app started with valid configuration."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def unconfigured_client(clean_env: None) -> Iterator[TestClient]:
    """A client whose app started with configuration missing.

    ``raise_server_exceptions=False`` is not needed: startup must *not* raise,
    which is precisely what this fixture asserts by existing at all.
    """
    with TestClient(create_app()) as test_client:
        yield test_client


class TestAppFactory:
    def test_creates_independent_instances(self, valid_env: dict[str, str]) -> None:
        assert create_app() is not create_app()

    def test_metadata(self, valid_env: dict[str, str]) -> None:
        app = create_app()
        assert app.version == __version__
        assert "Document Processing" in app.title

    def test_openapi_schema_is_generated(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "/health" in schema["paths"]
        assert "/health/ready" in schema["paths"]

    def test_docs_are_served(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200

    def test_root_redirects_to_docs(self, client: TestClient) -> None:
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert response.headers["location"] == "/docs"


class TestStartupResilience:
    """Missing configuration must not prevent the server from starting."""

    def test_starts_without_configuration(
        self, unconfigured_client: TestClient
    ) -> None:
        """The whole point: a misconfigured app is still diagnosable."""
        assert unconfigured_client.get("/health").status_code == 200

    def test_records_configuration_error_on_state(
        self, unconfigured_client: TestClient
    ) -> None:
        app = unconfigured_client.app
        assert app.state.settings is None
        assert app.state.settings_error is not None
        assert "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" in app.state.settings_error

    def test_records_settings_on_state_when_valid(self, client: TestClient) -> None:
        app = client.app
        assert app.state.settings is not None
        assert app.state.settings_error is None


class TestCorrelationId:
    def test_generates_request_id_when_absent(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.headers.get(REQUEST_ID_HEADER)

    def test_echoes_client_supplied_request_id(self, client: TestClient) -> None:
        """Lets a caller correlate its own logs with the server's."""
        response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})
        assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"

    def test_ids_differ_between_requests(self, client: TestClient) -> None:
        first = client.get("/health").headers[REQUEST_ID_HEADER]
        second = client.get("/health").headers[REQUEST_ID_HEADER]
        assert first != second

    def test_id_reaches_the_request_handler(self, valid_env: dict[str, str]) -> None:
        """Verifies ContextVar propagation into the endpoint's task."""
        from infrastructure.utilities.logging_config import get_correlation_id

        app = create_app()
        seen: dict[str, str] = {}

        @app.get("/_probe")
        async def _probe() -> dict[str, str]:
            seen["correlation_id"] = get_correlation_id()
            return {"ok": "true"}

        with TestClient(app) as probe_client:
            probe_client.get("/_probe", headers={REQUEST_ID_HEADER: "ctx-check"})

        assert seen["correlation_id"] == "ctx-check"


class TestErrorHandling:
    def test_unknown_route_returns_uniform_error_shape(
        self, client: TestClient
    ) -> None:
        response = client.get("/does-not-exist")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "http_404"
        assert "message" in body
        assert body["request_id"]

    def test_http_exception_uses_error_shape(
        self, valid_env: dict[str, str]
    ) -> None:
        app = create_app()

        @app.get("/_teapot")
        async def _teapot() -> None:
            raise HTTPException(status_code=418, detail="I am a teapot")

        with TestClient(app) as teapot_client:
            response = teapot_client.get("/_teapot")

        assert response.status_code == 418
        assert response.json()["error"] == "http_418"
        assert response.json()["message"] == "I am a teapot"

    def test_unhandled_exception_returns_500_without_leaking_details(
        self, valid_env: dict[str, str]
    ) -> None:
        """Internal details must never reach the client."""
        app = create_app()

        @app.get("/_boom")
        async def _boom() -> None:
            raise RuntimeError("secret internal detail: connection string xyz")

        with TestClient(app, raise_server_exceptions=False) as boom_client:
            response = boom_client.get("/_boom")

        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "internal_error"
        assert "secret internal detail" not in response.text
        assert body["request_id"]

    def test_validation_error_returns_422(self, valid_env: dict[str, str]) -> None:
        from pydantic import BaseModel

        app = create_app()

        class Payload(BaseModel):
            count: int

        @app.post("/_validate")
        async def _validate(payload: Payload) -> dict[str, int]:
            return {"count": payload.count}

        with TestClient(app) as validate_client:
            response = validate_client.post("/_validate", json={"count": "not-a-number"})

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "validation_error"
        assert body["details"]["errors"]
