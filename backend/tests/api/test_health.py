"""Tests for the health and readiness endpoints."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from version import __version__


@pytest.fixture
def client(valid_env: dict[str, str]) -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def unconfigured_client(clean_env: None) -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


class TestLiveness:
    def test_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__
        assert body["service"] == "doc-int-agent-system"

    def test_reports_environment_when_configured(self, client: TestClient) -> None:
        assert client.get("/health").json()["environment"] == "local"

    def test_stays_green_without_configuration(
        self, unconfigured_client: TestClient
    ) -> None:
        """Liveness must not depend on configuration -- that is its purpose."""
        response = unconfigured_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["environment"] is None

    def test_includes_timestamp(self, client: TestClient) -> None:
        assert client.get("/health").json()["timestamp"]


def _component(body: dict, name: str) -> dict:
    """Pull one named component out of a readiness body."""
    return next(item for item in body["components"] if item["name"] == name)


class TestReadiness:
    def test_ready_when_configured(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert _component(body, "configuration")["ready"] is True
        assert _component(body, "azure_services")["ready"] is True

    def test_returns_503_when_unconfigured(
        self, unconfigured_client: TestClient
    ) -> None:
        response = unconfigured_client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_names_the_missing_variable(
        self, unconfigured_client: TestClient
    ) -> None:
        """The probe must be actionable, not just red."""
        component = _component(
            unconfigured_client.get("/health/ready").json(), "configuration"
        )
        assert component["ready"] is False
        assert "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" in component["detail"]

    def test_services_not_built_when_unconfigured(
        self, unconfigured_client: TestClient
    ) -> None:
        component = _component(
            unconfigured_client.get("/health/ready").json(), "azure_services"
        )
        assert component["ready"] is False

    def test_probe_makes_no_azure_calls(self, client: TestClient) -> None:
        """A probe must stay fast, so it only performs local checks.

        The test credentials point at a host that does not resolve; if the
        probe called Azure this would hang on DNS and socket timeouts.
        """
        import time

        started = time.perf_counter()
        client.get("/health/ready")
        assert time.perf_counter() - started < 2.0

    def test_reports_version(self, client: TestClient) -> None:
        assert client.get("/health/ready").json()["version"] == __version__


class TestReadinessAfterPartialConfiguration:
    def test_missing_one_required_key_blocks_readiness(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any single missing credential must keep the service out of rotation."""
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)

        with TestClient(create_app()) as partial_client:
            response = partial_client.get("/health/ready")

        assert response.status_code == 503
        assert "AZURE_OPENAI_KEY" in response.json()["components"][0]["detail"]
