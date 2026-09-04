"""Tests for :mod:`infrastructure.utilities.logging_config`."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator

import pytest

from infrastructure.utilities import logging_config
from infrastructure.utilities.logging_config import (
    configure_logging,
    correlation_id_scope,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Snapshot and restore root logging state around each test.

    ``configure_logging`` mutates the root logger, which would otherwise leak
    into unrelated tests and into pytest's own capture handlers.
    """
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    original_configured = logging_config._configured

    yield

    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in original_handlers:
        root.addHandler(handler)
    root.setLevel(original_level)
    logging_config._configured = original_configured
    set_correlation_id(None)


class TestCorrelationId:
    """Correlation IDs tie every log line for one document together."""

    def test_defaults_to_placeholder(self) -> None:
        set_correlation_id(None)
        assert get_correlation_id() == "-"

    def test_set_and_read(self) -> None:
        set_correlation_id("doc-123")
        assert get_correlation_id() == "doc-123"

    def test_scope_restores_previous_value(self) -> None:
        set_correlation_id("outer")
        with correlation_id_scope("inner"):
            assert get_correlation_id() == "inner"
        assert get_correlation_id() == "outer"

    def test_scope_restores_on_exception(self) -> None:
        """A failing agent must not leave a stale correlation ID behind."""
        set_correlation_id("outer")
        with pytest.raises(RuntimeError):
            with correlation_id_scope("inner"):
                raise RuntimeError("agent blew up")
        assert get_correlation_id() == "outer"

    def test_nested_scopes(self) -> None:
        with correlation_id_scope("a"):
            with correlation_id_scope("b"):
                assert get_correlation_id() == "b"
            assert get_correlation_id() == "a"

    def test_isolated_across_concurrent_tasks(self) -> None:
        """ContextVar isolation matters once FastAPI handles requests concurrently."""
        observed: dict[str, str] = {}

        async def worker(name: str) -> None:
            with correlation_id_scope(name):
                await asyncio.sleep(0)  # force a task switch
                observed[name] = get_correlation_id()

        async def main() -> None:
            await asyncio.gather(worker("task-a"), worker("task-b"))

        asyncio.run(main())
        assert observed == {"task-a": "task-a", "task-b": "task-b"}


class TestConfigureLogging:
    def test_installs_single_handler(self) -> None:
        configure_logging(force=True)
        assert len(logging.getLogger().handlers) == 1

    def test_is_idempotent(self) -> None:
        """Both the API lifespan and the demo scripts call this; no double-logging."""
        configure_logging(force=True)
        configure_logging()
        configure_logging()
        assert len(logging.getLogger().handlers) == 1

    def test_force_reconfigures_without_stacking(self) -> None:
        configure_logging(force=True)
        configure_logging(log_format="json", force=True)
        assert len(logging.getLogger().handlers) == 1

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            ("DEBUG", logging.DEBUG),
            ("info", logging.INFO),
            ("WARNING", logging.WARNING),
            (logging.ERROR, logging.ERROR),
        ],
    )
    def test_sets_level(self, level: str | int, expected: int) -> None:
        configure_logging(level=level, force=True)
        assert logging.getLogger().level == expected

    def test_unknown_level_falls_back_to_info(self) -> None:
        configure_logging(level="NOT_A_LEVEL", force=True)
        assert logging.getLogger().level == logging.INFO

    def test_quietens_azure_http_logger(self) -> None:
        """Azure's HTTP policy logger can echo credential-bearing headers."""
        configure_logging(level="DEBUG", force=True)
        noisy = logging.getLogger(
            "azure.core.pipeline.policies.http_logging_policy"
        )
        assert noisy.level == logging.WARNING


class TestTextFormat:
    def test_includes_correlation_id(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(log_format="text", force=True)
        with correlation_id_scope("doc-abc"):
            get_logger("test.text").warning("classifying")

        out = capsys.readouterr().out
        assert "doc-abc" in out
        assert "classifying" in out
        assert "WARNING" in out


class TestJsonFormat:
    @staticmethod
    def _last_json_line(raw: str) -> dict[str, object]:
        lines = [line for line in raw.strip().splitlines() if line.startswith("{")]
        assert lines, f"no JSON lines in output: {raw!r}"
        return json.loads(lines[-1])

    def test_emits_valid_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(log_format="json", force=True)
        with correlation_id_scope("doc-json"):
            get_logger("test.json").warning("extracted")

        record = self._last_json_line(capsys.readouterr().out)
        assert record["message"] == "extracted"
        assert record["level"] == "WARNING"
        assert record["logger"] == "test.json"
        assert record["correlation_id"] == "doc-json"
        assert "timestamp" in record

    def test_promotes_extra_fields(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Structured context must stay queryable in a log aggregator."""
        configure_logging(log_format="json", force=True)
        get_logger("test.extra").warning(
            "field extracted",
            extra={"document_type": "invoice", "confidence": 0.93},
        )

        record = self._last_json_line(capsys.readouterr().out)
        assert record["document_type"] == "invoice"
        assert record["confidence"] == pytest.approx(0.93)

    def test_includes_exception_details(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(log_format="json", force=True)
        try:
            raise ValueError("extraction failed")
        except ValueError:
            get_logger("test.exc").exception("agent error")

        record = self._last_json_line(capsys.readouterr().out)
        assert "ValueError" in str(record["exception"])
        assert "extraction failed" in str(record["exception"])
