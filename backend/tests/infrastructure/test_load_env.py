"""Tests for :mod:`infrastructure.utilities.load_env`."""

from __future__ import annotations

from pathlib import Path

import pytest

from infrastructure.utilities.load_env import (
    MissingEnvironmentVariableError,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    load_env,
    load_required_env,
)


class TestGetEnv:
    """Optional variable reads."""

    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_VAR", "hello")
        assert get_env("SOME_VAR") == "hello"

    def test_returns_default_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOME_VAR", raising=False)
        assert get_env("SOME_VAR", "fallback") == "fallback"

    def test_returns_none_when_absent_and_no_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SOME_VAR", raising=False)
        assert get_env("SOME_VAR") is None

    def test_strips_surrounding_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOME_VAR", "  padded  ")
        assert get_env("SOME_VAR") == "padded"

    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    def test_blank_value_treated_as_absent(
        self, monkeypatch: pytest.MonkeyPatch, blank: str
    ) -> None:
        """A leftover ``KEY=`` line must not supply an empty credential.

        This is the case that matters most: an empty key would otherwise reach
        Azure and fail with an opaque authentication error.
        """
        monkeypatch.setenv("SOME_VAR", blank)
        assert get_env("SOME_VAR", "fallback") == "fallback"


class TestLoadRequiredEnv:
    """Required variable reads."""

    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REQUIRED_VAR", "present")
        assert load_required_env("REQUIRED_VAR") == "present"

    def test_raises_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REQUIRED_VAR", raising=False)
        with pytest.raises(MissingEnvironmentVariableError) as exc_info:
            load_required_env("REQUIRED_VAR")
        assert exc_info.value.variable_name == "REQUIRED_VAR"

    def test_raises_when_blank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REQUIRED_VAR", "   ")
        with pytest.raises(MissingEnvironmentVariableError):
            load_required_env("REQUIRED_VAR")

    def test_error_message_is_actionable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The message must name the variable and point at the template."""
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        with pytest.raises(MissingEnvironmentVariableError) as exc_info:
            load_required_env("AZURE_OPENAI_KEY")
        message = str(exc_info.value)
        assert "AZURE_OPENAI_KEY" in message
        assert ".env.example" in message


class TestGetEnvBool:
    """Boolean coercion."""

    @pytest.mark.parametrize(
        "raw", ["1", "true", "TRUE", "True", "t", "yes", "Y", "on", "ON"]
    )
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv("FLAG", raw)
        assert get_env_bool("FLAG") is True

    @pytest.mark.parametrize(
        "raw", ["0", "false", "FALSE", "f", "no", "N", "off", "OFF"]
    )
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv("FLAG", raw)
        assert get_env_bool("FLAG", default=True) is False

    def test_default_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FLAG", raising=False)
        assert get_env_bool("FLAG", default=True) is True

    def test_unrecognised_value_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo in a non-critical toggle should warn, not crash."""
        monkeypatch.setenv("FLAG", "maybe")
        assert get_env_bool("FLAG", default=True) is True
        assert get_env_bool("FLAG", default=False) is False


class TestGetEnvNumeric:
    """Integer and float coercion."""

    def test_int_parses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NUM", "42")
        assert get_env_int("NUM", 0) == 42

    def test_int_falls_back_on_garbage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NUM", "not-a-number")
        assert get_env_int("NUM", 7) == 7

    def test_float_parses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RATIO", "0.85")
        assert get_env_float("RATIO", 0.0) == pytest.approx(0.85)

    def test_float_falls_back_on_garbage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RATIO", "high")
        assert get_env_float("RATIO", 0.8) == pytest.approx(0.8)


class TestLoadEnv:
    """Dotenv file loading."""

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """A missing dotenv file is not an error -- production uses real env vars."""
        assert load_env(tmp_path / "does-not-exist.env") is None

    def test_loads_values_from_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FROM_FILE=file-value\n", encoding="utf-8")
        monkeypatch.delenv("FROM_FILE", raising=False)

        assert load_env(env_file) == env_file
        assert get_env("FROM_FILE") == "file-value"

    def test_does_not_override_real_env_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deployment environment variables must win over a stray local file.

        This protects against a developer's leftover ``.env`` silently
        overriding container or CI configuration.
        """
        env_file = tmp_path / ".env"
        env_file.write_text("CONFLICT=from-file\n", encoding="utf-8")
        monkeypatch.setenv("CONFLICT", "from-environment")

        load_env(env_file)
        assert get_env("CONFLICT") == "from-environment"

    def test_override_flag_lets_file_win(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("CONFLICT=from-file\n", encoding="utf-8")
        monkeypatch.setenv("CONFLICT", "from-environment")

        load_env(env_file, override=True)
        assert get_env("CONFLICT") == "from-file"
