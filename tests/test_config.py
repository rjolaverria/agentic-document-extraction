"""Tests for configuration management module."""

import logging
import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from agentic_document_extraction.config import Settings, validate_settings_on_startup


class TestSettings:
    """Tests for the Settings class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        # Create settings with no env vars to test defaults
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        # File upload defaults
        assert settings.max_file_size_mb == 10
        assert settings.temp_upload_dir == "/tmp/ade_uploads"

        # OpenAI defaults
        assert settings.get_openai_api_key() == ""
        assert settings.openai_model == "gpt-4o"
        assert settings.openai_vlm_model == "gpt-4o"
        assert settings.openai_temperature == 0.0
        assert settings.openai_max_tokens == 4096

        # Chunking defaults
        assert settings.chunk_size == 4000
        assert settings.chunk_overlap == 200

        # Quality threshold defaults
        assert settings.min_overall_confidence == 0.7
        assert settings.min_field_confidence == 0.5
        assert settings.required_field_coverage == 0.9

        # Agentic loop defaults
        assert settings.max_refinement_iterations == 3

        # Job management defaults
        assert settings.job_ttl_hours == 24
        assert settings.docket_name == "agentic-document-extraction"
        assert settings.docket_url == "redis://localhost:6379/0"
        assert settings.docket_result_storage_url is None
        assert settings.docket_execution_ttl_seconds is None
        assert settings.docket_execution_ttl.total_seconds() == settings.job_ttl_seconds

        # Logging defaults
        assert settings.log_level == "INFO"
        assert settings.debug is False

        # Server defaults
        assert settings.cors_origins == "*"
        assert settings.server_host == "0.0.0.0"
        assert settings.server_port == 8000

    def test_environment_variable_prefix(self) -> None:
        """Test that environment variables use ADE_ prefix."""
        env_vars = {
            "ADE_MAX_FILE_SIZE_MB": "25",
            "ADE_OPENAI_MODEL": "gpt-4-turbo",
            "ADE_LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.max_file_size_mb == 25
        assert settings.openai_model == "gpt-4-turbo"
        assert settings.log_level == "DEBUG"

    def test_secret_str_for_api_key(self) -> None:
        """Test that API key uses SecretStr for security."""
        env_vars = {"ADE_OPENAI_API_KEY": "sk-test-secret-key"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        # Direct access returns SecretStr
        assert isinstance(settings.openai_api_key, SecretStr)

        # get_openai_api_key() returns the actual value
        assert settings.get_openai_api_key() == "sk-test-secret-key"

        # String representation masks the key
        assert "sk-test-secret-key" not in str(settings.openai_api_key)
        assert "**" in str(settings.openai_api_key)

    def test_max_file_size_bytes_property(self) -> None:
        """Test max_file_size_bytes computed property."""
        env_vars = {"ADE_MAX_FILE_SIZE_MB": "10"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.max_file_size_bytes == 10 * 1024 * 1024

    def test_job_ttl_seconds_property(self) -> None:
        """Test job_ttl_seconds computed property."""
        env_vars = {"ADE_JOB_TTL_HOURS": "12"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.job_ttl_seconds == 12 * 3600

    def test_docket_execution_ttl_override(self) -> None:
        """Test Docket execution TTL override."""
        env_vars = {"ADE_DOCKET_EXECUTION_TTL_SECONDS": "900"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.docket_execution_ttl_seconds == 900
        assert settings.docket_execution_ttl.total_seconds() == 900

    def test_cors_origins_list_single(self) -> None:
        """Test CORS origins list with single origin."""
        env_vars = {"ADE_CORS_ORIGINS": "https://example.com"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.cors_origins_list == ["https://example.com"]

    def test_cors_origins_list_multiple(self) -> None:
        """Test CORS origins list with multiple origins."""
        env_vars = {"ADE_CORS_ORIGINS": "https://example.com, https://api.example.com"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.cors_origins_list == [
            "https://example.com",
            "https://api.example.com",
        ]

    def test_cors_origins_list_wildcard(self) -> None:
        """Test CORS origins list with wildcard."""
        env_vars = {"ADE_CORS_ORIGINS": "*"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.cors_origins_list == ["*"]

    def test_log_level_int_property(self) -> None:
        """Test log_level_int computed property."""
        test_cases = [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]

        for level_str, expected_int in test_cases:
            env_vars = {"ADE_LOG_LEVEL": level_str}
            with patch.dict(os.environ, env_vars, clear=True):
                settings = Settings(_env_file=None)
            assert settings.log_level_int == expected_int, f"Failed for {level_str}"

    def test_to_safe_dict_masks_api_key(self) -> None:
        """Test to_safe_dict masks sensitive values."""
        env_vars = {"ADE_OPENAI_API_KEY": "sk-actual-secret-key"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        safe_dict = settings.to_safe_dict()

        # API key should be masked
        assert safe_dict["openai_api_key"] == "***"
        assert "sk-actual-secret-key" not in str(safe_dict)

        # Other values should be present
        assert safe_dict["openai_model"] == "gpt-4o"
        assert safe_dict["max_file_size_mb"] == 10

    def test_to_safe_dict_shows_not_set_for_empty_key(self) -> None:
        """Test to_safe_dict shows (not set) for empty API key."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        safe_dict = settings.to_safe_dict()
        assert safe_dict["openai_api_key"] == "(not set)"


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_valid_log_level(self) -> None:
        """Test valid log levels are accepted."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            env_vars = {"ADE_LOG_LEVEL": level}
            with patch.dict(os.environ, env_vars, clear=True):
                settings = Settings(_env_file=None)
                assert settings.log_level == level

    def test_lowercase_log_level_normalized(self) -> None:
        """Test lowercase log levels are normalized to uppercase."""
        env_vars = {"ADE_LOG_LEVEL": "debug"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)
            assert settings.log_level == "DEBUG"

    def test_invalid_log_level_raises_error(self) -> None:
        """Test invalid log level raises validation error."""
        env_vars = {"ADE_LOG_LEVEL": "INVALID"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="Invalid log level"),
        ):
            Settings(_env_file=None)

    def test_threshold_must_be_between_0_and_1(self) -> None:
        """Test threshold values must be between 0 and 1."""
        # Too high
        env_vars = {"ADE_MIN_OVERALL_CONFIDENCE": "1.5"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="between 0.0 and 1.0"),
        ):
            Settings(_env_file=None)

        # Negative
        env_vars = {"ADE_MIN_FIELD_CONFIDENCE": "-0.1"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="between 0.0 and 1.0"),
        ):
            Settings(_env_file=None)

    def test_valid_threshold_values(self) -> None:
        """Test valid threshold values are accepted."""
        env_vars = {
            "ADE_MIN_OVERALL_CONFIDENCE": "0.8",
            "ADE_MIN_FIELD_CONFIDENCE": "0.6",
            "ADE_REQUIRED_FIELD_COVERAGE": "0.95",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.min_overall_confidence == 0.8
        assert settings.min_field_confidence == 0.6
        assert settings.required_field_coverage == 0.95

    def test_max_iterations_must_be_between_1_and_10(self) -> None:
        """Test max iterations must be between 1 and 10."""
        # Too high
        env_vars = {"ADE_MAX_REFINEMENT_ITERATIONS": "15"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="between 1 and 10"),
        ):
            Settings(_env_file=None)

        # Zero
        env_vars = {"ADE_MAX_REFINEMENT_ITERATIONS": "0"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="between 1 and 10"),
        ):
            Settings(_env_file=None)

    def test_valid_max_iterations(self) -> None:
        """Test valid max iterations values."""
        for iterations in [1, 5, 10]:
            env_vars = {"ADE_MAX_REFINEMENT_ITERATIONS": str(iterations)}
            with patch.dict(os.environ, env_vars, clear=True):
                settings = Settings(_env_file=None)
                assert settings.max_refinement_iterations == iterations

    def test_file_size_must_be_between_1_and_500(self) -> None:
        """Test file size limit validation."""
        # Too high
        env_vars = {"ADE_MAX_FILE_SIZE_MB": "1000"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="between 1 and 500"),
        ):
            Settings(_env_file=None)

        # Zero
        env_vars = {"ADE_MAX_FILE_SIZE_MB": "0"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="between 1 and 500"),
        ):
            Settings(_env_file=None)

    def test_valid_file_size(self) -> None:
        """Test valid file size values."""
        env_vars = {"ADE_MAX_FILE_SIZE_MB": "100"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)
            assert settings.max_file_size_mb == 100

    def test_port_must_be_valid(self) -> None:
        """Test port validation."""
        # Too high
        env_vars = {"ADE_SERVER_PORT": "70000"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="between 1 and 65535"),
        ):
            Settings(_env_file=None)

        # Zero
        env_vars = {"ADE_SERVER_PORT": "0"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="between 1 and 65535"),
        ):
            Settings(_env_file=None)

    def test_valid_port(self) -> None:
        """Test valid port values."""
        env_vars = {"ADE_SERVER_PORT": "8080"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)
            assert settings.server_port == 8080

    def test_job_ttl_must_be_positive(self) -> None:
        """Test job TTL must be positive."""
        env_vars = {"ADE_JOB_TTL_HOURS": "0"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="at least 1"),
        ):
            Settings(_env_file=None)

    def test_docket_execution_ttl_must_be_positive(self) -> None:
        """Test Docket execution TTL must be positive."""
        env_vars = {"ADE_DOCKET_EXECUTION_TTL_SECONDS": "0"}
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="at least 1"),
        ):
            Settings(_env_file=None)

    def test_chunk_overlap_must_be_less_than_chunk_size(self) -> None:
        """Test chunk overlap must be less than chunk size."""
        env_vars = {
            "ADE_CHUNK_SIZE": "1000",
            "ADE_CHUNK_OVERLAP": "1500",
        }
        with (
            patch.dict(os.environ, env_vars, clear=True),
            pytest.raises(ValueError, match="chunk_overlap.*must be less than"),
        ):
            Settings(_env_file=None)


class TestValidateSettingsOnStartup:
    """Tests for the validate_settings_on_startup function."""

    def test_warns_when_api_key_not_set(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning is logged when API key is not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        with caplog.at_level(logging.WARNING):
            validate_settings_on_startup(settings)

        assert "OPENAI_API_KEY is not configured" in caplog.text

    def test_no_warning_when_api_key_set(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test no API key warning when key is configured."""
        env_vars = {"ADE_OPENAI_API_KEY": "sk-test-key"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        with caplog.at_level(logging.WARNING):
            validate_settings_on_startup(settings)

        assert "OPENAI_API_KEY is not configured" not in caplog.text

    def test_warns_about_permissive_cors_in_production(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning for permissive CORS when not in debug mode."""
        env_vars = {
            "ADE_CORS_ORIGINS": "*",
            "ADE_DEBUG": "false",
            "ADE_OPENAI_API_KEY": "sk-test",  # Avoid API key warning
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        with caplog.at_level(logging.WARNING):
            validate_settings_on_startup(settings)

        assert "CORS is configured to allow all origins" in caplog.text

    def test_no_cors_warning_in_debug_mode(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test no CORS warning when in debug mode."""
        env_vars = {
            "ADE_CORS_ORIGINS": "*",
            "ADE_DEBUG": "true",
            "ADE_OPENAI_API_KEY": "sk-test",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        with caplog.at_level(logging.WARNING):
            validate_settings_on_startup(settings)

        assert "CORS is configured to allow all origins" not in caplog.text

    def test_no_cors_warning_with_specific_origins(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test no CORS warning when specific origins are configured."""
        env_vars = {
            "ADE_CORS_ORIGINS": "https://example.com",
            "ADE_OPENAI_API_KEY": "sk-test",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        with caplog.at_level(logging.WARNING):
            validate_settings_on_startup(settings)

        assert "CORS is configured to allow all origins" not in caplog.text

    def test_logs_configuration_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test configuration summary is logged on startup."""
        env_vars = {"ADE_OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        with caplog.at_level(logging.INFO):
            validate_settings_on_startup(settings)

        assert "Configuration loaded" in caplog.text
        assert "log_level=INFO" in caplog.text
