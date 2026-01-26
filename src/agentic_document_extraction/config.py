"""Configuration management for agentic document extraction.

This module provides centralized configuration using pydantic-settings.
All configuration options can be set via environment variables with the
ADE_ prefix, or via a .env file in the project root.

Environment Variables:
    ADE_MAX_FILE_SIZE_MB: Maximum file upload size in MB (default: 10)
    ADE_TEMP_UPLOAD_DIR: Directory for temporary file uploads
    ADE_OPENAI_API_KEY: OpenAI API key (required for LLM features)
    ADE_OPENAI_MODEL: Default OpenAI model for text processing (default: gpt-4o)
    ADE_OPENAI_VLM_MODEL: OpenAI model for vision processing (default: gpt-4o)
    ADE_OPENAI_TEMPERATURE: LLM temperature setting (default: 0.0)
    ADE_OPENAI_MAX_TOKENS: Maximum tokens for LLM responses (default: 4096)
    ADE_CHUNK_SIZE: Token chunk size for large documents (default: 4000)
    ADE_CHUNK_OVERLAP: Token overlap between chunks (default: 200)
    ADE_PADDLEOCR_LANGUAGE: PaddleOCR language code (default: en)
    ADE_PADDLEOCR_USE_GPU: Enable PaddleOCR GPU usage (default: false)
    ADE_PADDLEOCR_USE_ANGLE_CLS: Enable PaddleOCR angle classifier (default: true)
    ADE_PADDLEOCR_DET_MODEL_DIR: PaddleOCR detection model directory
    ADE_PADDLEOCR_REC_MODEL_DIR: PaddleOCR recognition model directory
    ADE_PADDLEOCR_CLS_MODEL_DIR: PaddleOCR classifier model directory
    ADE_PADDLEOCR_ENABLE_MKLDNN: Enable PaddleOCR MKL-DNN (default: false)
    ADE_PADDLEOCR_CPU_THREADS: PaddleOCR CPU thread count (default: 4)
    ADE_LAYOUTREADER_MODEL: LayoutReader model name (default: hantian/layoutreader)
    ADE_MIN_OVERALL_CONFIDENCE: Min confidence threshold (default: 0.7)
    ADE_MIN_FIELD_CONFIDENCE: Min per-field confidence (default: 0.5)
    ADE_REQUIRED_FIELD_COVERAGE: Required field coverage threshold (default: 0.9)
    ADE_MAX_REFINEMENT_ITERATIONS: Max agentic loop iterations (default: 3)
    ADE_JOB_TTL_HOURS: Job result retention time in hours (default: 24)
    ADE_DOCKET_NAME: Shared Docket name (default: agentic-document-extraction)
    ADE_DOCKET_URL: Redis/memory backend URL (default: redis://localhost:6379/0)
    ADE_DOCKET_RESULT_STORAGE_URL: Optional Redis URL for result storage
    ADE_DOCKET_EXECUTION_TTL_SECONDS: Optional TTL override for executions
    ADE_DOCKET_ENABLE_INTERNAL_INSTRUMENTATION: Enable Docket internal spans
    ADE_LOG_LEVEL: Logging level (default: INFO)
    ADE_DEBUG: Enable debug mode (default: false)
    ADE_CORS_ORIGINS: Comma-separated CORS origins (default: *)
    ADE_SERVER_HOST: Server bind host (default: 0.0.0.0)
    ADE_SERVER_PORT: Server bind port (default: 8000)
"""

import logging
from datetime import timedelta
from typing import Any

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be configured via environment variables prefixed with ADE_
    or via a .env file. Sensitive values like API keys use SecretStr to prevent
    accidental logging.

    Example .env file:
        ADE_OPENAI_API_KEY=sk-...
        ADE_LOG_LEVEL=DEBUG
        ADE_MAX_REFINEMENT_ITERATIONS=5
    """

    model_config = SettingsConfigDict(
        env_prefix="ADE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================================================================
    # File Upload Settings
    # =========================================================================

    max_file_size_mb: int = 10
    """Maximum file upload size in megabytes."""

    temp_upload_dir: str = "/tmp/ade_uploads"
    """Directory for storing temporary uploaded files."""

    # =========================================================================
    # OpenAI / LLM Settings
    # =========================================================================

    openai_api_key: SecretStr = SecretStr("")
    """OpenAI API key. Required for LLM-powered extraction features."""

    openai_model: str = "gpt-4o"
    """Default OpenAI model for text processing and planning."""

    openai_vlm_model: str = "gpt-4o"
    """OpenAI model for vision/multimodal processing."""

    openai_temperature: float = 0.0
    """Temperature for LLM sampling. 0.0 for deterministic, higher for creativity."""

    openai_max_tokens: int = 4096
    """Maximum tokens for LLM response generation."""

    # =========================================================================
    # Document Processing Settings
    # =========================================================================

    chunk_size: int = 4000
    """Token chunk size when splitting large documents."""

    chunk_overlap: int = 200
    """Token overlap between chunks for context continuity."""

    # =========================================================================
    # OCR Settings (PaddleOCR-VL)
    # =========================================================================

    paddleocr_language: str = "en"
    """PaddleOCR language code (e.g., en, ch, fr)."""

    paddleocr_use_gpu: bool = False
    """Enable GPU acceleration for PaddleOCR."""

    paddleocr_use_angle_cls: bool = True
    """Enable angle classification for PaddleOCR."""

    paddleocr_det_model_dir: str | None = None
    """Optional PaddleOCR detection model directory override."""

    paddleocr_rec_model_dir: str | None = None
    """Optional PaddleOCR recognition model directory override."""

    paddleocr_cls_model_dir: str | None = None
    """Optional PaddleOCR classifier model directory override."""

    paddleocr_enable_mkldnn: bool = False
    """Enable MKL-DNN acceleration for PaddleOCR on CPU."""

    paddleocr_cpu_threads: int = 4
    """CPU thread count for PaddleOCR."""

    # =========================================================================
    # LayoutReader Settings
    # =========================================================================

    layoutreader_model: str = "hantian/layoutreader"
    """LayoutReader model name for reading order detection."""

    # =========================================================================
    # Quality Threshold Settings
    # =========================================================================

    min_overall_confidence: float = 0.7
    """Minimum overall confidence score for extraction to pass (0.0-1.0)."""

    min_field_confidence: float = 0.5
    """Minimum confidence score per extracted field (0.0-1.0)."""

    required_field_coverage: float = 0.9
    """Minimum coverage of required schema fields (0.0-1.0)."""

    # =========================================================================
    # Agentic Loop Settings
    # =========================================================================

    max_refinement_iterations: int = 3
    """Maximum iterations for the agentic refinement loop (1-10)."""

    # =========================================================================
    # Job Management Settings
    # =========================================================================

    job_ttl_hours: int = 24
    """Time-to-live for job results in hours before cleanup."""

    docket_name: str = "agentic-document-extraction"
    """Shared Docket name for coordinating workers."""

    docket_url: str = "redis://localhost:6379/0"
    """Redis or memory backend URL for Docket (e.g., redis:// or memory://)."""

    docket_result_storage_url: str | None = None
    """Optional Redis URL for result storage (defaults to docket_url)."""

    docket_execution_ttl_seconds: int | None = None
    """Override for Docket execution TTL in seconds (defaults to job_ttl_seconds)."""

    docket_enable_internal_instrumentation: bool = False
    """Enable OpenTelemetry spans for Docket's internal Redis polling."""

    # =========================================================================
    # Logging Settings
    # =========================================================================

    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL."""

    debug: bool = False
    """Enable debug mode with additional logging and error details."""

    # =========================================================================
    # Server Settings
    # =========================================================================

    cors_origins: str = "*"
    """Comma-separated list of allowed CORS origins, or * for all."""

    server_host: str = "0.0.0.0"
    """Host address for the server to bind to."""

    server_port: int = 8000
    """Port for the server to listen on."""

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(
                f"Invalid log level: {v}. Must be one of: {', '.join(valid_levels)}"
            )
        return upper_v

    @field_validator(
        "min_overall_confidence", "min_field_confidence", "required_field_coverage"
    )
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("max_refinement_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Validate max iterations is reasonable."""
        if not 1 <= v <= 10:
            raise ValueError(
                f"max_refinement_iterations must be between 1 and 10, got {v}"
            )
        return v

    @field_validator("max_file_size_mb")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """Validate file size is positive and reasonable."""
        if not 1 <= v <= 500:
            raise ValueError(f"max_file_size_mb must be between 1 and 500, got {v}")
        return v

    @field_validator("paddleocr_cpu_threads")
    @classmethod
    def validate_paddleocr_threads(cls, v: int) -> int:
        """Validate PaddleOCR CPU thread count."""
        if v < 1:
            raise ValueError("paddleocr_cpu_threads must be at least 1")
        return v

    @field_validator("paddleocr_language")
    @classmethod
    def validate_paddleocr_language(cls, v: str) -> str:
        """Validate PaddleOCR language code is non-empty."""
        if not v.strip():
            raise ValueError("paddleocr_language must be a non-empty string")
        return v.strip()

    @field_validator("server_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"server_port must be between 1 and 65535, got {v}")
        return v

    @field_validator("job_ttl_hours")
    @classmethod
    def validate_job_ttl(cls, v: int) -> int:
        """Validate job TTL is positive."""
        if v < 1:
            raise ValueError(f"job_ttl_hours must be at least 1, got {v}")
        return v

    @field_validator("docket_execution_ttl_seconds")
    @classmethod
    def validate_docket_ttl(cls, v: int | None) -> int | None:
        """Validate Docket execution TTL override."""
        if v is not None and v < 1:
            raise ValueError(
                f"docket_execution_ttl_seconds must be at least 1, got {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        """Validate chunk overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def job_ttl_seconds(self) -> int:
        """Get job TTL in seconds."""
        return self.job_ttl_hours * 3600

    @property
    def docket_execution_ttl(self) -> timedelta:
        """Get Docket execution TTL as a timedelta."""
        ttl_seconds = (
            self.docket_execution_ttl_seconds
            if self.docket_execution_ttl_seconds is not None
            else self.job_ttl_seconds
        )
        return timedelta(seconds=ttl_seconds)

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def log_level_int(self) -> int:
        """Get log level as integer for logging module."""
        level: int = getattr(logging, self.log_level)
        return level

    def get_openai_api_key(self) -> str:
        """Get the OpenAI API key value.

        Returns:
            The API key string. Returns empty string if not set.

        Note:
            Use this method to access the API key value. Direct access to
            openai_api_key returns a SecretStr which prevents accidental logging.
        """
        return self.openai_api_key.get_secret_value()

    def to_safe_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary with sensitive values masked.

        Returns:
            Dictionary representation with API keys masked.
        """
        data = {
            "max_file_size_mb": self.max_file_size_mb,
            "temp_upload_dir": self.temp_upload_dir,
            "openai_api_key": "***" if self.get_openai_api_key() else "(not set)",
            "openai_model": self.openai_model,
            "openai_vlm_model": self.openai_vlm_model,
            "openai_temperature": self.openai_temperature,
            "openai_max_tokens": self.openai_max_tokens,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "layoutreader_model": self.layoutreader_model,
            "min_overall_confidence": self.min_overall_confidence,
            "min_field_confidence": self.min_field_confidence,
            "required_field_coverage": self.required_field_coverage,
            "max_refinement_iterations": self.max_refinement_iterations,
            "job_ttl_hours": self.job_ttl_hours,
            "docket_name": self.docket_name,
            "docket_url": self.docket_url,
            "docket_result_storage_url": self.docket_result_storage_url,
            "docket_execution_ttl_seconds": self.docket_execution_ttl_seconds,
            "docket_enable_internal_instrumentation": (
                self.docket_enable_internal_instrumentation
            ),
            "log_level": self.log_level,
            "debug": self.debug,
            "cors_origins": self.cors_origins,
            "server_host": self.server_host,
            "server_port": self.server_port,
        }
        return data


def validate_settings_on_startup(s: Settings) -> None:
    """Validate settings on application startup.

    This function performs additional validation that may require
    external checks or warnings for production readiness.

    Args:
        s: Settings instance to validate.

    Raises:
        ValueError: If critical configuration is invalid.
    """
    logger = logging.getLogger(__name__)

    # Warn if API key is not set
    if not s.get_openai_api_key():
        logger.warning(
            "OPENAI_API_KEY is not configured. LLM-powered features will not work. "
            "Set ADE_OPENAI_API_KEY environment variable."
        )

    # Warn about permissive CORS in non-debug mode
    if s.cors_origins == "*" and not s.debug:
        logger.warning(
            "CORS is configured to allow all origins (*). "
            "Consider restricting this in production."
        )

    # Log configuration summary (without sensitive values)
    logger.info(
        f"Configuration loaded: log_level={s.log_level}, debug={s.debug}, "
        f"max_file_size_mb={s.max_file_size_mb}, "
        f"max_iterations={s.max_refinement_iterations}"
    )


# Create the global settings instance
settings = Settings()
