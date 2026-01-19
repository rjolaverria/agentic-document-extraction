"""Configuration management for agentic document extraction."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ADE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # File upload settings
    max_file_size_mb: int = 10
    temp_upload_dir: str = "/tmp/ade_uploads"

    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 4096

    # Chunking settings for large documents
    chunk_size: int = 4000
    chunk_overlap: int = 200

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()
