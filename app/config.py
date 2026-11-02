"""Application settings, loaded from environment variables / the .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from the environment, never from code."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    dataset: str = "ml-latest-small"
    data_dir: str = "./data/raw"

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5-coder:7b"
    llm_timeout_seconds: float = 120.0

    admin_api_key: str = ""
    model_dir: str = "./models"

    # A rating at or above this value counts as "the user liked the movie".
    like_threshold: float = 4.0
    # Hard cap on rows any LLM-generated query may return.
    llm_max_rows: int = 50


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
