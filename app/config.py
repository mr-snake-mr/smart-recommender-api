"""Application configuration, loaded from environment variables / .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider (any OpenAI-compatible endpoint)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_mock_mode: bool = False

    # Catalog sources. The fallback is used only when the primary source is unavailable.
    fakestore_api_url: str = "https://fakestoreapi.com/products"
    fallback_catalog_api_url: str = "https://dummyjson.com/products?limit=100"
    catalog_cache_ttl_seconds: int = 300

    # Analytics logging (Path B)
    analytics_backend: str = ""  # "", "redis", or "supabase"
    analytics_queue_name: str = "unmet_constraints"
    redis_url: str = "redis://localhost:6379/0"
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_table: str = "unmet_constraints"

    # CORS
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def analytics_enabled(self) -> bool:
        return self.analytics_backend in {"redis", "supabase"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
