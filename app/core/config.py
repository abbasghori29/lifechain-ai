from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "LifeChain API"
    ENV: str = "development"
    DEBUG: bool = True

    # Security
    SECRET_KEY: str = "changeme"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = []

    # Database - Local PostgreSQL (must be set in .env file)
    DATABASE_URL: str  # Async connection URL (postgresql+asyncpg://...)
    DIRECT_DATABASE_URL: str  # Sync connection URL for Alembic migrations (postgresql+psycopg2://...)
    
    # AI/ML - Gemini API (must be set in .env file)
    GOOGLE_API_KEY: str
    
    # Translation - Groq API (must be set in .env file)
    GROQ_API_KEY: str


@lru_cache
def get_settings() -> Settings:
    return Settings()

def clear_settings_cache():
    """Clear the settings cache - useful when .env file changes"""
    get_settings.cache_clear()


