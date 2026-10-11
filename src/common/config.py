from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Database
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5434
    POSTGRES_DB: str = "rian"
    POSTGRES_USER: str = "rian"
    POSTGRES_PASSWORD: str = "rian_secret_password"
    DATABASE_URL: Optional[str] = None
    DATABASE_SYNC_URL: Optional[str] = None

    # Ingestion & CV
    SMTP_HOST: str = "0.0.0.0"
    SMTP_PORT: int = 2525
    CONFIDENCE_THRESHOLD: float = 0.45
    STORAGE_ROOT: str = "./data/snapshots"

    # Twilio Alerts
    TWILIO_ACCOUNT_SID: str = "mock_sid"
    TWILIO_AUTH_TOKEN: str = "mock_token"
    TWILIO_FROM_NUMBER: str = "+15555555555"
    DISPATCH_PHONE_NUMBERS: str = "+15555555556"

    # Domain profile (spec 13): ranch_livestock | industrial_terminal | critical_infrastructure
    OPERATION_PROFILE: str = "ranch_livestock"

    # Mapbox
    MAPBOX_ACCESS_TOKEN: Optional[str] = None

    # LLM Tactical Copilot
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    LLM_MODEL: str = "deepseek-chat"

    # JWT authentication (spec 10)
    JWT_SECRET_KEY: str = "rian-dev-jwt-secret-change-in-production"
    JWT_ACCESS_TOKEN_EXPIRE_HOURS: int = 24
    FIRST_SUPERUSER: str = "admin@rian.internal"
    FIRST_SUPERUSER_PASSWORD: str = "rian_admin_secret"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def async_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def sync_database_url(self) -> str:
        if self.DATABASE_SYNC_URL:
            return self.DATABASE_SYNC_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()
