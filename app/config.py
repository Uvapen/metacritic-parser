"""Настройки приложения через переменные окружения."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация сервиса-парсера Metacritic."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Metacritic Parser"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/metacritic.db",
        description="DSN SQLAlchemy. SQLite по умолчанию, Postgres: postgresql+asyncpg://...",
    )

    metacritic_base_url: str = "https://www.metacritic.com"
    browse_url: str = "https://www.metacritic.com/browse/game/all/all/all-time/new/"
    impersonate: str = "chrome120"
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5

    scheduler_enabled: bool = True
    pipeline_interval_hours: int = 1
    games_limit: int = 20

    llm_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GROQ_API_KEY", "LLM_API_KEY"),
    )
    llm_model: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias=AliasChoices("LLM_MODEL", "GROQ_MODEL"),
    )
    llm_fallback_models: str = Field(
        default="openai/gpt-oss-20b",
        validation_alias=AliasChoices("LLM_FALLBACK_MODELS", "GROQ_FALLBACK_MODELS"),
        description="Через запятую: если у текущей модели TPD/RPD, берём следующую.",
    )
    llm_timeout: float = 60.0
    llm_call_interval: float = Field(
        default=20.0,
        validation_alias=AliasChoices("LLM_CALL_INTERVAL"),
        description="Минимальная пауза между HTTP-вызовами Groq, секунды.",
    )
    llm_log_filename: str = "llm_logs.jsonl"

    youtube_enabled: bool = True
    youtube_timeout: float = 25.0
    youtube_call_interval: float = 1.5
    youtube_backfill_limit: int = 40
    youtube_sweep_limit: int = 10
    youtube_proxy: str = Field(
        default="",
        validation_alias=AliasChoices("YOUTUBE_PROXY"),
        description="HTTPS-прокси для yt-dlp/Innertube. Пусто — прямой выход Render.",
    )
    youtube_cookies_path: str = Field(
        default="",
        validation_alias=AliasChoices("YOUTUBE_COOKIES_PATH"),
        description="Путь к Netscape cookies.txt на диске.",
    )
    youtube_cookies: str = Field(
        default="",
        validation_alias=AliasChoices("YOUTUBE_COOKIES", "YOUTUBE_COOKIES_B64"),
        description="Содержимое cookies.txt или base64:... Секрет Render, не в git.",
    )
    whisper_enabled: bool = True
    whisper_model: str = Field(
        default="whisper-large-v3-turbo",
        validation_alias=AliasChoices("WHISPER_MODEL", "GROQ_WHISPER_MODEL"),
    )
    whisper_timeout: float = 180.0
    card_delay_seconds: float = 1.5
    enrichment_tick_seconds: float = 30

    @property
    def project_root(self) -> Path:
        """Корень репозитория (на уровень выше пакета app)."""
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def llm_log_path(self) -> Path:
        return self.data_dir / self.llm_log_filename

    @property
    def templates_dir(self) -> Path:
        return Path(__file__).resolve().parent / "web" / "templates"


@lru_cache
def get_settings() -> Settings:
    """Возвращает кэшированный экземпляр настроек."""
    return Settings()
