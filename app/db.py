"""Асинхронное подключение к SQLite или PostgreSQL."""

import json
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

from sqlalchemy import event, inspect, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.models import Base
from app.timeutil import as_local_date


def normalize_database_url(raw_url: str, project_root: Path) -> str:
    """Приводит DSN к async-драйверу и абсолютному пути для SQLite."""
    url = raw_url.strip()
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url.removeprefix("postgresql://")
    elif url.startswith("postgres://"):
        url = "postgresql+asyncpg://" + url.removeprefix("postgres://")
    elif url.startswith("sqlite://") and not url.startswith("sqlite+aiosqlite://"):
        url = "sqlite+aiosqlite://" + url.removeprefix("sqlite://")

    parsed = make_url(url)
    if parsed.drivername.startswith("sqlite"):
        database = parsed.database or "./data/metacritic.db"
        db_path = Path(database)
        if not db_path.is_absolute():
            db_path = (project_root / db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{db_path.as_posix()}"
    return url


def create_engine_and_session(
    settings: Settings | None = None,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Создаёт engine и фабрику сессий."""
    settings = settings or get_settings()
    url = normalize_database_url(settings.database_url, settings.project_root)
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["timeout"] = 30

    engine = create_async_engine(
        url,
        echo=settings.debug,
        connect_args=connect_args,
    )
    if url.startswith("sqlite"):
        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.fetchall()
            cursor.close()

    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return engine, session_factory


engine, SessionLocal = create_engine_and_session()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Зависимость FastAPI: сессия на запрос."""
    async with SessionLocal() as session:
        yield session


def _run_day_key(started_at: object) -> str:
    """Календарный день прогона в UTC+4."""
    if started_at is None:
        return ""
    if isinstance(started_at, datetime):
        day = as_local_date(started_at)
        return day.isoformat() if day else ""
    raw = str(started_at).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return raw[:10]
    day = as_local_date(dt)
    return day.isoformat() if day else raw[:10]


def _backfill_carousel_flags(sync_connection) -> None:
    """Первый успешный прогон каждого дня — карусель по ТЗ."""
    inspector = inspect(sync_connection)
    if "run_logs" not in inspector.get_table_names():
        return
    rows = sync_connection.execute(
        text(
            "SELECT started_at, details FROM run_logs "
            "WHERE status IN ('success', 'partial') AND games_processed > 0 "
            "ORDER BY started_at ASC"
        )
    ).fetchall()
    seen_days: set[str] = set()
    slugs: set[str] = set()
    for started_at, details in rows:
        day_key = _run_day_key(started_at)
        if not day_key or day_key in seen_days:
            continue
        parsed = details
        if isinstance(details, str):
            try:
                parsed = json.loads(details)
            except json.JSONDecodeError:
                continue
        if not isinstance(parsed, list):
            continue
        day_slugs = [
            str(item.get("slug"))
            for item in parsed
            if isinstance(item, dict) and item.get("slug") and item.get("action") != "error"
        ]
        if not day_slugs:
            continue
        seen_days.add(day_key)
        slugs.update(day_slugs)
    if not slugs:
        return
    params = {f"s{i}": slug for i, slug in enumerate(slugs)}
    placeholders = ", ".join(f":s{i}" for i in range(len(slugs)))
    sync_connection.execute(
        text(f"UPDATE games SET is_from_carousel = 1 WHERE slug IN ({placeholders})"),
        params,
    )


def _ensure_game_columns(sync_connection) -> None:
    """Добавляет новые колонки в уже существующую SQLite-таблицу games."""
    inspector = inspect(sync_connection)
    if "games" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("games")}
    statements = {
        "developers": "ALTER TABLE games ADD COLUMN developers JSON",
        "publishers": "ALTER TABLE games ADD COLUMN publishers JSON",
        "genres": "ALTER TABLE games ADD COLUMN genres JSON",
        "video_url": "ALTER TABLE games ADD COLUMN video_url VARCHAR(2048)",
        "video_title": "ALTER TABLE games ADD COLUMN video_title VARCHAR(512)",
        "cover_url": "ALTER TABLE games ADD COLUMN cover_url VARCHAR(1024)",
        "metascore": "ALTER TABLE games ADD COLUMN metascore INTEGER",
        "userscore": "ALTER TABLE games ADD COLUMN userscore FLOAT",
        "critic_count": "ALTER TABLE games ADD COLUMN critic_count INTEGER",
        "youtube_url": "ALTER TABLE games ADD COLUMN youtube_url VARCHAR(2048)",
        "youtube_title": "ALTER TABLE games ADD COLUMN youtube_title VARCHAR(512)",
        "youtube_summary": "ALTER TABLE games ADD COLUMN youtube_summary TEXT",
        "youtube_summary_source": "ALTER TABLE games ADD COLUMN youtube_summary_source VARCHAR(16)",
        "youtube_channel": "ALTER TABLE games ADD COLUMN youtube_channel VARCHAR(256)",
        "youtube_views": "ALTER TABLE games ADD COLUMN youtube_views INTEGER",
        "youtube_duration_sec": "ALTER TABLE games ADD COLUMN youtube_duration_sec INTEGER",
        "youtube_kind": "ALTER TABLE games ADD COLUMN youtube_kind VARCHAR(16)",
        "youtube_transcript_sample": "ALTER TABLE games ADD COLUMN youtube_transcript_sample TEXT",
        "related_games": "ALTER TABLE games ADD COLUMN related_games JSON",
        "summary_fingerprint_critic": "ALTER TABLE games ADD COLUMN summary_fingerprint_critic VARCHAR(64)",
        "summary_fingerprint_user": "ALTER TABLE games ADD COLUMN summary_fingerprint_user VARCHAR(64)",
        "review_count_critic": "ALTER TABLE games ADD COLUMN review_count_critic INTEGER DEFAULT 0",
        "review_count_user": "ALTER TABLE games ADD COLUMN review_count_user INTEGER DEFAULT 0",
        "is_from_carousel": "ALTER TABLE games ADD COLUMN is_from_carousel BOOLEAN DEFAULT 0",
        "ai_tags": "ALTER TABLE games ADD COLUMN ai_tags JSON",
        "ai_tags_fingerprint": "ALTER TABLE games ADD COLUMN ai_tags_fingerprint VARCHAR(64)",
        "tags": "ALTER TABLE games ADD COLUMN tags JSON",
        "genre_detailed": "ALTER TABLE games ADD COLUMN genre_detailed VARCHAR(128)",
        "key_features": "ALTER TABLE games ADD COLUMN key_features JSON",
        "target_audience": "ALTER TABLE games ADD COLUMN target_audience VARCHAR(256)",
    }
    added_carousel = "is_from_carousel" not in existing
    added_tags = "tags" not in existing
    for name, ddl in statements.items():
        if name not in existing:
            sync_connection.execute(text(ddl))
    if added_carousel:
        _backfill_carousel_flags(sync_connection)
    if added_tags:
        sync_connection.execute(
            text(
                "UPDATE games SET tags = ai_tags "
                "WHERE tags IS NULL AND ai_tags IS NOT NULL"
            )
        )

    if "reviews" in inspector.get_table_names():
        review_cols = {column["name"] for column in inspector.get_columns("reviews")}
        if "platform" not in review_cols:
            sync_connection.execute(text("ALTER TABLE reviews ADD COLUMN platform VARCHAR(64)"))

    if "platform_scores" in inspector.get_table_names():
        platform_cols = {column["name"] for column in inspector.get_columns("platform_scores")}
        if "userscore_count" not in platform_cols:
            sync_connection.execute(text("ALTER TABLE platform_scores ADD COLUMN userscore_count INTEGER"))
        if "critic_count" not in platform_cols:
            sync_connection.execute(text("ALTER TABLE platform_scores ADD COLUMN critic_count INTEGER"))

    if "summaries" in inspector.get_table_names():
        summary_cols = {column["name"] for column in inspector.get_columns("summaries")}
        if "kind" not in summary_cols:
            sync_connection.execute(text("ALTER TABLE summaries ADD COLUMN kind VARCHAR(16)"))

    if "run_logs" in inspector.get_table_names():
        run_cols = {column["name"] for column in inspector.get_columns("run_logs")}
        if "details" not in run_cols:
            sync_connection.execute(text("ALTER TABLE run_logs ADD COLUMN details JSON"))
        if "llm_errors" not in run_cols:
            sync_connection.execute(
                text("ALTER TABLE run_logs ADD COLUMN llm_errors INTEGER DEFAULT 0")
            )

    if "similar_game_links" in inspector.get_table_names():
        link_cols = {column["name"] for column in inspector.get_columns("similar_game_links")}
        if "fingerprint" not in link_cols:
            sync_connection.execute(
                text("ALTER TABLE similar_game_links ADD COLUMN fingerprint VARCHAR(32)")
            )
        if "source" not in link_cols:
            sync_connection.execute(
                text("ALTER TABLE similar_game_links ADD COLUMN source VARCHAR(16)")
            )

    if "pipeline_state" in inspector.get_table_names():
        state_cols = {column["name"] for column in inspector.get_columns("pipeline_state")}
        if "used_main" not in state_cols:
            sync_connection.execute(
                text("ALTER TABLE pipeline_state ADD COLUMN used_main BOOLEAN DEFAULT 0")
            )
        if "source" not in state_cols:
            sync_connection.execute(
                text("ALTER TABLE pipeline_state ADD COLUMN source VARCHAR(16) DEFAULT 'main'")
            )
        if "followup_run_id" not in state_cols:
            sync_connection.execute(text("ALTER TABLE pipeline_state ADD COLUMN followup_run_id INTEGER"))
        if "followup_slugs" not in state_cols:
            sync_connection.execute(text("ALTER TABLE pipeline_state ADD COLUMN followup_slugs JSON"))
        if "similar_due_at" not in state_cols:
            sync_connection.execute(text("ALTER TABLE pipeline_state ADD COLUMN similar_due_at DATETIME"))
        if "youtube_due_at" not in state_cols:
            sync_connection.execute(text("ALTER TABLE pipeline_state ADD COLUMN youtube_due_at DATETIME"))
        if "similar_done" not in state_cols:
            sync_connection.execute(
                text("ALTER TABLE pipeline_state ADD COLUMN similar_done BOOLEAN DEFAULT 1")
            )
        if "youtube_done" not in state_cols:
            sync_connection.execute(
                text("ALTER TABLE pipeline_state ADD COLUMN youtube_done BOOLEAN DEFAULT 1")
            )


async def init_db() -> None:
    """Создаёт каталог data/ и таблицы, если их ещё нет."""
    get_settings().data_dir.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(_ensure_game_columns)
