"""SQLAlchemy 2.0 модели хранилища игр, отзывов и прогонов."""

from datetime import date, datetime

from sqlalchemy import Boolean, CheckConstraint, Index, JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Базовый класс декларативных моделей."""


class Game(Base):
    """Игра с Metacritic (карточка + агрегированные оценки)."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(512))
    cover_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    developer: Mapped[str | None] = mapped_column(String(512), nullable=True)
    developers: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    publishers: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    genres: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    video_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    video_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    youtube_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    youtube_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    youtube_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    youtube_summary_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    youtube_channel: Mapped[str | None] = mapped_column(String(256), nullable=True)
    youtube_views: Mapped[int | None] = mapped_column(Integer, nullable=True)
    youtube_duration_sec: Mapped[int | None] = mapped_column(Integer, nullable=True)
    youtube_kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    youtube_transcript_sample: Mapped[str | None] = mapped_column(Text, nullable=True)
    related_games: Mapped[list | None] = mapped_column(JSON, nullable=True)
    release_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    metascore: Mapped[int | None] = mapped_column(Integer, nullable=True)
    userscore: Mapped[float | None] = mapped_column(Float, nullable=True)
    critic_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary_fingerprint_critic: Mapped[str | None] = mapped_column(String(64), nullable=True)
    summary_fingerprint_user: Mapped[str | None] = mapped_column(String(64), nullable=True)
    review_count_critic: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    review_count_user: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_from_carousel: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    ai_tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    ai_tags_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    genre_detailed: Mapped[str | None] = mapped_column(String(128), nullable=True)
    key_features: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    target_audience: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    platform_scores: Mapped[list["PlatformScore"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )
    summaries: Mapped[list["Summary"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )


class PlatformScore(Base):
    """Оценки игры в разрезе платформы."""

    __tablename__ = "platform_scores"
    __table_args__ = (UniqueConstraint("game_id", "platform", name="uq_platform_score_game_platform"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    platform: Mapped[str] = mapped_column(String(128))
    metascore: Mapped[int | None] = mapped_column(Integer, nullable=True)
    userscore: Mapped[float | None] = mapped_column(Float, nullable=True)
    userscore_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    critic_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    release_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    game: Mapped[Game] = relationship(back_populates="platform_scores")


class Review(Base):
    """Отзыв критика или пользователя."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(16), index=True)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    publication: Mapped[str | None] = mapped_column(String(255), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    platform: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    game: Mapped[Game] = relationship(back_populates="reviews")

    @property
    def quote(self) -> str | None:
        return self.body

    @property
    def publicationName(self) -> str | None:
        return self.publication

    @property
    def date(self) -> date | None:
        return self.reviewed_at


class Summary(Base):
    """LLM-саммари по отзывам игры."""

    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    game: Mapped[Game] = relationship(back_populates="summaries")


class RunLog(Base):
    """Журнал hourly-прогона пайплайна."""

    __tablename__ = "run_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    games_found: Mapped[int] = mapped_column(Integer, default=0)
    games_processed: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[list | None] = mapped_column(JSON, nullable=True)
    llm_errors: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class SimilarGameLink(Base):
    """Ненаправленная связь похожих игр: всегда game_a_id < game_b_id."""

    __tablename__ = "similar_game_links"
    __table_args__ = (
        UniqueConstraint("game_a_id", "game_b_id", name="uq_similar_game_pair"),
        CheckConstraint("game_a_id < game_b_id", name="ck_similar_game_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_a_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    game_b_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    fingerprint: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class SimilarVerdict(Base):
    """LLM-вердикт для смешанного сходства: показывать ли other рядом с origin."""

    __tablename__ = "similar_verdicts"
    __table_args__ = (UniqueConstraint("origin_slug", "other_slug", name="uq_similar_verdict_pair"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    origin_slug: Mapped[str] = mapped_column(String(255), index=True)
    other_slug: Mapped[str] = mapped_column(String(255), index=True)
    fingerprint: Mapped[str] = mapped_column(String(32))
    keep: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class PipelineState(Base):
    """Дневной курсор обхода ленты new: один ряд id=1."""

    __tablename__ = "pipeline_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    day: Mapped[date | None] = mapped_column(Date, nullable=True)
    offset: Mapped[int] = mapped_column(Integer, default=0)
    used_main: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    source: Mapped[str] = mapped_column(String(16), default="main", server_default="main")
    followup_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    followup_slugs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    similar_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    youtube_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    similar_done: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    youtube_done: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class PipelineJob(Base):
    """Одна задача похожих или летсплея: ставится во время прогона, съедается тиком."""

    __tablename__ = "pipeline_jobs"
    __table_args__ = (Index("ix_pipeline_jobs_status_kind_id", "status", "kind", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    slug: Mapped[str] = mapped_column(String(255), index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", server_default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
