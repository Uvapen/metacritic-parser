"""Pydantic-схемы API и ответа LLM."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class LlmSummaryResponse(BaseModel):
    """JSON саммари + признаки для рекомендаций."""

    summary: str = ""
    tags: list[str] | None = None
    genre_detailed: str | None = None
    key_features: list[str] | None = None
    target_audience: str | None = None


class GameOut(BaseModel):
    """Карточка игры в JSON API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    slug: str
    title: str
    cover_url: str | None = None
    description: str | None = None
    developer: str | None = None
    developers: list[str] | None = None
    publishers: list[str] | None = None
    genres: list[str] | None = None
    tags: list[str] = Field(default_factory=list)
    genre_detailed: str | None = None
    key_features: list[str] = Field(default_factory=list)
    target_audience: str | None = None
    release_date: date | None = None
    metascore: int | None = None
    userscore: float | None = None
    is_from_carousel: bool = False
    last_processed_at: datetime | None = None
