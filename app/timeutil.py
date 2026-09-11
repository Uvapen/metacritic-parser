"""Единая локальная зона сервиса: UTC+4."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

APP_TZ = timezone(timedelta(hours=4))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_local() -> datetime:
    return datetime.now(APP_TZ)


def today_local() -> date:
    return now_local().date()


def to_local(value: Any) -> datetime | None:
    """datetime/ISO-строка → UTC+4. Naive считается UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(APP_TZ)


def as_local_date(value: datetime | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        localized = to_local(value)
        return localized.date() if localized else None
    return value
