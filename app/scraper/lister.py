"""Парсинг главной (New Releases) и browse-ленты new из __NUXT_DATA__."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup

from app.config import Settings, get_settings
from app.scraper.client import MetacriticClient
from app.scraper.nuxt import (
    NuxtDataNotFoundError,
    cover_url_from_image,
    extract_nuxt_data,
    find_component,
    rehydrate_nuxt,
    walk,
)


@dataclass(slots=True)
class ListedGame:
    """Краткая карточка игры из каталога или карусели New Releases."""

    title: str
    slug: str
    url: str
    release_date: str | None
    metascore: int | None
    userscore: float | None
    cover_url: str | None = None
    description: str | None = None
    genres: list[str] = field(default_factory=list)
    critic_count: int | None = None
    from_carousel: bool = False


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(item["name"]) for item in items if isinstance(item, dict) and item.get("name")]


def _listed_from_item(item: dict[str, Any], *, base_url: str) -> ListedGame | None:
    slug = item.get("slug")
    if not isinstance(slug, str) or not slug:
        return None
    critic = item.get("criticScoreSummary") if isinstance(item.get("criticScoreSummary"), dict) else {}
    user = item.get("userScore") if isinstance(item.get("userScore"), dict) else {}
    description = item.get("description") if isinstance(item.get("description"), str) else None
    userscore = _as_float(user.get("score"))
    if userscore == 0:
        userscore = None
    return ListedGame(
        title=str(item.get("title") or slug),
        slug=slug,
        url=f"{base_url.rstrip('/')}/game/{slug}/",
        release_date=item.get("releaseDate") if isinstance(item.get("releaseDate"), str) else None,
        metascore=_as_int(critic.get("score")),
        userscore=userscore,
        cover_url=cover_url_from_image(item.get("image")),
        description=description,
        genres=_names(item.get("genres")),
        critic_count=_as_int(critic.get("reviewCount")),
    )


def _hydrate(html: str) -> Any | None:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NUXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        return rehydrate_nuxt(json.loads(tag.string))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _collect_from_items(
    items: list[Any],
    *,
    base_url: str,
    limit: int | None = None,
) -> list[ListedGame]:
    games: list[ListedGame] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {None, "game-title"} and "slug" not in item:
            continue
        listed = _listed_from_item(item, base_url=base_url)
        if listed is None or listed.slug in seen:
            continue
        seen.add(listed.slug)
        games.append(listed)
        if limit is not None and len(games) >= limit:
            break
    return games


def parse_new_releases(
    html: str,
    limit: int = 20,
    base_url: str = "https://www.metacritic.com",
) -> list[ListedGame]:
    """New Releases с главной ``/game/``: карусель, иначе type==game-title по порядку."""
    data = _hydrate(html)
    if data is None:
        return []

    items: list[Any] = []
    component = find_component(data, "new-releases-carousel")
    if component and isinstance(component.get("data"), dict):
        raw = component["data"].get("items")
        if isinstance(raw, list):
            items = [item for item in raw if isinstance(item, dict) and item.get("type") == "game-title"]

    if not items:
        for obj in walk(data):
            if isinstance(obj, dict) and obj.get("type") == "game-title" and obj.get("slug"):
                items.append(obj)

    return _collect_from_items(items, base_url=base_url, limit=limit)


def parse_browse_page(
    html: str,
    base_url: str = "https://www.metacritic.com",
) -> list[ListedGame]:
    """Все игры browse-страницы в порядке payload, без сортировки и лимита."""
    data = _hydrate(html)
    if data is None:
        return []

    for obj in walk(data):
        if not isinstance(obj, dict):
            continue
        for key, value in obj.items():
            if not (isinstance(key, str) and key.startswith("browse-game-") and isinstance(value, dict)):
                continue
            raw_items = value.get("items", [])
            if not isinstance(raw_items, list) or not raw_items:
                continue
            return _collect_from_items(raw_items, base_url=base_url)
    return []


def parse_games_list(
    html: str,
    limit: int = 20,
    base_url: str = "https://www.metacritic.com",
) -> list[ListedGame]:
    """Парсит список игр из browse ``__NUXT_DATA__`` (ключ browse-game-…)."""
    games = parse_browse_page(html, base_url=base_url)
    games.sort(key=lambda game: (game.title or "").lower())
    games.sort(key=lambda game: game.release_date or "1900-01-01", reverse=True)
    return games[:limit]


def parse_main_new_releases(
    html: str,
    base_url: str = "https://www.metacritic.com",
) -> list[ListedGame]:
    """Только карусель New Releases на главной, порядок страницы, без других блоков."""
    try:
        data = extract_nuxt_data(html)
    except (NuxtDataNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return []

    component = find_component(data, "new-releases-carousel")
    if not component or not isinstance(component.get("data"), dict):
        return []
    raw = component["data"].get("items")
    if not isinstance(raw, list):
        return []
    titles = [
        item
        for item in raw
        if isinstance(item, dict) and item.get("type") == "game-title"
    ]
    return _collect_from_items(titles, base_url=base_url)


async def fetch_main_new_releases(
    client: MetacriticClient,
    *,
    settings: Settings | None = None,
) -> list[ListedGame]:
    """GET ``/game/``: карусель New Releases в порядке страницы, без сортировки."""
    settings = settings or get_settings()
    url = f"{settings.metacritic_base_url.rstrip('/')}/game/"
    html = await client.fetch_page(url)
    return parse_main_new_releases(html, base_url=settings.metacritic_base_url)


async def fetch_home_new_releases(
    client: MetacriticClient,
    *,
    settings: Settings | None = None,
    limit: int = 20,
) -> list[ListedGame]:
    """Качает главную /game/ и возвращает блок New Releases."""
    settings = settings or get_settings()
    url = f"{settings.metacritic_base_url.rstrip('/')}/game/"
    html = await client.fetch_page(url)
    return parse_new_releases(html, limit=limit, base_url=settings.metacritic_base_url)


def browse_page_url(settings: Settings, page: int) -> str:
    """URL страницы ленты all-time/new. page=1 — первая страница."""
    base = settings.browse_url.rstrip("/") + "/"
    if page <= 1:
        return base
    return f"{base}?page={page}"


async def fetch_browse_page(
    client: MetacriticClient,
    page: int,
    *,
    settings: Settings | None = None,
) -> list[ListedGame]:
    """Одна страница browse/new в порядке сайта."""
    settings = settings or get_settings()
    html = await client.fetch_page(browse_page_url(settings, page))
    return parse_browse_page(html, base_url=settings.metacritic_base_url)


async def fetch_latest_games(
    client: MetacriticClient,
    *,
    settings: Settings | None = None,
    limit: int | None = None,
) -> list[ListedGame]:
    """Скачивает browse-страницу и возвращает список игр."""
    settings = settings or get_settings()
    html = await client.fetch_page(settings.browse_url)
    return parse_games_list(
        html,
        limit=limit or settings.games_limit,
        base_url=settings.metacritic_base_url,
    )
