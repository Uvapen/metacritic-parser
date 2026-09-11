"""Парсинг карточки игры через find_component и __NUXT_DATA__."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup

from app.config import Settings, get_settings
from app.scraper.client import MetacriticClient

logger = logging.getLogger(__name__)
from app.scraper.nuxt import cover_url_from_image, extract_nuxt_data, find_component, rehydrate_nuxt


@dataclass(slots=True)
class PlatformInfo:
    """Оценки и дата релиза на конкретной платформе."""

    name: str
    slug: str | None = None
    release_date: str | None = None
    metascore: int | None = None
    userscore: float | None = None
    critic_count: int | None = None
    userscore_count: int | None = None


@dataclass(slots=True)
class CriticReview:
    publication: str | None = None
    author: str | None = None
    score: float | None = None
    quote: str | None = None
    date: str | None = None
    url: str | None = None
    platform: str | None = None


@dataclass(slots=True)
class UserReview:
    author: str | None = None
    score: float | None = None
    quote: str | None = None
    date: str | None = None
    platform: str | None = None


@dataclass(slots=True)
class RelatedGame:
    title: str | None = None
    slug: str | None = None
    url: str | None = None
    metascore: int | None = None
    cover_url: str | None = None


@dataclass(slots=True)
class GameCard:
    """Полные данные карточки игры."""

    slug: str
    url: str
    title: str | None = None
    description: str | None = None
    release_date: str | None = None
    premiere_year: int | None = None
    developers: list[str] = field(default_factory=list)
    publishers: list[str] = field(default_factory=list)
    cover_url: str | None = None
    genres: list[str] = field(default_factory=list)
    platforms: list[PlatformInfo] = field(default_factory=list)
    video_url: str | None = None
    video_title: str | None = None
    metascore: int | None = None
    metascore_count: int | None = None
    userscore: float | None = None
    userscore_count: int | None = None
    critic_reviews: list[CriticReview] = field(default_factory=list)
    user_reviews: list[UserReview] = field(default_factory=list)
    related_games: list[RelatedGame] = field(default_factory=list)


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


def _append_unique(target: list[str], name: str | None) -> None:
    if name and name not in target:
        target.append(name)


def _as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def extract_production_companies(item: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Разработчики и издатели лежат в item.production.companies."""
    developers: list[str] = []
    publishers: list[str] = []
    production = item.get("production")
    if not isinstance(production, dict):
        return developers, publishers
    companies = production.get("companies")
    if not isinstance(companies, list):
        return developers, publishers
    for company in companies:
        if not isinstance(company, dict):
            continue
        name = _as_str(company.get("name"))
        if not name:
            continue
        type_name = company.get("typeName")
        if type_name == "Developer":
            _append_unique(developers, name)
        elif type_name == "Publisher":
            _append_unique(publishers, name)
    return developers, publishers


def extract_developers(item: dict[str, Any]) -> list[str]:
    developers, _publishers = extract_production_companies(item)
    if developers:
        return developers
    developers.extend(_names(item.get("developers")))
    if developers:
        return developers
    dev = item.get("developer")
    if isinstance(dev, dict) and _as_str(dev.get("name")):
        return [str(dev["name"])]
    if isinstance(dev, str) and dev.strip():
        return [dev.strip()]
    return developers


def extract_publishers(item: dict[str, Any]) -> list[str]:
    _developers, publishers = extract_production_companies(item)
    if publishers:
        return publishers
    return _names(item.get("publishers"))


def _cover_from_product(item: dict[str, Any]) -> str | None:
    """Обложка в item.images[], не в item.image."""
    images = item.get("images")
    if isinstance(images, list):
        preferred: str | None = None
        for image in images:
            if not isinstance(image, dict) or not image.get("bucketPath"):
                continue
            url = cover_url_from_image(image)
            if not url:
                continue
            if image.get("typeName") in {"mainImage", "cardImage"}:
                return url
            if preferred is None:
                preferred = url
        if preferred:
            return preferred
    return cover_url_from_image(item.get("image"))


def _reviews_bucket(data: Any, component_name: str, category: str = "default") -> list[dict[str, Any]]:
    """Берёт первый компонент и только указанную категорию (обычно default)."""
    component = find_component(data, component_name)
    if not component or not isinstance(component.get("data"), dict):
        return []
    item = component["data"].get("item")
    if not isinstance(item, dict):
        return []
    bucket = item.get(category)
    if not isinstance(bucket, list):
        return []
    return [review for review in bucket if isinstance(review, dict)]


def _dedupe_reviews(reviews: list[CriticReview] | list[UserReview], *, limit: int = 10):
    """Убирает дубликаты по (quote, score) и обрезает до limit."""
    unique = []
    seen: set[tuple[str | None, float | None]] = set()
    for review in reviews:
        key = (review.quote, review.score)
        if key in seen:
            continue
        seen.add(key)
        unique.append(review)
        if len(unique) >= limit:
            break
    return unique


def _component_item(data: Any, name: str) -> dict[str, Any] | None:
    component = find_component(data, name)
    if not component or "data" not in component:
        return None
    payload = component["data"]
    if not isinstance(payload, dict) or "item" not in payload:
        return None
    item = payload["item"]
    return item if isinstance(item, dict) else None


def parse_game_card(html: str, slug: str, url: str) -> GameCard | None:
    """Парсит карточку игры из ``__NUXT_DATA__``."""
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NUXT_DATA__")
    if not tag or not tag.string:
        return None

    try:
        payload = json.loads(tag.string)
        data = rehydrate_nuxt(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    product = find_component(data, "product")
    if not product or "data" not in product or not isinstance(product["data"], dict):
        return None
    item = product["data"].get("item")
    if not isinstance(item, dict):
        return None

    platforms: list[PlatformInfo] = []
    raw_platforms = item.get("platforms")
    if isinstance(raw_platforms, list):
        for plat in raw_platforms:
            if not isinstance(plat, dict) or not plat.get("name"):
                continue
            critic = plat.get("criticScoreSummary") if isinstance(plat.get("criticScoreSummary"), dict) else {}
            user = plat.get("userScore") if isinstance(plat.get("userScore"), dict) else {}
            platforms.append(
                PlatformInfo(
                    name=str(plat["name"]),
                    slug=_as_str(plat.get("slug")),
                    release_date=plat.get("releaseDate") if isinstance(plat.get("releaseDate"), str) else None,
                    metascore=_as_int(critic.get("score")),
                    userscore=_as_float(user.get("score")),
                    critic_count=_as_int(critic.get("reviewCount")),
                    userscore_count=_as_int(user.get("reviewCount")),
                )
            )

    critic_reviews: list[CriticReview] = []
    for rev in _reviews_bucket(data, "critic-reviews", "default"):
        critic_reviews.append(
            CriticReview(
                publication=_as_str(rev.get("publicationName")),
                author=_as_str(rev.get("author")),
                score=_as_float(rev.get("score")),
                quote=_as_str(rev.get("quote")) or _as_str(rev.get("snippet")),
                date=_as_str(rev.get("date")),
                url=_as_str(rev.get("url")),
                platform=_as_str(rev.get("platform")),
            )
        )

    user_reviews: list[UserReview] = []
    for rev in _reviews_bucket(data, "user-reviews", "default"):
        user_reviews.append(
            UserReview(
                author=_as_str(rev.get("author")),
                score=_as_float(rev.get("score")),
                quote=_as_str(rev.get("quote")) or _as_str(rev.get("text")),
                date=_as_str(rev.get("date")),
                platform=_as_str(rev.get("platform")),
            )
        )

    product_critic = item.get("criticScoreSummary") if isinstance(item.get("criticScoreSummary"), dict) else {}
    product_user = item.get("userScore") if isinstance(item.get("userScore"), dict) else {}
    metascore = _as_int(product_critic.get("score"))
    userscore = _as_float(product_user.get("score"))
    metascore_count = _as_int(product_critic.get("reviewCount"))
    userscore_count = _as_int(product_user.get("reviewCount"))

    critic_summary = _component_item(data, "critic-score-summary")
    if critic_summary:
        metascore = _as_int(critic_summary.get("score")) if critic_summary.get("score") is not None else metascore
        metascore_count = _as_int(critic_summary.get("reviewCount"))

    user_summary = _component_item(data, "user-score-summary")
    if user_summary:
        userscore = _as_float(user_summary.get("score")) if user_summary.get("score") is not None else userscore
        userscore_count = _as_int(user_summary.get("reviewCount"))
        if userscore_count == 0:
            userscore = None

    video = item.get("video") if isinstance(item.get("video"), dict) else {}
    video_url = _as_str(video.get("url")) or _as_str(video.get("embedUrl"))
    video_title = _as_str(video.get("videoTitle")) or _as_str(video.get("title"))
    related_games: list[RelatedGame] = []
    related = find_component(data, "related-carousel")
    related_items = related["data"].get("items") if related and isinstance(related.get("data"), dict) else None
    if isinstance(related_items, list):
        for game in related_items[:10]:
            if not isinstance(game, dict):
                continue
            related_slug = game.get("slug") if isinstance(game.get("slug"), str) else None
            critic = game.get("criticScoreSummary") if isinstance(game.get("criticScoreSummary"), dict) else {}
            related_games.append(
                RelatedGame(
                    title=game.get("title") if isinstance(game.get("title"), str) else None,
                    slug=related_slug,
                    url=f"https://www.metacritic.com/game/{related_slug}/" if related_slug else None,
                    metascore=_as_int(critic.get("score")),
                    cover_url=cover_url_from_image(game.get("image")),
                )
            )

    card = GameCard(
        slug=item["slug"] if isinstance(item.get("slug"), str) and item["slug"] else slug,
        url=url,
        title=item.get("title") if isinstance(item.get("title"), str) else slug,
        description=item.get("description") if isinstance(item.get("description"), str) else None,
        release_date=item.get("releaseDate") if isinstance(item.get("releaseDate"), str) else None,
        premiere_year=_as_int(item.get("premiereYear")),
        developers=extract_developers(item),
        publishers=extract_publishers(item),
        cover_url=_cover_from_product(item),
        genres=_names(item.get("genres")),
        platforms=platforms,
        video_url=video_url,
        video_title=video_title,
        metascore=metascore,
        metascore_count=metascore_count,
        userscore=userscore,
        userscore_count=userscore_count,
        critic_reviews=_dedupe_reviews(critic_reviews, limit=10),
        user_reviews=_dedupe_reviews(user_reviews, limit=10),
        related_games=related_games,
    )
    if card.userscore_count in (None, 0):
        card.userscore = None
    return card


async def fetch_game_card(
    client: MetacriticClient,
    slug: str,
    *,
    settings: Settings | None = None,
) -> GameCard | None:
    """Скачивает и разбирает карточку игры по slug."""
    settings = settings or get_settings()
    page_url = f"{settings.metacritic_base_url.rstrip('/')}/game/{slug}/"
    html = await client.fetch_page(page_url)
    return parse_game_card(html, slug=slug, url=page_url)


async def fetch_platform_userscores(
    client: MetacriticClient,
    slug: str,
    platforms: list[PlatformInfo],
    *,
    settings: Settings | None = None,
) -> dict[str, dict]:
    """Userscore с /game/{slug}/user-reviews/?platform={plat_slug} (user-score-summary)."""
    settings = settings or get_settings()
    base = settings.metacritic_base_url.rstrip("/")
    result: dict[str, dict] = {}
    for platform in platforms:
        plat_slug = platform.slug
        if not plat_slug:
            continue
        url = f"{base}/game/{slug}/user-reviews/?platform={plat_slug}"
        try:
            html = await client.fetch_page(url)
            data = extract_nuxt_data(html)
        except Exception:
            logger.warning("Не удалось получить userscore %s / %s", slug, plat_slug)
            result[plat_slug] = {"userscore": None, "userscore_count": None}
            continue
        item = _component_item(data, "user-score-summary")
        count = _as_int(item.get("reviewCount")) if item else None
        score = _as_float(item.get("score")) if item else None
        if not item or score is None or count == 0:
            result[plat_slug] = {"userscore": None, "userscore_count": count}
            continue
        result[plat_slug] = {"userscore": score, "userscore_count": count}
    return result
