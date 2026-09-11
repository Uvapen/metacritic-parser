"""Похожие игры из своей БД: набор очков с порогом, чипы совпадений."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Iterable

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Game, SimilarGameLink, SimilarVerdict

logger = logging.getLogger(__name__)

SIMILAR_LIMIT = 10
SCORE_DEVELOPER = 3
SCORE_GENRE = 0.5
SCORE_TAG = 1
SCORE_CONFIDENT_TAG = 2
SCORE_PLATFORM = 0.5
SCORE_AUDIENCE = 2
MAX_TAG_SCORE = 3
MIN_SHOW_SCORE = 3
CONFIDENT_SCORE = 5
MIXED_REVIEW_LIMIT = 5
FALLBACK_CATALOG_LIMIT = 50
FALLBACK_MIN_SCORE = MIN_SHOW_SCORE
VIA_CONFIDENT = "confident"
VIA_LLM = "llm"
VIA_HEURISTIC = "heuristic"
SOURCE_RANK = {VIA_HEURISTIC: 0, VIA_CONFIDENT: 1, VIA_LLM: 2}
LLM_BLOCK_STATUSES = {401, 403, 429}

# Слишком широкие теги: сами по себе не означают сходство.
GENERIC_TAG_COMPACT = {
    "2d",
    "3d",
    "action",
    "actionadventure",
    "adventure",
    "atmospheric",
    "boardgame",
    "casual",
    "cinematic",
    "coop",
    "customization",
    "exploration",
    "graphics",
    "handheld",
    "horror",
    "indie",
    "management",
    "multiplayer",
    "openworld",
    "pixelart",
    "port",
    "puzzle",
    "relaxing",
    "remaster",
    "sandbox",
    "singleplayer",
    "stealth",
    "storyrich",
    "survival",
    "voiceacting",
}
GENERIC_GENRE_COMPACT = {
    "action",
    "actionadventure",
    "adventure",
    "arcade",
    "casual",
    "family",
    "fighting",
    "indie",
    "mmo",
    "music",
    "party",
    "puzzle",
    "racing",
    "roleplaying",
    "rpg",
    "shooter",
    "simulation",
    "sports",
    "strategy",
}
# Узкий тег: 2 очка, можно пройти порог вместе с жанром и платформой.
# tactical-rpg специально не здесь: слишком широкий (SRPG и инди-тактики).
CONFIDENT_TAG_COMPACT = {
    "metroidvania",
    "soulslike",
}
_AUDIENCE_STOP = {
    "также",
    "для",
    "игра",
    "играм",
    "играми",
    "игры",
    "игр",
    "игроки",
    "любители",
    "оценит",
    "серии",
    "фанаты",
    "аркадных",
    "аркадные",
    "время",
    "времени",
    "геймплея",
    "динамичных",
    "динамичные",
    "казуальных",
    "казуальные",
    "казуальной",
    "кооператива",
    "кооперативных",
    "лёгких",
    "легких",
    "механикам",
    "механиками",
    "мультиплеер",
    "мультиплеера",
    "музыкальных",
    "оригинального",
    "оригинальной",
    "приключений",
    "приключения",
    "простых",
    "реальном",
    "симулятора",
    "симуляторов",
    "симуляторы",
    "стратегий",
    "стратегии",
    "стратегических",
    "стратегические",
    "стратегический",
    "тактических",
    "тактические",
    "тактический",
    "тактической",
    "тактическими",
    "тактическое",
    "элементами",
    "элементов",
}
# Семья из genre_detailed: разные семьи не считаем похожими.
_FAMILY_MARKERS = (
    ("puzzle", ("solitaire", "puzzleboard", "match3", "boardgame", "cardgame", "hiddenobject")),
    ("sport", ("sport", "basketball", "soccer", "football", "racing", "golf")),
    ("sim", ("vehiclesim", "managementsim", "managementsimulation", "tycoon", "careersimulator", "farmingsim")),
    ("horror", ("horror",)),
    ("platform", ("platform", "metroidvania")),
    ("rhythm", ("rhythm",)),
    ("strategy", ("tacticalrpg", "turnbased", "grandstrategy", "4x")),
    ("rpg", ("actionrpg", "jrpg", "crpg")),
    ("action", ("action", "souls", "shooter", "fps", "hackandslash")),
    ("survival", ("survival", "sandbox")),
)


def _norm_text(value: str | None) -> str:
    return " ".join(str(value or "").lower().replace("&", "and").split())


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value)


def developer_keys(game: Any) -> set[str]:
    names: list[str] = []
    developers = getattr(game, "developers", None) or []
    if isinstance(developers, list):
        names.extend(str(item) for item in developers if item)
    developer = getattr(game, "developer", None)
    if developer:
        names.extend(part.strip() for part in str(developer).split(",") if part.strip())
    keys: set[str] = set()
    for name in names:
        text = _norm_text(name)
        if not text:
            continue
        keys.add(text)
        compact = _compact(text)
        if compact:
            keys.add(compact)
    return keys


def developer_label(game: Any) -> str | None:
    developers = getattr(game, "developers", None) or []
    if isinstance(developers, list):
        names = [str(item).strip() for item in developers if str(item).strip()]
        if names:
            return ", ".join(names)
    developer = getattr(game, "developer", None)
    if developer and str(developer).strip():
        return str(developer).strip()
    return None


def platform_names(game: Any) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in getattr(game, "platform_scores", None) or []:
        name = str(getattr(item, "platform", None) or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _genre_label_keys(label: str) -> set[str]:
    text = _norm_text(label).replace("-", " ")
    compact = _compact(text)
    if not compact or compact in GENERIC_GENRE_COMPACT:
        return set()
    keys = {compact}
    if text:
        keys.add(text)
    return keys


def genre_family(game: Any) -> set[str]:
    """Семья по уточнённому жанру LLM, не по ярлыку Metacritic."""
    compact = _compact(_norm_text(str(getattr(game, "genre_detailed", None) or "")))
    if not compact:
        return set()
    families: set[str] = set()
    for family, markers in _FAMILY_MARKERS:
        if any(marker in compact for marker in markers):
            families.add(family)
    return families


def families_conflict(origin: Any, other: Any) -> bool:
    left = genre_family(origin)
    right = genre_family(other)
    return bool(left and right and left.isdisjoint(right))


def genre_keys(game: Any) -> set[str]:
    """Только уточнённый жанр от LLM. Ярлык Metacritic слишком грубый."""
    detailed = getattr(game, "genre_detailed", None)
    if not detailed:
        return set()
    return _genre_label_keys(str(detailed))


def _tag_values(game: Any) -> list[str]:
    return list(getattr(game, "tags", None) or getattr(game, "ai_tags", None) or [])


def tag_keys(game: Any) -> set[str]:
    keys: set[str] = set()
    for item in _tag_values(game):
        text = _norm_text(str(item)).replace(" ", "-")
        compact = _compact(text)
        if compact and compact not in GENERIC_TAG_COMPACT:
            keys.add(compact)
    return keys


def _tag_label_map(game: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in _tag_values(game):
        raw = str(item).strip()
        if not raw:
            continue
        compact = _compact(_norm_text(raw).replace(" ", "-"))
        if compact and compact not in GENERIC_TAG_COMPACT and compact not in mapping:
            mapping[compact] = raw
    return mapping


def shared_tag_labels(origin: Any, other: Any) -> list[str]:
    left = _tag_label_map(origin)
    right = _tag_label_map(other)
    return [left[key] for key in sorted(set(left) & set(right))]


def feature_keys(game: Any) -> set[str]:
    keys: set[str] = set()
    for item in getattr(game, "key_features", None) or []:
        text = _norm_text(str(item))
        if len(text) >= 3:
            keys.add(text)
    return keys


def audience_tokens(game: Any) -> set[str]:
    raw = _norm_text(str(getattr(game, "target_audience", None) or ""))
    tokens: set[str] = set()
    for part in re.split(r"[^a-z0-9а-яё]+", raw):
        if len(part) >= 5 and part not in _AUDIENCE_STOP:
            tokens.add(part)
    return tokens


def audiences_close(origin: Any, other: Any) -> bool:
    overlap = audience_tokens(origin) & audience_tokens(other)
    if not overlap:
        return False
    if len(overlap) >= 2:
        return True
    return any(len(token) >= 7 for token in overlap)


def has_similar_profile(game: Any) -> bool:
    """В похожие не берём карточки без LLM-профиля (теги, жанр или особенности)."""
    return bool(tag_keys(game) or genre_keys(game) or feature_keys(game))


def _genre_chip(origin: Any, other: Any) -> str | None:
    shared = genre_keys(origin) & genre_keys(other)
    if not shared:
        return None
    for game in (origin, other):
        label = str(getattr(game, "genre_detailed", None) or "").strip()
        if label and _compact(_norm_text(label)) in {_compact(item) for item in shared}:
            return label
    return next(iter(shared), None)


def _as_score(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _tag_weight(compact_tag: str) -> float:
    if compact_tag in CONFIDENT_TAG_COMPACT:
        return SCORE_CONFIDENT_TAG
    return SCORE_TAG


def compare_similarity(origin: Any, other: Any) -> tuple[float, list[str], list[str]]:
    """Очки: студия 3, узкий тег 2, обычный тег 1 (макс 3), аудитория 2, жанр 0.5, платформа 0.5.

    Широкие ярлыки вроде action-adventure не считаются. Порог показа — отдельно.
    """
    if families_conflict(origin, other) and not (developer_keys(origin) & developer_keys(other)):
        return 0.0, [], []

    score = 0.0
    reasons: list[str] = []
    weighted_chips: list[tuple[int, int, str]] = []
    order = 0

    def add_chip(weight: int, label: str) -> None:
        nonlocal order
        text = str(label or "").strip()
        if not text:
            return
        order += 1
        weighted_chips.append((-weight, order, text))

    if developer_keys(origin) & developer_keys(other):
        score += SCORE_DEVELOPER
        reasons.append("разработчик")
        add_chip(100, developer_label(other) or "студия")

    shared_tags = tag_keys(origin) & tag_keys(other)
    if shared_tags:
        tag_score = sum(_tag_weight(item) for item in shared_tags)
        tag_score = min(tag_score, float(MAX_TAG_SCORE))
        score += tag_score
        tag_n = min(len(shared_tags), 3)
        reasons.append("теги" if tag_n == 1 else f"теги ×{tag_n}")
        labels = shared_tag_labels(origin, other)
        strong = []
        regular = []
        for label in labels:
            if _compact(_norm_text(label).replace(" ", "-")) in CONFIDENT_TAG_COMPACT:
                strong.append(label)
            else:
                regular.append(label)
        for label in strong:
            add_chip(80, label)
        for label in regular:
            add_chip(50, label)

    genre_chip = _genre_chip(origin, other)
    genre_compact = _compact(_norm_text(genre_chip)) if genre_chip else ""
    if genre_chip and genre_compact not in shared_tags:
        score += SCORE_GENRE
        reasons.append("жанр")
        add_chip(30, genre_chip)

    # «Для кого» — бонус, не самостоятельный путь в блок. Иначе NBA + симулятор
    # набирают аудитория 2 + платформа 0.5 и проходят порог.
    if audiences_close(origin, other) and score > 0:
        score += SCORE_AUDIENCE
        reasons.append("аудитория")

    origin_platforms = platform_names(origin)
    other_platforms = platform_names(other)
    shared_platforms = [name for name in other_platforms if name in origin_platforms]
    if shared_platforms:
        score += SCORE_PLATFORM
        reasons.append("платформа")
        add_chip(10, shared_platforms[0])

    chips = [label for _weight, _order, label in sorted(weighted_chips)]
    if score <= 0:
        return 0.0, [], []
    return score, reasons, chips


def similarity_score(origin: Any, other: Any) -> tuple[float, list[str]]:
    score, reasons, _chips = compare_similarity(origin, other)
    return score, reasons


def _hit_dict(origin: Any, other: Any, score: float, reasons: list[str], chips: list[str] | None = None) -> dict[str, Any]:
    origin_platforms = platform_names(origin)
    other_platforms = platform_names(other)
    shared_platforms = [name for name in other_platforms if name in origin_platforms]
    if chips is None:
        _score, _reasons, chips = compare_similarity(origin, other)
        score = _score
        reasons = _reasons
    return {
        "id": getattr(other, "id", None),
        "slug": other.slug,
        "title": other.title or other.slug,
        "cover_url": getattr(other, "cover_url", None),
        "metascore": getattr(other, "metascore", None),
        "is_from_carousel": bool(getattr(other, "is_from_carousel", False)),
        "score": score,
        "reasons": reasons,
        "chips": chips,
        "developer": developer_label(other),
        "platforms": other_platforms,
        "shared_platforms": shared_platforms,
        "shared_tags": shared_tag_labels(origin, other),
        "genre_detailed": getattr(other, "genre_detailed", None),
    }


def rank_similar(origin: Any, others: Iterable[Any], *, limit: int = SIMILAR_LIMIT) -> list[dict[str, Any]]:
    """Топ-10 похожих: перебор каталога, только кто набрал порог."""
    ranked: list[tuple[float, str, Any, list[str], list[str]]] = []
    origin_id = getattr(origin, "id", None)
    origin_slug = getattr(origin, "slug", None)
    for other in others:
        if origin_id is not None and getattr(other, "id", None) == origin_id:
            continue
        if origin_slug and getattr(other, "slug", None) == origin_slug:
            continue
        if not has_similar_profile(other):
            continue
        score, reasons, chips = compare_similarity(origin, other)
        if score < MIN_SHOW_SCORE:
            continue
        ranked.append((score, getattr(other, "title", "") or "", other, reasons, chips))
    ranked.sort(key=lambda item: (-item[0], item[1].lower()))
    return [
        _hit_dict(origin, other, score, reasons, chips)
        for score, _title, other, reasons, chips in ranked[:limit]
    ]


def is_confident_hit(hit: dict[str, Any]) -> bool:
    """Сильный набор очков, общая студия или узкий тег при пороге."""
    if _as_score(hit.get("score")) >= CONFIDENT_SCORE:
        return True
    if "разработчик" in (hit.get("reasons") or []):
        return True
    shared = {_compact(str(item)) for item in (hit.get("shared_tags") or [])}
    return bool(shared & CONFIDENT_TAG_COMPACT) and _as_score(hit.get("score")) >= MIN_SHOW_SCORE


def heuristic_keep_mixed(hit: dict[str, Any], *, catalog_size: int) -> bool:
    """Показать смешанных без LLM: в маленьком каталоге от порога, в большом — крепче."""
    score = _as_score(hit.get("score"))
    if score < FALLBACK_MIN_SCORE:
        return False
    if catalog_size < FALLBACK_CATALOG_LIMIT:
        return True
    return score >= CONFIDENT_SCORE - 1


def apply_mixed_filter(
    hits: list[dict[str, Any]],
    verdicts: dict[str, bool],
    *,
    catalog_size: int = 0,
) -> list[dict[str, Any]]:
    """Уверенные сразу; смешанные — LLM, иначе эвристика; keep=False скрывает."""
    kept: list[dict[str, Any]] = []
    for hit in hits:
        payload = dict(hit)
        if is_confident_hit(hit):
            payload["via"] = VIA_CONFIDENT
            kept.append(payload)
            continue
        slug = str(hit.get("slug") or "")
        if slug in verdicts:
            if verdicts[slug] is True:
                payload["via"] = VIA_LLM
                kept.append(payload)
            continue
        if heuristic_keep_mixed(hit, catalog_size=catalog_size):
            payload["via"] = VIA_HEURISTIC
            kept.append(payload)
    return kept


def profile_fingerprint(game: Any) -> str:
    payload = {
        "g": getattr(game, "genre_detailed", None),
        "t": list(getattr(game, "tags", None) or getattr(game, "ai_tags", None) or []),
        "f": list(getattr(game, "key_features", None) or []),
        "a": getattr(game, "target_audience", None),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def pair_fingerprint(origin: Any, other: Any) -> str:
    """Не зависит от направления пары: A↔B и B↔A — один отпечаток."""
    fps = sorted((profile_fingerprint(origin), profile_fingerprint(other)))
    return hashlib.sha1(f"v3|{'|'.join(fps)}".encode("utf-8")).hexdigest()[:24]


def _brief_game(game: Any) -> str:
    tags = ", ".join(str(item) for item in (getattr(game, "tags", None) or [])[:8]) or "—"
    features = "; ".join(str(item) for item in (getattr(game, "key_features", None) or [])[:4]) or "—"
    description = " ".join(str(getattr(game, "description", None) or "").split())[:240]
    return (
        f"{getattr(game, 'title', None) or getattr(game, 'slug', '')} "
        f"[жанр: {getattr(game, 'genre_detailed', None) or '—'}; "
        f"теги: {tags}; особенности: {features}] "
        f"{description}"
    )


def _ordered_ids(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


async def _load_games(session: AsyncSession) -> list[Game]:
    result = await session.execute(select(Game).options(selectinload(Game.platform_scores)))
    return list(result.scalars().all())


async def _load_verdicts(
    session: AsyncSession,
    origin: Game,
    others: dict[str, Game],
    slugs: list[str],
) -> dict[str, bool]:
    if not slugs:
        return {}
    result = await session.execute(
        select(SimilarVerdict).where(
            SimilarVerdict.origin_slug == origin.slug,
            SimilarVerdict.other_slug.in_(slugs),
        )
    )
    verdicts: dict[str, bool] = {}
    for row in result.scalars():
        other = others.get(row.other_slug)
        if other is None:
            continue
        if row.fingerprint != pair_fingerprint(origin, other):
            continue
        verdicts[row.other_slug] = bool(row.keep)
    return verdicts


async def _upsert_verdict(
    session: AsyncSession,
    *,
    origin_slug: str,
    other_slug: str,
    fingerprint: str,
    keep: bool,
) -> None:
    row = await session.scalar(
        select(SimilarVerdict).where(
            SimilarVerdict.origin_slug == origin_slug,
            SimilarVerdict.other_slug == other_slug,
        )
    )
    if row is None:
        session.add(
            SimilarVerdict(
                origin_slug=origin_slug,
                other_slug=other_slug,
                fingerprint=fingerprint,
                keep=keep,
            )
        )
        return
    row.fingerprint = fingerprint
    row.keep = keep


async def _upsert_verdict_pair(
    session: AsyncSession,
    origin: Game,
    other: Game,
    *,
    keep: bool,
) -> None:
    fingerprint = pair_fingerprint(origin, other)
    await _upsert_verdict(
        session,
        origin_slug=origin.slug,
        other_slug=other.slug,
        fingerprint=fingerprint,
        keep=keep,
    )
    await _upsert_verdict(
        session,
        origin_slug=other.slug,
        other_slug=origin.slug,
        fingerprint=fingerprint,
        keep=keep,
    )


async def _upsert_link(
    session: AsyncSession,
    origin: Game,
    other: Game,
    *,
    score: float,
    source: str,
) -> None:
    origin_id = getattr(origin, "id", None)
    other_id = getattr(other, "id", None)
    if origin_id is None or other_id is None or origin_id == other_id:
        return
    low, high = _ordered_ids(int(origin_id), int(other_id))
    fingerprint = pair_fingerprint(origin, other)
    row = await session.scalar(
        select(SimilarGameLink).where(
            SimilarGameLink.game_a_id == low,
            SimilarGameLink.game_b_id == high,
        )
    )
    if row is None:
        session.add(
            SimilarGameLink(
                game_a_id=low,
                game_b_id=high,
                score=float(score),
                fingerprint=fingerprint,
                source=source,
            )
        )
        return
    if row.fingerprint == fingerprint and SOURCE_RANK.get(row.source or "", -1) > SOURCE_RANK.get(source, -1):
        return
    row.score = float(score)
    row.fingerprint = fingerprint
    row.source = source


async def _drop_link(session: AsyncSession, origin: Game, other: Game) -> None:
    origin_id = getattr(origin, "id", None)
    other_id = getattr(other, "id", None)
    if origin_id is None or other_id is None:
        return
    low, high = _ordered_ids(int(origin_id), int(other_id))
    row = await session.scalar(
        select(SimilarGameLink).where(
            SimilarGameLink.game_a_id == low,
            SimilarGameLink.game_b_id == high,
        )
    )
    if row is not None:
        await session.delete(row)


async def _persist_kept_pairs(
    session: AsyncSession,
    origin: Game,
    by_slug: dict[str, Game],
    kept: list[dict[str, Any]],
) -> None:
    for hit in kept:
        other = by_slug.get(str(hit.get("slug") or ""))
        if other is None:
            continue
        await _upsert_link(
            session,
            origin,
            other,
            score=_as_score(hit.get("score")),
            source=str(hit.get("via") or VIA_HEURISTIC),
        )


async def _linked_hits(
    session: AsyncSession,
    origin: Game,
    by_id: dict[int, Game],
    already: set[str],
) -> list[dict[str, Any]]:
    origin_id = getattr(origin, "id", None)
    if origin_id is None:
        return []
    result = await session.execute(
        select(SimilarGameLink).where(
            or_(
                SimilarGameLink.game_a_id == origin_id,
                SimilarGameLink.game_b_id == origin_id,
            )
        )
    )
    extra: list[dict[str, Any]] = []
    for row in result.scalars():
        other_id = row.game_b_id if row.game_a_id == origin_id else row.game_a_id
        other = by_id.get(other_id)
        if other is None or other.slug in already:
            continue
        if row.fingerprint and row.fingerprint != pair_fingerprint(origin, other):
            continue
        score, reasons, chips = compare_similarity(origin, other)
        if score < MIN_SHOW_SCORE:
            continue
        hit = _hit_dict(origin, other, score, reasons, chips)
        hit["via"] = row.source or VIA_HEURISTIC
        extra.append(hit)
    return extra


async def review_mixed_similars(
    session: AsyncSession,
    llm: Any | None,
    *,
    slugs: list[str] | None = None,
) -> int:
    """Один короткий LLM-вызов на игру: только смешанные кандидаты без вердикта."""
    from app.llm.prompts import (
        SIMILAR_REVIEW_SYSTEM_PROMPT,
        build_similar_review_prompt,
        parse_similar_review,
    )

    games = await _load_games(session)
    by_slug = {game.slug: game for game in games}
    targets = [by_slug[slug] for slug in slugs or [] if slug in by_slug] or list(games)
    catalog_size = len(games)
    reviewed = 0
    llm_blocked = False
    for origin in targets:
        if not has_similar_profile(origin):
            continue
        hits = rank_similar(origin, games, limit=max(SIMILAR_LIMIT * 2, 12))
        mixed = [hit for hit in hits if not is_confident_hit(hit)][:MIXED_REVIEW_LIMIT]
        pending: list[tuple[dict[str, Any], Game]] = []
        cached = await _load_verdicts(session, origin, by_slug, [hit["slug"] for hit in mixed])
        for hit in mixed:
            if hit["slug"] in cached:
                continue
            other = by_slug.get(hit["slug"])
            if other is not None:
                pending.append((hit, other))
        if llm is not None and pending and not llm_blocked:
            prompt = build_similar_review_prompt(
                origin_brief=_brief_game(origin),
                candidates=[
                    {
                        "slug": other.slug,
                        "brief": _brief_game(other),
                        "score": hit["score"],
                        "reasons": hit.get("reasons") or [],
                    }
                    for hit, other in pending
                ],
            )
            result = await llm.complete(
                prompt,
                system=SIMILAR_REVIEW_SYSTEM_PROMPT,
                slug=origin.slug,
                kind="similar",
                json_object=True,
            )
            blocked = bool(result.stub or result.error or not result.text)
            if blocked:
                logger.warning("LLM-проверка похожих не удалась для %s: %s", origin.slug, result.error)
                if result.error_status in LLM_BLOCK_STATUSES:
                    llm_blocked = True
            else:
                keep_slugs = parse_similar_review(result.text)
                pending_slugs = {other.slug for _hit, other in pending}
                for hit, other in pending:
                    keep = other.slug in keep_slugs
                    await _upsert_verdict_pair(session, origin, other, keep=keep)
                    if not keep:
                        await _drop_link(session, origin, other)
                extra_keep = keep_slugs - pending_slugs
                if extra_keep:
                    logger.info("LLM keep вне кандидатов %s: %s", origin.slug, sorted(extra_keep))
                reviewed += 1
        verdicts = await _load_verdicts(session, origin, by_slug, [hit["slug"] for hit in hits])
        kept = apply_mixed_filter(hits, verdicts, catalog_size=catalog_size)
        await _persist_kept_pairs(session, origin, by_slug, kept)
        await session.commit()
    return reviewed


async def similar_payload(
    session: AsyncSession,
    game: Game,
    *,
    limit: int = SIMILAR_LIMIT,
) -> list[dict[str, Any]]:
    """On-the-fly топ похожих. Пары в БД пишет только пайплайн, не GET."""
    games = await _load_games(session)
    origin = next((item for item in games if item.id == game.id), game)
    by_slug = {item.slug: item for item in games if item.id != origin.id}
    by_id = {item.id: item for item in games if item.id != origin.id}
    hits = rank_similar(origin, games, limit=max(limit * 2, 12))
    mixed_slugs = [hit["slug"] for hit in hits if not is_confident_hit(hit)]
    verdicts = await _load_verdicts(session, origin, by_slug, mixed_slugs)
    kept = apply_mixed_filter(hits, verdicts, catalog_size=len(games))
    already = {str(hit.get("slug") or "") for hit in kept}
    extra = await _linked_hits(session, origin, by_id, already)
    merged = kept + extra
    merged.sort(key=lambda item: (-int(item.get("score") or 0), str(item.get("title") or "").lower()))
    return merged[:limit]
