"""Hourly-пайплайн: дневной курсор ленты new → smart/light/deep апдейт."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.db import SessionLocal, init_db
from app.timeutil import as_local_date, now_utc, today_local
from app.llm.client import LLMClient, groq_chat_blocked, groq_chat_retry_in
from app.llm.prompts import (
    CRITIC_SUMMARY_SYSTEM_PROMPT,
    SUMMARY_TEXT_SYSTEM_PROMPT,
    USER_SUMMARY_SYSTEM_PROMPT,
    build_critic_summary_prompt,
    build_user_summary_prompt,
    looks_russian,
    normalize_ai_tag,
    parse_summary_payload,
)
from app.models import Game, PipelineJob, PipelineState, PlatformScore, Review, RunLog, Summary
from app.scraper.card import GameCard, fetch_game_card, fetch_platform_userscores
from app.scraper.client import MetacriticClient
from app.scraper.lister import (
    ListedGame,
    fetch_browse_page,
    fetch_main_new_releases,
)
from app.scraper.nuxt import normalize_cover_url
from app.services.similar import review_mixed_similars
from app.services.youtube import needs_letsplay_job, process_letsplay_slug

logger = logging.getLogger(__name__)

_running = False
_active_card_run_id: int | None = None
_run_lock = asyncio.Lock()
_enriching = False
_enrich_lock = asyncio.Lock()
JOB_SIMILAR = "similar"
JOB_YOUTUBE = "youtube"
JOB_OPEN = ("pending", "running")
RUN_IN_FLIGHT = ("running", "enriching")

BATCH_SIZE = 20
BROWSE_PAGE_LIMIT = 20
LLM_MIN_REVIEWS = 3
LLM_INFO_KINDS: tuple[tuple[str, str], ...] = (
    ("critic", "критики"),
    ("user", "игроки"),
)
MAX_GAME_TAGS = 8
MAX_GAME_FEATURES = 6
STRUCTURED_EXTRACTED_MARK = "extracted"


def pipeline_is_running() -> bool:
    """True, если hourly-прогон уже выполняется."""
    return _running


def enrichment_is_running() -> bool:
    """True, пока тик или кнопка ест задачу из очереди."""
    return _enriching


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _as_local_date(value: datetime | date | None) -> date | None:
    return as_local_date(value)


def _processed_today(value: datetime | date | None, today: date) -> bool:
    return _as_local_date(value) == today


async def _write_run(
    *,
    status: str,
    finished_at: datetime | None = None,
    error_message: str | None = None,
) -> RunLog:
    async with SessionLocal() as session:
        run = RunLog(status=status, finished_at=finished_at, error_message=error_message)
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run


def _status_from_card_details(details: Any, games_processed: int) -> str:
    """Итог прогона по карточкам: очередь похожих/летсплеев на статус не влияет."""
    items = details if isinstance(details, list) else []
    game_errors = sum(
        1 for item in items if isinstance(item, dict) and item.get("action") == "error"
    )
    if game_errors and games_processed:
        return "partial"
    if game_errors:
        return "error"
    return "success"


async def _finish_run(
    run_id: int,
    *,
    status: str,
    games_found: int,
    games_processed: int,
    error_message: str | None = None,
    details: list[dict[str, Any]] | None = None,
    llm_errors: int = 0,
    complete: bool = True,
) -> RunLog:
    async with SessionLocal() as session:
        run = await session.get(RunLog, run_id)
        if run is None:
            run = RunLog(status=status)
            session.add(run)
        if complete:
            run.status = status
            run.finished_at = now_utc()
        else:
            run.status = status
            run.finished_at = None
        run.games_found = games_found
        run.games_processed = games_processed
        run.error_message = error_message
        run.details = details
        run.llm_errors = llm_errors
        await session.commit()
        await session.refresh(run)
        return run


async def _finalize_run_if_idle(run_id: int) -> RunLog | None:
    """Закрывает прогон только когда пуста очередь похожих/летсплеев именно этого run_id."""
    rid = int(run_id or 0)
    if not rid:
        return None
    if _active_card_run_id == rid:
        return None
    async with SessionLocal() as session:
        run = await session.get(RunLog, rid)
        if run is None or run.status not in RUN_IN_FLIGHT:
            return None
        open_n = await session.scalar(
            select(func.count())
            .select_from(PipelineJob)
            .where(
                PipelineJob.run_id == rid,
                PipelineJob.status.in_(JOB_OPEN),
            )
        )
        if int(open_n or 0):
            return None
        status = _status_from_card_details(run.details, int(run.games_processed or 0))
        run.status = status
        run.finished_at = now_utc()
        await session.commit()
        await session.refresh(run)
        logger.info("Прогон #%s закрыт: %s (очередь этого прогона пуста)", rid, status)
        return run


async def _load_state(session: AsyncSession, today: date) -> PipelineState:
    state = await session.get(PipelineState, 1)
    if state is None:
        state = PipelineState(id=1, day=today, offset=0, used_main=False, source="main")
        session.add(state)
        await session.flush()
    if state.day != today:
        state.day = today
        state.offset = 0
        state.used_main = False
        state.source = "main"
    elif not (state.source or "").strip():
        state.source = "browse" if state.used_main else "main"
    return state


async def _save_pipeline_state(today: date, *, source: str, offset: int) -> None:
    """Пишет курсор дня: после карусели всегда browse, дальше offset SEE ALL."""
    async with SessionLocal() as session:
        state = await session.get(PipelineState, 1)
        if state is None:
            state = PipelineState(id=1, day=today, offset=0, used_main=False, source="main")
            session.add(state)
        state.day = today
        state.source = source
        state.used_main = source != "main"
        state.offset = offset
        await session.commit()


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


async def load_hourly_due_at() -> datetime | None:
    """Когда должен сработать следующий hourly. Ручной запуск это не двигает."""
    async with SessionLocal() as session:
        state = await session.get(PipelineState, 1)
        if state is None:
            return None
        return _aware_utc(state.hourly_due_at)


async def save_hourly_due_at(when: datetime) -> None:
    due = _aware_utc(when)
    async with SessionLocal() as session:
        state = await session.get(PipelineState, 1)
        if state is None:
            state = PipelineState(
                id=1,
                day=today_local(),
                offset=0,
                used_main=False,
                source="main",
            )
            session.add(state)
        state.hourly_due_at = due
        await session.commit()


def choose_hourly_due(
    stored: datetime | None, *, now: datetime, interval: timedelta
) -> datetime:
    """Какой слот hourly поставить: сохранённый, просроченный или слишком далёкий."""
    if stored is None:
        return now + interval
    if stored <= now:
        return now + timedelta(seconds=30)
    if stored - now > interval * 2:
        return now + interval
    return stored


async def next_hourly_due_at(*, interval_hours: int, now: datetime | None = None) -> datetime:
    """Следующий hourly: сохранённый слот или now+интервал. Просроченный — скоро."""
    hours = max(1, int(interval_hours or 1))
    interval = timedelta(hours=hours)
    now = _aware_utc(now) or now_utc()
    stored = await load_hourly_due_at()
    due = choose_hourly_due(stored, now=now, interval=interval)
    await save_hourly_due_at(due)
    return due


async def bump_hourly_due_at(*, interval_hours: int) -> datetime:
    """Сдвиг слота после срабатывания планировщика. Ручной POST сюда не ходит."""
    hours = max(1, int(interval_hours or 1))
    due = now_utc() + timedelta(hours=hours)
    await save_hourly_due_at(due)
    return due


async def _game_index(session: AsyncSession) -> dict[str, Game]:
    result = await session.execute(select(Game))
    return {game.slug: game for game in result.scalars()}


def _norm_score(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _scores_changed(game: Game, listed: ListedGame) -> bool:
    """Deep если в ленте появился или изменился metascore."""
    if listed.metascore is None:
        return False
    if game.metascore is None:
        return True
    return listed.metascore != game.metascore


def _needs_llm_backfill(game: Game | None) -> bool:
    """Hourly должен дожать саммари, если прошлый вызов LLM не оставил fingerprint."""
    if game is None:
        return False
    return not game.summary_fingerprint_critic and not game.summary_fingerprint_user


def _worth_processing(existing: Game | None, item: ListedGame, today: date) -> bool:
    """В пачку 20: новые игры или те, кому реально нужно обновление. No-op skip слот не занимает."""
    if existing is None:
        return True
    if _processed_today(existing.last_processed_at, today):
        return False
    return _scores_changed(existing, item) or _needs_llm_backfill(existing)


def _fingerprint_reviews(reviews: list[Any]) -> str:
    """sha256 по сортированному списку строк score|text."""
    lines = []
    for item in reviews:
        text = (
            getattr(item, "quote", None)
            or getattr(item, "text", None)
            or getattr(item, "body", None)
            or ""
        )
        text = " ".join(str(text).split())
        lines.append(f"{_norm_score(getattr(item, 'score', None))}|{text}")
    lines.sort()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _fingerprint_llm_input(reviews: list[Any], description: str | None) -> str:
    """Отпечаток отзывов + описания: смена описания тоже пересобирает саммари."""
    review_fp = _fingerprint_reviews(reviews)
    desc = " ".join(str(description or "").split())
    return hashlib.sha256(f"{review_fp}\nDESC:{desc}".encode("utf-8")).hexdigest()


def _norm_list(value: list | None) -> list[str]:
    return sorted(str(item) for item in (value or []))


def _platform_sig_from_game(game: Game) -> list[tuple]:
    return sorted((item.platform, item.metascore, item.userscore) for item in game.platform_scores)


def _snippet(text: str | None, limit: int = 100) -> str:
    if not text:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "…"


def _join_names(value: list | None) -> str:
    return ", ".join(str(item) for item in (value or []) if item)


def _format_platforms_card(card: GameCard) -> str:
    parts: list[str] = []
    for platform in card.platforms:
        if platform.metascore is not None:
            parts.append(f"{platform.name} ({platform.metascore})")
        else:
            parts.append(platform.name)
    return ", ".join(parts)


def _empty_llm() -> dict[str, Any]:
    return {kind: {"status": "none"} for kind, _label in LLM_INFO_KINDS}


def _llm_kind(
    status: str,
    *,
    reason: str | None = None,
    log_id: int | None = None,
    log_ids: list[int] | None = None,
    error: str | None = None,
    error_status: int | None = None,
    attempt: int | None = None,
    extracted: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status}
    ids = [item for item in (log_ids or []) if item is not None]
    if ids:
        payload["log_ids"] = ids
        payload["log_id"] = ids[-1]
    elif log_id is not None:
        payload["log_id"] = log_id
    if reason:
        payload["reason"] = reason
    if error:
        payload["error"] = error
    if error_status is not None:
        payload["error_status"] = error_status
    if attempt is not None:
        payload["attempt"] = attempt
    if extracted is not None:
        payload["extracted"] = extracted
    return payload


def _detail_llm_errors(detail: dict[str, Any] | None) -> int:
    if not detail:
        return 0
    llm = detail.get("llm")
    if not isinstance(llm, dict):
        return 0
    count = 0
    for kind, _label in LLM_INFO_KINDS:
        item = llm.get(kind)
        if isinstance(item, dict) and item.get("status") == "error":
            count += 1
    return count


def _llm_generated(llm_info: dict[str, Any]) -> bool:
    return any(
        isinstance(llm_info.get(kind), dict) and llm_info[kind].get("status") == "generated"
        for kind, _label in LLM_INFO_KINDS
    )


def _llm_reason_summary(llm: dict[str, Any] | None) -> str:
    if not isinstance(llm, dict):
        return ""
    parts: list[str] = []
    for kind, label in LLM_INFO_KINDS:
        info = llm.get(kind)
        if not isinstance(info, dict):
            continue
        status = info.get("status")
        reason = str(info.get("reason") or "").strip()
        if status == "skipped":
            parts.append(f"{label}: {reason or 'без изменений'}")
        elif status == "none" and reason:
            parts.append(f"{label}: {reason}")
        elif status == "error":
            parts.append(f"{label}: {reason or 'ошибка саммари'}")
        elif status == "generated":
            parts.append(f"{label}: {reason or 'сгенерировано'}")
    return "; ".join(parts)


def build_status_reason(
    action: str,
    *,
    changes: list | None = None,
    llm: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    llm_text = _llm_reason_summary(llm)
    if action == "error":
        return error or "ошибка обработки"
    if action == "new":
        return f"новая карточка · {llm_text}" if llm_text else "новая карточка"
    if action == "updated":
        has_changes = bool(changes)
        generated = _llm_generated(llm or {})
        if has_changes and generated:
            return f"изменились поля и саммари · {llm_text}" if llm_text else "изменились поля и саммари"
        if has_changes:
            return f"изменились поля карточки · {llm_text}" if llm_text else "изменились поля карточки"
        if generated:
            return llm_text or "сгенерировано саммари"
        return llm_text or "обновлена"
    if llm_text:
        return f"карточка без изменений · {llm_text}"
    return "карточка без изменений"


def _run_detail(
    *,
    slug: str,
    title: str,
    action: str,
    changes: list | None = None,
    error: str | None = None,
    llm: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    llm_payload = llm or _empty_llm()
    change_items = changes or []
    return {
        "slug": slug,
        "title": title,
        "action": action,
        "changes": change_items,
        "error": error,
        "llm": llm_payload,
        "reason": reason
        or build_status_reason(action, changes=change_items, llm=llm_payload, error=error),
    }


def _light_change_items(game: Game, listed: ListedGame) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if listed.metascore is not None and game.metascore is None:
        items.append({"field": "metascore", "text": str(listed.metascore), "kind": "added"})
    if listed.userscore is not None and game.userscore is None:
        items.append({"field": "userscore", "text": str(listed.userscore), "kind": "added"})
    if listed.cover_url and not game.cover_url:
        items.append({"field": "cover_url", "kind": "added"})
    return items


def _new_change_items(game: Game, card: GameCard) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if game.description:
        items.append({"field": "description", "text": _snippet(game.description), "kind": "added"})
    if game.genres:
        items.append({"field": "genres", "text": _join_names(game.genres)})
    if game.developers:
        items.append({"field": "developers", "text": _join_names(game.developers)})
    platforms = _format_platforms_card(card)
    if platforms:
        items.append({"field": "platforms", "text": platforms})
    if game.cover_url:
        items.append({"field": "cover_url", "kind": "added"})
    if game.metascore is not None:
        items.append({"field": "metascore", "text": str(game.metascore), "kind": "added"})
    if game.userscore is not None:
        items.append({"field": "userscore", "text": str(game.userscore), "kind": "added"})
    if card.critic_reviews or card.user_reviews:
        items.append({"field": "reviews", "kind": "added"})
    return items


def _deep_change_items(
    before: dict[str, Any],
    game: Game,
    card: GameCard,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if before["description"] != game.description:
        kind = "added" if not before["description"] and game.description else "updated"
        items.append({"field": "description", "text": _snippet(game.description), "kind": kind})
    if before["genres"] != _norm_list(game.genres):
        items.append({"field": "genres", "text": _join_names(game.genres)})
    if before["developers"] != _norm_list(game.developers):
        items.append({"field": "developers", "text": _join_names(game.developers)})
    if before["platforms"] != _platform_sig_from_game(game):
        items.append({"field": "platforms", "text": _format_platforms_card(card)})
    if before["cover_url"] != game.cover_url:
        kind = "added" if not before["cover_url"] and game.cover_url else "updated"
        items.append({"field": "cover_url", "kind": kind})
    if before["metascore"] != game.metascore:
        kind = "added" if before["metascore"] is None else "updated"
        items.append(
            {
                "field": "metascore",
                "text": "" if game.metascore is None else str(game.metascore),
                "kind": kind,
            }
        )
    if before["userscore"] != game.userscore:
        kind = "added" if before["userscore"] is None else "updated"
        items.append(
            {
                "field": "userscore",
                "text": "" if game.userscore is None else str(game.userscore),
                "kind": kind,
            }
        )
    new_fp_c = _fingerprint_llm_input(card.critic_reviews, game.description)
    new_fp_u = _fingerprint_llm_input(card.user_reviews, game.description)
    reviews_changed = (
        (before["fp_c"] is not None and before["fp_c"] != new_fp_c)
        or (before["fp_u"] is not None and before["fp_u"] != new_fp_u)
    )
    if reviews_changed and before["description"] == game.description:
        items.append({"field": "reviews", "kind": "updated"})
    return items


async def _collect_candidates(
    client: MetacriticClient,
    *,
    settings: Settings,
    today: date,
    start_offset: int,
    source: str,
    index: dict[str, Game],
    ignore_processed_today: bool = False,
) -> tuple[list[ListedGame], int, str, list[int | None]]:
    """Первый прогон дня — карусель, незаполненное добираем из SEE ALL. До 20 игр."""
    selected: list[ListedGame] = []
    positions: list[int | None] = []
    seen: set[str] = set()

    def _accept(item: ListedGame) -> bool:
        if item.slug in seen:
            return False
        existing = index.get(item.slug)
        if not ignore_processed_today and not _worth_processing(existing, item, today):
            return False
        seen.add(item.slug)
        selected.append(item)
        return True

    if source == "main":
        home = await fetch_main_new_releases(client, settings=settings)
        logger.info("New Releases карусель: %s игр", len(home))
        for item in home:
            item.from_carousel = True
            if _accept(item):
                positions.append(None)
            if len(selected) >= BATCH_SIZE:
                break
        if len(selected) >= BATCH_SIZE:
            return selected, len(selected), "browse", [None] * len(selected)
        source = "browse"
        start_offset = 0
        logger.info(
            "Карусель дала %s из %s — добираем новые из SEE ALL",
            len(selected),
            BATCH_SIZE,
        )

    first_page = await fetch_browse_page(client, 1, settings=settings)
    page_size = len(first_page) or 24
    start_page = start_offset // page_size + 1
    abs_index = (start_page - 1) * page_size
    new_offset = start_offset
    pages_fetched = 0
    page = start_page
    while pages_fetched < BROWSE_PAGE_LIMIT and len(selected) < BATCH_SIZE:
        items = first_page if page == 1 else await fetch_browse_page(client, page, settings=settings)
        if page == 1:
            first_page = []
        pages_fetched += 1
        if not items:
            break
        start_in_page = start_offset % page_size if page == start_page else 0
        for i, item in enumerate(items):
            if page == start_page and i < start_in_page:
                abs_index += 1
                continue
            if abs_index < start_offset:
                abs_index += 1
                continue
            pos = abs_index
            new_offset = abs_index + 1
            abs_index += 1
            if _accept(item):
                positions.append(pos)
            if len(selected) >= BATCH_SIZE:
                break
        page += 1

    return selected, new_offset, "browse", positions


def _apply_carousel_flag(game: Game, from_carousel: bool) -> None:
    """Карусель помечает игру навсегда; SEE ALL не снимает флаг."""
    if from_carousel:
        game.is_from_carousel = True
    elif not game.is_from_carousel:
        game.is_from_carousel = False


async def _light_update(session: AsyncSession, game: Game, listed: ListedGame) -> Game:
    if listed.metascore is not None and game.metascore is None:
        game.metascore = listed.metascore
    if listed.userscore is not None and game.userscore is None:
        game.userscore = listed.userscore
    if listed.critic_count is not None and game.critic_count is None:
        game.critic_count = listed.critic_count
    if listed.cover_url and not game.cover_url:
        game.cover_url = normalize_cover_url(listed.cover_url)
    game.last_processed_at = now_utc()
    await session.flush()
    return game


async def _upsert_game(
    session: AsyncSession,
    card: GameCard,
    listed: ListedGame | None = None,
    platform_userscores: dict[str, dict] | None = None,
) -> Game:
    slug = card.slug or (listed.slug if listed else "")
    game = await session.scalar(select(Game).where(Game.slug == slug))
    if game is None and listed is not None and listed.slug != slug:
        game = await session.scalar(select(Game).where(Game.slug == listed.slug))
    if game is None:
        try:
            async with session.begin_nested():
                game = Game(slug=slug, title=card.title or slug)
                session.add(game)
                await session.flush()
        except IntegrityError:
            game = await session.scalar(select(Game).where(Game.slug == slug))
            if game is None:
                raise

    developers = list(card.developers)
    genres = list(card.genres) or (list(listed.genres) if listed else [])

    cover_url = card.cover_url or (listed.cover_url if listed else None)
    description = card.description or (listed.description if listed else None)
    metascore = card.metascore if card.metascore is not None else (listed.metascore if listed else None)
    userscore = card.userscore if card.userscore is not None else (listed.userscore if listed else None)
    release_date = card.release_date or (listed.release_date if listed else None)
    critic_count = card.metascore_count
    if critic_count is None and listed is not None:
        critic_count = listed.critic_count

    if userscore == 0:
        userscore = None

    game.title = card.title or (listed.title if listed else card.slug)
    game.cover_url = normalize_cover_url(cover_url)
    game.description = description
    game.developers = developers or None
    game.developer = ", ".join(developers) if developers else None
    game.publishers = list(card.publishers) or None
    game.genres = genres or None
    game.video_url = card.video_url
    game.video_title = card.video_title
    game.related_games = [
        {
            "title": item.title,
            "slug": item.slug,
            "url": item.url,
            "metascore": item.metascore,
            "cover_url": normalize_cover_url(item.cover_url),
        }
        for item in card.related_games
        if item.slug
    ] or None
    game.release_date = _parse_date(release_date)
    game.metascore = metascore
    game.userscore = userscore
    game.critic_count = critic_count
    game.last_processed_at = now_utc()

    await session.execute(delete(PlatformScore).where(PlatformScore.game_id == game.id))
    await session.execute(delete(Review).where(Review.game_id == game.id))

    extras = platform_userscores or {}
    for platform in card.platforms:
        extra = extras.get(platform.slug or "") if platform.slug else None
        if extra is not None:
            plat_userscore = extra.get("userscore")
            plat_userscore_count = extra.get("userscore_count")
        else:
            plat_userscore = platform.userscore
            plat_userscore_count = platform.userscore_count
        if plat_userscore_count == 0 or plat_userscore == 0:
            plat_userscore = None
        session.add(
            PlatformScore(
                game_id=game.id,
                platform=platform.name,
                metascore=platform.metascore,
                userscore=plat_userscore,
                userscore_count=plat_userscore_count,
                critic_count=platform.critic_count,
                release_date=_parse_date(platform.release_date),
            )
        )

    for review in card.critic_reviews:
        session.add(
            Review(
                game_id=game.id,
                source="critic",
                publication=review.publication,
                author=review.author,
                score=review.score,
                body=review.quote,
                reviewed_at=_parse_date(review.date),
                url=review.url,
                platform=getattr(review, "platform", None),
            )
        )

    for review in card.user_reviews:
        session.add(
            Review(
                game_id=game.id,
                source="user",
                publication=None,
                author=getattr(review, "author", None) or getattr(review, "username", None),
                score=review.score,
                body=getattr(review, "quote", None) or getattr(review, "text", None),
                reviewed_at=_parse_date(review.date),
                url=None,
                platform=getattr(review, "platform", None),
            )
        )

    await session.flush()
    return game


def _generated_reason(attempt: int) -> str:
    if attempt > 1:
        return f"сгенерировано (попытка {attempt})"
    return "сгенерировано"


def _error_reason(attempt: int) -> str:
    return f"ошибка после {attempt} попыток"


def _merge_game_tags(game: Game, incoming: list[str]) -> None:
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(game.tags or game.ai_tags or []) + list(incoming or []):
        tag = normalize_ai_tag(str(item)) if item else None
        if not tag or tag in seen:
            continue
        seen.add(tag)
        merged.append(tag)
        if len(merged) >= MAX_GAME_TAGS:
            break
    game.tags = merged
    game.ai_tags = merged


def _merge_key_features(game: Game, incoming: list[str]) -> None:
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(game.key_features or []) + list(incoming or []):
        text = " ".join(str(item).split()).strip()
        if len(text) < 3:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
        if len(merged) >= MAX_GAME_FEATURES:
            break
    game.key_features = merged


def _clear_structured_fields(game: Game) -> None:
    game.tags = None
    game.ai_tags = None
    game.ai_tags_fingerprint = None
    game.genre_detailed = None
    game.key_features = None
    game.target_audience = None


def _structured_extracted(game: Game) -> bool:
    return bool(getattr(game, "ai_tags_fingerprint", None))


def _apply_llm_extraction(game: Game, extracted: Any) -> bool:
    """Пишет tags/жанр/фичи один раз. Повторный JSON (другое саммари) игнорируется."""
    if not getattr(extracted, "parsed", False):
        return False
    if _structured_extracted(game):
        return False
    if extracted.tags:
        _merge_game_tags(game, extracted.tags)
    elif not (game.tags or game.ai_tags):
        game.tags = []
        game.ai_tags = []
    if extracted.genre_detailed:
        game.genre_detailed = extracted.genre_detailed
    if extracted.key_features:
        _merge_key_features(game, extracted.key_features)
    if extracted.target_audience:
        game.target_audience = extracted.target_audience
    game.ai_tags_fingerprint = STRUCTURED_EXTRACTED_MARK
    return True


async def _summarize_kind(
    session: AsyncSession,
    game: Game,
    llm: LLMClient,
    *,
    kind: str,
    reviews: list,
    snippets: list[str],
    fingerprint: str,
    force_resummarize: bool,
    notes: list[str],
    title: str,
    description: str | None,
    metascore: int | None,
    userscore: float | None,
    genres: list[str] | None = None,
    want_extraction: bool = True,
) -> dict[str, Any]:
    """Финальный статус одного саммари: skipped / none / generated / error."""
    snippets = [str(item).strip() for item in snippets if str(item or "").strip()]
    description_text = (description or "").strip()
    enough_reviews = len(reviews) >= LLM_MIN_REVIEWS and bool(snippets)
    # Критики: при нехватке отзывов всё равно вызываем LLM, если есть описание.
    # Игроки: описание-only не дублируем — теги уже снимет critic-вызов.
    can_run = enough_reviews or (
        bool(description_text) and (kind == "critic" or bool(snippets))
    )
    if not can_run:
        if not description_text and not snippets:
            return _llm_kind("none", reason="нет текста отзывов")
        return _llm_kind(
            "none",
            reason=f"мало отзывов ({len(reviews)}, нужно ≥{LLM_MIN_REVIEWS})",
        )

    fp_attr = "summary_fingerprint_critic" if kind == "critic" else "summary_fingerprint_user"
    count_attr = "review_count_critic" if kind == "critic" else "review_count_user"
    existing_id = await session.scalar(
        select(Summary.id).where(Summary.game_id == game.id, Summary.kind == kind).limit(1)
    )
    if (
        not force_resummarize
        and existing_id is not None
        and getattr(game, fp_attr) == fingerprint
    ):
        if "skipped (reviews unchanged)" not in notes:
            notes.append("skipped (reviews unchanged)")
        logger.info("LLM %s skipped %s (reviews unchanged)", kind, game.slug)
        return _llm_kind("skipped", reason="отзывы без изменений")

    if kind == "critic":
        prompt = build_critic_summary_prompt(
            title=title,
            description=description,
            metascore=metascore,
            critic_snippets=snippets,
            genres=genres,
            include_features=want_extraction,
        )
        system = CRITIC_SUMMARY_SYSTEM_PROMPT if want_extraction else SUMMARY_TEXT_SYSTEM_PROMPT
    else:
        prompt = build_user_summary_prompt(
            title=title,
            description=description,
            userscore=userscore,
            user_snippets=snippets,
            genres=genres,
            include_features=want_extraction,
        )
        system = USER_SUMMARY_SYSTEM_PROMPT if want_extraction else SUMMARY_TEXT_SYSTEM_PROMPT

    result = await llm.complete(
        prompt,
        system=system,
        slug=game.slug,
        run_id=llm.run_id,
        kind=kind,
        json_object=True,
    )
    attempt = result.attempt or 1
    if result.error:
        logger.warning("Саммари %s пропущено для %s: %s", kind, game.slug, result.error)
        return _llm_kind(
            "error",
            reason=_error_reason(attempt),
            log_id=result.log_id,
            log_ids=result.log_ids,
            error=result.error,
            error_status=result.error_status,
            attempt=attempt,
        )
    if result.text:
        extracted = parse_summary_payload(result.text)
        content = extracted.summary or result.text.strip()
        if content and not looks_russian(content):
            logger.warning("Саммари %s/%s без кириллицы — модель ответила не по-русски", game.slug, kind)
        await session.execute(
            delete(Summary).where(Summary.game_id == game.id, Summary.kind == kind)
        )
        session.add(
            Summary(game_id=game.id, kind=kind, content=content, model=result.model)
        )
        setattr(game, fp_attr, fingerprint)
        setattr(game, count_attr, len(reviews))
        applied = False
        if want_extraction:
            applied = _apply_llm_extraction(game, extracted)
        if not extracted.parsed:
            logger.warning("JSON саммари не разобрали для %s/%s — сохранили сырой текст", game.slug, kind)
        return _llm_kind(
            "generated",
            reason=_generated_reason(attempt),
            log_id=result.log_id,
            log_ids=result.log_ids,
            attempt=attempt,
            extracted=applied,
        )
    return _llm_kind("none", reason="пустой ответ LLM", attempt=attempt)


async def _maybe_summarize(
    session: AsyncSession,
    game: Game,
    card: GameCard,
    llm: LLMClient,
    *,
    force_resummarize: bool = False,
) -> tuple[list[str], dict[str, Any], int]:
    """Генерирует саммари, если отзывы изменились. Ошибка LLM не прерывает игру."""
    notes: list[str] = []
    if force_resummarize:
        _clear_structured_fields(game)
    critic_fp = _fingerprint_llm_input(card.critic_reviews, game.description)
    user_fp = _fingerprint_llm_input(card.user_reviews, game.description)
    critic_snippets = [item.quote or "" for item in card.critic_reviews if item.quote]
    user_snippets = [item.quote or "" for item in card.user_reviews if item.quote]
    critic_info = await _summarize_kind(
        session,
        game,
        llm,
        kind="critic",
        reviews=card.critic_reviews,
        snippets=critic_snippets,
        fingerprint=critic_fp,
        force_resummarize=force_resummarize,
        notes=notes,
        title=game.title,
        description=game.description,
        metascore=game.metascore,
        userscore=game.userscore,
        genres=game.genres,
        want_extraction=not _structured_extracted(game),
    )
    user_info = await _summarize_kind(
        session,
        game,
        llm,
        kind="user",
        reviews=card.user_reviews,
        snippets=user_snippets,
        fingerprint=user_fp,
        force_resummarize=force_resummarize,
        notes=notes,
        title=game.title,
        description=game.description,
        metascore=game.metascore,
        userscore=game.userscore,
        genres=game.genres,
        want_extraction=not _structured_extracted(game),
    )
    llm_info = {"critic": critic_info, "user": user_info}
    error_count = sum(
        1 for item in (critic_info, user_info) if item.get("status") == "error"
    )
    if critic_info.get("status") == "none" and user_info.get("status") == "none":
        logger.info(
            "LLM не вызван %s: critic=%s user=%s (порог %s)",
            game.slug,
            len(card.critic_reviews),
            len(card.user_reviews),
            LLM_MIN_REVIEWS,
        )
    return notes, llm_info, error_count


async def _process_listed(
    client: MetacriticClient,
    item: ListedGame,
    existing: Game | None,
    llm: LLMClient,
    settings: Settings,
    *,
    recheck_card: bool = False,
    force_resummarize: bool = False,
    from_carousel: bool = False,
) -> tuple[bool, dict[str, Any] | None]:
    is_insert = existing is None
    need_deep = is_insert or _scores_changed(existing, item) or _needs_llm_backfill(existing)
    async with SessionLocal() as session:
        if not need_deep and existing is not None and not recheck_card:
            game = await session.get(Game, existing.id)
            if game is None:
                return False, None
            changes = _light_change_items(game, item)
            await _light_update(session, game, item)
            _apply_carousel_flag(game, from_carousel)
            await session.commit()
            action = "updated" if changes else "skipped"
            logger.info("Light-апдейт %s (%s)", item.slug, action)
            return True, _run_detail(
                slug=item.slug,
                title=game.title or item.title,
                action=action,
                changes=changes,
                reason=None
                if changes
                else "карточка без изменений (metascore тот же, глубокая проверка не нужна)",
            )

        card = await fetch_game_card(client, item.slug, settings=settings)
        if card is None:
            logger.warning("Карточка не разобрана: %s", item.slug)
            return False, _run_detail(
                slug=item.slug,
                title=item.title,
                action="error",
                changes=[],
                error="Карточка не разобрана",
            )

        if not need_deep and existing is not None and recheck_card:
            game = await session.get(Game, existing.id)
            if game is None:
                return False, None
            changes = _light_change_items(game, item)
            await _light_update(session, game, item)
            _apply_carousel_flag(game, from_carousel)
            _llm_notes, llm_info, _error_count = await _maybe_summarize(
                session,
                game,
                card,
                llm,
                force_resummarize=force_resummarize,
            )
            await session.commit()
            action = "updated" if changes or _llm_generated(llm_info) else "skipped"
            logger.info("Light-апдейт %s (%s)", item.slug, action)
            return True, _run_detail(
                slug=item.slug,
                title=game.title or item.title,
                action=action,
                changes=changes,
                llm=llm_info,
            )

        before: dict[str, Any] | None = None
        if existing is not None:
            db_game = await session.scalar(
                select(Game)
                .where(Game.id == existing.id)
                .options(selectinload(Game.platform_scores), selectinload(Game.reviews))
            )
            if db_game is not None:
                before = {
                    "metascore": db_game.metascore,
                    "userscore": db_game.userscore,
                    "description": db_game.description,
                    "cover_url": db_game.cover_url,
                    "genres": _norm_list(db_game.genres),
                    "developers": _norm_list(db_game.developers),
                    "platforms": _platform_sig_from_game(db_game),
                    "fp_c": db_game.summary_fingerprint_critic,
                    "fp_u": db_game.summary_fingerprint_user,
                }

        platform_userscores = await fetch_platform_userscores(
            client,
            card.slug or item.slug,
            card.platforms,
            settings=settings,
        )
        game = await _upsert_game(session, card, listed=item, platform_userscores=platform_userscores)
        _apply_carousel_flag(game, from_carousel)
        _llm_notes, llm_info, _error_count = await _maybe_summarize(
            session,
            game,
            card,
            llm,
            force_resummarize=force_resummarize,
        )
        if is_insert:
            changes = _new_change_items(game, card)
        else:
            changes = _deep_change_items(before, game, card) if before else []
        await session.commit()

        if is_insert:
            action = "new"
        else:
            action = "updated" if changes or _llm_generated(llm_info) else "skipped"
        logger.info("%s %s", "Insert" if is_insert else "Deep-апдейт", item.slug)
        return True, _run_detail(
            slug=item.slug,
            title=game.title or item.title,
            action=action,
            changes=changes,
            llm=llm_info,
        )


ENRICHMENT_KINDS = (JOB_SIMILAR, JOB_YOUTUBE)


def _job_slugs_from_details(details: Any) -> list[str]:
    if not isinstance(details, list):
        return []
    slugs: list[str] = []
    seen: set[str] = set()
    for item in details:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip()
        if not slug or item.get("action") == "error" or slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    return slugs


async def last_run_slugs() -> tuple[int, list[str]]:
    """Slug'и последнего прогона — кнопки и recover."""
    async with SessionLocal() as session:
        last_run = await session.scalar(select(RunLog).order_by(RunLog.id.desc()).limit(1))
        if last_run is None:
            return 0, []
        return int(last_run.id), _job_slugs_from_details(last_run.details)


async def enrichment_queue_counts(*, run_id: int | None = None) -> dict[str, int]:
    """Сколько задач similar/youtube ждут или уже в работе."""
    counts = {JOB_SIMILAR: 0, JOB_YOUTUBE: 0}
    async with SessionLocal() as session:
        stmt = (
            select(PipelineJob.kind, func.count())
            .where(PipelineJob.status.in_(JOB_OPEN))
            .group_by(PipelineJob.kind)
        )
        rid = int(run_id or 0)
        if rid:
            stmt = stmt.where(PipelineJob.run_id == rid)
        rows = await session.execute(stmt)
        for kind, total in rows.all():
            if kind in counts:
                counts[kind] = int(total or 0)
    return counts


async def enrichment_queue_counts_for_run(run_id: int) -> dict[str, int]:
    """Очередь похожих/летсплеев одного прогона."""
    return await enrichment_queue_counts(run_id=run_id)


async def enqueue_enrichment(
    run_id: int,
    slugs: list[str],
    *,
    kinds: tuple[str, ...] | list[str] = ENRICHMENT_KINDS,
    force_youtube: bool = False,
) -> int:
    """Ставит similar/youtube на slug, если такой задачи ещё нет в полёте."""
    if not int(run_id or 0):
        run_id, _ = await last_run_slugs()
    if not int(run_id or 0):
        logger.warning("Очередь enrichment без номера прогона — пропуск")
        return 0
    clean = [item for item in slugs if item]
    wanted = tuple(kind for kind in kinds if kind in ENRICHMENT_KINDS)
    if not clean or not wanted:
        return 0
    async with SessionLocal() as session:
        by_slug: dict[str, Game] = {}
        if JOB_YOUTUBE in wanted:
            rows = await session.execute(select(Game).where(Game.slug.in_(clean)))
            by_slug = {game.slug: game for game in rows.scalars().all()}
        open_rows = await session.execute(
            select(PipelineJob.slug, PipelineJob.kind).where(
                PipelineJob.status.in_(JOB_OPEN),
                PipelineJob.slug.in_(clean),
                PipelineJob.kind.in_(wanted),
            )
        )
        open_set = {(slug, kind) for slug, kind in open_rows.all()}
        added = 0
        for slug in clean:
            for kind in wanted:
                if (slug, kind) in open_set:
                    continue
                if kind == JOB_YOUTUBE:
                    game = by_slug.get(slug)
                    if game is not None and not needs_letsplay_job(
                        game, force=force_youtube
                    ):
                        continue
                session.add(
                    PipelineJob(
                        run_id=run_id or None,
                        slug=slug,
                        kind=kind,
                        status="pending",
                    )
                )
                open_set.add((slug, kind))
                added += 1
        await session.commit()
    if added:
        logger.info(
            "В очередь прогона #%s: +%s задач (%s), slugs=%s",
            run_id,
            added,
            ",".join(wanted),
            clean[:20],
        )
    return added


async def _youtube_hole_slugs(limit: int, *, include_stubs: bool = False) -> list[str]:
    cap = max(0, int(limit or 0))
    if not cap:
        return []
    slugs: list[str] = []
    async with SessionLocal() as session:
        rows = await session.execute(
            select(Game.slug, Game.youtube_summary, Game.youtube_summary_source).order_by(Game.id)
        )
        for slug, summary, source in rows.all():
            if not needs_letsplay_job(
                SimpleNamespace(youtube_summary=summary, youtube_summary_source=source),
                force=include_stubs,
            ):
                continue
            slugs.append(slug)
            if len(slugs) >= cap:
                break
    return slugs


async def recover_enrichment_on_startup() -> None:
    """После рестарта: зависшие running → pending. Прогон с живой очередью не убиваем."""
    leftover: list[str] = []
    run_id = 0
    n_stuck = 0
    n_runs = 0
    n_kept = 0
    n_reopened = 0
    async with SessionLocal() as session:
        stuck = await session.execute(select(PipelineJob).where(PipelineJob.status == "running"))
        for job in stuck.scalars().all():
            job.status = "pending"
            n_stuck += 1
        hung = await session.execute(select(RunLog).where(RunLog.status.in_(RUN_IN_FLIGHT)))
        for run in hung.scalars().all():
            open_n = await session.scalar(
                select(func.count())
                .select_from(PipelineJob)
                .where(
                    PipelineJob.run_id == run.id,
                    PipelineJob.status.in_(JOB_OPEN),
                )
            )
            has_details = isinstance(run.details, list) and bool(run.details)
            if int(open_n or 0) or has_details:
                if run.status == "running" and has_details:
                    run.status = "enriching"
                n_kept += 1
                continue
            run.status = "error"
            run.finished_at = now_utc()
            if not run.error_message:
                run.error_message = "Прогон оборвался при рестарте процесса"
            n_runs += 1
        open_run_ids = await session.execute(
            select(PipelineJob.run_id)
            .where(
                PipelineJob.status.in_(JOB_OPEN),
                PipelineJob.run_id.is_not(None),
            )
            .distinct()
        )
        for (rid,) in open_run_ids.all():
            owned = await session.get(RunLog, int(rid))
            if owned is None or owned.status not in {"success", "partial"}:
                continue
            owned.status = "enriching"
            owned.finished_at = None
            n_reopened += 1
        state = await session.get(PipelineState, 1)
        if state and isinstance(state.followup_slugs, list):
            leftover = [str(item) for item in state.followup_slugs if item]
            run_id = int(state.followup_run_id or 0)
            state.followup_slugs = None
            state.similar_done = True
            state.youtube_done = True
        await session.commit()
    if n_stuck:
        logger.info("Вернули в очередь %s зависших задач", n_stuck)
    if n_reopened:
        logger.info("Вернули в работу %s прогонов: очередь этого прогона ещё не пуста", n_reopened)
    if n_kept:
        logger.info("Оставили %s прогонов в работе: очередь похожих/летсплеев ещё жива", n_kept)
    if n_runs:
        logger.info("Закрыли %s оборванных прогонов после рестарта", n_runs)
    if leftover:
        await enqueue_enrichment(run_id, leftover)
    holes = await _youtube_hole_slugs(get_settings().youtube_backfill_limit)
    if holes:
        attach_id = int(run_id or 0)
        run_slugs: set[str] = set()
        attach_running = False
        if not attach_id:
            attach_id, slug_list = await last_run_slugs()
            run_slugs = set(slug_list)
        if attach_id:
            async with SessionLocal() as session:
                target = await session.get(RunLog, attach_id)
                if target is not None:
                    attach_running = target.status in RUN_IN_FLIGHT
                    if not run_slugs:
                        run_slugs = set(_job_slugs_from_details(target.details))
        if attach_running:
            holes = [slug for slug in holes if slug in run_slugs]
        if holes:
            await enqueue_enrichment(attach_id, holes, kinds=(JOB_YOUTUBE,))
    hung_ids: list[int] = []
    async with SessionLocal() as session:
        hung = await session.execute(select(RunLog).where(RunLog.status.in_(RUN_IN_FLIGHT)))
        hung_ids = [int(run.id) for run in hung.scalars().all()]
    for rid in hung_ids:
        await _finalize_run_if_idle(rid)
    pending = await enrichment_queue_counts()
    logger.info(
        "Очередь обогащения: похожие %s, летсплеи %s",
        pending[JOB_SIMILAR],
        pending[JOB_YOUTUBE],
    )


async def recover_followups_on_startup() -> None:
    await recover_enrichment_on_startup()


async def _pending_job_ids(kind: str) -> list[int]:
    async with SessionLocal() as session:
        rows = await session.scalars(
            select(PipelineJob.id)
            .where(PipelineJob.status == "pending", PipelineJob.kind == kind)
            .order_by(PipelineJob.id)
        )
        return [int(item) for item in rows.all()]


async def _pop_job(
    kind: str | None = None,
    *,
    job_id: int | None = None,
) -> dict[str, Any] | None:
    async with SessionLocal() as session:
        stmt = select(PipelineJob).where(PipelineJob.status == "pending")
        if job_id is not None:
            stmt = stmt.where(PipelineJob.id == int(job_id))
        else:
            stmt = stmt.order_by(PipelineJob.id)
            if kind:
                stmt = stmt.where(PipelineJob.kind == kind)
        job = await session.scalar(stmt.limit(1))
        if job is None:
            return None
        payload = {
            "id": job.id,
            "run_id": int(job.run_id or 0),
            "slug": job.slug,
            "kind": job.kind,
        }
        job.status = "running"
        await session.commit()
        return payload


async def _finish_job(job_id: int, *, status: str, error: str | None = None) -> None:
    async with SessionLocal() as session:
        job = await session.get(PipelineJob, job_id)
        if job is None:
            return
        job.status = status
        job.finished_at = now_utc()
        job.error_message = error
        await session.commit()


async def _requeue_job(job_id: int) -> None:
    """Снова в pending, но в хвост очереди: тот же id блокировал бы всех остальных."""
    async with SessionLocal() as session:
        job = await session.get(PipelineJob, job_id)
        if job is None:
            return
        session.add(
            PipelineJob(
                run_id=job.run_id,
                slug=job.slug,
                kind=job.kind,
                status="pending",
            )
        )
        await session.delete(job)
        await session.commit()


async def _execute_job(job: dict[str, Any]) -> None:
    settings = get_settings()
    run_id = int(job.get("run_id") or 0)
    if not run_id:
        run_id, _ = await last_run_slugs()
        job["run_id"] = run_id
    llm = LLMClient(settings, run_id=run_id or None)
    slug = str(job.get("slug") or "")
    kind = str(job.get("kind") or "")
    try:
        if kind == JOB_SIMILAR:
            async with SessionLocal() as session:
                reviewed = await review_mixed_similars(session, llm, slugs=[slug])
            logger.info("Очередь similar %s run=%s: %s", slug, job.get("run_id"), reviewed)
        elif kind == JOB_YOUTUBE:
            filled = await process_letsplay_slug(slug, settings=settings, llm=llm)
            logger.info("Очередь youtube %s run=%s: %s", slug, job.get("run_id"), filled)
            if not filled:
                await _requeue_job(int(job["id"]))
                return
        else:
            raise ValueError(f"unknown job kind {kind}")
        await _finish_job(int(job["id"]), status="done")
        await _finalize_run_if_idle(run_id)
    except Exception as exc:
        logger.exception("Задача %s %s упала", kind, slug)
        await _finish_job(int(job["id"]), status="error", error=str(exc)[:500])
        await _finalize_run_if_idle(run_id)


async def drain_one_job(*, kind: str | None = None, job_id: int | None = None) -> bool:
    """Одна задача. Сначала летсплеи (поиск без Groq), похожие — если chat жив."""
    global _enriching
    if groq_chat_blocked() and kind == JOB_SIMILAR:
        return False
    async with _enrich_lock:
        _enriching = True
        try:
            if job_id is not None:
                job = await _pop_job(kind=kind, job_id=job_id)
            elif kind:
                job = await _pop_job(kind=kind)
            else:
                job = await _pop_job(kind=JOB_YOUTUBE)
                if job is None and not groq_chat_blocked():
                    job = await _pop_job(kind=JOB_SIMILAR)
            if job is None:
                return False
            if groq_chat_blocked() and job.get("kind") != JOB_YOUTUBE:
                await _requeue_job(int(job["id"]))
                return False
            await _execute_job(job)
            return True
        finally:
            _enriching = False


async def drain_kind(kind: str) -> int:
    """Съедает текущий снимок очереди. Реqueue не крутит тот же id по кругу."""
    ids = await _pending_job_ids(kind)
    done = 0
    for jid in ids:
        if await drain_one_job(kind=kind, job_id=jid):
            done += 1
    return done


async def tick_pipeline_stages() -> None:
    """Раз в ~20 с: одна similar или youtube. При суточном лимите Groq — только летсплеи."""
    if groq_chat_blocked():
        logger.info(
            "Очередь LLM на паузе Groq ещё %s с — берём только летсплеи",
            int(groq_chat_retry_in()),
        )
        await drain_one_job(kind=JOB_YOUTUBE)
        return
    await drain_one_job()


async def run_similar_now() -> int | None:
    """Кнопка: поставить похожих последнего прогона и съесть очередь similar."""
    run_id, slugs = await last_run_slugs()
    logger.info("Ручной этап похожих run=%s, игр %s", run_id, len(slugs))
    await enqueue_enrichment(run_id, slugs, kinds=(JOB_SIMILAR,))
    done = await drain_kind(JOB_SIMILAR)
    await _finalize_run_if_idle(run_id)
    return done


async def run_youtube_now() -> int | None:
    """Кнопка: летсплеи последнего прогона + дырки, включая заглушки после сбоя Whisper."""
    settings = get_settings()
    run_id, slugs = await last_run_slugs()
    holes = await _youtube_hole_slugs(settings.youtube_backfill_limit, include_stubs=True)
    attach_running = False
    if run_id:
        async with SessionLocal() as session:
            target = await session.get(RunLog, run_id)
            attach_running = target is not None and target.status in RUN_IN_FLIGHT
    merged = list(dict.fromkeys(slugs if attach_running else [*slugs, *holes]))
    logger.info("Ручной этап YouTube run=%s, игр %s", run_id, len(merged))
    await enqueue_enrichment(run_id, merged, kinds=(JOB_YOUTUBE,), force_youtube=True)
    done = await drain_kind(JOB_YOUTUBE)
    await _finalize_run_if_idle(run_id)
    return done


async def run_hourly_pipeline(
    settings: Settings | None = None,
    *,
    manual: bool = False,
    force_resummarize: bool = False,
    source_override: str | None = None,
) -> RunLog:
    """Один прогон: курсор дня, до 20 необработанных сегодня игр.

    Ручной запуск не сбрасывает таймер планировщика, но двигает курсор дня
    (первый прогон — карусель, следующие — SEE ALL). Уже обработанные сегодня
    игры в выборку не попадают, чтобы добирались новые.
    """
    global _running, _active_card_run_id
    settings = settings or get_settings()

    async with _run_lock:
        if _running:
            logger.info("Пайплайн уже запущен — пропуск")
            return await _write_run(
                status="skipped",
                finished_at=now_utc(),
                error_message="Пайплайн уже выполняется",
            )
        _running = True

    await init_db()
    run = await _write_run(status="running")
    _active_card_run_id = run.id
    games_found = 0
    games_processed = 0
    llm_errors = 0
    details: list[dict[str, Any]] = []
    today = today_local()
    llm: LLMClient | None = None
    new_offset = 0
    source = "main"
    result: RunLog | None = None

    try:
        async with SessionLocal() as session:
            state = await _load_state(session, today)
            start_offset = state.offset
            source = (state.source or "main").strip() or "main"
            index = await _game_index(session)
            await session.commit()
            if source_override in {"main", "browse"}:
                source = source_override
            elif source == "main" and any(
                bool(game.is_from_carousel) and _processed_today(game.last_processed_at, today)
                for game in index.values()
            ):
                source = "browse"
                logger.info("Карусель сегодня уже обработана — выборка SEE ALL")

        async with MetacriticClient(settings) as client:
            source_before = source
            selected, new_offset, source, positions = await _collect_candidates(
                client,
                settings=settings,
                today=today,
                start_offset=start_offset,
                source=source_before,
                index=index,
                ignore_processed_today=False,
            )
            games_found = len(selected)
            llm = LLMClient(settings, run_id=run.id)
            logger.info(
                "Выборка%s: %s игр, source %s → %s, offset %s → %s, slugs=%s",
                " (ручной)" if manual else "",
                games_found,
                source_before,
                source,
                start_offset,
                new_offset,
                [item.slug for item in selected],
            )

            first_fail: tuple[str, int | None] | None = None
            delay = max(0.0, float(getattr(settings, "card_delay_seconds", 0) or 0))
            for index_in_batch, item in enumerate(selected):
                if index_in_batch and delay:
                    await asyncio.sleep(delay)
                failed = False
                try:
                    ok, detail = await _process_listed(
                        client,
                        item,
                        index.get(item.slug),
                        llm,
                        settings,
                        recheck_card=manual,
                        force_resummarize=force_resummarize,
                        from_carousel=bool(item.from_carousel),
                    )
                    if ok:
                        games_processed += 1
                    if detail:
                        details.append(detail)
                    failed = (not ok) or bool(detail and detail.get("action") == "error")
                    if (
                        not failed
                        and detail
                        and detail.get("action") in {"new", "updated"}
                        and item.slug
                    ):
                        await enqueue_enrichment(run.id, [item.slug])
                except Exception as exc:
                    logger.exception("Не удалось обработать игру %s", item.slug)
                    details.append(
                        _run_detail(
                            slug=item.slug,
                            title=item.title,
                            action="error",
                            changes=[],
                            error=str(exc),
                        )
                    )
                    failed = True
                if failed and first_fail is None:
                    pos = positions[index_in_batch] if index_in_batch < len(positions) else None
                    if item.from_carousel:
                        first_fail = ("main", None)
                    else:
                        first_fail = ("browse", pos if isinstance(pos, int) else start_offset)

            if first_fail is not None:
                kind, pos = first_fail
                if kind == "main":
                    source = "main"
                    new_offset = 0
                else:
                    source = "browse"
                    new_offset = int(pos or 0)

        await _save_pipeline_state(today, source=source, offset=new_offset)

        llm_errors = sum(_detail_llm_errors(item) for item in details)
        result = await _finish_run(
            run.id,
            status="enriching",
            games_found=games_found,
            games_processed=games_processed,
            details=details,
            llm_errors=llm_errors,
            complete=False,
        )
        logger.info(
            "Карточки прогона #%s в базе (%s из %s). Ждём очередь похожих/летсплеев этого прогона.",
            run.id,
            games_processed,
            games_found,
        )
    except Exception as exc:
        logger.exception("Пайплайн завершился с ошибкой")
        llm_errors = sum(_detail_llm_errors(item) for item in details)
        result = await _finish_run(
            run.id,
            status="error",
            games_found=games_found,
            games_processed=games_processed,
            error_message=str(exc),
            details=details,
            llm_errors=llm_errors,
        )
    finally:
        _running = False
        _active_card_run_id = None

    if result is not None and result.status in RUN_IN_FLIGHT:
        closed = await _finalize_run_if_idle(run.id)
        if closed is not None:
            result = closed

    return result if result is not None else run


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_hourly_pipeline())
