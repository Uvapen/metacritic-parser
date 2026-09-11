"""HTTP-роуты: Jinja2 UI и JSON API."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.db import get_db
from app.models import Game, PlatformScore, RunLog
from app.timeutil import to_local
from app.llm.client import chat_model_chain, groq_chat_retry_in, humanize_llm_error
from app.scraper.nuxt import normalize_cover_url
from app.services.pipeline import (
    LLM_INFO_KINDS,
    build_status_reason,
    enrichment_is_running,
    enrichment_queue_counts,
    pipeline_is_running,
    run_hourly_pipeline,
    run_similar_now,
    run_youtube_now,
)
from app.services.similar import similar_payload
from app.services.youtube import letsplay_has_summary

router = APIRouter()
templates = Jinja2Templates(directory=str(get_settings().templates_dir))
templates.env.filters["cover"] = normalize_cover_url
_YOUTUBE_ID = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|embed/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)
ACTION_RU = {
    "new": "новая",
    "updated": "обновлена",
    "skipped": "пропущена",
    "error": "ошибка",
}
STATUS_RU = {
    "running": "идёт",
    "success": "успех",
    "partial": "частично",
    "error": "ошибка",
    "skipped": "пропущен",
}
KIND_RU = {"added": "добавлено", "updated": "обновлено"}
COVER_RU = {"added": "добавлена", "updated": "обновлена"}
REVIEWS_RU = {"added": "добавлены", "updated": "обновлены"}
LLM_PER_PAGE = 15
LLM_KIND_RU = {
    "critic": "Summary critics",
    "user": "Summary players",
    "tags": "Tags",
    "similar": "Similar games",
    "youtube": "Летсплей (саммари)",
    "whisper": "Whisper",
}
LLM_KIND_FILTERS = frozenset(LLM_KIND_RU)
LLM_KIND_ORDER = ("critic", "user", "tags", "similar", "youtube", "whisper")


def _configured_llm_models() -> list[str]:
    """Модели из .env (120B, запасные, Whisper), даже если в логе их ещё нет."""
    settings = get_settings()
    names: list[str] = []
    whisper = str(getattr(settings, "whisper_model", "") or "").strip()
    for name in [*chat_model_chain(settings), whisper]:
        item = name.strip()
        if item and item not in names:
            names.append(item)
    return names


def youtube_id(url: str | None) -> str | None:
    """Достаёт id ролика из youtube_url."""
    if not url:
        return None
    match = _YOUTUBE_ID.search(url)
    return match.group(1) if match else None


templates.env.filters["youtube_id"] = youtube_id
templates.env.tests["letsplay_ready"] = letsplay_has_summary
DbSession = Annotated[AsyncSession, Depends(get_db)]


def _next_run_at_iso() -> str | None:
    from app.main import get_next_pipeline_run_at

    return get_next_pipeline_run_at()


async def _queue_runtime() -> dict[str, Any]:
    counts = await enrichment_queue_counts()
    return {
        "pipeline_running": pipeline_is_running(),
        "enrichment_running": enrichment_is_running(),
        "queue_similar": counts.get("similar", 0),
        "queue_youtube": counts.get("youtube", 0),
        "groq_pause_sec": int(groq_chat_retry_in()),
    }


def _llm_log_count() -> int:
    path = get_settings().llm_log_path
    if not path.exists():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8"))


def _action_counts(details: list | None) -> dict[str, int]:
    counts = {"new": 0, "updated": 0, "skipped": 0, "error": 0}
    if not isinstance(details, list):
        return counts
    for item in details:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if action in counts:
            counts[action] += 1
    return counts


def format_localtime(value: Any, time_only: bool = False) -> str:
    """Локальное время UTC+4: ГГГГ-ММ-ДД ЧЧ:ММ:СС без микросекунд."""
    dt = to_local(value)
    if dt is None:
        return "—"
    if time_only:
        return dt.strftime("%H:%M:%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def action_ru(value: str | None) -> str:
    if not value:
        return "—"
    return ACTION_RU.get(value, value)


def status_ru(value: str | None) -> str:
    if not value:
        return "—"
    return STATUS_RU.get(value, value)


def format_change_line(item: Any) -> str:
    if isinstance(item, str):
        if item == "skipped (reviews unchanged)":
            return "отзывы без изменений"
        return item
    if not isinstance(item, dict):
        return str(item)
    field = str(item.get("field") or "")
    kind = item.get("kind")
    text = str(item.get("text") or "")
    if field == "description":
        quoted = f"«{text}»" if text else "—"
        suffix = f" ({KIND_RU[kind]})" if kind in KIND_RU else ""
        return f"description: {quoted}{suffix}"
    if field == "cover_url":
        return f"cover_url: {COVER_RU.get(kind, kind or '—')}"
    if field == "reviews":
        return f"reviews: {REVIEWS_RU.get(kind, kind or '—')}"
    if text:
        suffix = f" ({KIND_RU[kind]})" if kind in KIND_RU and field in {"metascore", "userscore"} else ""
        return f"{field}: {text}{suffix}"
    if kind in KIND_RU:
        return f"{field}: {KIND_RU[kind]}"
    return field or "—"


def llm_lines(llm: Any) -> list[dict[str, Any]]:
    if not isinstance(llm, dict):
        return [{"text": "—", "href": None, "link": None}]
    lines: list[dict[str, Any]] = []
    for kind, label in LLM_INFO_KINDS:
        info = llm.get(kind) if isinstance(llm.get(kind), dict) else {}
        status = info.get("status")
        log_id = info.get("log_id")
        href = f"/llm/{log_id}" if log_id else None
        link = "диалог" if href else None
        if status == "generated":
            lines.append({"text": f"{label}: сгенерировано", "href": href, "link": link})
        elif status == "error":
            code = info.get("error_status")
            short = str(code) if code is not None else (info.get("error") or "ошибка")
            if isinstance(short, str) and len(short) > 40:
                short = short[:40].rstrip() + "…"
            lines.append({"text": f"{label}: {short}, повтор в след. прогоне", "href": href, "link": link})
        elif status == "skipped" and kind != "tags":
            lines.append({"text": f"{label}: без изменений", "href": None, "link": None})
    return lines or [{"text": "—", "href": None, "link": None}]


def pagination_window(page: int, total_pages: int) -> list[int | None]:
    """Номера страниц для макроса: 1 … 4 [5] 6 … N. None = многоточие."""
    if total_pages <= 0:
        return []
    page = max(1, min(int(page), int(total_pages)))
    total_pages = int(total_pages)
    if total_pages <= 7:
        return list(range(1, total_pages + 1))
    start = max(2, page - 1)
    end = min(total_pages - 1, page + 1)
    pages: list[int | None] = [1]
    if start > 2:
        pages.append(None)
    pages.extend(range(start, end + 1))
    if end < total_pages - 1:
        pages.append(None)
    if pages[-1] != total_pages:
        pages.append(total_pages)
    return pages


def change_summary(changes: Any) -> str:
    if not isinstance(changes, list) or not changes:
        return ""
    names: list[str] = []
    for item in changes:
        if isinstance(item, dict) and item.get("field"):
            names.append(str(item["field"]))
        elif isinstance(item, str) and item:
            names.append(item)
    labels = ", ".join(names)
    if labels:
        return f"{len(changes)} изменений: {labels}"
    return f"{len(changes)} изменений"


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _llm_game_slug(item: dict[str, Any]) -> str:
    slug = item.get("game_slug") or item.get("slug") or ""
    return str(slug) if slug else ""


def _run_detail_slugs(details: Any) -> frozenset[str]:
    if not isinstance(details, list):
        return frozenset()
    return frozenset(
        str(item.get("slug"))
        for item in details
        if isinstance(item, dict) and item.get("slug")
    )


def _bind_llm_run_id(
    recorded: int | None,
    slug: str,
    ts: Any,
    runs: list[Any] | None,
) -> int | None:
    """Номер прогона из лога или из игры, которая в этом прогоне обрабатывалась."""
    if recorded:
        return recorded
    if not runs:
        return None
    ts_local = to_local(ts)
    timed: int | None = None
    for run in runs:
        started = to_local(getattr(run, "started_at", None))
        if ts_local and started and started > ts_local:
            continue
        run_id = _coerce_int(getattr(run, "id", None))
        if not run_id:
            continue
        if timed is None:
            timed = run_id
        if slug and slug in _run_detail_slugs(getattr(run, "details", None)):
            return run_id
    return timed


def _unique_ids(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for item in values:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _ids_from_llm_info(llm: Any) -> list[int]:
    ids: list[int] = []
    if not isinstance(llm, dict):
        return ids
    for kind, _label in LLM_INFO_KINDS:
        info = llm.get(kind)
        if not isinstance(info, dict):
            continue
        raw_ids = info.get("log_ids")
        if isinstance(raw_ids, list) and raw_ids:
            for value in raw_ids:
                n = _coerce_int(value)
                if n:
                    ids.append(n)
            continue
        n = _coerce_int(info.get("log_id"))
        if n:
            ids.append(n)
    return _unique_ids(ids)


def _llm_ok_by_id_for_run(run_id: int, runs: list[Any] | None = None) -> dict[int, bool]:
    path = get_settings().llm_log_path
    mapping: dict[int, bool] = {}
    if not path.exists():
        return mapping
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _bind_llm_run_id(_coerce_int(item.get("run_id")), _llm_game_slug(item), item.get("ts"), runs) != run_id:
            continue
        mapping[line_no] = bool(item.get("ok"))
    return mapping


def _llm_entries_by_slug_for_run(run_id: int, runs: list[Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    path = get_settings().llm_log_path
    mapping: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return mapping
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        slug = _llm_game_slug(item)
        if _bind_llm_run_id(_coerce_int(item.get("run_id")), slug, item.get("ts"), runs) != run_id:
            continue
        if not slug:
            continue
        mapping.setdefault(slug, []).append(
            {
                "id": line_no,
                "kind": str(item.get("kind") or ""),
                "ok": bool(item.get("ok")),
            }
        )
    return mapping


def _llm_ids_by_slug_for_run(run_id: int) -> dict[str, list[int]]:
    return {
        slug: [int(entry["id"]) for entry in entries]
        for slug, entries in _llm_entries_by_slug_for_run(run_id).items()
    }


def _llm_jsonl_stats_for_run(run_id: int, runs: list[Any] | None = None) -> tuple[int, int, int]:
    """Вызовы, ретраи (attempt > 1) и неуспешные вызовы прогона."""
    path = get_settings().llm_log_path
    if not path.exists():
        return 0, 0, 0
    calls = 0
    retries = 0
    fails = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _bind_llm_run_id(_coerce_int(item.get("run_id")), _llm_game_slug(item), item.get("ts"), runs) != run_id:
            continue
        calls += 1
        if (_coerce_int(item.get("attempt")) or 1) > 1:
            retries += 1
        if not item.get("ok"):
            fails += 1
    return calls, retries, fails


def _llm_call_count_for_run(run_id: int) -> int:
    calls, _retries, _fails = _llm_jsonl_stats_for_run(run_id)
    return calls


def _llm_ids_for_game(item: dict[str, Any], mapping: dict[str, list[int]]) -> list[int]:
    slug = str(item.get("slug") or "")
    ids = list(mapping.get(slug) or [])
    if ids:
        return ids
    return _ids_from_llm_info(item.get("llm"))


def _llm_status_count(details: list | None, status: str) -> int:
    if not isinstance(details, list):
        return 0
    count = 0
    for item in details:
        if not isinstance(item, dict):
            continue
        llm = item.get("llm")
        if not isinstance(llm, dict):
            continue
        for kind, _label in LLM_INFO_KINDS:
            info = llm.get(kind)
            if isinstance(info, dict) and info.get("status") == status:
                count += 1
    return count


def _llm_errors_of(run: RunLog) -> int:
    run_id = getattr(run, "id", None)
    if run_id:
        _calls, _retries, fails = _llm_jsonl_stats_for_run(int(run_id))
        if fails:
            return fails
    details = run.details if isinstance(run.details, list) else []
    from_details = _llm_status_count(details, "error")
    if from_details:
        return from_details
    stored = int(getattr(run, "llm_errors", None) or 0)
    if stored:
        return stored
    legacy = 0
    for item in details:
        if not isinstance(item, dict):
            continue
        if item.get("action") != "error" and item.get("error"):
            legacy += 1
    return legacy


def _llm_calls_of(details: list | None) -> int:
    if not isinstance(details, list):
        return 0
    count = 0
    for item in details:
        if not isinstance(item, dict):
            continue
        llm = item.get("llm")
        if not isinstance(llm, dict):
            continue
        for kind, _label in LLM_INFO_KINDS:
            info = llm.get(kind)
            if isinstance(info, dict) and info.get("status") in {"generated", "error"}:
                count += 1
    return count


def format_yt_duration(value: Any) -> str:
    try:
        sec = int(value)
    except (TypeError, ValueError):
        return ""
    if sec <= 0:
        return ""
    hours, rem = divmod(sec, 3600)
    minutes, _seconds = divmod(rem, 60)
    if hours:
        return f"{hours} ч {minutes} мин"
    return f"{minutes} мин"


def format_yt_views(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return ""
    if count >= 1_000_000:
        text = f"{count / 1_000_000:.1f}".replace(".0", "")
        return f"{text} млн"
    if count >= 1_000:
        text = f"{count / 1_000:.1f}".replace(".0", "")
        return f"{text} тыс."
    return str(count)


templates.env.filters["action_counts"] = _action_counts
templates.env.filters["localtime"] = format_localtime
templates.env.filters["action_ru"] = action_ru
templates.env.filters["status_ru"] = status_ru
templates.env.filters["change_line"] = format_change_line
templates.env.filters["change_summary"] = change_summary
templates.env.filters["llm_lines"] = llm_lines
templates.env.filters["llm_error_count"] = _llm_errors_of
templates.env.filters["llm_kind_ru"] = lambda value: LLM_KIND_RU.get(
    str(value or ""), str(value or "")
)
templates.env.filters["yt_duration"] = format_yt_duration
templates.env.filters["yt_views"] = format_yt_views
templates.env.globals["pagination_window"] = pagination_window


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def game_to_dict(game: Game, *, detailed: bool = False) -> dict[str, Any]:
    """Сериализует игру для JSON API."""
    payload: dict[str, Any] = {
        "id": game.id,
        "slug": game.slug,
        "title": game.title,
        "cover_url": normalize_cover_url(game.cover_url),
        "description": game.description,
        "developer": game.developer,
        "developers": game.developers,
        "publishers": game.publishers,
        "genres": game.genres,
        "video_url": game.video_url,
        "video_title": game.video_title,
        "youtube_url": game.youtube_url if letsplay_has_summary(game) else None,
        "youtube_title": game.youtube_title if letsplay_has_summary(game) else None,
        "youtube_summary": game.youtube_summary if letsplay_has_summary(game) else None,
        "youtube_summary_source": getattr(game, "youtube_summary_source", None)
        if letsplay_has_summary(game)
        else None,
        "youtube_channel": getattr(game, "youtube_channel", None),
        "youtube_views": getattr(game, "youtube_views", None),
        "youtube_duration_sec": getattr(game, "youtube_duration_sec", None),
        "youtube_kind": getattr(game, "youtube_kind", None),
        "related_games": game.related_games,
        "release_date": _jsonable(game.release_date),
        "metascore": game.metascore,
        "userscore": game.userscore,
        "is_from_carousel": bool(game.is_from_carousel),
        "tags": list(game.tags or game.ai_tags or []),
        "ai_tags": list(game.tags or game.ai_tags or []),
        "genre_detailed": game.genre_detailed,
        "key_features": list(game.key_features or []),
        "target_audience": game.target_audience,
        "last_processed_at": _jsonable(game.last_processed_at),
    }
    if detailed:
        payload["platforms"] = [
            {
                "platform": item.platform,
                "metascore": item.metascore,
                "userscore": item.userscore,
                "userscore_count": item.userscore_count,
                "critic_count": item.critic_count,
                "release_date": _jsonable(item.release_date),
            }
            for item in game.platform_scores
        ]
        payload["reviews"] = [
            {
                "source": item.source,
                "author": item.author,
                "publication": item.publication,
                "publicationName": item.publication,
                "score": item.score,
                "quote": item.body,
                "date": _jsonable(item.reviewed_at),
                "platform": item.platform,
                "url": item.url,
            }
            for item in game.reviews
        ]
        payload["summaries"] = [
            {
                "id": item.id,
                "kind": item.kind,
                "content": item.content,
                "model": item.model,
                "created_at": _jsonable(item.created_at),
            }
            for item in game.summaries
        ]
        payload["similar_games"] = []
    return payload


def _qs(**kwargs: Any) -> str:
    params: dict[str, str] = {}
    for key, value in kwargs.items():
        if value is None or value == "" or value is False:
            continue
        params[key] = str(value)
    return ("?" + urlencode(params)) if params else ""


GAMES_SORTS = {"carousel", "rating", "date"}
GAMES_SCORES = {"all", "high", "mixed", "low", "tbd"}


def _games_qs(
    *,
    sort: str = "carousel",
    score: str = "all",
    q: str = "",
    platform: str = "",
    page: int = 1,
) -> str:
    return _qs(
        sort=sort if sort != "carousel" else None,
        score=score if score not in {"", "all"} else None,
        q=q or None,
        platform=platform or None,
        page=page if page > 1 else None,
    )


def like_needle(q: str) -> str | None:
    """Шаблон LIKE по названию: нижний регистр, экранированы % и _."""
    text = " ".join((q or "").split()).lower()
    if not text:
        return None
    escaped = text.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    return f"%{escaped}%"


def _games_title_filter(q: str):
    needle = like_needle(q)
    if not needle:
        return None
    return func.lower(Game.title).like(needle, escape="\\")


def _games_score_filter(score: str):
    if score == "high":
        return Game.metascore >= 75
    if score == "mixed":
        return Game.metascore.between(50, 74)
    if score == "low":
        return (Game.metascore.is_not(None)) & (Game.metascore < 50)
    if score == "tbd":
        return Game.metascore.is_(None)
    return None


def _games_order(sort: str):
    if sort == "rating":
        return (Game.metascore.desc().nullslast(), Game.title.asc())
    if sort == "date":
        return (Game.release_date.desc().nullslast(), Game.title.asc())
    return (
        Game.is_from_carousel.desc(),
        Game.created_at.desc(),
        Game.title.asc(),
    )


def _monitor_qs(
    *,
    page: int = 1,
    llm_page: int = 1,
    model: str = "",
    ok: str | None = None,
    date_from: str = "",
    date_to: str = "",
    run_status: str = "",
    run_from: str = "",
    run_to: str = "",
    llm_run: str = "",
    kind: str = "",
) -> str:
    return _qs(
        page=page if page > 1 else None,
        llm_page=llm_page if llm_page > 1 else None,
        model=model or None,
        ok=ok if ok in {"0", "1"} else None,
        date_from=date_from or None,
        date_to=date_to or None,
        run_status=run_status if run_status in STATUS_RU else None,
        run_from=run_from or None,
        run_to=run_to or None,
        llm_run=llm_run or None,
        kind=kind if kind in LLM_KIND_FILTERS else None,
    )


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _short_error(text: str | None) -> str:
    if not text:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= 80:
        return compact
    return compact[:80].rstrip() + "…"


def _parse_llm_run(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("#"):
        text = text[1:].strip()
    return _coerce_int(text)


def _collect_llm_records(
    *,
    model: str = "",
    ok: str | None = None,
    date_from: str = "",
    date_to: str = "",
    llm_run: str = "",
    kind: str = "",
    runs: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    path = get_settings().llm_log_path
    configured_models = _configured_llm_models()
    if not path.exists():
        return [], configured_models, list(LLM_KIND_ORDER)
    records: list[dict[str, Any]] = []
    models: set[str] = set(configured_models)
    kinds: set[str] = set(LLM_KIND_ORDER)
    model_exact = model.strip()
    kind_exact = kind.strip() if kind.strip() in LLM_KIND_FILTERS else ""
    from_day = _parse_iso_date(date_from)
    to_day = _parse_iso_date(date_to)
    run_raw = llm_run.strip()
    run_filter = _parse_llm_run(run_raw) if run_raw else None
    last_start: dict[tuple[Any, ...], int] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_error = item.get("error")
        status = item.get("error_status")
        try:
            status_int = int(status) if status is not None else None
        except (TypeError, ValueError):
            status_int = None
        model_name = item.get("model") or ""
        if model_name:
            models.add(str(model_name))
        is_ok = bool(item.get("ok"))
        attempt = _coerce_int(item.get("attempt")) or 1
        rec_kind = str(item.get("kind") or "")
        if rec_kind:
            kinds.add(rec_kind)
        slug = _llm_game_slug(item)
        rec_run_id = _bind_llm_run_id(_coerce_int(item.get("run_id")), slug, item.get("ts"), runs)
        chain_key = (rec_run_id, slug, rec_kind)
        origin_id = last_start.get(chain_key) if attempt > 1 else None
        if attempt <= 1:
            last_start[chain_key] = line_no
        if model_exact and str(model_name) != model_exact:
            continue
        if kind_exact and rec_kind != kind_exact:
            continue
        if ok == "1" and not is_ok:
            continue
        if ok == "0" and is_ok:
            continue
        local_dt = to_local(item.get("ts"))
        local_day = local_dt.date() if local_dt else None
        if from_day and (local_day is None or local_day < from_day):
            continue
        if to_day and (local_day is None or local_day > to_day):
            continue
        if run_raw:
            if run_filter is None or rec_run_id != run_filter:
                continue
        records.append(
            {
                "id": line_no,
                "ts": item.get("ts"),
                "model": model_name,
                "ok": is_ok,
                "stub": item.get("stub"),
                "latency_ms": item.get("latency_ms"),
                "game_slug": slug,
                "run_id": rec_run_id,
                "attempt": attempt,
                "kind": rec_kind,
                "origin_id": origin_id,
                "error": _short_error(humanize_llm_error(raw_error, status_int, compact=True)),
            }
        )
    records.reverse()
    model_order = configured_models + sorted(name for name in models if name not in configured_models)
    kind_order = list(LLM_KIND_ORDER)
    kind_order.extend(sorted(name for name in kinds if name not in kind_order))
    return records, model_order, kind_order


def _read_llm_logs(
    *,
    page: int = 1,
    per_page: int = LLM_PER_PAGE,
    model: str = "",
    ok: str | None = None,
    date_from: str = "",
    date_to: str = "",
    llm_run: str = "",
    kind: str = "",
    runs: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], int, int, int, list[str], list[str]]:
    records, models, kinds = _collect_llm_records(
        model=model,
        ok=ok,
        date_from=date_from,
        date_to=date_to,
        llm_run=llm_run,
        kind=kind,
        runs=runs,
    )
    total = len(records)
    total_pages = max(1, (total + per_page - 1) // per_page) if total else 1
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    page_records = records[start : start + per_page]
    id_pos = {item["id"]: index for index, item in enumerate(records)}
    for item in page_records:
        oid = item.get("origin_id")
        if not oid:
            continue
        pos = id_pos.get(oid)
        origin_page = (
            (pos // per_page + 1)
            if pos is not None
            else _llm_page_for_id(oid, llm_run=llm_run, kind=kind, runs=runs)
        )
        item["origin_href"] = (
            f"/monitor{_monitor_qs(llm_page=origin_page, model=model, ok=ok, date_from=date_from, date_to=date_to, llm_run=llm_run, kind=kind)}#llm-{oid}"
        )
    return page_records, page, total_pages, total, models, kinds


def _llm_page_for_id(
    log_id: int,
    *,
    llm_run: str = "",
    kind: str = "",
    per_page: int = LLM_PER_PAGE,
    runs: list[Any] | None = None,
) -> int:
    records, _models, _kinds = _collect_llm_records(llm_run=llm_run, kind=kind, runs=runs)
    for index, item in enumerate(records):
        if item["id"] == log_id:
            return index // per_page + 1
    if llm_run or kind:
        records, _models, _kinds = _collect_llm_records(runs=runs)
        for index, item in enumerate(records):
            if item["id"] == log_id:
                return index // per_page + 1
    return 1


def _llm_page_map(
    *, llm_run: str = "", per_page: int = LLM_PER_PAGE, runs: list[Any] | None = None
) -> dict[int, int]:
    records, _models, _kinds = _collect_llm_records(llm_run=llm_run, runs=runs)
    return {item["id"]: index // per_page + 1 for index, item in enumerate(records)}


def _llm_table_href(log_id: int, run_id: int | None = None, *, page: int | None = None) -> str:
    if page:
        llm_run = str(run_id) if run_id else ""
        return f"/monitor{_monitor_qs(llm_page=page, llm_run=llm_run)}#llm-{log_id}"
    resolved = _llm_page_for_id(log_id)
    return f"/monitor{_monitor_qs(llm_page=resolved)}#llm-{log_id}"


def _read_llm_record(log_id: int, runs: list[Any] | None = None) -> dict[str, Any] | None:
    path = get_settings().llm_log_path
    if not path.exists() or log_id < 1:
        return None
    last_start: dict[tuple[Any, ...], int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = _llm_game_slug(item)
            rec_run_id = _bind_llm_run_id(_coerce_int(item.get("run_id")), slug, item.get("ts"), runs)
            kind = str(item.get("kind") or "")
            attempt = _coerce_int(item.get("attempt")) or 1
            chain_key = (rec_run_id, slug, kind)
            origin_id = last_start.get(chain_key) if attempt > 1 else None
            if attempt <= 1:
                last_start[chain_key] = line_no
            if line_no != log_id:
                continue
            status = item.get("error_status")
            try:
                status_int = int(status) if status is not None else None
            except (TypeError, ValueError):
                status_int = None
            return {
                "id": log_id,
                "ts": item.get("ts"),
                "model": item.get("model"),
                "ok": bool(item.get("ok")),
                "stub": item.get("stub"),
                "latency_ms": item.get("latency_ms"),
                "slug": slug,
                "game_slug": slug,
                "run_id": rec_run_id,
                "attempt": attempt,
                "kind": kind,
                "origin_id": origin_id,
                "origin_href": _llm_table_href(origin_id, rec_run_id) if origin_id else "",
                "system": item.get("system") or "",
                "prompt": item.get("prompt") or "",
                "response": item.get("response") or "",
                "error": humanize_llm_error(item.get("error"), status_int),
                "error_status": status_int,
            }
    return None


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: DbSession) -> HTMLResponse:
    result = await db.execute(
        select(Game)
        .order_by(Game.created_at.desc(), Game.id.desc())
        .limit(20)
    )
    games = result.scalars().all()
    return templates.TemplateResponse(request, "index.html", {"games": games})


@router.get("/games", response_class=HTMLResponse)
async def games_page(
    request: Request,
    db: DbSession,
    page: int = 1,
    sort: str = "carousel",
    score: str = "all",
    q: str = "",
    platform: str = "",
) -> HTMLResponse:
    per_page = 20
    page = max(1, page)
    sort = sort if sort in GAMES_SORTS else "carousel"
    score = score if score in GAMES_SCORES else "all"
    query = " ".join((q or "").split())
    platform_name = " ".join((platform or "").split())
    filt = _games_score_filter(score)
    title_filt = _games_title_filter(query)
    count_stmt = select(func.count()).select_from(Game)
    stmt = select(Game)
    if filt is not None:
        count_stmt = count_stmt.where(filt)
        stmt = stmt.where(filt)
    if title_filt is not None:
        count_stmt = count_stmt.where(title_filt)
        stmt = stmt.where(title_filt)
    if platform_name:
        plat_ids = select(PlatformScore.game_id).where(PlatformScore.platform == platform_name)
        count_stmt = count_stmt.where(Game.id.in_(plat_ids))
        stmt = stmt.where(Game.id.in_(plat_ids))
    total = int(await db.scalar(count_stmt) or 0)
    total_pages = max(1, (total + per_page - 1) // per_page) if total else 1
    page = min(page, total_pages)
    offset = (page - 1) * per_page
    result = await db.execute(
        stmt.order_by(*_games_order(sort)).offset(offset).limit(per_page)
    )
    games = result.scalars().all()
    platform_rows = await db.execute(
        select(PlatformScore.platform).distinct().order_by(PlatformScore.platform.asc())
    )
    platforms = [name for name in platform_rows.scalars().all() if name]
    games_qs = _games_qs(sort=sort, score=score, q=query, platform=platform_name)
    return templates.TemplateResponse(
        request,
        "games_list.html",
        {
            "games": games,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "sort": sort,
            "score": score,
            "q": query,
            "platform": platform_name,
            "platforms": platforms,
            "games_base": "/games" + games_qs,
        },
    )


@router.get("/games/{slug}", response_class=HTMLResponse)
async def game_page(slug: str, request: Request, db: DbSession) -> HTMLResponse:
    game = await db.scalar(
        select(Game)
        .where(Game.slug == slug)
        .options(
            selectinload(Game.platform_scores),
            selectinload(Game.reviews),
            selectinload(Game.summaries),
        )
    )
    if game is None:
        raise HTTPException(status_code=404, detail="Игра не найдена")
    critic_reviews = [item for item in game.reviews if item.source == "critic"]
    user_reviews = [item for item in game.reviews if item.source == "user"]
    critic_summary = None
    user_summary = None
    for item in game.summaries:
        if item.kind == "critic" and (critic_summary is None or item.created_at > critic_summary.created_at):
            critic_summary = item
        elif item.kind == "user" and (user_summary is None or item.created_at > user_summary.created_at):
            user_summary = item
    return templates.TemplateResponse(
        request,
        "game.html",
        {
            "game": game,
            "critic_reviews": critic_reviews,
            "user_reviews": user_reviews,
            "critic_summary": critic_summary,
            "user_summary": user_summary,
            "similar_games": await similar_payload(db, game),
        },
    )


@router.get("/monitor", response_class=HTMLResponse)
async def monitor_page(
    request: Request,
    db: DbSession,
    page: int = 1,
    llm_page: int = 1,
    model: str = "",
    ok: str | None = None,
    date_from: str = "",
    date_to: str = "",
    run_status: str = "",
    run_from: str = "",
    run_to: str = "",
    llm_run: str = "",
    kind: str = "",
) -> HTMLResponse:
    runs_per_page = 10
    page = max(1, page)
    status_filter = run_status if run_status in STATUS_RU else ""
    from_day = _parse_iso_date(run_from)
    to_day = _parse_iso_date(run_to)
    result = await db.execute(select(RunLog).order_by(RunLog.started_at.desc()))
    all_runs = list(result.scalars().all())
    filtered_runs: list[RunLog] = []
    for run in all_runs:
        if status_filter and run.status != status_filter:
            continue
        local_dt = to_local(run.started_at)
        local_day = local_dt.date() if local_dt else None
        if from_day and (local_day is None or local_day < from_day):
            continue
        if to_day and (local_day is None or local_day > to_day):
            continue
        filtered_runs.append(run)
    total_runs = len(filtered_runs)
    total_pages = max(1, (total_runs + runs_per_page - 1) // runs_per_page) if total_runs else 1
    page = min(page, total_pages)
    start = (page - 1) * runs_per_page
    runs = filtered_runs[start : start + runs_per_page]
    last_run = await db.scalar(select(RunLog).order_by(RunLog.started_at.desc()).limit(1))
    games_count = int(await db.scalar(select(func.count()).select_from(Game)) or 0)
    last_counts = _action_counts(last_run.details if last_run else None)
    last_llm_errors = _llm_errors_of(last_run) if last_run else 0
    ok_filter = ok if ok in {"0", "1"} else None
    llm_run_value = llm_run.strip()
    kind_filter = kind.strip() if kind.strip() in LLM_KIND_FILTERS else ""
    llm_logs, llm_page, llm_total_pages, llm_total, llm_models, llm_kinds = _read_llm_logs(
        page=llm_page,
        per_page=LLM_PER_PAGE,
        model=model,
        ok=ok_filter,
        date_from=date_from,
        date_to=date_to,
        llm_run=llm_run_value,
        kind=kind_filter,
        runs=all_runs,
    )
    runs_qs = _monitor_qs(
        llm_page=llm_page,
        model=model,
        ok=ok_filter,
        date_from=date_from,
        date_to=date_to,
        run_status=status_filter,
        run_from=run_from,
        run_to=run_to,
        llm_run=llm_run_value,
        kind=kind_filter,
    )
    llm_qs = _monitor_qs(
        page=page,
        model=model,
        ok=ok_filter,
        date_from=date_from,
        date_to=date_to,
        run_status=status_filter,
        run_from=run_from,
        run_to=run_to,
        llm_run=llm_run_value,
        kind=kind_filter,
    )
    runtime = await _queue_runtime()
    return templates.TemplateResponse(
        request,
        "monitor.html",
        {
            "runs": runs,
            "page": page,
            "total_pages": total_pages,
            "total_runs": total_runs,
            "pipeline_running": runtime["pipeline_running"],
            "enrichment_running": runtime["enrichment_running"],
            "queue_similar": runtime["queue_similar"],
            "queue_youtube": runtime["queue_youtube"],
            "groq_pause_sec": runtime.get("groq_pause_sec") or 0,
            "llm_logs": llm_logs,
            "llm_page": llm_page,
            "llm_total_pages": llm_total_pages,
            "llm_total": llm_total,
            "llm_model": model,
            "llm_ok": ok_filter or "",
            "llm_date_from": date_from,
            "llm_date_to": date_to,
            "llm_run": llm_run_value,
            "llm_kind": kind_filter,
            "llm_kinds": llm_kinds,
            "llm_models": llm_models,
            "llm_calls": _llm_log_count(),
            "games_count": games_count,
            "last_run": last_run,
            "last_counts": last_counts,
            "last_llm_errors": last_llm_errors,
            "next_run_at": _next_run_at_iso(),
            "monitor_qs": _monitor_qs,
            "run_status": status_filter,
            "run_from": run_from,
            "run_to": run_to,
            "runs_base": "/monitor" + runs_qs,
            "llm_base": "/monitor" + llm_qs,
        },
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail_page(run_id: int, request: Request, db: DbSession) -> HTMLResponse:
    run = await db.get(RunLog, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    details = run.details if isinstance(run.details, list) else []
    all_runs = list(
        (await db.execute(select(RunLog).order_by(RunLog.started_at.desc()))).scalars().all()
    )
    entries_map = _llm_entries_by_slug_for_run(run.id, all_runs)
    ok_map = _llm_ok_by_id_for_run(run.id, all_runs)
    pages = _llm_page_map(llm_run=str(run.id), runs=all_runs)
    rows: list[dict[str, Any]] = []
    for item in details:
        row = dict(item) if isinstance(item, dict) else {"title": str(item), "changes": []}
        entries = list(entries_map.get(str(row.get("slug") or "")) or [])
        if not entries:
            for log_id in _ids_from_llm_info(row.get("llm")):
                entries.append(
                    {
                        "id": log_id,
                        "kind": "",
                        "ok": ok_map.get(log_id) is not False,
                    }
                )
        ok_entries = [entry for entry in entries if entry.get("ok") is not False]
        err_entries = [entry for entry in entries if entry.get("ok") is False]
        row["llm_log_ids"] = [int(entry["id"]) for entry in ok_entries]
        row["llm_links"] = [
            {
                "id": entry["id"],
                "kind": entry.get("kind") or "",
                "href": _llm_table_href(int(entry["id"]), run.id, page=pages.get(int(entry["id"]))),
            }
            for entry in ok_entries
        ]
        row["llm_error_links"] = [
            {
                "id": entry["id"],
                "kind": entry.get("kind") or "",
                "href": _llm_table_href(int(entry["id"]), run.id, page=pages.get(int(entry["id"]))),
            }
            for entry in err_entries
        ]
        row["reason"] = build_status_reason(
            str(row.get("action") or ""),
            changes=row.get("changes") if isinstance(row.get("changes"), list) else [],
            llm=row.get("llm") if isinstance(row.get("llm"), dict) else None,
            error=str(row.get("error") or "") or None,
        )
        rows.append(row)
    calls, _retries, fails = _llm_jsonl_stats_for_run(run.id, all_runs)
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "run": run,
            "details": rows,
            "counts": _action_counts(details),
            "llm_errors": fails or _llm_errors_of(run),
            "llm_calls": calls or _llm_calls_of(details),
        },
    )


@router.get("/llm/{log_id}", response_class=HTMLResponse)
async def llm_dialog_page(log_id: int, request: Request, db: DbSession) -> HTMLResponse:
    all_runs = list(
        (await db.execute(select(RunLog).order_by(RunLog.started_at.desc()))).scalars().all()
    )
    item = _read_llm_record(log_id, all_runs)
    if item is None:
        raise HTTPException(status_code=404, detail="Запись LLM не найдена")
    run_id = item.get("run_id")
    pages = _llm_page_map(llm_run=str(run_id), runs=all_runs) if run_id else {}
    return templates.TemplateResponse(
        request,
        "llm_detail.html",
        {
            "item": item,
            "llm_table_href": _llm_table_href(item["id"], run_id, page=pages.get(item["id"])),
        },
    )


@router.get("/api/docs")
async def api_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get("/api/health")
async def health() -> dict[str, Any]:
    runtime = await _queue_runtime()
    return {
        "status": "ok",
        "next_run_at": _next_run_at_iso(),
        **runtime,
    }


@router.get("/api/games")
async def api_games(db: DbSession) -> dict[str, Any]:
    result = await db.execute(
        select(Game).order_by(Game.release_date.desc().nullslast(), Game.title.asc())
    )
    games = result.scalars().all()
    return {"items": [game_to_dict(game) for game in games]}


@router.get("/api/games/{slug}")
async def api_game(slug: str, db: DbSession) -> dict[str, Any]:
    game = await db.scalar(
        select(Game)
        .where(Game.slug == slug)
        .options(
            selectinload(Game.platform_scores),
            selectinload(Game.reviews),
            selectinload(Game.summaries),
        )
    )
    if game is None:
        raise HTTPException(status_code=404, detail="Игра не найдена")
    payload = game_to_dict(game, detailed=True)
    payload["similar_games"] = await similar_payload(db, game)
    return payload


@router.get("/api/runs")
async def api_runs(db: DbSession) -> dict[str, Any]:
    result = await db.execute(select(RunLog).order_by(RunLog.started_at.desc()).limit(50))
    runs = result.scalars().all()
    runtime = await _queue_runtime()
    return {
        "next_run_at": _next_run_at_iso(),
        **runtime,
        "items": [
            {
                "id": run.id,
                "started_at": _jsonable(run.started_at),
                "finished_at": _jsonable(run.finished_at),
                "status": run.status,
                "games_found": run.games_found,
                "games_processed": run.games_processed,
                "error_message": run.error_message,
                "details": run.details,
                "llm_errors": getattr(run, "llm_errors", 0) or 0,
            }
            for run in runs
        ],
    }


@router.post("/api/pipeline/run")
async def trigger_pipeline(request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
    if pipeline_is_running():
        return {"status": "already_running"}
    force_resummarize = False
    source_override = None
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            force_resummarize = bool(payload.get("force_resummarize"))
            raw_source = payload.get("source_override")
            if raw_source in {"main", "browse"}:
                source_override = raw_source
    background_tasks.add_task(
        run_hourly_pipeline,
        manual=True,
        force_resummarize=force_resummarize,
        source_override=source_override,
    )
    return {"status": "accepted"}


@router.post("/api/pipeline/similar")
async def trigger_similar(background_tasks: BackgroundTasks) -> dict[str, str]:
    background_tasks.add_task(run_similar_now)
    return {"status": "accepted"}


@router.post("/api/pipeline/youtube")
async def trigger_youtube(background_tasks: BackgroundTasks) -> dict[str, str]:
    background_tasks.add_task(run_youtube_now)
    return {"status": "accepted"}
