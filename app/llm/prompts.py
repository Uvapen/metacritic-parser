"""Промпты для саммари по отзывам Metacritic."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

PROMPT_REVIEW_LIMIT = 6
PROMPT_REVIEW_MAX_CHARS = 350
SUMMARY_MAX_CHARS = 600
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def clip_review_snippets(
    snippets: Sequence[str],
    *,
    limit: int = PROMPT_REVIEW_LIMIT,
    max_chars: int = PROMPT_REVIEW_MAX_CHARS,
) -> list[str]:
    """Обрезает отзывы для промпта: не больше limit штук и max_chars символов."""
    clipped: list[str] = []
    for raw in snippets:
        text = " ".join(str(raw or "").split())
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        clipped.append(text)
        if len(clipped) >= limit:
            break
    return clipped


def clip_summary_text(text: str | None, *, limit: int = SUMMARY_MAX_CHARS) -> str:
    """Обрезает саммари до limit символов, по возможности по пробелу."""
    raw = " ".join(str(text or "").split()).strip()
    if len(raw) <= limit:
        return raw
    cut = raw[:limit].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "…"


def looks_russian(text: str | None) -> bool:
    """True, если в тексте есть кириллица."""
    return bool(_CYRILLIC.search(str(text or "")))


ANALYST_SYSTEM_PROMPT = """Ты эксперт-аналитик видеоигр. Твоя задача:
1. Проанализировать описание игры и отзывы (критиков/игроков)
2. Составить нейтральное саммари СТРОГО на русском языке (3-4 предложения, ≤600 символов)
3. Извлечь структурированные данные для системы рекомендаций

Отвечай СТРОГО в формате JSON без markdown-обёрток:
{
  "summary": "текст саммари на русском",
  "tags": ["тег1", "тег2"],
  "genre_detailed": "уточнённый жанр",
  "key_features": ["фича1", "фича2"],
  "target_audience": "для кого игра"
}

ПРАВИЛА:

summary: 3-4 предложения на русском, не более 600 символов. Только суть: что хвалят, что ругают. Без вступлений. Даже если отзывы на английском — пиши summary кириллицей, не на английском.

tags (3-7 штук): конкретные механики и особенности на английском kebab-case.
Примеры: souls-like, open-world, stealth, boss-rush, crafting, multiplayer, story-rich, dark-fantasy, remaster, turn-based.
Не используй общие слова: game, fun, good, interesting.

genre_detailed: один уточнённый жанр kebab-case.
Примеры: action-rpg, hack-and-slash, tactical-rpg, stealth-action, platformer-metroidvania.

key_features (2-4): что делает игру уникальной, коротко на русском.
Примеры: "эпические битвы с боссами", "нелинейный сюжет", "кооператив до 4 игроков".

target_audience: кто оценит игру, на русском.
Примеры: "фанаты серии", "любители сложных игр", "ценители сюжета", "казуальные игроки".

Не выдумывай факты, которых нет во входных данных. Если отзывов нет — опирайся только на описание. Если поле нельзя определить — пустой массив или null.

Пример корректного ответа:
{"summary":"Критики хвалят боёвку и мир, ругают оптимизацию.","tags":["souls-like","open-world","dark-fantasy","boss-rush"],"genre_detailed":"action-rpg","key_features":["открытый мир","сложные боссы"],"target_audience":"любители сложных игр"}"""

SUMMARY_TEXT_SYSTEM_PROMPT = """Ты эксперт-аналитик видеоигр.
Составь нейтральное саммари СТРОГО на русском языке: 3-4 предложения, ≤600 символов.
Только суть: что хвалят, что ругают. Без вступлений.
Даже если отзывы на английском — пиши кириллицей, не на английском.
Структурные признаки (tags, genre, features) уже извлечены — не повторяй их.

Отвечай СТРОГО JSON без markdown-обёрток:
{"summary": "текст саммари на русском"}"""

CRITIC_SUMMARY_SYSTEM_PROMPT = ANALYST_SYSTEM_PROMPT
USER_SUMMARY_SYSTEM_PROMPT = ANALYST_SYSTEM_PROMPT
SUMMARY_SYSTEM_PROMPT = ANALYST_SYSTEM_PROMPT


def _format_genres(genres: Sequence[str] | None) -> str:
    names = [str(item).strip() for item in (genres or []) if str(item).strip()]
    return ", ".join(names) if names else "n/a"


def build_summary_prompt(
    *,
    title: str,
    description: str | None,
    metascore: int | None,
    userscore: float | None,
    critic_snippets: Sequence[str],
    user_snippets: Sequence[str],
    genres: Sequence[str] | None = None,
) -> str:
    """Собирает пользовательский промпт для генерации саммари."""
    critics = (
        "\n".join(f"- {text}" for text in clip_review_snippets(critic_snippets))
        or "- (нет отзывов критиков)"
    )
    users = (
        "\n".join(f"- {text}" for text in clip_review_snippets(user_snippets))
        or "- (нет отзывов игроков)"
    )
    description_block = description.strip() if description else "(описание отсутствует)"
    return (
        f"Игра: {title}\n"
        f"Жанры Metacritic: {_format_genres(genres)}\n"
        f"Metascore: {metascore if metascore is not None else 'n/a'}\n"
        f"Userscore: {userscore if userscore is not None else 'n/a'}\n\n"
        f"Описание:\n{description_block}\n\n"
        f"Отзывы критиков:\n{critics}\n\n"
        f"Отзывы игроков:\n{users}\n"
    )


def _json_instruction(*, include_features: bool) -> str:
    if include_features:
        return (
            "Верни JSON с полями summary, tags, genre_detailed, key_features, "
            "target_audience. Поле summary — только на русском.\n"
        )
    return (
        "Признаки уже есть. Верни JSON только с полем summary на русском: "
        '{"summary": "..."}. Не добавляй tags, genre_detailed, key_features, target_audience.\n'
    )


def build_critic_summary_prompt(
    *,
    title: str,
    description: str | None,
    metascore: int | None,
    critic_snippets: Sequence[str],
    genres: Sequence[str] | None = None,
    include_features: bool = True,
) -> str:
    """Промпт только по отзывам критиков."""
    critics = "\n".join(f"- {text}" for text in clip_review_snippets(critic_snippets))
    description_block = description.strip() if description else "(описание отсутствует)"
    return (
        f"Игра: {title}\n"
        f"Жанры Metacritic: {_format_genres(genres)}\n"
        f"Metascore: {metascore if metascore is not None else 'n/a'}\n\n"
        f"Описание:\n{description_block}\n\n"
        f"Отзывы критиков:\n{critics or '- (нет отзывов критиков)'}\n\n"
        "Источник: отзывы критиков (если их нет — только описание). "
        f"{_json_instruction(include_features=include_features)}"
    )


def build_user_summary_prompt(
    *,
    title: str,
    description: str | None,
    userscore: float | None,
    user_snippets: Sequence[str],
    genres: Sequence[str] | None = None,
    include_features: bool = True,
) -> str:
    """Промпт только по отзывам игроков."""
    users = "\n".join(f"- {text}" for text in clip_review_snippets(user_snippets))
    description_block = description.strip() if description else "(описание отсутствует)"
    return (
        f"Игра: {title}\n"
        f"Жанры Metacritic: {_format_genres(genres)}\n"
        f"Userscore: {userscore if userscore is not None else 'n/a'}\n\n"
        f"Описание:\n{description_block}\n\n"
        f"Отзывы игроков:\n{users or '- (нет отзывов игроков)'}\n\n"
        "Источник: отзывы игроков (если их нет — только описание). "
        f"{_json_instruction(include_features=include_features)}"
    )


_TAG_ALIAS = {
    "soulslike": "souls-like",
    "openworld": "open-world",
    "darkfantasy": "dark-fantasy",
    "actionrpg": "action-rpg",
    "scifi": "sci-fi",
    "thirdperson": "third-person",
    "firstperson": "first-person",
}
_TAG_SLUG = re.compile(r"[^a-z0-9]+")


def normalize_ai_tag(value: str) -> str | None:
    raw = str(value or "").strip().lower().replace("_", "-")
    compact = re.sub(r"[^a-z0-9]", "", raw)
    if compact in _TAG_ALIAS:
        return _TAG_ALIAS[compact]
    slug = _TAG_SLUG.sub("-", raw).strip("-")
    if len(slug) < 2 or len(slug) > 48:
        return None
    return slug


def _strip_markdown_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return raw


def _normalize_tag_list(values: object, *, limit: int = 7) -> list[str]:
    if not isinstance(values, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str | int | float):
            continue
        tag = normalize_ai_tag(str(item))
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def _normalize_features(values: object, *, limit: int = 4) -> list[str]:
    if not isinstance(values, list):
        return []
    features: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split()).strip()
        if len(text) < 3 or len(text) > 80:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        features.append(text)
        if len(features) >= limit:
            break
    return features


def _normalize_genre_detailed(value: object) -> str | None:
    if value is None:
        return None
    tag = normalize_ai_tag(str(value))
    if not tag or tag in {"null", "none", "n-a", "na"}:
        return None
    return tag


def _normalize_audience(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text[:256]


@dataclass
class SummaryExtraction:
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    genre_detailed: str | None = None
    key_features: list[str] = field(default_factory=list)
    target_audience: str | None = None
    parsed: bool = False


def parse_ai_tags(text: str | None, *, limit: int = 7) -> list[str]:
    """Теги из JSON-массива или из объекта саммари."""
    return parse_summary_payload(text, tag_limit=limit).tags


def parse_summary_payload(text: str | None, *, tag_limit: int = 7) -> SummaryExtraction:
    """Разбирает JSON саммари. При поломке — сырой текст, пустые признаки."""
    raw = _strip_markdown_fences(str(text or ""))
    if not raw:
        return SummaryExtraction()
    blob = raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        blob = raw[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return SummaryExtraction(summary=clip_summary_text(raw), parsed=False)
    if isinstance(data, list):
        return SummaryExtraction(
            summary=clip_summary_text(raw),
            tags=_normalize_tag_list(data, limit=tag_limit),
            parsed=False,
        )
    if not isinstance(data, dict):
        return SummaryExtraction(summary=clip_summary_text(raw), parsed=False)
    try:
        from app.schemas import LlmSummaryResponse

        payload = LlmSummaryResponse.model_validate(data)
        summary = (payload.summary or "").strip()
        tags = _normalize_tag_list(payload.tags or [], limit=tag_limit)
        genre = _normalize_genre_detailed(payload.genre_detailed)
        features = _normalize_features(payload.key_features)
        audience = _normalize_audience(payload.target_audience)
    except Exception:
        summary = str(data.get("summary") or "").strip()
        tags = _normalize_tag_list(data.get("tags"), limit=tag_limit)
        genre = _normalize_genre_detailed(data.get("genre_detailed"))
        features = _normalize_features(data.get("key_features"))
        audience = _normalize_audience(data.get("target_audience"))
    if not summary:
        summary = raw
    summary = clip_summary_text(summary)
    return SummaryExtraction(
        summary=summary,
        tags=tags,
        genre_detailed=genre,
        key_features=features,
        target_audience=audience,
        parsed=True,
    )


SIMILAR_REVIEW_SYSTEM_PROMPT = """Ты проверяешь рекомендации похожих игр.
Нужно оставить только те кандидаты, которые близки по сути: тот же род опыта
(боевой экшен к боевому экшену, тактическая RPG к тактической RPG).
Нельзя считать похожими разные семьи: экшен с боями и пасьянс/детская головоломка,
симулятор автобуса и souls-like, спорт и survival.
Слабое пересечение одного широкого слова (puzzle, story, exploration) — не повод оставить.
Если сомневаешься — не оставляй.
Ответ СТРОГО JSON: {"keep": ["slug", ...]} без markdown."""


def build_similar_review_prompt(
    *,
    origin_brief: str,
    candidates: Sequence[dict[str, Any]],
) -> str:
    lines: list[str] = []
    for item in candidates:
        reasons = ", ".join(str(part) for part in (item.get("reasons") or []) if part) or "слабый score"
        lines.append(
            f"- {item.get('slug')}: score {item.get('score')}, {reasons}; {item.get('brief')}"
        )
    block = "\n".join(lines) if lines else "- (нет кандидатов)"
    return (
        f"Исходная игра:\n{origin_brief}\n\n"
        f"Кандидаты со смешанным сходством:\n{block}\n\n"
        'Верни JSON {"keep": ["slug", ...]} только подходящих кандидатов.'
    )


def parse_similar_review(text: str | None) -> set[str]:
    raw = _strip_markdown_fences(str(text or ""))
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return set()
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return set()
    keep = data.get("keep") if isinstance(data, dict) else None
    if not isinstance(keep, list):
        return set()
    return {str(item).strip() for item in keep if str(item).strip()}


YOUTUBE_SUMMARY_SYSTEM_PROMPT = """Ты читаешь расшифровку УСТНОЙ речи автора летсплея (то, что блогер говорит вслух: комментарии, реакции, выводы по геймплею).
Для длинных роликов это три окна: начало, середина, конец.

Напиши короткое заключение СТРОГО на русском: что автор говорит про игру, что замечает в бою/мире, какие эмоции и выводы.
3–5 предложений, ≤600 символов. Без вступлений и покадрового пересказа.

Это НЕ описание ролика, НЕ синопсис уровня и НЕ карточка YouTube.
Запрещено: «ролик представляет собой», призывы скачать демо, Discord, лайк/подписка, «в конце зрителям предлагается», пересказ описания под видео.
Если живой речи автора нет (титры, маркетинг, описание) — верни {"summary": ""}.

Даже если субтитры на английском — пиши кириллицей.

Отвечай СТРОГО JSON без markdown:
{"summary": "заключение на русском"}"""

YOUTUBE_TRANSCRIPT_LIMIT = 25000


def build_youtube_summary_prompt(*, title: str, transcript: str, kind: str = "letsplay") -> str:
    body = " ".join((transcript or "").split())
    if len(body) > YOUTUBE_TRANSCRIPT_LIMIT:
        cut = body[:YOUTUBE_TRANSCRIPT_LIMIT].rsplit(" ", 1)[0]
        body = cut + "…"
    label = "обзора" if kind == "review" else "летсплея"
    return (
        f"Игра: {title or '—'}\n"
        f"Тип ролика: {label}\n\n"
        f"Расшифровка речи автора {label} (не описание ролика):\n"
        f"{body or '(пусто)'}"
    )

