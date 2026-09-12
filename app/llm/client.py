"""HTTP-клиент LLM с журналированием запросов в JSONL."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from app.config import Settings, get_settings

LLM_HTTP_ERROR_MESSAGES = {
    401: "Неверный API-ключ (401)",
    403: "Доступ запрещён (403): проверь ключ LLM_API_KEY, либо провайдер блокирует регион/IP",
    429: "Лимит запросов провайдера (429)",
}
LLM_HTTP_ERROR_SHORT = {
    401: "Неверный API-ключ (401)",
    403: "Доступ запрещён (403)",
    429: "Лимит запросов провайдера (429)",
}

MIN_CALL_INTERVAL_SEC = 20.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SEC = (60.0, 120.0, 240.0)
RETRY_AFTER_CAP_SEC = 300.0
CHAT_PAUSE_CAP_SEC = 3600.0
_TRY_AGAIN_HM = re.compile(
    r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+)?s",
    re.IGNORECASE,
)

_last_call_ts: float | None = None
_pace_lock: asyncio.Lock | None = None
_log_write_lock = threading.Lock()
_log_line_counts: dict[str, int] = {}


def classify_llm_http_error(exc: httpx.HTTPStatusError) -> tuple[str, int]:
    """Возвращает (человекочитаемое сообщение, HTTP-статус)."""
    status = exc.response.status_code
    detail = ""
    try:
        body = exc.response.json()
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            detail = str(err.get("message") or "").strip()
        elif isinstance(err, str):
            detail = err.strip()
        elif isinstance(body, dict) and body.get("message"):
            detail = str(body.get("message") or "").strip()
    except Exception:
        detail = (exc.response.text or "").strip()[:400]
    if status in LLM_HTTP_ERROR_MESSAGES:
        human = LLM_HTTP_ERROR_MESSAGES[status]
        return (f"{human}: {detail}" if detail else human), status
    if 500 <= status <= 599:
        human = f"Ошибка сервера провайдера ({status})"
        return (f"{human}: {detail}" if detail else human), status
    if status == 400:
        return (
            f"Некорректный запрос (400): {detail}"
            if detail
            else "Некорректный запрос (400): Whisper не принял аудиофайл"
        ), status
    return (f"{exc}: {detail}" if detail else str(exc)), status


def humanize_llm_error(
    message: str | None,
    status: int | None = None,
    *,
    compact: bool = False,
) -> str | None:
    """Маппит сырой текст/статус ошибки LLM в сообщение для монитора."""
    table = LLM_HTTP_ERROR_SHORT if compact else LLM_HTTP_ERROR_MESSAGES
    if status in table:
        return table[status]
    if status is not None and 500 <= status <= 599:
        return f"Ошибка сервера провайдера ({status})"
    if not message:
        return None
    if compact:
        for code, text in LLM_HTTP_ERROR_SHORT.items():
            if f"({code})" in message or str(code) in message[:40]:
                return text
    if message in LLM_HTTP_ERROR_MESSAGES.values() or message in LLM_HTTP_ERROR_SHORT.values():
        if compact:
            for code, text in LLM_HTTP_ERROR_MESSAGES.items():
                if message == text:
                    return LLM_HTTP_ERROR_SHORT.get(code, message)
        return message
    for code, text in LLM_HTTP_ERROR_MESSAGES.items():
        token = f"{code}"
        if f"'{code} " in message or f" {code} " in message or message.startswith(token) or f"({code})" in message:
            return LLM_HTTP_ERROR_SHORT[code] if compact else text
    return message


def _is_retryable_status(status: int | None) -> bool:
    if status is None:
        return False
    return status == 429 or status >= 500


def is_daily_token_quota(message: str | None) -> bool:
    """TPD не отпустит через минуту — ретраи только жгут время."""
    text = (message or "").lower()
    return "tokens per day" in text or "(tpd)" in text


def is_daily_quota(message: str | None) -> bool:
    """Суточный лимит модели (TPD/RPD/аудио), не RPM."""
    text = (message or "").lower()
    return (
        is_daily_token_quota(message)
        or "requests per day" in text
        or "(rpd)" in text
        or "audio seconds" in text
    )


def parse_try_again_seconds(message: str | None) -> float | None:
    """Достаёт паузу из текста Groq: «try again in 49m27s»."""
    match = _TRY_AGAIN_HM.search(message or "")
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    total = hours * 3600 + minutes * 60 + seconds
    return total if total > 0 else None


def quota_wait_seconds(message: str | None, header: float | None) -> float:
    parsed = parse_try_again_seconds(message)
    if parsed is not None:
        return min(max(parsed, 30.0), CHAT_PAUSE_CAP_SEC)
    if header is not None and header > 0:
        return min(max(float(header), 30.0), CHAT_PAUSE_CAP_SEC)
    return 60.0


def chat_model_chain(settings: Settings | None = None) -> list[str]:
    cfg = settings or get_settings()
    primary = (cfg.llm_model or "").strip()
    extras = [
        item.strip()
        for item in str(getattr(cfg, "llm_fallback_models", "") or "").split(",")
        if item.strip()
    ]
    chain: list[str] = []
    for name in [primary, *extras]:
        if name and name not in chain:
            chain.append(name)
    return chain or ["openai/gpt-oss-120b"]


class GroqQuota:
    """Общий лимит процесса: смена модели, потом сон очереди до retry-after."""

    def __init__(self) -> None:
        self.chat_index = 0
        self.chat_paused_until = 0.0
        self.whisper_paused_until = 0.0

    def reset(self) -> None:
        self.chat_index = 0
        self.chat_paused_until = 0.0
        self.whisper_paused_until = 0.0

    def chat_retry_in(self) -> float:
        left = self.chat_paused_until - time.monotonic()
        return left if left > 0 else 0.0

    def chat_ready(self) -> bool:
        if self.chat_retry_in() > 0:
            return False
        if self.chat_paused_until:
            self.chat_paused_until = 0.0
            self.chat_index = 0
        return True

    def whisper_ready(self) -> bool:
        if self.whisper_paused_until - time.monotonic() > 0:
            return False
        self.whisper_paused_until = 0.0
        return True

    def current_chat_model(self, settings: Settings | None = None) -> str:
        models = chat_model_chain(settings)
        index = min(max(self.chat_index, 0), len(models) - 1)
        return models[index]

    def note_chat_daily(self, message: str | None, header: float | None, settings: Settings | None = None) -> str | None:
        models = chat_model_chain(settings)
        nxt = self.chat_index + 1
        if nxt < len(models):
            self.chat_index = nxt
            logger.warning("Groq суточный лимит, переключаемся на %s", models[nxt])
            return models[nxt]
        wait = quota_wait_seconds(message, header)
        self.chat_paused_until = time.monotonic() + wait
        self.chat_index = 0
        logger.warning("Groq chat исчерпан на всех моделях, очередь спит %ss", int(wait))
        return None

    def pause_whisper(self, message: str | None, header: float | None) -> None:
        wait = quota_wait_seconds(message, header)
        self.whisper_paused_until = time.monotonic() + wait
        logger.warning("Groq Whisper суточный лимит, пауза %ss", int(wait))


groq_quota = GroqQuota()


def groq_chat_blocked() -> bool:
    return groq_quota.chat_retry_in() > 0


def groq_chat_retry_in() -> float:
    return groq_quota.chat_retry_in()


def _parse_retry_after(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    raw = (response.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), RETRY_AFTER_CAP_SEC))
    except ValueError:
        return None


def _retry_wait(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return retry_after
    index = min(max(attempt, 1), len(RETRY_BACKOFF_SEC)) - 1
    return RETRY_BACKOFF_SEC[index]


def _get_pace_lock() -> asyncio.Lock:
    global _pace_lock
    if _pace_lock is None:
        _pace_lock = asyncio.Lock()
    return _pace_lock


async def _wait_for_slot(interval: float) -> None:
    """Ждёт, пока с прошлого вызова LLM не пройдёт interval секунд."""
    global _last_call_ts
    async with _get_pace_lock():
        now = time.monotonic()
        if _last_call_ts is not None:
            wait = interval - (now - _last_call_ts)
            if wait > 0:
                await asyncio.sleep(wait)
        _last_call_ts = time.monotonic()


def _mark_call() -> None:
    global _last_call_ts
    _last_call_ts = time.monotonic()


logger = logging.getLogger(__name__)

WHISPER_MAX_BYTES = 25 * 1024 * 1024
_AUDIO_MIME = {
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".mpga": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


def groq_transcriptions_url(chat_completions_url: str) -> str:
    """Chat Completions → тот же хост, эндпоинт audio/transcriptions."""
    raw = (chat_completions_url or "").strip().rstrip("/")
    suffix = "/chat/completions"
    if raw.endswith(suffix):
        return raw[: -len(suffix)] + "/audio/transcriptions"
    return "https://api.groq.com/openai/v1/audio/transcriptions"


def whisper_upload_name(filename: str) -> str:
    """Имя с расширением, которое Groq умеет: opus → ogg, неизвестное → wav."""
    name = Path(filename).name or "audio.wav"
    if name.lower().endswith(".part"):
        name = name[: -len(".part")]
    suffix = Path(name).suffix.lower()
    if suffix == ".opus":
        return f"{Path(name).stem}.ogg"
    if suffix in _AUDIO_MIME:
        return name
    return f"{Path(name).stem or 'audio'}.wav"


def _audio_mime(filename: str) -> str:
    suffix = Path(whisper_upload_name(filename)).suffix.lower()
    return _AUDIO_MIME.get(suffix, "audio/wav")


STUB_RESPONSE = (
    "[stub] LLM API не настроен. Укажите GROQ_API_KEY в .env "
    "(https://console.groq.com/keys), чтобы генерировать саммари."
)


@dataclass(slots=True)
class LLMResult:
    """Ответ модели и метаданные вызова."""

    text: str
    model: str
    stub: bool
    latency_ms: int
    error: str | None = None
    error_status: int | None = None
    log_id: int | None = None
    log_ids: list[int] = field(default_factory=list)
    attempt: int = 1
    kind: str | None = None
    retry_after: float | None = None


class LLMClient:
    """Вызывает chat-completions API (или заглушку) и дописывает лог в JSONL."""

    def __init__(self, settings: Settings | None = None, *, run_id: int | None = None) -> None:
        self._settings = settings or get_settings()
        self.run_id = run_id
        self.final_errors = 0
        self.retries = 0
        self._hard_block: tuple[str, int] | None = None

    @property
    def log_path(self) -> Path:
        return self._settings.llm_log_path

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        slug: str | None = None,
        run_id: int | None = None,
        kind: str | None = None,
        json_object: bool = False,
    ) -> LLMResult:
        """Отправляет промпт в LLM и пишет запись в ``data/llm_logs.jsonl``."""
        effective_run_id = run_id if run_id is not None else self.run_id
        if self._hard_block:
            msg, status = self._hard_block
            return LLMResult(
                text="",
                model=groq_quota.current_chat_model(self._settings),
                stub=False,
                latency_ms=0,
                error=msg,
                error_status=status,
                kind=kind,
            )
        if not groq_quota.chat_ready():
            wait = groq_quota.chat_retry_in()
            return LLMResult(
                text="",
                model=groq_quota.current_chat_model(self._settings),
                stub=False,
                latency_ms=0,
                error=f"Лимит запросов провайдера (429): очередь Groq спит ещё {int(wait)}с",
                error_status=429,
                kind=kind,
            )
        interval = float(getattr(self._settings, "llm_call_interval", None) or MIN_CALL_INTERVAL_SEC)
        if not self._settings.llm_api_key:
            await _wait_for_slot(interval)
            started = perf_counter()
            stub_text = STUB_RESPONSE
            if json_object:
                stub_text = json.dumps(
                    {
                        "summary": STUB_RESPONSE,
                        "tags": [],
                        "genre_detailed": None,
                        "key_features": [],
                        "target_audience": None,
                    },
                    ensure_ascii=False,
                )
            result = LLMResult(
                text=stub_text,
                model="stub",
                stub=True,
                latency_ms=int((perf_counter() - started) * 1000),
                attempt=1,
                kind=kind,
            )
            _mark_call()
            result.log_id = await self._append_log(
                prompt=prompt,
                system=system,
                result=result,
                slug=slug,
                run_id=effective_run_id,
                attempt=1,
                kind=kind,
            )
            result.log_ids = [result.log_id]
            return result

        log_ids: list[int] = []
        last: LLMResult | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt > 1:
                self.retries += 1
            model = groq_quota.current_chat_model(self._settings)
            await _wait_for_slot(interval)
            started = perf_counter()
            result = await self._one_request(
                prompt, system=system, started=started, json_object=json_object, model=model
            )
            _mark_call()
            result.attempt = attempt
            result.kind = kind
            result.log_id = await self._append_log(
                prompt=prompt,
                system=system,
                result=result,
                slug=slug,
                run_id=effective_run_id,
                attempt=attempt,
                kind=kind,
            )
            log_ids.append(result.log_id)
            result.log_ids = list(log_ids)
            last = result
            if not result.error:
                return result
            if result.error_status == 429 and is_daily_quota(result.error):
                nxt = groq_quota.note_chat_daily(
                    result.error, result.retry_after, self._settings
                )
                if nxt:
                    continue
                self.final_errors += 1
                return result
            if not _is_retryable_status(result.error_status) or attempt >= MAX_ATTEMPTS:
                self.final_errors += 1
                if result.error_status in {401, 403}:
                    self._hard_block = (result.error or "LLM заблокирован", result.error_status)
                return result
            wait = _retry_wait(attempt, result.retry_after if result.error_status == 429 else None)
            logger.warning(
                "LLM HTTP %s, попытка %s/%s, пауза %ss",
                result.error_status,
                attempt,
                MAX_ATTEMPTS,
                int(wait),
            )
            if wait > 0:
                await asyncio.sleep(wait)
        assert last is not None
        self.final_errors += 1
        return last

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        slug: str | None = None,
        run_id: int | None = None,
        kind: str = "whisper",
    ) -> LLMResult:
        """Groq Whisper: multipart на /audio/transcriptions, тот же API-ключ."""
        effective_run_id = run_id if run_id is not None else self.run_id
        model = str(getattr(self._settings, "whisper_model", None) or "whisper-large-v3-turbo")
        prompt = f"[whisper {filename} {len(audio)} bytes]"
        if self._hard_block and self._hard_block[1] in {401, 403}:
            msg, status = self._hard_block
            return LLMResult(
                text="",
                model=model,
                stub=False,
                latency_ms=0,
                error=msg,
                error_status=status,
                kind=kind,
            )
        if not groq_quota.whisper_ready():
            return LLMResult(
                text="",
                model=model,
                stub=False,
                latency_ms=0,
                error="Лимит запросов провайдера (429): Whisper на паузе",
                error_status=429,
                kind=kind,
            )
        interval = float(getattr(self._settings, "llm_call_interval", None) or MIN_CALL_INTERVAL_SEC)
        if not self._settings.llm_api_key:
            await _wait_for_slot(interval)
            result = LLMResult(
                text="",
                model="stub",
                stub=True,
                latency_ms=0,
                error="Нет GROQ_API_KEY для Whisper",
                attempt=1,
                kind=kind,
            )
            _mark_call()
            result.log_id = await self._append_log(
                prompt=prompt,
                system=None,
                result=result,
                slug=slug,
                run_id=effective_run_id,
                attempt=1,
                kind=kind,
            )
            result.log_ids = [result.log_id]
            return result
        if len(audio) > WHISPER_MAX_BYTES:
            result = LLMResult(
                text="",
                model=model,
                stub=False,
                latency_ms=0,
                error=f"Аудио больше {WHISPER_MAX_BYTES} байт",
                attempt=1,
                kind=kind,
            )
            result.log_id = await self._append_log(
                prompt=prompt,
                system=None,
                result=result,
                slug=slug,
                run_id=effective_run_id,
                attempt=1,
                kind=kind,
            )
            result.log_ids = [result.log_id]
            return result

        log_ids: list[int] = []
        last: LLMResult | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt > 1:
                self.retries += 1
            await _wait_for_slot(interval)
            started = perf_counter()
            result = await self._one_transcription(
                audio, filename=filename, started=started, model=model
            )
            _mark_call()
            result.attempt = attempt
            result.kind = kind
            result.log_id = await self._append_log(
                prompt=prompt,
                system=None,
                result=result,
                slug=slug,
                run_id=effective_run_id,
                attempt=attempt,
                kind=kind,
            )
            log_ids.append(result.log_id)
            result.log_ids = list(log_ids)
            last = result
            if not result.error:
                return result
            if result.error_status == 429 and is_daily_quota(result.error):
                groq_quota.pause_whisper(result.error, result.retry_after)
                self.final_errors += 1
                return result
            if not _is_retryable_status(result.error_status) or attempt >= MAX_ATTEMPTS:
                self.final_errors += 1
                if result.error_status in {401, 403}:
                    self._hard_block = (result.error or "LLM заблокирован", result.error_status)
                return result
            wait = _retry_wait(attempt, result.retry_after if result.error_status == 429 else None)
            logger.warning(
                "Whisper HTTP %s, попытка %s/%s, пауза %ss",
                result.error_status,
                attempt,
                MAX_ATTEMPTS,
                int(wait),
            )
            if wait > 0:
                await asyncio.sleep(wait)
        assert last is not None
        self.final_errors += 1
        return last

    async def note(
        self,
        *,
        prompt: str,
        error: str | None = None,
        response: str = "",
        slug: str | None = None,
        kind: str | None = None,
        model: str = "youtube",
    ) -> int:
        """Строка в JSONL без HTTP: пустой поиск YouTube или похожие без LLM."""
        result = LLMResult(
            text=response,
            model=model,
            stub=True,
            latency_ms=0,
            error=(error or "").strip() or None,
            kind=kind,
        )
        if not result.text:
            result.text = (error or "").strip()
        result.log_id = await self._append_log(
            prompt=prompt,
            system=None,
            result=result,
            slug=slug,
            run_id=self.run_id,
            attempt=1,
            kind=kind,
        )
        result.log_ids = [result.log_id]
        return result.log_id

    async def _one_transcription(
        self,
        audio: bytes,
        *,
        filename: str,
        started: float,
        model: str,
    ) -> LLMResult:
        try:
            text = await self._post_transcription(audio, filename=filename, model=model)
            return LLMResult(
                text=text,
                model=model,
                stub=False,
                latency_ms=int((perf_counter() - started) * 1000),
            )
        except httpx.HTTPStatusError as exc:
            human, status = classify_llm_http_error(exc)
            logger.warning("Ошибка Whisper: %s", human)
            return LLMResult(
                text="",
                model=model,
                stub=False,
                latency_ms=int((perf_counter() - started) * 1000),
                error=human,
                error_status=status,
                retry_after=_parse_retry_after(exc.response) if status == 429 else None,
            )
        except Exception as exc:
            logger.exception("Ошибка Whisper")
            return LLMResult(
                text="",
                model=model,
                stub=False,
                latency_ms=int((perf_counter() - started) * 1000),
                error=str(exc),
            )

    async def _one_request(
        self,
        prompt: str,
        *,
        system: str | None,
        started: float,
        json_object: bool = False,
        model: str | None = None,
    ) -> LLMResult:
        model = model or groq_quota.current_chat_model(self._settings)
        try:
            text = await self._post(prompt, system=system, json_object=json_object, model=model)
            return LLMResult(
                text=text,
                model=model,
                stub=False,
                latency_ms=int((perf_counter() - started) * 1000),
            )
        except httpx.HTTPStatusError as exc:
            if json_object and exc.response.status_code == 400:
                logger.warning("response_format json_object отклонён, повтор без него")
                return await self._one_request(
                    prompt,
                    system=system,
                    started=started,
                    json_object=False,
                    model=model,
                )
            human, status = classify_llm_http_error(exc)
            logger.warning("Ошибка вызова LLM: %s", human)
            return LLMResult(
                text="",
                model=model,
                stub=False,
                latency_ms=int((perf_counter() - started) * 1000),
                error=human,
                error_status=status,
                retry_after=_parse_retry_after(exc.response) if status == 429 else None,
            )
        except Exception as exc:
            logger.exception("Ошибка вызова LLM")
            return LLMResult(
                text="",
                model=model,
                stub=False,
                latency_ms=int((perf_counter() - started) * 1000),
                error=str(exc),
            )

    async def _post(
        self,
        prompt: str,
        *,
        system: str | None,
        json_object: bool = False,
        model: str | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model or groq_quota.current_chat_model(self._settings),
            "messages": messages,
            "temperature": 0.3,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=self._settings.llm_timeout) as client:
            response = await client.post(
                self._settings.llm_api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"].strip()
        raise RuntimeError(f"Неожиданный ответ LLM: {body!r}")

    async def _post_transcription(self, audio: bytes, *, filename: str, model: str) -> str:
        timeout = float(getattr(self._settings, "whisper_timeout", None) or 180.0)
        headers = {"Authorization": f"Bearer {self._settings.llm_api_key}"}
        name = whisper_upload_name(filename)
        files = {"file": (name, audio, _audio_mime(name))}
        data = {
            "model": model,
            "response_format": "json",
        }
        url = groq_transcriptions_url(self._settings.llm_api_url)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            body: dict[str, Any] = response.json()
        text = body.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        raise RuntimeError(f"Неожиданный ответ Whisper: {body!r}")

    async def _append_log(
        self,
        *,
        prompt: str,
        system: str | None,
        result: LLMResult,
        slug: str | None = None,
        run_id: int | None = None,
        attempt: int = 1,
        kind: str | None = None,
    ) -> int:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": result.model,
            "stub": result.stub,
            "latency_ms": result.latency_ms,
            "ok": result.error is None,
            "error": result.error,
            "error_status": result.error_status,
            "system": system,
            "prompt": prompt,
            "response": result.text,
            "game_slug": slug,
            "run_id": run_id,
            "attempt": attempt,
            "kind": kind,
        }
        return await asyncio.to_thread(_write_jsonl, self.log_path, record)


def _write_jsonl(path: Path, record: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path)
    with _log_write_lock:
        if key not in _log_line_counts:
            n = 0
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    n = sum(1 for _ in handle)
            _log_line_counts[key] = n
        _log_line_counts[key] += 1
        log_id = _log_line_counts[key]
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return log_id
