"""Поиск летсплея на YouTube: Innertube, без Data API ключа."""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import quote_plus

from curl_cffi.requests import AsyncSession
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import async_object_session

from app.config import Settings, get_settings
from app.db import SessionLocal
from app.models import Game
from app.services.yt_audio import (
    ANTIBOT_RENDER_MESSAGE,
    as_media_result,
    download_letsplay_audio,
    download_letsplay_captions,
    is_render_block,
    prepare_whisper_audio,
    search_youtube_ytdlp,
    youtube_cookie_dict,
)

logger = logging.getLogger(__name__)

INNERTUBE_URL = "https://www.youtube.com/youtubei/v1/search"
INNERTUBE_PLAYER_URL = "https://www.youtube.com/youtubei/v1/player"
INNERTUBE_KEY = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"
INNERTUBE_CLIENT_VERSION = "2.20241218.01.00"
INNERTUBE_ANDROID_KEY = "AIzaSyA8eiZmM1FaDVjRy-df2KTyQ_vz_yYM39w"
INNERTUBE_ANDROID_VERSION = "19.47.14"
INNERTUBE_TV_VERSION = "2.0"
RESULTS_URL = "https://www.youtube.com/results"
WATCH_URL = "https://www.youtube.com/watch?v={video_id}"
VIDEO_FILTER = "EgIQAQ%3D%3D"

YOUTUBE_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Origin": "https://www.youtube.com",
    "Referer": "https://www.youtube.com/",
}

_YT_INITIAL = re.compile(r"ytInitialData\s*=\s*(\{.*?\});\s*</script>", re.DOTALL)
_VIDEO_ID = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|embed/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)
_STOP_TITLE = {
    "a",
    "and",
    "edition",
    "game",
    "of",
    "the",
    "complete",
    "definitive",
    "remastered",
    "remake",
}
_PLAY_HINTS = (
    "gameplay",
    "let's play",
    "lets play",
    "letsplay",
    "playthrough",
    "walkthrough",
    "first hour",
    "прохождение",
    "летсплей",
    "летс плей",
    "геймплей",
)
_REVIEW_HINTS = (
    " review",
    "reviewed",
    "обзор",
    "ревью",
)
_SKIP_HINTS = (
    "official trailer",
    "launch trailer",
    "reveal trailer",
    "teaser trailer",
    "game trailer",
    "trailer",
    "soundtrack",
    " ost",
    "music video",
    "music mix",
    "lyric",
    "audio only",
    "song ",
)

_last_call_ts: float | None = None
_pace_lock: asyncio.Lock | None = None
LETS_PLAY_PROBE_LIMIT = 3
LETS_PLAY_READY_SOURCES = frozenset({"transcript", "whisper"})
LETS_PLAY_ANTIBOT_SOURCE = "antibot"
LETS_PLAY_STUB = "Подходящий летсплей не найден."
LETS_PLAY_STUB_CAPTIONS = "Подходящий летсплей с субтитрами не найден."
_BLURB_MARKERS = re.compile(
    r"ролик представляет|в этом видео|this video (?:is|shows|presents)|"
    r"в конце зрител|скачать.{0,48}демо|бесплатн\w*\s+демо|"
    r"присоединиться к сообществу|join (?:our |the )?discord|"
    r"link in (?:the )?description|wishlist|"
    r"ставьте лайк|подписывайтесь",
    re.I,
)


@dataclass(frozen=True, slots=True)
class YoutubeVideo:
    video_id: str
    title: str
    duration_sec: int | None = None
    description: str | None = None
    is_short: bool = False
    views: int | None = None
    channel: str | None = None

    @property
    def url(self) -> str:
        return WATCH_URL.format(video_id=self.video_id)


@dataclass(frozen=True, slots=True)
class CaptionCue:
    start: float
    text: str


@dataclass(frozen=True, slots=True)
class LetsPlayHit:
    url: str
    title: str
    summary: str | None = None
    channel: str | None = None
    views: int | None = None
    duration_sec: int | None = None
    kind: str = "letsplay"
    transcript_sample: str | None = None


@dataclass
class WhisperOutcome:
    text: str | None = None
    error: str | None = None
    cause: str = "ok"

    @property
    def blocked(self) -> bool:
        return self.cause in {"bot", "http", "token", "timeout"}

    @property
    def render_blocked(self) -> bool:
        return self.cause in {"bot", "http", "token"}


def _pace_lock_get() -> asyncio.Lock:
    global _pace_lock
    if _pace_lock is None:
        _pace_lock = asyncio.Lock()
    return _pace_lock


async def _pace(interval: float) -> None:
    global _last_call_ts
    if interval <= 0:
        return
    async with _pace_lock_get():
        now = time.monotonic()
        if _last_call_ts is not None:
            wait = interval - (now - _last_call_ts)
            if wait > 0:
                await asyncio.sleep(wait)
        _last_call_ts = time.monotonic()


def _visible_text(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        simple = node.get("simpleText")
        if simple:
            return str(simple).strip()
        runs = node.get("runs")
        if isinstance(runs, list):
            return "".join(str(item.get("text") or "") for item in runs if isinstance(item, dict)).strip()
        text = node.get("text")
        if text:
            return str(text).strip()
    return ""


def _parse_duration(value: str | None) -> int | None:
    text = (value or "").strip()
    if not text or not re.fullmatch(r"\d+:\d{2}(?::\d{2})?", text):
        return None
    parts = [int(part) for part in text.split(":")]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _iter_nodes(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_nodes(item)


def extract_search_videos(payload: Any) -> list[YoutubeVideo]:
    """Достаёт ролики из Innertube / ytInitialData."""
    videos: list[YoutubeVideo] = []
    seen: set[str] = set()
    for obj in _iter_nodes(payload):
        renderer = obj.get("videoRenderer") or obj.get("compactVideoRenderer")
        if not isinstance(renderer, dict):
            continue
        video_id = str(renderer.get("videoId") or "").strip()
        if len(video_id) != 11 or video_id in seen:
            continue
        title = _visible_text(renderer.get("title"))
        if not title:
            continue
        length = _visible_text(renderer.get("lengthText"))
        duration = _parse_duration(length)
        badges = renderer.get("badges") or renderer.get("thumbnailOverlays") or []
        badge_text = json.dumps(badges, ensure_ascii=False).lower() if badges else ""
        is_short = "shorts" in badge_text or (duration is not None and duration < 60)
        description = _visible_text(renderer.get("descriptionSnippet")) or None
        views = parse_view_count(
            _visible_text(renderer.get("viewCountText"))
            or _visible_text(renderer.get("shortViewCountText"))
        )
        if views is None:
            views = parse_view_count(str(renderer.get("viewCount") or ""))
        channel = (
            _visible_text(renderer.get("ownerText"))
            or _visible_text(renderer.get("shortBylineText"))
            or None
        )
        seen.add(video_id)
        videos.append(
            YoutubeVideo(
                video_id=video_id,
                title=title,
                duration_sec=duration,
                description=description,
                is_short=is_short,
                views=views,
                channel=channel,
            )
        )
    return videos


def parse_view_count(text: str | None) -> int | None:
    """«1,234,567 views», «1.2M views», «12 тыс. просмотров» → int."""
    raw = (text or "").strip().lower().replace("\xa0", " ")
    if not raw:
        return None
    abbr = re.search(r"([\d]+(?:[.,]\d+)?)\s*([kmb]|тыс\.?|млн|млрд)", raw)
    if abbr:
        number = float(abbr.group(1).replace(",", "."))
        suffix = abbr.group(2).lower()
        if suffix.startswith("k") or suffix.startswith("тыс"):
            number *= 1_000
        elif suffix.startswith("m") or suffix.startswith("млн"):
            number *= 1_000_000
        elif suffix.startswith("b") or suffix.startswith("млрд"):
            number *= 1_000_000_000
        return int(number)
    digits = re.sub(r"[^\d]", "", raw)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _fold(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9а-яё]+", (text or "").lower()))


def _core_title(title: str) -> str:
    head = re.split(r"[:–—|/]", title or "", maxsplit=1)[0]
    if " - " in head:
        head = head.split(" - ", 1)[0]
    return _fold(head)


def _title_tokens(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9а-яё]+", title.lower())
    return {word for word in words if len(word) > 2 and word not in _STOP_TITLE}


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _title_matches(game_title: str, video: YoutubeVideo) -> bool:
    video_fold = _fold(video.title)
    core = _core_title(game_title)
    if core and len(core) >= 4 and core not in video_fold:
        return False
    game_tokens = _title_tokens(game_title)
    if game_tokens and not (game_tokens & _title_tokens(video.title)):
        return False
    return True


def classify_youtube_video(game_title: str, video: YoutubeVideo) -> str | None:
    """letsplay | review | None (трейлер/шортс/чужая игра)."""
    if not _title_matches(game_title, video):
        return None
    if video.is_short:
        return None
    title = f" {video.title.lower()} "
    play = _has_any(title, _PLAY_HINTS)
    review = _has_any(title, _REVIEW_HINTS)
    trailer = _has_any(title, _SKIP_HINTS)
    if play:
        return "letsplay"
    if trailer:
        return None
    if review:
        return "review"
    if video.duration_sec is None:
        if _has_any(title, (" shorts", " #short", "short film")):
            return None
        return "letsplay"
    if video.duration_sec >= 8 * 60:
        return "letsplay"
    return None


def video_tier(video: YoutubeVideo, kind: str | None) -> int | None:
    """1: летсплей 8–75 мин; 2: любой летсплей; 3: обзор."""
    if kind == "letsplay":
        duration = video.duration_sec
        if duration is not None and 8 * 60 <= duration <= 75 * 60:
            return 1
        return 2
    if kind == "review":
        return 3
    return None


def score_letsplay_candidate(game_title: str, video: YoutubeVideo) -> int:
    """Совместимость с тестами: >0 только живой летсплей, не трейлер."""
    kind = classify_youtube_video(game_title, video)
    if kind != "letsplay":
        return -40 if kind is None else -10
    score = 25
    if video.duration_sec is not None:
        if 8 * 60 <= video.duration_sec <= 75 * 60:
            score += 10
        elif video.duration_sec < 3 * 60:
            score -= 8
    return score


def rank_letsplay_candidates(game_title: str, videos: Iterable[YoutubeVideo]) -> list[YoutubeVideo]:
    """Ярусы 1→2→3, внутри — по просмотрам, при равенстве ближе к 30 мин."""
    buckets: dict[int, list[YoutubeVideo]] = {1: [], 2: [], 3: []}
    for video in videos:
        kind = classify_youtube_video(game_title, video)
        tier = video_tier(video, kind)
        if tier is None:
            continue
        buckets[tier].append(video)
    ranked: list[YoutubeVideo] = []
    for tier in (1, 2, 3):
        pool = buckets[tier]
        pool.sort(
            key=lambda item: (
                -(item.views or 0),
                abs((item.duration_sec or 1800) - 1800),
            )
        )
        ranked.extend(pool)
    return ranked


def choose_letsplay(game_title: str, videos: Iterable[YoutubeVideo]) -> YoutubeVideo | None:
    """Самый популярный ролик внутри первого непустого яруса."""
    ranked = rank_letsplay_candidates(game_title, videos)
    return ranked[0] if ranked else None


def _play_queries(game_title: str) -> list[str]:
    title = " ".join((game_title or "").split())
    if not title:
        return []
    return [
        f"{title} gameplay",
        f"{title} let's play",
        f"{title} прохождение",
        f"{title} летсплей",
    ]


def _review_queries(game_title: str) -> list[str]:
    title = " ".join((game_title or "").split())
    if not title:
        return []
    return [f"{title} review", f"{title} обзор"]


def _parse_initial_data(html: str) -> Any | None:
    match = _YT_INITIAL.search(html or "")
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


class YoutubeClient:
    """Поиск роликов через публичный Innertube и HTML выдачи."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._session: AsyncSession | None = None
        self.last_search_count = 0
        self.last_caption_error: str | None = None
        self.last_caption_cause: str | None = None

    async def __aenter__(self) -> YoutubeClient:
        kwargs: dict[str, Any] = {
            "headers": YOUTUBE_HEADERS,
            "timeout": self._settings.youtube_timeout,
            "impersonate": self._settings.impersonate or "chrome120",
        }
        proxy = (getattr(self._settings, "youtube_proxy", "") or "").strip()
        if proxy:
            kwargs["proxy"] = proxy
            logger.info("YouTube Innertube: прокси включён")
        cookies = youtube_cookie_dict()
        if cookies:
            kwargs["cookies"] = cookies
            logger.info("YouTube Innertube: cookies загружены (%s шт.)", len(cookies))
        try:
            self._session = AsyncSession(**kwargs)
        except TypeError:
            kwargs.pop("proxy", None)
            kwargs.pop("cookies", None)
            self._session = AsyncSession(
                headers=YOUTUBE_HEADERS,
                timeout=self._settings.youtube_timeout,
                impersonate=self._settings.impersonate or "chrome120",
            )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _require_session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("YoutubeClient нужно использовать как async context manager")
        return self._session

    async def list_letsplays(self, game_title: str) -> list[YoutubeVideo]:
        """Популярные летсплеи: ярус 8–75 мин, затем любой, внутри — по просмотрам."""
        collected: list[YoutubeVideo] = []
        seen: set[str] = set()

        async def _collect(queries: list[str]) -> None:
            for query in queries:
                for video in await self.search_videos(query):
                    if video.video_id in seen:
                        continue
                    seen.add(video.video_id)
                    collected.append(video)
                ranked = rank_letsplay_candidates(game_title, collected)
                letsplays = [
                    video
                    for video in ranked
                    if classify_youtube_video(game_title, video) == "letsplay"
                ]
                if letsplays:
                    return

        await _collect(_play_queries(game_title))
        ranked = rank_letsplay_candidates(game_title, collected)
        letsplays = [
            video
            for video in ranked
            if classify_youtube_video(game_title, video) == "letsplay"
        ]
        if letsplays:
            self.last_search_count = len(collected)
            return letsplays[:LETS_PLAY_PROBE_LIMIT]
        await _collect(_review_queries(game_title))
        ranked = rank_letsplay_candidates(game_title, collected)
        self.last_search_count = len(collected)
        if ranked:
            logger.info(
                "Летсплей %s: берём ярус обзора (%s)",
                game_title,
                ranked[0].title,
            )
            return ranked[:LETS_PLAY_PROBE_LIMIT]
        logger.info(
            "Летсплей не найден: %s (собрали %s роликов, ни один не прошёл фильтр)",
            game_title,
            len(collected),
        )
        return []

    async def find_letsplay(self, game_title: str) -> LetsPlayHit | None:
        ranked = await self.list_letsplays(game_title)
        if not ranked:
            logger.info("Летсплей не найден: %s", game_title)
            return None
        video = ranked[0]
        sample = None
        try:
            sample = await self.fetch_transcript(
                video.video_id,
                duration_sec=video.duration_sec,
            )
        except Exception:
            logger.exception("Субтитры не разобрались для %s", video.video_id)
        kind = "letsplay"
        logger.info(
            "YouTube %s для %s: %s (%s, views=%s, %ss, captions=%s)",
            kind,
            game_title,
            video.title,
            video.url,
            video.views,
            video.duration_sec,
            "yes" if sample else "no",
        )
        return LetsPlayHit(
            url=video.url,
            title=video.title,
            channel=video.channel,
            views=video.views,
            duration_sec=video.duration_sec,
            kind=kind,
            transcript_sample=sample,
        )

    async def search_videos(self, query: str) -> list[YoutubeVideo]:
        videos = await self._search_innertube(query)
        if videos:
            return videos
        videos = await self._search_html(query)
        if videos:
            return videos
        return await self._search_ytdlp(query)

    async def _search_ytdlp(self, query: str) -> list[YoutubeVideo]:
        await _pace(self._settings.youtube_call_interval)
        try:
            rows = await search_youtube_ytdlp(
                query,
                limit=10,
                timeout=max(20.0, float(self._settings.youtube_timeout or 25.0)),
            )
        except Exception as exc:
            logger.warning("yt-dlp search ошибка %s: %s", query, exc)
            return []
        videos = [
            YoutubeVideo(
                video_id=str(row["video_id"]),
                title=str(row["title"]),
                duration_sec=row.get("duration_sec"),
                description=row.get("description"),
                views=row.get("views"),
                channel=row.get("channel"),
            )
            for row in rows
        ]
        if videos:
            logger.info("yt-dlp search %s: %s роликов", query, len(videos))
        return videos

    async def _search_innertube(self, query: str) -> list[YoutubeVideo]:
        session = self._require_session()
        await _pace(self._settings.youtube_call_interval)
        payload = {
            "context": {
                "client": {
                    "clientName": "WEB",
                    "clientVersion": INNERTUBE_CLIENT_VERSION,
                    "hl": "en",
                    "gl": "US",
                }
            },
            "query": query,
            "params": "EgIQAQ==",
        }
        try:
            response = await session.post(
                INNERTUBE_URL,
                params={"key": INNERTUBE_KEY, "prettyPrint": "false"},
                json=payload,
                headers={**YOUTUBE_HEADERS, "Content-Type": "application/json"},
                timeout=self._settings.youtube_timeout,
                impersonate=self._settings.impersonate or "chrome120",
            )
            if response.status_code != 200:
                logger.warning("YouTube Innertube HTTP %s для %s", response.status_code, query)
                return []
            data = response.json()
        except Exception as exc:
            logger.warning("YouTube Innertube ошибка %s: %s", query, exc)
            return []
        return extract_search_videos(data)

    async def _search_html(self, query: str) -> list[YoutubeVideo]:
        session = self._require_session()
        await _pace(self._settings.youtube_call_interval)
        url = f"{RESULTS_URL}?search_query={quote_plus(query)}&hl=en&gl=US&sp={VIDEO_FILTER}"
        try:
            response = await session.get(
                url,
                headers=YOUTUBE_HEADERS,
                timeout=self._settings.youtube_timeout,
                impersonate=self._settings.impersonate or "chrome120",
            )
            if response.status_code != 200:
                logger.warning("YouTube HTML HTTP %s для %s", response.status_code, query)
                return []
            payload = _parse_initial_data(response.text)
        except Exception as exc:
            logger.warning("YouTube HTML ошибка %s: %s", query, exc)
            return []
        if payload is None:
            return []
        return extract_search_videos(payload)

    async def fetch_transcript(
        self,
        video_id: str,
        *,
        duration_sec: int | None = None,
    ) -> str | None:
        """Субтитры ролика: watch page / Innertube player → timedtext XML."""
        self.last_caption_error = None
        self.last_caption_cause = None
        video_id = (video_id or "").strip()
        if len(video_id) != 11:
            self.last_caption_cause = "empty"
            self.last_caption_error = "некорректный video_id"
            return None
        tracks = await self._caption_tracks(video_id)
        track_url = pick_caption_track(tracks)
        if not tracks:
            self.last_caption_cause = "empty"
            self.last_caption_error = "у ролика нет дорожки субтитров (watch/player)"
        elif not track_url:
            self.last_caption_cause = "empty"
            self.last_caption_error = "дорожки субтитров без URL"
        else:
            session = self._require_session()
            await _pace(self._settings.youtube_call_interval)
            try:
                response = await session.get(
                    track_url,
                    headers=YOUTUBE_HEADERS,
                    timeout=self._settings.youtube_timeout,
                    impersonate=self._settings.impersonate or "chrome120",
                )
                if response.status_code == 200:
                    text = parse_caption_xml(response.text)
                    if text:
                        cues = parse_caption_cues(response.text)
                        sampled = sample_transcript(cues, duration_sec=duration_sec)
                        return sampled or text
                    self.last_caption_cause = "token"
                    self.last_caption_error = "timedtext пустой (часто без PO-токена)"
                else:
                    self.last_caption_cause = "http"
                    self.last_caption_error = f"timedtext HTTP {response.status_code}"
                    logger.warning("YouTube captions HTTP %s для %s", response.status_code, video_id)
            except Exception as exc:
                self.last_caption_cause = "unknown"
                self.last_caption_error = f"timedtext ошибка: {exc}"
                logger.warning("YouTube captions ошибка %s: %s", video_id, exc)
        return await self._captions_via_ytdlp(video_id, duration_sec=duration_sec)

    async def _caption_tracks(self, video_id: str) -> list[dict[str, Any]]:
        tracks = await self._caption_tracks_watch(video_id)
        if tracks:
            return tracks
        return await self._caption_tracks_player(video_id)

    async def _caption_tracks_watch(self, video_id: str) -> list[dict[str, Any]]:
        session = self._require_session()
        await _pace(self._settings.youtube_call_interval)
        url = WATCH_URL.format(video_id=video_id) + "&hl=en"
        try:
            response = await session.get(
                url,
                headers=YOUTUBE_HEADERS,
                timeout=self._settings.youtube_timeout,
                impersonate=self._settings.impersonate or "chrome120",
            )
            if response.status_code != 200:
                return []
            payload = _extract_player_response(response.text)
        except Exception as exc:
            logger.warning("YouTube watch ошибка %s: %s", video_id, exc)
            return []
        return caption_tracks_from_player(payload)

    async def _caption_tracks_player(self, video_id: str) -> list[dict[str, Any]]:
        session = self._require_session()
        clients = (
            ("WEB", INNERTUBE_CLIENT_VERSION, INNERTUBE_KEY, None),
            ("ANDROID", INNERTUBE_ANDROID_VERSION, INNERTUBE_ANDROID_KEY, None),
            ("TVHTML5_SIMPLY_EMBEDDED_PLAYER", INNERTUBE_TV_VERSION, INNERTUBE_KEY, "https://www.youtube.com/"),
        )
        for name, version, key, embed in clients:
            await _pace(self._settings.youtube_call_interval)
            client: dict[str, Any] = {
                "clientName": name,
                "clientVersion": version,
                "hl": "en",
                "gl": "US",
            }
            payload: dict[str, Any] = {"context": {"client": client}, "videoId": video_id}
            if embed:
                payload["context"]["thirdParty"] = {"embedUrl": embed}
            try:
                response = await session.post(
                    INNERTUBE_PLAYER_URL,
                    params={"key": key, "prettyPrint": "false"},
                    json=payload,
                    headers={**YOUTUBE_HEADERS, "Content-Type": "application/json"},
                    timeout=self._settings.youtube_timeout,
                    impersonate=self._settings.impersonate or "chrome120",
                )
                if response.status_code != 200:
                    logger.warning("YouTube player HTTP %s (%s) для %s", response.status_code, name, video_id)
                    continue
                tracks = caption_tracks_from_player(response.json())
            except Exception as exc:
                logger.warning("YouTube player ошибка %s (%s): %s", video_id, name, exc)
                continue
            if tracks:
                return tracks
        return []

    async def _captions_via_ytdlp(
        self,
        video_id: str,
        *,
        duration_sec: int | None = None,
    ) -> str | None:
        """timedtext часто пустой без PO-токена; yt-dlp android_vr отдаёт VTT."""
        dest = self._settings.data_dir / "tmp" / "captions" / video_id
        try:
            media = as_media_result(await download_letsplay_captions(video_id, dest))
            if media.path is None:
                if media.error:
                    self.last_caption_cause = media.cause
                    self.last_caption_error = f"yt-dlp субтитры: {media.error}"
                return None
            raw = media.path.read_text(encoding="utf-8", errors="replace")
            suffix = media.path.suffix.lower()
            cues = parse_vtt_cues(raw) if suffix == ".vtt" else parse_caption_cues(raw)
            sampled = sample_transcript(cues, duration_sec=duration_sec)
            text = sampled or parse_caption_xml(raw) or None
            if text:
                logger.info("yt-dlp субтитры %s: %s символов (%s)", video_id, len(text), media.path.name)
                self.last_caption_error = None
                self.last_caption_cause = None
            elif media.error:
                self.last_caption_cause = media.cause
                self.last_caption_error = f"yt-dlp субтитры пустые: {media.error}"
            else:
                self.last_caption_cause = "empty"
                self.last_caption_error = "yt-dlp скачал субтитры, но текст пустой"
            return text
        except Exception:
            logger.exception("yt-dlp субтитры не разобрались для %s", video_id)
            self.last_caption_cause = "unknown"
            self.last_caption_error = "yt-dlp субтитры упали с исключением"
            return None
        finally:
            shutil.rmtree(dest, ignore_errors=True)


def video_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = _VIDEO_ID.search(url)
    return match.group(1) if match else None


def parse_caption_cues(payload: str) -> list[CaptionCue]:
    """Реплики timedtext XML с таймкодами."""
    cues: list[CaptionCue] = []
    for match in re.finditer(
        r"<(?:text|p)\b([^>]*)>(.*?)</(?:text|p)>",
        payload or "",
        flags=re.DOTALL | re.IGNORECASE,
    ):
        attrs, raw = match.group(1), match.group(2)
        raw = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
        raw = re.sub(r"<[^>]+>", "", raw)
        text = " ".join(html_lib.unescape(raw).split())
        if not text:
            continue
        start_match = re.search(r'\bstart="([\d.]+)"', attrs or "", flags=re.IGNORECASE)
        start = float(start_match.group(1)) if start_match else (cues[-1].start if cues else 0.0)
        cues.append(CaptionCue(start=start, text=text))
    return cues


def parse_caption_xml(payload: str) -> str:
    """Склеивает реплики из timedtext XML (srv1 / transcript)."""
    return " ".join(cue.text for cue in parse_caption_cues(payload))


_VTT_ARROW = re.compile(
    r"^(?:(\d{2,}):)?(\d{2}):(\d{2})[.,](\d{3})\s+-->",
)


def _vtt_seconds(line: str) -> float:
    match = _VTT_ARROW.match((line or "").strip())
    if not match:
        return 0.0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    millis = int(match.group(4) or 0)
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def parse_vtt_cues(payload: str) -> list[CaptionCue]:
    """Реплики WEBVTT, в том числе автогенерация YouTube."""
    cues: list[CaptionCue] = []
    lines = (payload or "").replace("\r\n", "\n").split("\n")
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if "-->" not in line:
            index += 1
            continue
        start = _vtt_seconds(line)
        index += 1
        parts: list[str] = []
        while index < len(lines) and lines[index].strip() and "-->" not in lines[index]:
            cleaned = re.sub(r"<[^>]+>", "", lines[index])
            cleaned = " ".join(html_lib.unescape(cleaned).split())
            if cleaned:
                parts.append(cleaned)
            index += 1
        text = " ".join(parts)
        if text and (not cues or cues[-1].text != text):
            cues.append(CaptionCue(start=start, text=text))
    return cues


def _clip_chars(text: str, limit: int) -> str:
    raw = " ".join((text or "").split())
    if len(raw) <= limit:
        return raw
    cut = raw[:limit].rsplit(" ", 1)[0]
    return (cut or raw[:limit]).rstrip() + "…"


def sample_transcript(
    cues: list[CaptionCue],
    *,
    duration_sec: int | None = None,
    limit: int = 25_000,
) -> str:
    """Короткое видео — целиком; длиннее 45 мин — начало/середина/конец ~10 мин."""
    if not cues:
        return ""
    total = float(duration_sec or cues[-1].start or 0)
    if total <= 45 * 60:
        return _clip_chars(collapse_rolling_caption_text(cues), limit)
    window = 10 * 60
    mid = total / 2
    spans = (
        (0.0, float(window)),
        (max(0.0, mid - window / 2), mid + window / 2),
        (max(0.0, total - window), total),
    )
    parts: list[str] = []
    for start, end in spans:
        chunk = collapse_rolling_caption_text(
            [cue for cue in cues if start <= cue.start < end]
        )
        if chunk:
            parts.append(chunk)
    return _clip_chars(" … ".join(parts), limit)


def collapse_rolling_caption_text(cues: list[CaptionCue] | Iterable[CaptionCue]) -> str:
    """Убирает эхо автосубтитров YouTube: каждая фраза повторяется 2–3 раза."""
    words_out: list[str] = []
    prev: list[str] = []
    for cue in cues:
        words = (getattr(cue, "text", None) or str(cue) or "").split()
        if not words:
            continue
        overlap = 0
        for count in range(min(len(words), len(prev)), 0, -1):
            if prev[-count:] == words[:count]:
                overlap = count
                break
        words_out.extend(words[overlap:])
        prev = words
    return " ".join(words_out)


def pick_caption_track(tracks: list[dict[str, Any]]) -> str | None:
    """Русский, затем английский, ручные дорожки выше ASR."""
    scored: list[tuple[int, str]] = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        url = str(track.get("baseUrl") or "").strip()
        if not url:
            continue
        lang = str(track.get("languageCode") or "").lower()
        kind = str(track.get("kind") or "").lower()
        score = 0
        if lang.startswith("ru"):
            score += 30
        elif lang.startswith("en"):
            score += 20
        if kind != "asr":
            score += 5
        scored.append((score, url))
    if not scored:
        return None
    scored.sort(key=lambda item: -item[0])
    url = scored[0][1]
    if "fmt=" not in url:
        url = f"{url}{'&' if '?' in url else '?'}fmt=srv1"
    return url


def caption_tracks_from_player(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    captions = payload.get("captions")
    if not isinstance(captions, dict):
        return []
    renderer = captions.get("playerCaptionsTracklistRenderer")
    if not isinstance(renderer, dict):
        return []
    tracks = renderer.get("captionTracks")
    if not isinstance(tracks, list):
        return []
    return [item for item in tracks if isinstance(item, dict)]


def _extract_player_response(html: str) -> Any | None:
    token = "ytInitialPlayerResponse"
    start = (html or "").find(token)
    if start < 0:
        return None
    eq = html.find("=", start)
    if eq < 0:
        return None
    try:
        data, _end = json.JSONDecoder().raw_decode(html[eq + 1 :].lstrip())
    except json.JSONDecodeError:
        return None
    return data


def is_raw_caption_summary(summary: str | None, transcript: str | None) -> bool:
    """Обрезка субтитров вместо Groq: тот же текст, часто с тройным ASR-эхом."""
    s = " ".join((summary or "").split())
    t = " ".join((transcript or "").split())
    if len(s) < 40 or len(t) < 40:
        return False
    return t.startswith(s[: min(120, len(s))])


def is_video_blurb_summary(text: str | None) -> bool:
    """Описание ролика / CTA вместо заключения по речи автора."""
    raw = " ".join((text or "").split())
    if len(raw) < 40:
        return False
    return bool(_BLURB_MARKERS.search(raw))


def transcript_is_listing_copy(transcript: str | None, description: str | None = None) -> bool:
    """Субтитры совпадают с карточкой YouTube или это короткий маркетинг."""
    spoken = " ".join((transcript or "").split())
    if not spoken:
        return True
    if is_video_blurb_summary(spoken) and spoken.count(" ") < 90:
        return True
    desc = " ".join((description or "").split()).lower()
    body = spoken.lower()
    if len(desc) < 40:
        return False
    desc_words = {word for word in re.findall(r"[a-zа-яё0-9]{3,}", desc) if word}
    if len(desc_words) < 8:
        return False
    body_words = set(re.findall(r"[a-zа-яё0-9]{3,}", body))
    return (len(desc_words & body_words) / len(desc_words)) >= 0.7


def youtube_egress_configured(settings: Settings | None = None) -> bool:
    """Прокси или cookies — есть шанс обойти датацентр Render."""
    cfg = settings or get_settings()
    if (getattr(cfg, "youtube_proxy", "") or "").strip():
        return True
    if (getattr(cfg, "youtube_cookies", "") or "").strip():
        return True
    path = (getattr(cfg, "youtube_cookies_path", "") or "").strip()
    return bool(path)


def letsplay_is_antibot(game: Any) -> bool:
    return getattr(game, "youtube_summary_source", None) == LETS_PLAY_ANTIBOT_SOURCE


def apply_letsplay_antibot(game: Game) -> None:
    """Ролик оставляем, саммари нет: Render/датацентр режет аудио и субтитры."""
    game.youtube_summary = None
    game.youtube_summary_source = LETS_PLAY_ANTIBOT_SOURCE


def letsplay_has_summary(game: Any) -> bool:
    source = getattr(game, "youtube_summary_source", None)
    summary = " ".join((getattr(game, "youtube_summary", None) or "").split())
    if source not in LETS_PLAY_READY_SOURCES or not summary:
        return False
    if is_raw_caption_summary(summary, getattr(game, "youtube_transcript_sample", None)):
        return False
    if is_video_blurb_summary(summary):
        return False
    return True


def letsplay_has_video(game: Any) -> bool:
    """Ролик можно показать, даже если Groq ещё не дал саммари."""
    return bool(video_id_from_url(getattr(game, "youtube_url", None)))


async def persist_letsplay_progress(game: Any) -> None:
    """Сразу пишем URL в БД, чтобы карточка не ждала саммари Groq."""
    try:
        session = async_object_session(game)
    except Exception:
        return
    if session is None:
        return
    await session.commit()
    await session.refresh(game)


def needs_letsplay_job(game: Any, *, force: bool = False) -> bool:
    """True, пока нет саммари по субтитрам/Whisper. Старая заглушка «субтитры» — перепрос.

    force: кнопка «Запустить летсплеи» заново берёт и честные заглушки
    (Whisper 400 / антибот не должны навсегда закрывать карточку).
    """
    if letsplay_has_summary(game):
        return False
    if force:
        return True
    source = getattr(game, "youtube_summary_source", None)
    summary = " ".join((getattr(game, "youtube_summary", None) or "").split())
    if source == LETS_PLAY_ANTIBOT_SOURCE:
        return youtube_egress_configured()
    if source == "none":
        return summary == LETS_PLAY_STUB_CAPTIONS
    return True


def _video_from_game(game: Any) -> YoutubeVideo | None:
    video_id = video_id_from_url(getattr(game, "youtube_url", None))
    if not video_id:
        return None
    title = (getattr(game, "youtube_title", None) or getattr(game, "title", None) or video_id).strip()
    return YoutubeVideo(
        video_id=video_id,
        title=title,
        duration_sec=getattr(game, "youtube_duration_sec", None),
        views=getattr(game, "youtube_views", None),
        channel=getattr(game, "youtube_channel", None),
    )


def apply_letsplay(game: Game, hit: LetsPlayHit) -> None:
    prev = video_id_from_url(getattr(game, "youtube_url", None))
    nxt = video_id_from_url(hit.url)
    if prev != nxt:
        game.youtube_transcript_sample = None
    game.youtube_url = hit.url
    game.youtube_title = hit.title
    game.youtube_channel = hit.channel
    game.youtube_views = hit.views
    game.youtube_duration_sec = hit.duration_sec
    game.youtube_kind = hit.kind


def apply_letsplay_stub(game: Game) -> bool:
    """ТЗ: саммари обязательно. Нет ролика с текстом — заглушка, без видео."""
    changed = (
        game.youtube_url is not None
        or game.youtube_title is not None
        or game.youtube_channel is not None
        or game.youtube_views is not None
        or game.youtube_duration_sec is not None
        or game.youtube_kind is not None
        or game.youtube_transcript_sample is not None
        or game.youtube_summary != LETS_PLAY_STUB
        or getattr(game, "youtube_summary_source", None) != "none"
    )
    game.youtube_url = None
    game.youtube_title = None
    game.youtube_channel = None
    game.youtube_views = None
    game.youtube_duration_sec = None
    game.youtube_kind = None
    game.youtube_transcript_sample = None
    game.youtube_summary = LETS_PLAY_STUB
    game.youtube_summary_source = "none"
    return changed


def clear_letsplay_attempt(game: Game) -> None:
    """Сброс ролика без заглушки: техсбой (Whisper 400, антибот) — карточку ещё разберём."""
    game.youtube_url = None
    game.youtube_title = None
    game.youtube_channel = None
    game.youtube_views = None
    game.youtube_duration_sec = None
    game.youtube_kind = None
    game.youtube_transcript_sample = None


async def whisper_letsplay_text(
    video_id: str,
    *,
    duration_sec: int | None,
    llm: Any,
    slug: str | None = None,
) -> WhisperOutcome:
    """Скачать аудио популярного ролика и расшифровать Groq Whisper."""
    settings = get_settings()
    if not getattr(settings, "whisper_enabled", True):
        return WhisperOutcome(error="Whisper выключен", cause="empty")
    if llm is None or not hasattr(llm, "transcribe"):
        return WhisperOutcome(error="Whisper: нет LLM-клиента", cause="empty")
    dest = settings.data_dir / "tmp" / "whisper" / video_id
    try:
        media = as_media_result(
            await download_letsplay_audio(video_id, dest, duration_sec=duration_sec)
        )
        if not media.files:
            error = f"Whisper не запустился: {media.error or 'yt-dlp не скачал аудио'}"
            logger.info("Whisper: нет аудио для %s (%s)", video_id, error)
            if hasattr(llm, "note"):
                try:
                    await llm.note(
                        prompt=f"[whisper {video_id}]",
                        error=error,
                        slug=slug,
                        kind="whisper",
                        model=str(getattr(settings, "whisper_model", None) or "whisper-large-v3-turbo"),
                    )
                except Exception:
                    logger.exception("Не записали whisper-заметку для %s", video_id)
            return WhisperOutcome(error=error, cause=media.cause)
        logger.info("Whisper: %s → %s", video_id, [path.name for path in media.files])
        parts: list[str] = []
        api_error: str | None = None
        for path in media.files:
            if path.suffix.lower() == ".part" or ".part" in path.suffixes:
                continue
            ready = prepare_whisper_audio(path)
            if ready.stat().st_size > 25 * 1024 * 1024:
                api_error = f"аудио {ready.name} всё ещё больше 25 МБ"
                continue
            result = await llm.transcribe(
                ready.read_bytes(),
                filename=ready.name,
                slug=slug,
                kind="whisper",
            )
            if getattr(result, "error", None):
                api_error = str(result.error)
                logger.warning("Whisper %s: %s", ready.name, result.error)
            text = " ".join((result.text or "").split())
            if text:
                parts.append(text)
        if not parts:
            error = f"Whisper API: {api_error}" if api_error else "Whisper не вернул текст"
            logger.warning("Whisper: нет текста для %s (%s)", video_id, error)
            return WhisperOutcome(error=error, cause="unknown")
        return WhisperOutcome(text=_clip_chars(" … ".join(parts), 25_000))
    except Exception as exc:
        logger.exception("Whisper не расшифровал %s", video_id)
        return WhisperOutcome(error=f"Whisper упал: {exc}", cause="unknown")
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def _hit_from_video(video: YoutubeVideo, sample: str | None = None) -> LetsPlayHit:
    return LetsPlayHit(
        url=video.url,
        title=video.title,
        channel=video.channel,
        views=video.views,
        duration_sec=video.duration_sec,
        kind="letsplay",
        transcript_sample=sample,
    )


async def conclude_letsplay(
    game: Game,
    client: YoutubeClient,
    llm: Any | None,
    *,
    sample: str | None = None,
    source: str = "transcript",
    description: str | None = None,
) -> str:
    """Субтитры или Whisper → Groq-саммари по речи автора. Без текста саммари не пишем.

    Возвращает: saved | exists | no_text | listing | blurb | llm_error
    """
    from app.llm.prompts import (
        YOUTUBE_SUMMARY_SYSTEM_PROMPT,
        build_youtube_summary_prompt,
        clip_summary_text,
        parse_summary_payload,
    )

    if letsplay_has_summary(game):
        return "exists"
    video_id = video_id_from_url(game.youtube_url)
    text = " ".join((sample or getattr(game, "youtube_transcript_sample", None) or "").split()) or None
    if not text and video_id:
        try:
            text = await client.fetch_transcript(
                video_id,
                duration_sec=getattr(game, "youtube_duration_sec", None),
            )
        except Exception:
            logger.exception("Субтитры летсплея не разобрались для %s", game.slug)
    if text:
        game.youtube_transcript_sample = text
    if not text:
        return "no_text"
    if transcript_is_listing_copy(text, description):
        logger.info("Пропуск летсплея %s: в субтитрах описание ролика, не речь", game.slug)
        return "listing"
    kind = getattr(game, "youtube_kind", None) or "letsplay"
    origin = source if source in LETS_PLAY_READY_SOURCES else "transcript"
    summary = None
    if llm is None:
        return "llm_error"
    prompt = build_youtube_summary_prompt(
        title=game.title or game.slug or "",
        transcript=text,
        kind=kind,
    )
    result = await llm.complete(
        prompt,
        system=YOUTUBE_SUMMARY_SYSTEM_PROMPT,
        slug=game.slug,
        kind="youtube",
        json_object=True,
    )
    if result.error:
        logger.warning("Groq не дал саммари летсплея для %s: %s", game.slug, result.error)
        return "llm_error"
    if result.text:
        parsed = parse_summary_payload(result.text)
        summary = (parsed.summary or "").strip()
        if not parsed.parsed:
            summary = clip_summary_text(result.text)
        if is_video_blurb_summary(summary) or not summary:
            logger.warning("Groq вернул описание ролика вместо речи автора для %s", game.slug)
            return "blurb"
    if not summary:
        logger.warning("Groq не дал саммари летсплея для %s — сырые субтитры не пишем", game.slug)
        return "blurb"
    game.youtube_summary = summary
    game.youtube_summary_source = origin
    game.youtube_kind = kind
    return "saved"


async def attach_letsplay(
    game: Game,
    client: YoutubeClient | None,
    llm: Any | None = None,
) -> bool:
    """Популярный летсплей: субтитры или Whisper + саммари. Иначе заглушка."""
    if client is None:
        return False
    title = (game.title or game.slug or "").strip()
    if not title:
        return False
    if letsplay_has_summary(game):
        return False
    if getattr(game, "youtube_summary_source", None) in {"none", LETS_PLAY_ANTIBOT_SOURCE}:
        game.youtube_summary = None
        game.youtube_summary_source = None

    async def _trace(message: str, *, failed: bool = False) -> None:
        if llm is None or not hasattr(llm, "note"):
            return
        try:
            await llm.note(
                prompt=f"[youtube search] {title}",
                error=message if failed else None,
                response="" if failed else message,
                slug=getattr(game, "slug", None),
                kind="youtube",
            )
        except Exception:
            logger.exception("Не записали youtube-заметку для %s", getattr(game, "slug", None))

    known = _video_from_game(game)
    search_failed = False
    try:
        videos = await client.list_letsplays(title)
    except Exception:
        logger.exception("Поиск летсплея сломался для %s", game.slug)
        search_failed = True
        videos = []
    if known is not None:
        videos = [known, *[item for item in videos if item.video_id != known.video_id]]
    if not videos:
        found = int(getattr(client, "last_search_count", 0) or 0)
        if search_failed:
            await _trace("Поиск YouTube упал с ошибкой", failed=True)
            clear_letsplay_attempt(game)
            return False
        await _trace(
            f"Поиск YouTube: {found} роликов, ни один не прошёл фильтр летсплея"
            if found
            else "Поиск YouTube не вернул ролики (Innertube/HTML пустые, yt-dlp тоже)"
        )
        return apply_letsplay_stub(game)
    from app.llm.client import groq_chat_blocked

    technical_block = False
    saw_non_speech = False
    caption_error: str | None = None
    whisper_error: str | None = None
    for video in videos:
        apply_letsplay(game, _hit_from_video(video))
        await persist_letsplay_progress(game)
        text = None
        origin = "transcript"
        try:
            text = await client.fetch_transcript(
                video.video_id,
                duration_sec=video.duration_sec,
            )
        except Exception:
            logger.exception("Субтитры не разобрались для %s", video.video_id)
            technical_block = True
            caption_error = "субтитры упали с исключением"
            text = None
        else:
            caption_error = getattr(client, "last_caption_error", None) or caption_error
        if not text:
            spoken = await whisper_letsplay_text(
                video.video_id,
                duration_sec=video.duration_sec,
                llm=llm,
                slug=game.slug,
            )
            if spoken.text:
                text = spoken.text
                origin = "whisper"
            else:
                technical_block = True
                whisper_error = spoken.error or whisper_error
                if spoken.render_blocked:
                    apply_letsplay_antibot(game)
                    parts: list[str] = []
                    if caption_error:
                        parts.append(
                            caption_error
                            if caption_error.lower().startswith("субтитр")
                            else f"Субтитры: {caption_error}"
                        )
                    parts.append(ANTIBOT_RENDER_MESSAGE)
                    await _trace(". ".join(parts), failed=True)
                    return letsplay_has_video(game)
                if spoken.blocked:
                    break
                continue
        status = await conclude_letsplay(
            game,
            client,
            llm,
            sample=text,
            source=origin,
            description=video.description,
        )
        if status == "saved" or letsplay_has_summary(game):
            return True
        if status == "llm_error" or groq_chat_blocked():
            logger.warning(
                "Летсплей %s найден, саммари Groq нет — карточку не заглушаем",
                game.slug,
            )
            return letsplay_has_video(game)
        saw_non_speech = True
        logger.info("Летсплей %s без речи автора (%s), следующий ролик", game.slug, status)
    if technical_block and not saw_non_speech:
        logger.warning(
            "Летсплей %s: нет текста из‑за субтитров/Whisper — заглушку не ставим",
            game.slug,
        )
        parts: list[str] = []
        if caption_error:
            parts.append(
                caption_error
                if caption_error.lower().startswith("субтитр")
                else f"Субтитры: {caption_error}"
            )
        if whisper_error:
            parts.append(whisper_error)
        caption_cause = getattr(client, "last_caption_cause", None)
        if is_render_block(caption_cause, caption_error) or is_render_block(None, whisper_error):
            apply_letsplay_antibot(game)
            if not any(is_render_block(None, item) for item in parts):
                parts.append(ANTIBOT_RENDER_MESSAGE)
        await _trace(
            ". ".join(parts) if parts else "Ролик найден, но субтитры и Whisper не дали текст",
            failed=True,
        )
        return letsplay_has_video(game)
    return apply_letsplay_stub(game)


async def backfill_letsplays(
    settings: Settings | None = None,
    *,
    client: YoutubeClient | None = None,
    llm: Any | None = None,
    limit: int | None = None,
    slugs: list[str] | None = None,
) -> int:
    """Добирает летсплеи: сначала slug'и прогона, затем дырки без саммари."""
    settings = settings or get_settings()
    if not settings.youtube_enabled:
        return 0
    cap = settings.youtube_sweep_limit if limit is None else limit
    wanted = [item for item in (slugs or []) if item]
    async with SessionLocal() as session:
        pending: list[int] = []
        seen: set[int] = set()
        if wanted:
            rows = await session.execute(select(Game.id).where(Game.slug.in_(wanted)).order_by(Game.id))
            for game_id in rows.scalars().all():
                if game_id not in seen:
                    seen.add(game_id)
                    pending.append(game_id)
        holes = await session.execute(
            select(Game.id)
            .where(
                or_(
                    Game.youtube_summary.is_(None),
                    Game.youtube_summary == "",
                )
            )
            .order_by(Game.id)
            .limit(cap)
        )
        for game_id in holes.scalars().all():
            if game_id in seen:
                continue
            seen.add(game_id)
            pending.append(game_id)
            if len(pending) >= len(wanted) + cap:
                break
    if not pending:
        return 0
    filled = 0
    owns_client = client is None
    finder = client or YoutubeClient(settings)
    try:
        if owns_client:
            await finder.__aenter__()
        for game_id in pending:
            async with SessionLocal() as session:
                game = await session.get(Game, game_id)
                if game is None:
                    continue
                if await attach_letsplay(game, finder, llm=llm):
                    await session.commit()
                    filled += 1
    finally:
        if owns_client:
            await finder.__aexit__(None, None, None)
    logger.info("YouTube этап: обработано %s из %s", filled, len(pending))
    return filled


async def process_letsplay_slug(
    slug: str,
    *,
    settings: Settings | None = None,
    llm: Any | None = None,
    client: YoutubeClient | None = None,
) -> bool:
    """Одна игра из очереди: популярный летсплей и саммари, иначе заглушка."""
    settings = settings or get_settings()
    if not settings.youtube_enabled or not slug:
        return False
    owns_client = client is None
    finder = client or YoutubeClient(settings)
    try:
        if owns_client:
            await finder.__aenter__()
        async with SessionLocal() as session:
            game = await session.scalar(select(Game).where(Game.slug == slug))
            if game is None:
                return False
            if letsplay_has_summary(game):
                return True
            done = await attach_letsplay(game, finder, llm=llm)
            await session.commit()
            return bool(done) or letsplay_has_video(game)
    finally:
        if owns_client:
            await finder.__aexit__(None, None, None)
    return False
