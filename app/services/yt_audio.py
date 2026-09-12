"""Аудио летсплея через yt-dlp: любое скачанное окно сжимаем под лимит Groq Whisper 25 МБ."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

WHISPER_MAX_BYTES = 25 * 1024 * 1024
WHISPER_CONVERT_MIN_BYTES = 8 * 1024
WHISPER_WAV_BYTES_PER_SEC = 16000 * 2
WHISPER_MAX_SECONDS = max(30, int(WHISPER_MAX_BYTES / WHISPER_WAV_BYTES_PER_SEC * 0.95))
PLAYER_CLIENTS = ("android", "android_vr", "ios", "mweb", "tv_embedded", "web")
_AUDIO_FORMAT = (
    "bestaudio[vcodec=none]/bestaudio[ext=m4a]/bestaudio[ext=webm]"
    "/140/251/250/249/18/bestaudio/ba"
)
_BOT_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "confirm you're not a bot",
    "confirm you are a bot",
    "use --cookies",
    "please sign in",
    "bot check",
    "captcha",
)
_CAUSE_WEIGHT = {
    "ok": 0,
    "empty": 1,
    "unknown": 2,
    "format": 3,
    "oversized": 4,
    "unavailable": 5,
    "js": 6,
    "timeout": 7,
    "http": 8,
    "token": 9,
    "bot": 10,
}
ANTIBOT_RENDER_MESSAGE = (
    "Антибот YouTube: Render (датацентр) не пускает скачивать аудио и субтитры. "
    "Функционал Whisper и саммари по ролику есть — с домашнего IP. "
    "Примеры для проверяющего: вкладка «Примеры» — Valheim и Elden Ring."
)
_RENDER_BLOCK_CAUSES = frozenset({"bot", "http", "token"})
_HOSTILE_CAUSES = frozenset({"bot", "http", "token", "timeout"})


@dataclass
class YtMediaResult:
    """Файлы yt-dlp плюс причина, если скачать не вышло."""

    files: list[Path] = field(default_factory=list)
    cause: str = "ok"
    error: str | None = None

    @property
    def path(self) -> Path | None:
        return self.files[0] if self.files else None

    @property
    def blocked(self) -> bool:
        return self.cause in _HOSTILE_CAUSES


def as_media_result(raw: Any) -> YtMediaResult:
    if isinstance(raw, YtMediaResult):
        return raw
    if isinstance(raw, Path):
        return YtMediaResult(files=[raw])
    if isinstance(raw, list):
        paths = [item for item in raw if isinstance(item, Path)]
        if paths:
            return YtMediaResult(files=paths)
        return YtMediaResult(cause="empty", error="yt-dlp не скачал аудио")
    return YtMediaResult(cause="empty", error="yt-dlp не скачал аудио")


_SKIP_SUFFIXES = {".part", ".ytdl", ".tmp", ".temp"}
_AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".ogg", ".opus", ".wav", ".webm"}
_CREATE_NO_WINDOW = 0x08000000


def audio_windows(duration_sec: int | None, *, compact: bool | None = None) -> list[tuple[float, float] | None]:
    """None — целиком; иначе начало / середина / конец по 10 минут.

    Без ffmpeg целый ролик >10 мин может не влезть в 25 МБ, поэтому режем окна.
    """
    if compact is None:
        compact = _has_ffmpeg()
    if duration_sec is None:
        return [(0.0, 10 * 60.0)]
    if duration_sec <= 10 * 60:
        return [None]
    if duration_sec <= 45 * 60 and compact:
        return [None]
    total = float(duration_sec or 3600)
    window = 10 * 60.0
    mid = total / 2.0
    return [
        (0.0, window),
        (max(0.0, mid - window / 2.0), min(total, mid + window / 2.0)),
        (max(0.0, total - window), total),
    ]


def youtube_download_blocked(stderr: str) -> bool:
    return classify_ytdlp_error(stderr)[0] == "bot"


def classify_ytdlp_error(stderr: str, *, code: int | None = None) -> tuple[str, str]:
    """Причина сбоя yt-dlp: bot / timeout / http / format / token / … и текст для журнала."""
    raw = stderr or ""
    text = raw.lower().replace("’", "'").replace("`", "'")
    if code == 124 or "yt-dlp timeout" in text:
        return "timeout", "таймаут yt-dlp"
    if any(marker in text for marker in _BOT_MARKERS):
        return "bot", ANTIBOT_RENDER_MESSAGE
    if "po token" in text or "potoken" in text:
        return "token", ANTIBOT_RENDER_MESSAGE
    if "http error 429" in text or "too many requests" in text:
        return "http", "YouTube HTTP 429"
    if "http error 403" in text or "403: forbidden" in text or "http error 401" in text:
        return "http", ANTIBOT_RENDER_MESSAGE
    if "requested format is not available" in text:
        return "format", "yt-dlp: нужный формат аудио недоступен"
    if "this video is unavailable" in text or "video unavailable" in text:
        return "unavailable", "ролик недоступен"
    if "no subtitle" in text or "no automatic caption" in text or "subtitles are not available" in text:
        return "empty", "у ролика нет субтитров"
    if "nsig" in text or ("signature" in text and "fail" in text):
        return "js", "yt-dlp не расшифровал подпись (JS challenge)"
    snippet = " ".join(raw.split())
    if snippet:
        return "unknown", f"yt-dlp не скачал: {snippet[-180:]}"
    return "empty", "yt-dlp не скачал файл"


def is_antibot_error(message: str | None) -> bool:
    text = (message or "").lower()
    return "антибот youtube" in text or "бот-стена" in text


def is_render_block(cause: str | None, message: str | None = None) -> bool:
    if cause in _RENDER_BLOCK_CAUSES:
        return True
    return is_antibot_error(message)


def _decode_cookie_blob(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.lower().startswith("base64:"):
        text = text[7:].strip()
        return base64.b64decode(text).decode("utf-8")
    looks_netscape = text.lstrip().startswith("# Netscape") or "\t" in text[:800]
    if looks_netscape:
        return raw
    try:
        decoded = base64.b64decode(text, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return raw
    if decoded.lstrip().startswith("# Netscape") or "\t" in decoded[:800] or "youtube" in decoded.lower():
        return decoded
    return raw


def youtube_cookies_file() -> Path | None:
    """Netscape cookies.txt: путь на диске или секрет из env. Не логировать содержимое."""
    settings = get_settings()
    path = (getattr(settings, "youtube_cookies_path", "") or "").strip()
    if path:
        candidate = Path(path)
        return candidate if candidate.is_file() else None
    raw = (getattr(settings, "youtube_cookies", "") or "").strip()
    if not raw:
        return None
    dest = settings.data_dir / "tmp" / "youtube_cookies.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_decode_cookie_blob(raw), encoding="utf-8")
    return dest


def youtube_cookie_dict() -> dict[str, str] | None:
    path = youtube_cookies_file()
    if path is None:
        return None
    cookies: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("#HttpOnly_"):
                raw = raw[len("#HttpOnly_") :]
            elif raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) < 7:
                continue
            domain, name, value = parts[0], parts[5], parts[6]
            if name and ("youtube" in domain or "google" in domain):
                cookies[name] = value
    except OSError:
        return None
    return cookies or None


def _ytdlp_network_args() -> list[str]:
    """Прокси и cookies — реальный обход датацентра YouTube, не player_client."""
    args: list[str] = ["--geo-bypass"]
    settings = get_settings()
    proxy = (getattr(settings, "youtube_proxy", "") or "").strip()
    if proxy:
        args.extend(["--proxy", proxy])
    cookies = youtube_cookies_file()
    if cookies is not None:
        args.extend(["--cookies", str(cookies)])
    return args


def _prefer_cause(current: str, current_error: str | None, cause: str, error: str | None) -> tuple[str, str | None]:
    if _CAUSE_WEIGHT.get(cause, 0) >= _CAUSE_WEIGHT.get(current, 0):
        return cause, error or current_error
    return current, current_error


def fit_whisper_audio(src: Path, *, timeout: float = 90.0) -> Path | None:
    """Любой контейнер → WAV 16 kHz mono под 25 МБ (обрезаем длительность, если надо)."""
    if not src.is_file() or src.stat().st_size <= 0:
        return None
    size = src.stat().st_size
    if size < WHISPER_CONVERT_MIN_BYTES:
        return src if size <= WHISPER_MAX_BYTES else None
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        if size <= WHISPER_MAX_BYTES:
            return src
        logger.warning("Аудио %s больше 25 МБ (%s) и нет ffmpeg", src.name, size)
        return None
    ready = prepare_whisper_audio(src, timeout=timeout)
    if ready.is_file() and 0 < ready.stat().st_size <= WHISPER_MAX_BYTES:
        return ready
    dest = src.with_name(f"{src.stem}_groq25.wav")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-t",
        str(WHISPER_MAX_SECONDS),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    code, stderr = _run_ytdlp_sync(cmd, timeout)
    if code == 0 and dest.is_file() and 0 < dest.stat().st_size <= WHISPER_MAX_BYTES:
        return dest
    if size <= WHISPER_MAX_BYTES:
        return src
    logger.warning("Аудио %s больше 25 МБ (%s), пропуск", src.name, size)
    if stderr:
        logger.warning("ffmpeg trim %s: %s %s", src.name, code, stderr[-200:])
    return None


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and Path(path).exists():
            return path
    except Exception:
        logger.warning("imageio-ffmpeg не дал бинарник ffmpeg")
    return None


def _has_ffmpeg() -> bool:
    return ffmpeg_path() is not None


def prepare_whisper_audio(src: Path, *, timeout: float = 60.0) -> Path:
    """WAV 16 kHz mono: Groq часто отвечает 400 на opus/webm и кривые контейнеры."""
    if not src.is_file() or src.stat().st_size <= 0:
        return src
    if src.stat().st_size < WHISPER_CONVERT_MIN_BYTES:
        return src
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return src
    dest = src.with_name(f"{src.stem}_groq.wav")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    code, stderr = _run_ytdlp_sync(cmd, timeout)
    if code == 0 and dest.is_file() and dest.stat().st_size > 0:
        return dest
    logger.warning("ffmpeg→wav %s: %s %s", src.name, code, (stderr or "")[-200:])
    return src


def _js_runtime_args() -> list[str]:
    node = shutil.which("node")
    if node:
        return ["--js-runtimes", f"node:{node}"]
    return []


def _player_extractor_args(player_client: str | None) -> str:
    if not player_client:
        return "youtube:"
    args = f"youtube:player_client={player_client}"
    if player_client in {"android", "ios"}:
        args += ",player_skip=webpage"
    return args


def _ytdlp_cmd(
    url: str,
    outtmpl: str,
    *,
    player_client: str,
    section: tuple[float, float] | None,
) -> list[str]:
    ffmpeg = ffmpeg_path()
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-warnings",
        "--no-progress",
        "--retries",
        "2",
        "--fragment-retries",
        "2",
        *(_ytdlp_network_args()),
        "--extractor-args",
        _player_extractor_args(player_client),
        "-o",
        outtmpl,
    ]
    cmd.extend(_js_runtime_args())
    # Без -x: imageio-ffmpeg на Windows падает на mp3, а Groq Whisper ест webm/m4a.
    cmd.extend(["-f", _AUDIO_FORMAT])
    if ffmpeg:
        cmd.extend(["--ffmpeg-location", ffmpeg])
    if section is not None and ffmpeg:
        start, end = section
        cmd.extend(["--download-sections", f"*{start:.0f}-{end:.0f}", "--force-keyframes-at-cuts"])
    cmd.append(url)
    return cmd


def _run_ytdlp_sync(cmd: list[str], timeout: float) -> tuple[int, str]:
    """Обычный subprocess: на Windows SelectorEventLoop не умеет asyncio-процессы."""
    kwargs: dict = {
        "capture_output": True,
        "timeout": timeout,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = _CREATE_NO_WINDOW
    try:
        proc = subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired:
        return 124, "yt-dlp timeout"
    text = ((proc.stderr or b"") + b"\n" + (proc.stdout or b"")).decode("utf-8", errors="replace")
    return int(proc.returncode or 0), text


async def _run_ytdlp(cmd: list[str], timeout: float) -> tuple[int, str]:
    return await asyncio.to_thread(_run_ytdlp_sync, cmd, timeout)


def _run_ytdlp_stdout_sync(cmd: list[str], timeout: float) -> tuple[int, str, str]:
    kwargs: dict = {"capture_output": True, "timeout": timeout}
    if sys.platform == "win32":
        kwargs["creationflags"] = _CREATE_NO_WINDOW
    try:
        proc = subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired:
        return 124, "", "yt-dlp timeout"
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    return int(proc.returncode or 0), stdout, stderr


def parse_ytdlp_search_json(raw: str) -> list[dict[str, Any]]:
    """Разбор `yt-dlp -J ytsearchN:...` в список роликов."""
    blob = (raw or "").strip()
    start = blob.find("{")
    if start < 0:
        return []
    try:
        data = json.loads(blob[start:])
    except json.JSONDecodeError:
        return []
    entries: list[Any]
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        entries = data["entries"]
    elif isinstance(data, dict) and data.get("id"):
        entries = [data]
    else:
        return []
    videos: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        video_id = str(item.get("id") or "").strip()
        title = str(item.get("title") or "").strip()
        if len(video_id) != 11 or not title:
            continue
        duration = item.get("duration")
        try:
            duration_sec = int(float(duration)) if duration is not None else None
        except (TypeError, ValueError):
            duration_sec = None
        views = item.get("view_count")
        try:
            views_i = int(views) if views is not None else None
        except (TypeError, ValueError):
            views_i = None
        channel = str(item.get("channel") or item.get("uploader") or "").strip() or None
        desc = str(item.get("description") or "").strip() or None
        videos.append(
            {
                "video_id": video_id,
                "title": title,
                "duration_sec": duration_sec,
                "views": views_i,
                "channel": channel,
                "description": desc,
            }
        )
    return videos


async def search_youtube_ytdlp(
    query: str,
    *,
    limit: int = 10,
    timeout: float = 45.0,
) -> list[dict[str, Any]]:
    """Поиск через yt-dlp: на Render Innertube/HTML часто пустые."""
    q = " ".join((query or "").split())
    if not q:
        return []
    n = max(1, min(int(limit or 10), 15))
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--no-warnings",
        "--no-progress",
        "--skip-download",
        *(_ytdlp_network_args()),
        "-J",
        f"ytsearch{n}:{q}",
    ]
    cmd.extend(_js_runtime_args())
    code, stdout, stderr = await asyncio.to_thread(_run_ytdlp_stdout_sync, cmd, timeout)
    videos = parse_ytdlp_search_json(stdout)
    if videos:
        return videos
    logger.warning("yt-dlp search %s для %s: %s", code, q, (stderr or stdout or "")[-300:])
    return []


def _is_ready_audio(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    suffixes = [item.lower() for item in path.suffixes]
    if any(item in _SKIP_SUFFIXES for item in suffixes):
        return False
    if path.suffix.lower() not in _AUDIO_SUFFIXES:
        return False
    return True


def _pick_output(folder: Path, stem: str) -> Path | None:
    found = [path for path in folder.glob(f"{stem}.*") if _is_ready_audio(path)]
    if not found:
        return None
    under = [path for path in found if path.stat().st_size <= WHISPER_MAX_BYTES]
    candidate = max(under, key=lambda item: item.stat().st_size) if under else min(
        found, key=lambda item: item.stat().st_size
    )
    return fit_whisper_audio(candidate)


def _caption_rank(path: Path) -> tuple[int, int]:
    name = path.name.lower()
    lang = 0
    if ".ru." in name:
        lang = 2
    elif ".en." in name:
        lang = 1
    return (lang, path.stat().st_size)


def _pick_caption(folder: Path, video_id: str) -> Path | None:
    found = [
        path
        for path in folder.glob(f"{video_id}*")
        if path.is_file()
        and path.stat().st_size > 0
        and path.suffix.lower() in {".vtt", ".srv1", ".srv3", ".srt"}
    ]
    if not found:
        return None
    return max(found, key=_caption_rank)


def _ytdlp_captions_cmd(url: str, outtmpl: str, player_client: str | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-warnings",
        "--no-progress",
        "--no-update",
        "--skip-download",
        "--ignore-errors",
        "--write-auto-sub",
        "--write-sub",
        "--sub-langs",
        "en,ru",
        "--sub-format",
        "vtt",
        "--retries",
        "2",
        *(_ytdlp_network_args()),
        "-o",
        outtmpl,
    ]
    cmd.extend(_js_runtime_args())
    cmd.extend(["--extractor-args", _player_extractor_args(player_client)])
    ffmpeg = ffmpeg_path()
    if ffmpeg:
        cmd.extend(["--ffmpeg-location", ffmpeg])
    cmd.append(url)
    return cmd


async def download_letsplay_captions(video_id: str, dest_dir: Path, *, timeout: float = 90.0) -> YtMediaResult:
    """Субтитры через yt-dlp. timedtext без PO-токена часто приходит пустым."""
    video_id = (video_id or "").strip()
    if len(video_id) != 11:
        return YtMediaResult(cause="empty", error="некорректный video_id")
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    outtmpl = str(dest_dir / f"{video_id}.%(ext)s")
    cause, error = "empty", "yt-dlp не скачал субтитры"
    for client in (*PLAYER_CLIENTS, None):
        code, stderr = await _run_ytdlp(_ytdlp_captions_cmd(url, outtmpl, client), timeout)
        path = _pick_caption(dest_dir, video_id)
        if path is not None:
            return YtMediaResult(files=[path])
        nxt_cause, nxt_error = classify_ytdlp_error(stderr, code=code)
        cause, error = _prefer_cause(cause, error, nxt_cause, nxt_error)
        logger.warning("yt-dlp субтитры %s для %s (%s): %s", code, video_id, client, (stderr or "")[-300:])
        if nxt_cause in _HOSTILE_CAUSES:
            break
    return YtMediaResult(cause=cause, error=error)


async def download_letsplay_audio(
    video_id: str,
    dest_dir: Path,
    *,
    duration_sec: int | None = None,
    timeout: float = 180.0,
) -> YtMediaResult:
    """Качает 1 или 3 окна аудио и сразу сжимает каждое под 25 МБ."""
    video_id = (video_id or "").strip()
    if len(video_id) != 11:
        return YtMediaResult(cause="empty", error="некорректный video_id")
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    windows = audio_windows(duration_sec)
    if not _has_ffmpeg():
        windows = [None]
    files: list[Path] = []
    cause, error = "empty", "yt-dlp не скачал аудио"
    for index, section in enumerate(windows):
        outtmpl = str(dest_dir / f"{video_id}_{index}.%(ext)s")
        path = None
        for client in PLAYER_CLIENTS:
            for leftover in dest_dir.glob(f"{video_id}_{index}.*"):
                leftover.unlink(missing_ok=True)
            code, stderr = await _run_ytdlp(
                _ytdlp_cmd(url, outtmpl, player_client=client, section=section),
                timeout,
            )
            path = _pick_output(dest_dir, f"{video_id}_{index}")
            if path is not None:
                if code != 0:
                    logger.info(
                        "Аудио %s после yt-dlp %s (%s)",
                        path.name,
                        code,
                        client,
                    )
                break
            nxt_cause, nxt_error = classify_ytdlp_error(stderr, code=code)
            cause, error = _prefer_cause(cause, error, nxt_cause, nxt_error)
            logger.warning(
                "yt-dlp %s для %s (%s): %s",
                code,
                video_id,
                client,
                (stderr or "")[-400:],
            )
            if nxt_cause in _HOSTILE_CAUSES:
                return YtMediaResult(cause=nxt_cause, error=nxt_error)
        if path is None:
            if files:
                break
            return YtMediaResult(cause=cause, error=error)
        files.append(path)
    if files:
        return YtMediaResult(files=files)
    return YtMediaResult(cause=cause, error=error)
