"""Аудио летсплея через yt-dlp: короткие окна под лимит Groq Whisper 25 МБ."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WHISPER_MAX_BYTES = 25 * 1024 * 1024
WHISPER_CONVERT_MIN_BYTES = 8 * 1024
PLAYER_CLIENTS = ("tv_embedded", "web", "android")
_BOT_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "confirm you're not a bot",
    "use --cookies",
)
_SKIP_SUFFIXES = {".part", ".ytdl", ".tmp", ".temp"}
_AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".ogg", ".opus", ".wav", ".webm"}
_AUDIO_FORMAT = (
    "bestaudio[vcodec=none]/bestaudio[ext=m4a]/bestaudio[ext=webm]"
    "/140/251/250/249/bestaudio/ba"
)
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
    text = (stderr or "").lower()
    return any(marker in text for marker in _BOT_MARKERS)


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
        "--extractor-args",
        f"youtube:player_client={player_client}",
        "-o",
        outtmpl,
    ]
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
        "-J",
        f"ytsearch{n}:{q}",
    ]
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
    chosen = max(found, key=lambda item: item.stat().st_size)
    if chosen.stat().st_size > WHISPER_MAX_BYTES:
        logger.warning("Аудио %s больше 25 МБ (%s), пропуск", chosen.name, chosen.stat().st_size)
        return None
    return chosen


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
        "-o",
        outtmpl,
    ]
    if player_client:
        cmd.extend(["--extractor-args", f"youtube:player_client={player_client}"])
    ffmpeg = ffmpeg_path()
    if ffmpeg:
        cmd.extend(["--ffmpeg-location", ffmpeg])
    cmd.append(url)
    return cmd


async def download_letsplay_captions(video_id: str, dest_dir: Path, *, timeout: float = 90.0) -> Path | None:
    """Субтитры через yt-dlp. timedtext без PO-токена часто приходит пустым."""
    video_id = (video_id or "").strip()
    if len(video_id) != 11:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    outtmpl = str(dest_dir / f"{video_id}.%(ext)s")
    for client in ("android_vr", None):
        code, stderr = await _run_ytdlp(_ytdlp_captions_cmd(url, outtmpl, client), timeout)
        path = _pick_caption(dest_dir, video_id)
        if path is not None:
            return path
        logger.warning("yt-dlp субтитры %s для %s (%s): %s", code, video_id, client, (stderr or "")[-300:])
    return None


async def download_letsplay_audio(
    video_id: str,
    dest_dir: Path,
    *,
    duration_sec: int | None = None,
    timeout: float = 180.0,
) -> list[Path]:
    """Качает 1 или 3 окна аудио. Пустой список — бот-стена или ошибка yt-dlp."""
    video_id = (video_id or "").strip()
    if len(video_id) != 11:
        return []
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    windows = audio_windows(duration_sec)
    if not _has_ffmpeg():
        windows = [None]
    files: list[Path] = []
    for index, section in enumerate(windows):
        outtmpl = str(dest_dir / f"{video_id}_{index}.%(ext)s")
        path = None
        blocked = False
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
            if youtube_download_blocked(stderr):
                blocked = True
                logger.warning("YouTube бот-стена для %s (%s)", video_id, client)
                continue
            logger.warning(
                "yt-dlp %s для %s (%s): %s",
                code,
                video_id,
                client,
                (stderr or "")[-400:],
            )
        if path is None:
            if blocked:
                return []
            continue
        files.append(path)
    return files
