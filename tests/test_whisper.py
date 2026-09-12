"""Groq Whisper URL и окна аудио без сети."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.llm.client import groq_transcriptions_url, whisper_upload_name
from app.services.yt_audio import (
    WHISPER_MAX_BYTES,
    _AUDIO_FORMAT,
    _pick_output,
    _ytdlp_cmd,
    audio_windows,
    youtube_download_blocked,
)
from app.services.youtube import whisper_letsplay_text


def test_groq_transcriptions_url_from_chat_completions():
    assert (
        groq_transcriptions_url("https://api.groq.com/openai/v1/chat/completions")
        == "https://api.groq.com/openai/v1/audio/transcriptions"
    )
    assert groq_transcriptions_url("") == "https://api.groq.com/openai/v1/audio/transcriptions"


def test_whisper_upload_name_maps_opus_not_octet_stream():
    from app.llm.client import _audio_mime

    assert whisper_upload_name("chunk.opus") == "chunk.ogg"
    assert whisper_upload_name("chunk.webm") == "chunk.webm"
    assert whisper_upload_name("chunk.unknown") == "chunk.wav"
    assert _audio_mime("chunk.opus") == "audio/ogg"
    assert _audio_mime("chunk.bin") == "audio/wav"


def test_audio_windows_short_vs_long():
    assert audio_windows(None, compact=True) == [(0.0, 600.0)]
    assert audio_windows(5 * 60, compact=True) == [None]
    assert audio_windows(20 * 60, compact=True) == [None]
    assert audio_windows(20 * 60, compact=False)[0] == (0.0, 600.0)
    assert len(audio_windows(20 * 60, compact=False)) == 3
    windows = audio_windows(2 * 3600, compact=True)
    assert len(windows) == 3
    assert windows[0] == (0.0, 600.0)
    assert windows[-1][1] == 7200.0


def test_ytdlp_cmd_keeps_source_audio():
    cmd = _ytdlp_cmd(
        "https://www.youtube.com/watch?v=abcdefghijk",
        "out.%(ext)s",
        player_client="tv_embedded",
        section=None,
    )
    assert "-x" not in cmd
    assert "--audio-format" not in cmd
    assert _AUDIO_FORMAT in cmd


def test_pick_output_skips_partial_downloads(tmp_path: Path):
    part = tmp_path / "abc_0.mp4.part"
    part.write_bytes(b"x" * 1000)
    assert _pick_output(tmp_path, "abc_0") is None
    ready = tmp_path / "abc_0.mp3"
    ready.write_bytes(b"y" * 500)
    chosen = _pick_output(tmp_path, "abc_0")
    assert chosen == ready
    assert youtube_download_blocked("ERROR: Sign in to confirm you’re not a bot")
    assert youtube_download_blocked("Please use --cookies-from-browser")
    assert not youtube_download_blocked("Downloading audio")


def test_pick_output_prefers_file_under_whisper_limit(tmp_path: Path):
    oversized = tmp_path / "abc_0.mp4"
    compact = tmp_path / "abc_0.m4a"
    oversized.write_bytes(b"x" * (WHISPER_MAX_BYTES + 200))
    compact.write_bytes(b"y" * 8000)
    assert _pick_output(tmp_path, "abc_0") == compact


def test_pick_output_converts_oversized_download(tmp_path: Path):
    from app.services import yt_audio

    oversized = tmp_path / "abc_0.mp4"
    oversized.write_bytes(b"x" * (WHISPER_MAX_BYTES + 200))
    fitted = tmp_path / "abc_0_groq.wav"
    fitted.write_bytes(b"y" * 4000)

    def _fake_fit(src: Path, *, timeout: float = 90.0) -> Path | None:
        assert src == oversized
        return fitted

    with patch.object(yt_audio, "fit_whisper_audio", _fake_fit):
        assert yt_audio._pick_output(tmp_path, "abc_0") == fitted


def test_pick_caption_prefers_russian(tmp_path: Path):
    from app.services.yt_audio import _pick_caption

    (tmp_path / "vid.en.vtt").write_text("en", encoding="utf-8")
    (tmp_path / "vid.ru.vtt").write_text("ru-text", encoding="utf-8")
    chosen = _pick_caption(tmp_path, "vid")
    assert chosen is not None
    assert chosen.name.endswith(".ru.vtt")


def test_whisper_letsplay_joins_chunks(tmp_path: Path):
    first = tmp_path / "vid_0.mp3"
    second = tmp_path / "vid_1.mp3"
    first.write_bytes(b"aaa")
    second.write_bytes(b"bbb")
    llm = SimpleNamespace(
        transcribe=AsyncMock(
            side_effect=[
                SimpleNamespace(text=" intro combat "),
                SimpleNamespace(text="boss fight"),
            ]
        )
    )

    async def _run():
        with (
            patch("app.services.youtube.get_settings") as settings,
            patch(
                "app.services.youtube.download_letsplay_audio",
                new=AsyncMock(return_value=[first, second]),
            ),
        ):
            settings.return_value = SimpleNamespace(
                whisper_enabled=True,
                data_dir=tmp_path,
            )
            return await whisper_letsplay_text(
                "abcdefghijk",
                duration_sec=7200,
                llm=llm,
                slug="valheim",
            )

    text = asyncio.run(_run())
    assert text is not None
    assert "intro combat" in text
    assert "boss fight" in text
    assert llm.transcribe.await_count == 2


def test_whisper_logs_when_audio_missing(tmp_path: Path):
    llm = SimpleNamespace(transcribe=AsyncMock(), note=AsyncMock(return_value=1))

    async def _run():
        with (
            patch("app.services.youtube.get_settings") as settings,
            patch(
                "app.services.youtube.download_letsplay_audio",
                new=AsyncMock(return_value=[]),
            ),
        ):
            settings.return_value = SimpleNamespace(
                whisper_enabled=True,
                whisper_model="whisper-large-v3-turbo",
                data_dir=tmp_path,
            )
            return await whisper_letsplay_text(
                "abcdefghijk",
                duration_sec=600,
                llm=llm,
                slug="valheim",
            )

    assert asyncio.run(_run()) is None
    assert llm.transcribe.await_count == 0
    llm.note.assert_awaited()
    kwargs = llm.note.await_args.kwargs
    assert kwargs["kind"] == "whisper"
    assert "не скачал аудио" in kwargs["error"]
