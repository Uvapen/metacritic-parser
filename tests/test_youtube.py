"""Выбор летсплея из выдачи YouTube."""

from app.services.youtube import (
    CaptionCue,
    YoutubeVideo,
    choose_letsplay,
    classify_youtube_video,
    extract_search_videos,
    parse_view_count,
    sample_transcript,
    score_letsplay_candidate,
)


def test_extract_and_prefer_gameplay_over_trailer():
    payload = {
        "contents": {
            "twoColumnSearchResultsRenderer": {
                "primaryContents": {
                    "sectionListRenderer": {
                        "contents": [
                            {
                                "itemSectionRenderer": {
                                    "contents": [
                                        {
                                            "videoRenderer": {
                                                "videoId": "trailer12ab",
                                                "title": {"runs": [{"text": "Valheim Official Trailer"}]},
                                                "lengthText": {"simpleText": "2:10"},
                                            }
                                        },
                                        {
                                            "videoRenderer": {
                                                "videoId": "playthrough",
                                                "title": {"simpleText": "Valheim Gameplay Let's Play Episode 1"},
                                                "lengthText": {"simpleText": "32:18"},
                                                "descriptionSnippet": {
                                                    "runs": [{"text": "Survival in the tenth world."}]
                                                },
                                            }
                                        },
                                        {
                                            "videoRenderer": {
                                                "videoId": "short12abcd",
                                                "title": {"simpleText": "Valheim Shorts"},
                                                "lengthText": {"simpleText": "0:45"},
                                                "badges": [{"metadataBadgeRenderer": {"label": "SHORTS"}}],
                                            }
                                        },
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
    }
    videos = extract_search_videos(payload)
    assert [item.video_id for item in videos] == ["trailer12ab", "playthrough", "short12abcd"]
    hit = choose_letsplay("Valheim", videos)
    assert hit is not None
    assert hit.video_id == "playthrough"
    assert hit.url.endswith("playthrough")
    assert score_letsplay_candidate("Valheim", videos[0]) < 0


def test_unrelated_or_trailer_only_is_skipped():
    videos = [
        YoutubeVideo(video_id="nbaunrelate", title="NBA 2K27 Gameplay", duration_sec=1200),
        YoutubeVideo(video_id="valheimtrai", title="Valheim Launch Trailer", duration_sec=150),
        YoutubeVideo(
            video_id="destinyspir",
            title="Destiny - Gameplay Walkthrough Part 23 - The Garden's Spire!",
            duration_sec=1800,
        ),
        YoutubeVideo(
            video_id="dota2chaosk",
            title="OP Critical Rate Chaos Knight EPIC Pro Dota 2 Gameplay",
            duration_sec=900,
        ),
    ]
    assert choose_letsplay("Valheim", videos) is None
    assert choose_letsplay("Destiny Spire", videos) is None
    assert choose_letsplay("Critical Chaos", videos) is None


def test_parse_view_count():
    assert parse_view_count("1,234,567 views") == 1_234_567
    assert parse_view_count("1.2M views") == 1_200_000
    assert parse_view_count("12 тыс. просмотров") == 12_000
    assert parse_view_count("") is None


def test_choose_prefers_mid_length_over_marathon():
    marathon = YoutubeVideo(
        video_id="ninehoursxx",
        title="Valheim Gameplay Full Game No Commentary",
        duration_sec=9 * 3600,
        views=1_000_000,
    )
    mid = YoutubeVideo(
        video_id="thirtyminxx",
        title="Valheim Gameplay Let's Play Episode 1",
        duration_sec=32 * 60,
        views=8_000,
    )
    hit = choose_letsplay("Valheim", [marathon, mid])
    assert hit is not None
    assert hit.video_id == "thirtyminxx"


def test_choose_marathon_if_only_long_letsplay():
    marathon = YoutubeVideo(
        video_id="ninehoursxx",
        title="Valheim Gameplay Walkthrough FULL GAME",
        duration_sec=9 * 3600,
        views=50_000,
    )
    hit = choose_letsplay("Valheim", [marathon])
    assert hit is not None
    assert hit.video_id == "ninehoursxx"


def test_choose_review_fallback_and_views():
    trailer = YoutubeVideo(
        video_id="trailervalh",
        title="Valheim Official Trailer",
        duration_sec=150,
        views=5_000_000,
    )
    review = YoutubeVideo(
        video_id="reviewvalhe",
        title="Valheim Review",
        duration_sec=18 * 60,
        views=20_000,
    )
    weak_review = YoutubeVideo(
        video_id="tinyreviewx",
        title="Valheim Review",
        duration_sec=18 * 60,
        views=200,
    )
    assert choose_letsplay("Valheim", [trailer]) is None
    hit = choose_letsplay("Valheim", [trailer, review, weak_review])
    assert hit is not None
    assert hit.video_id == "reviewvalhe"
    assert classify_youtube_video("Valheim", review) == "review"


def test_sample_transcript_windows_for_long_video():
    cues = [
        CaptionCue(start=0, text="start-window"),
        CaptionCue(start=60, text="still-start"),
        CaptionCue(start=3600, text="middle-window"),
        CaptionCue(start=1200, text="too-late"),
        CaptionCue(start=7000, text="end-window"),
    ]
    sampled = sample_transcript(cues, duration_sec=7200)
    assert "start-window" in sampled
    assert "middle-window" in sampled
    assert "end-window" in sampled
    assert "too-late" not in sampled
    short = sample_transcript(
        [CaptionCue(start=0, text="hello"), CaptionCue(start=10, text="there")],
        duration_sec=600,
    )
    assert short == "hello there"


def test_parse_caption_xml_and_language_preference():
    from app.services.youtube import parse_caption_xml, pick_caption_track

    xml = """
    <transcript>
      <text start="0">Hello &amp; welcome</text>
      <text start="1">to <br/> Valheim</text>
    </transcript>
    """
    assert parse_caption_xml(xml) == "Hello & welcome to Valheim"
    url = pick_caption_track(
        [
            {"baseUrl": "https://ex.test/en", "languageCode": "en", "kind": "asr"},
            {"baseUrl": "https://ex.test/ru", "languageCode": "ru"},
        ]
    )
    assert url is not None
    assert "ru" in url
    assert "fmt=srv1" in url


def test_parse_vtt_cues_skips_duplicates():
    from app.services.youtube import parse_vtt_cues

    vtt = """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello <c>Valheim</c>

00:00:02.000 --> 00:00:04.000
Hello Valheim

00:00:04.000 --> 00:00:06.000
Let's build a house
"""
    cues = parse_vtt_cues(vtt)
    assert [cue.text for cue in cues] == ["Hello Valheim", "Let's build a house"]
    assert cues[0].start == 0.0
    assert cues[1].start == 4.0


def test_video_id_from_url():
    from app.services.youtube import video_id_from_url

    assert video_id_from_url("https://www.youtube.com/watch?v=abcdefghijk") == "abcdefghijk"
    assert video_id_from_url("https://youtu.be/abcdefghijk") == "abcdefghijk"
    assert video_id_from_url(None) is None


def test_letsplay_job_needed_until_transcript_or_no_captions():
    from types import SimpleNamespace

    from app.services.youtube import (
        LETS_PLAY_STUB,
        LETS_PLAY_STUB_CAPTIONS,
        needs_letsplay_job,
    )

    empty = SimpleNamespace(youtube_summary=None, youtube_summary_source=None)
    clipped = SimpleNamespace(
        youtube_summary="Great gameplay in this video",
        youtube_summary_source="description",
    )
    none = SimpleNamespace(youtube_summary=None, youtube_summary_source="none")
    done = SimpleNamespace(youtube_summary="ok", youtube_summary_source="transcript")
    whisper_done = SimpleNamespace(youtube_summary="ok", youtube_summary_source="whisper")
    old_stub = SimpleNamespace(
        youtube_summary=LETS_PLAY_STUB_CAPTIONS,
        youtube_summary_source="none",
    )
    new_stub = SimpleNamespace(
        youtube_summary=LETS_PLAY_STUB,
        youtube_summary_source="none",
    )
    assert needs_letsplay_job(empty) is True
    assert needs_letsplay_job(clipped) is True
    assert needs_letsplay_job(none) is False
    assert needs_letsplay_job(done) is False
    assert needs_letsplay_job(whisper_done) is False
    assert needs_letsplay_job(old_stub) is True
    assert needs_letsplay_job(new_stub) is False
    assert needs_letsplay_job(new_stub, force=True) is True
    dump = SimpleNamespace(
        youtube_summary="Привет и добро пожаловать обратно на Rage Gaming, и чёрт возьми",
        youtube_summary_source="transcript",
        youtube_transcript_sample=(
            "Привет и добро пожаловать обратно на Rage Gaming, и чёрт возьми, "
            "после столь долгого перерыва Elden Ring"
        ),
    )
    assert needs_letsplay_job(dump) is True


def test_letsplay_stub_clears_video_and_keeps_summary():
    from types import SimpleNamespace

    from app.services.youtube import LETS_PLAY_STUB, apply_letsplay_stub

    game = SimpleNamespace(
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        youtube_title="Valheim Gameplay",
        youtube_channel="ch",
        youtube_views=100,
        youtube_duration_sec=1200,
        youtube_kind="letsplay",
        youtube_transcript_sample="hi",
        youtube_summary="old clip",
        youtube_summary_source="description",
    )
    assert apply_letsplay_stub(game) is True
    assert game.youtube_url is None
    assert game.youtube_title is None
    assert game.youtube_summary == LETS_PLAY_STUB
    assert game.youtube_summary_source == "none"
    assert apply_letsplay_stub(game) is False


def test_conclude_letsplay_does_not_dump_captions():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.services.youtube import conclude_letsplay

    game = SimpleNamespace(
        slug="elden-ring-tarnished-edition",
        title="Elden Ring",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        youtube_kind="letsplay",
        youtube_transcript_sample=None,
        youtube_summary=None,
        youtube_summary_source=None,
        youtube_duration_sec=3600,
    )
    llm = SimpleNamespace(
        complete=AsyncMock(return_value=SimpleNamespace(error="429 TPD", text="")),
    )

    changed = asyncio.run(
        conclude_letsplay(
            game,
            SimpleNamespace(),
            llm,
            sample="Привет и добро пожаловать обратно на Rage Gaming, " * 20,
            source="transcript",
        )
    )
    assert changed == "llm_error"
    assert game.youtube_summary is None
    assert game.youtube_summary_source is None
    assert (game.youtube_transcript_sample or "").startswith("Привет")


def test_video_blurb_is_not_a_letsplay_summary():
    from types import SimpleNamespace

    from app.services.youtube import (
        is_video_blurb_summary,
        letsplay_has_summary,
        needs_letsplay_job,
        transcript_is_listing_copy,
    )

    blurb = (
        "Ролик представляет собой летсплей от создателей Sprawl Zero, где показан "
        "фрагмент пятого уровня. В конце зрителям предлагается скачать бесплатную "
        "демо‑версию и присоединиться к сообществу на Discord."
    )
    assert is_video_blurb_summary(blurb) is True
    game = SimpleNamespace(
        youtube_summary=blurb,
        youtube_summary_source="transcript",
        youtube_transcript_sample="hello welcome back to the channel " * 15,
    )
    assert letsplay_has_summary(game) is False
    assert needs_letsplay_job(game) is True
    spoken = "Alright, let's push into this room. I love these gravity gloves."
    assert transcript_is_listing_copy(spoken, "Download the free demo and join Discord") is False
    listing = "Download the free demo now and join our Discord community for updates"
    assert transcript_is_listing_copy(listing, listing) is True


def test_conclude_letsplay_rejects_video_blurb():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.services.youtube import conclude_letsplay

    game = SimpleNamespace(
        slug="sprawl-zero",
        title="Sprawl Zero",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        youtube_kind="letsplay",
        youtube_transcript_sample=None,
        youtube_summary=None,
        youtube_summary_source=None,
        youtube_duration_sec=900,
    )
    llm = SimpleNamespace(
        complete=AsyncMock(
            return_value=SimpleNamespace(
                error=None,
                text='{"summary": "Ролик представляет собой летсплей. В конце зрителям предлагается Discord."}',
            )
        ),
    )
    status = asyncio.run(
        conclude_letsplay(
            game,
            SimpleNamespace(),
            llm,
            sample="Alright let's go into this level, I am using the gravity gloves " * 10,
            source="transcript",
        )
    )
    assert status == "blurb"
    assert game.youtube_summary is None


def test_collapse_rolling_youtube_asr():
    from app.services.youtube import CaptionCue, collapse_rolling_caption_text, sample_transcript

    cues = [
        CaptionCue(start=0.0, text="С момента создания этого канала"),
        CaptionCue(start=1.0, text="С момента создания этого канала На YouTube почти все игры"),
        CaptionCue(
            start=2.0,
            text="С момента создания этого канала На YouTube почти все игры серии Souls",
        ),
    ]
    text = collapse_rolling_caption_text(cues)
    assert text.count("С момента создания этого канала") == 1
    assert "серии Souls" in text
    sampled = sample_transcript(cues, duration_sec=120)
    assert sampled.count("С момента создания этого канала") == 1


def test_groq_letsplay_summary_is_shown_on_card():
    from types import SimpleNamespace

    from app.services.youtube import letsplay_has_summary

    game = SimpleNamespace(
        youtube_summary=(
            "В ролике автор знакомит зрителей с игрой Crimson Moon, отмечая её "
            "вдохновение серией Souls и рассказывая о ценах на PS5, Xbox и ПК."
        ),
        youtube_summary_source="transcript",
        youtube_transcript_sample="С момента создания этого канала " * 40,
    )
    assert letsplay_has_summary(game) is True


def test_attach_does_not_stub_when_whisper_fails():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from app.services.youtube import LETS_PLAY_STUB, YoutubeVideo, attach_letsplay

    game = SimpleNamespace(
        slug="crimson-moon",
        title="Crimson Moon",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        youtube_title="old",
        youtube_channel="ch",
        youtube_views=1,
        youtube_duration_sec=600,
        youtube_kind="letsplay",
        youtube_transcript_sample="x",
        youtube_summary=LETS_PLAY_STUB,
        youtube_summary_source="none",
    )
    video = YoutubeVideo(
        video_id="abcdefghijk",
        title="Crimson Moon Gameplay Let's Play",
        duration_sec=600,
        views=1000,
    )
    client = SimpleNamespace(
        list_letsplays=AsyncMock(return_value=[video]),
        fetch_transcript=AsyncMock(return_value=None),
    )

    async def _run():
        with patch(
            "app.services.youtube.whisper_letsplay_text",
            new=AsyncMock(return_value=None),
        ):
            return await attach_letsplay(game, client, llm=SimpleNamespace())

    changed = asyncio.run(_run())
    assert changed is True
    assert game.youtube_summary_source is None
    assert game.youtube_summary is None
    assert game.youtube_url is None


def test_long_title_match_without_gameplay_word_is_letsplay():
    from app.services.youtube import YoutubeVideo, classify_youtube_video, choose_letsplay

    video = YoutubeVideo(
        video_id="valheimnight",
        title="Valheim — First Night in the Tenth World",
        duration_sec=22 * 60,
        views=50_000,
    )
    assert classify_youtube_video("Valheim", video) == "letsplay"
    assert choose_letsplay("Valheim", [video]) is video


def test_parse_ytdlp_search_json_reads_entries():
    import json

    from app.services.yt_audio import parse_ytdlp_search_json

    payload = json.dumps(
        {
            "entries": [
                {
                    "id": "abcdefghijk",
                    "title": "Valheim Gameplay",
                    "duration": 1800,
                    "view_count": 12345,
                    "channel": "PlayChan",
                },
                {"id": "short", "title": "skip"},
            ]
        }
    )
    rows = parse_ytdlp_search_json(payload)
    assert len(rows) == 1
    assert rows[0]["video_id"] == "abcdefghijk"
    assert rows[0]["views"] == 12345
    assert rows[0]["channel"] == "PlayChan"




