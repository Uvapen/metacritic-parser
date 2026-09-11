"""Таймзона UTC+4, fingerprint deep-diff и hourly LLM-backfill."""

from datetime import datetime, timezone
from types import SimpleNamespace

from app.scraper.lister import ListedGame
from app.services.pipeline import (
    _deep_change_items,
    _fingerprint_llm_input,
    _needs_llm_backfill,
    _scores_changed,
)
from app.timeutil import APP_TZ, as_local_date, to_local, today_local


def _listed(**kwargs) -> ListedGame:
    return ListedGame(
        title=kwargs.get("title", "Game"),
        slug=kwargs.get("slug", "game"),
        url=kwargs.get("url", "https://example.test/game"),
        release_date=kwargs.get("release_date"),
        metascore=kwargs.get("metascore"),
        userscore=kwargs.get("userscore"),
    )


def test_today_local_is_utc_plus_four():
    utc_now = datetime.now(timezone.utc)
    local = today_local()
    assert local == utc_now.astimezone(APP_TZ).date()


def test_naive_datetime_treated_as_utc():
    naive = datetime(2026, 9, 10, 22, 30, 0)
    local = to_local(naive)
    assert local is not None
    assert local.hour == 2
    assert local.day == 11
    assert as_local_date(naive) == local.date()


def test_scores_changed_when_tbd_gets_score():
    game = SimpleNamespace(metascore=None)
    listed = _listed(metascore=81)
    assert _scores_changed(game, listed) is True
    listed_none = _listed(metascore=None)
    assert _scores_changed(SimpleNamespace(metascore=80), listed_none) is False
    assert _scores_changed(SimpleNamespace(metascore=80), _listed(metascore=80)) is False
    assert _scores_changed(SimpleNamespace(metascore=80), _listed(metascore=81)) is True


def test_needs_llm_backfill_without_fingerprints():
    empty = SimpleNamespace(summary_fingerprint_critic=None, summary_fingerprint_user=None)
    done = SimpleNamespace(summary_fingerprint_critic="abc", summary_fingerprint_user=None)
    assert _needs_llm_backfill(empty) is True
    assert _needs_llm_backfill(done) is False
    assert _needs_llm_backfill(None) is False


def test_worth_processing_skips_unchanged_existing():
    from datetime import date

    from app.services.pipeline import _worth_processing

    today = date(2026, 9, 11)
    known = SimpleNamespace(
        last_processed_at=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
        metascore=80,
        summary_fingerprint_critic="abc",
        summary_fingerprint_user=None,
    )
    listed = _listed(metascore=80)
    assert _worth_processing(None, listed, today) is True
    assert _worth_processing(known, listed, today) is False
    assert _worth_processing(known, _listed(metascore=91), today) is True
    fresh = SimpleNamespace(
        last_processed_at=datetime(2026, 9, 11, 8, 0, tzinfo=timezone.utc),
        metascore=80,
        summary_fingerprint_critic=None,
        summary_fingerprint_user=None,
    )
    assert _worth_processing(fresh, _listed(metascore=91), today) is False


def test_deep_diff_same_reviews_not_flagged():
    reviews = [SimpleNamespace(quote="great combat", score=80)]
    desc = "samurai action"
    fp = _fingerprint_llm_input(reviews, desc)
    before = {
        "description": desc,
        "genres": [],
        "developers": [],
        "platforms": [],
        "cover_url": None,
        "metascore": 80,
        "userscore": 8.0,
        "fp_c": fp,
        "fp_u": _fingerprint_llm_input([], desc),
    }
    game = SimpleNamespace(
        description=desc,
        genres=[],
        developers=[],
        platform_scores=[],
        cover_url=None,
        metascore=80,
        userscore=8.0,
    )
    card = SimpleNamespace(critic_reviews=reviews, user_reviews=[], platforms=[])
    assert _deep_change_items(before, game, card) == []


def test_llm_hard_block_does_not_call_or_log():
    import asyncio

    from app.llm.client import LLMClient

    client = LLMClient()
    client._hard_block = ("Доступ запрещён (403)", 403)

    async def _run():
        return await client.complete("ping", kind="critic")

    result = asyncio.run(_run())
    assert result.error_status == 403
    assert result.log_id is None
    assert result.text == ""


def test_daily_token_quota_is_not_retried():
    from app.llm.client import is_daily_quota, is_daily_token_quota, parse_try_again_seconds

    assert is_daily_token_quota(
        "Rate limit reached for model x on tokens per day (TPD): Limit 200000"
    )
    assert is_daily_quota("Rate limit reached on requests per day (RPD)")
    assert not is_daily_token_quota("Rate limit reached on tokens per minute (TPM)")
    assert not is_daily_token_quota(None)
    wait = parse_try_again_seconds("Please try again in 49m27.8s. Need more tokens?")
    assert wait is not None
    assert abs(wait - (49 * 60 + 27.8)) < 0.2


def test_chat_fallback_chain_and_quota_pause():
    from types import SimpleNamespace

    from app.llm.client import chat_model_chain, groq_quota

    groq_quota.reset()
    settings = SimpleNamespace(
        llm_model="openai/gpt-oss-120b",
        llm_fallback_models="openai/gpt-oss-20b, openai/gpt-oss-120b",
    )
    assert chat_model_chain(settings) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert groq_quota.current_chat_model(settings).endswith("120b")
    nxt = groq_quota.note_chat_daily(
        "tokens per day (TPD). Please try again in 40m0s",
        None,
        settings,
    )
    assert nxt == "openai/gpt-oss-20b"
    assert groq_quota.current_chat_model(settings).endswith("20b")
    assert groq_quota.note_chat_daily("tokens per day (TPD) try again in 40m0s", None, settings) is None
    assert groq_quota.chat_retry_in() > 30
    groq_quota.reset()
    assert groq_quota.chat_ready() is True


def test_tick_skips_when_groq_paused():
    import asyncio
    import time
    from unittest.mock import AsyncMock, patch

    from app.llm.client import groq_quota
    from app.services.pipeline import tick_pipeline_stages

    groq_quota.reset()
    groq_quota.chat_paused_until = time.monotonic() + 120
    with patch("app.services.pipeline.drain_one_job", new=AsyncMock()) as drain:
        asyncio.run(tick_pipeline_stages())
        drain.assert_not_called()
    groq_quota.reset()


def test_deep_diff_flags_reviews_when_quotes_change():
    old = [SimpleNamespace(quote="old take", score=80)]
    new = [SimpleNamespace(quote="new take", score=80)]
    desc = "same"
    before = {
        "description": desc,
        "genres": [],
        "developers": [],
        "platforms": [],
        "cover_url": None,
        "metascore": 80,
        "userscore": 8.0,
        "fp_c": _fingerprint_llm_input(old, desc),
        "fp_u": None,
    }
    game = SimpleNamespace(
        description=desc,
        genres=[],
        developers=[],
        platform_scores=[],
        cover_url=None,
        metascore=80,
        userscore=8.0,
    )
    card = SimpleNamespace(critic_reviews=new, user_reviews=[], platforms=[])
    items = _deep_change_items(before, game, card)
    assert items == [{"field": "reviews", "kind": "updated"}]


def test_job_slugs_from_details_skips_errors_and_dupes():
    from app.services.pipeline import _job_slugs_from_details

    assert _job_slugs_from_details(
        [
            {"slug": "a", "action": "new"},
            {"slug": "b", "action": "error"},
            {"slug": "a", "action": "updated"},
            {"slug": "c", "action": "skipped"},
            {"slug": "", "action": "new"},
        ]
    ) == ["a", "c"]


def test_status_from_card_details():
    from app.services.pipeline import _status_from_card_details

    assert _status_from_card_details([{"action": "new"}], 1) == "success"
    assert _status_from_card_details([{"action": "error"}], 0) == "error"
    assert _status_from_card_details(
        [{"action": "new"}, {"action": "error"}], 1
    ) == "partial"
    assert _status_from_card_details(None, 0) == "success"


def test_choose_hourly_due_snaps_far_and_overdue():
    from datetime import datetime, timedelta, timezone

    from app.services.pipeline import choose_hourly_due

    now = datetime(2026, 9, 12, 2, 0, tzinfo=timezone.utc)
    hour = timedelta(hours=1)
    assert choose_hourly_due(None, now=now, interval=hour) == now + hour
    assert choose_hourly_due(now - timedelta(minutes=5), now=now, interval=hour) == now + timedelta(
        seconds=30
    )
    far = now + timedelta(hours=23)
    assert choose_hourly_due(far, now=now, interval=hour) == now + hour
    soon = now + timedelta(minutes=40)
    assert choose_hourly_due(soon, now=now, interval=hour) == soon
