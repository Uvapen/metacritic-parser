from types import SimpleNamespace

from app.web.routes import like_needle
from app.web import routes as web_routes


def test_like_needle_escapes_wildcards():
    assert like_needle("  Elden Ring  ") == "%elden ring%"
    assert like_needle("100%") == r"%100\%%"
    assert like_needle("") is None
    assert like_needle("   ") is None


def test_llm_monitor_filters_include_fallback_model_and_tags(tmp_path, monkeypatch):
    log_path = tmp_path / "llm_logs.jsonl"
    settings = SimpleNamespace(
        llm_model="openai/gpt-oss-120b",
        llm_fallback_models="openai/gpt-oss-20b",
        whisper_model="whisper-large-v3-turbo",
        llm_log_path=log_path,
    )
    monkeypatch.setattr(web_routes, "get_settings", lambda: settings)

    records, models, kinds = web_routes._collect_llm_records()
    assert records == []
    assert models[:2] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert "whisper-large-v3-turbo" in models
    assert kinds[:6] == ["critic", "user", "tags", "similar", "youtube", "whisper"]

    log_path.write_text(
        '{"ts":"2026-09-11T10:00:00+00:00","model":"openai/gpt-oss-20b","ok":true,'
        '"kind":"tags","attempt":1}\n',
        encoding="utf-8",
    )
    records, models, kinds = web_routes._collect_llm_records(kind="tags")
    assert [item["kind"] for item in records] == ["tags"]
    assert "openai/gpt-oss-20b" in models
    assert "tags" in kinds


def test_bind_llm_run_id_uses_game_from_run():
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from app.web.routes import _bind_llm_run_id

    ts = datetime(2026, 9, 11, 14, 0, tzinfo=timezone.utc)
    runs = [
        SimpleNamespace(
            id=5,
            started_at=datetime(2026, 9, 11, 13, 0, tzinfo=timezone.utc),
            details=[{"slug": "valheim"}],
        ),
        SimpleNamespace(
            id=4,
            started_at=datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc),
            details=[{"slug": "sprawl-zero"}],
        ),
    ]
    assert _bind_llm_run_id(None, "sprawl-zero", ts, runs) == 4
    assert _bind_llm_run_id(7, "sprawl-zero", ts, runs) == 7
