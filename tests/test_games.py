from types import SimpleNamespace

from app.web.routes import like_needle, status_ru
from app.web import routes as web_routes


def test_like_needle_escapes_wildcards():
    assert like_needle("  Elden Ring  ") == "%elden ring%"
    assert like_needle("100%") == r"%100\%%"
    assert like_needle("") is None
    assert like_needle("   ") is None


def test_status_ru_includes_run_number():
    assert status_ru("enriching", 1) == "очередь обогащения прогона №1"
    assert status_ru("running", 3) == "идёт прогон №3"
    assert status_ru("success") == "успех"


def test_youtube_search_notes_are_not_llm_errors(tmp_path, monkeypatch):
    from app.web.routes import _is_llm_note, llm_row_status

    log_path = tmp_path / "llm_logs.jsonl"
    log_path.write_text(
        '{"ts":"2026-09-12T01:00:00+00:00","model":"youtube","ok":false,'
        '"kind":"youtube","run_id":3,"error":"Поиск YouTube не вернул летсплей",'
        '"game_slug":"nba-2k27"}\n'
        '{"ts":"2026-09-12T01:01:00+00:00","model":"openai/gpt-oss-20b","ok":false,'
        '"kind":"youtube","run_id":3,"error":"429"}\n',
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        llm_model="openai/gpt-oss-120b",
        llm_fallback_models="openai/gpt-oss-20b",
        whisper_model="whisper-large-v3-turbo",
        llm_log_path=log_path,
    )
    monkeypatch.setattr(web_routes, "get_settings", lambda: settings)
    assert _is_llm_note({"model": "youtube"}) is True
    assert llm_row_status({"model": "youtube", "ok": True, "note": True}) == "заметка"
    assert (
        llm_row_status(
            {
                "model": "youtube",
                "ok": False,
                "note": True,
                "error": "Ролик найден, но субтитры и Whisper не дали текст",
            }
        )
        == "ошибка"
    )
    calls, _retries, fails = web_routes._llm_jsonl_stats_for_run(3)
    assert calls == 1
    assert fails == 1
    records, _models, _kinds = web_routes._collect_llm_records()
    youtube_rows = [item for item in records if item.get("model") == "youtube"]
    assert youtube_rows and youtube_rows[0]["ok"] is False
    assert youtube_rows[0]["error"].startswith("Поиск YouTube")


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
