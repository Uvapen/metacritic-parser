"""Детерминированный score похожих игр и разбор LLM-тегов."""

from types import SimpleNamespace

from app.llm.prompts import parse_ai_tags, parse_similar_review, parse_summary_payload
from app.services.similar import (
    apply_mixed_filter,
    is_confident_hit,
    pair_fingerprint,
    rank_similar,
    similarity_score,
)


def _plat(name: str) -> SimpleNamespace:
    return SimpleNamespace(platform=name)


def _game(**kwargs) -> SimpleNamespace:
    platforms = kwargs.pop("platforms", [])
    return SimpleNamespace(
        id=kwargs.get("id", 1),
        slug=kwargs.get("slug", "game"),
        title=kwargs.get("title", "Game"),
        cover_url=kwargs.get("cover_url"),
        developer=kwargs.get("developer"),
        developers=kwargs.get("developers"),
        genres=kwargs.get("genres"),
        genre_detailed=kwargs.get("genre_detailed"),
        tags=kwargs.get("tags", kwargs.get("ai_tags")),
        ai_tags=kwargs.get("ai_tags", kwargs.get("tags")),
        key_features=kwargs.get("key_features"),
        target_audience=kwargs.get("target_audience"),
        metascore=kwargs.get("metascore"),
        is_from_carousel=kwargs.get("is_from_carousel", False),
        platform_scores=[_plat(name) for name in platforms],
    )


def test_developer_genre_detailed_and_tags():
    origin = _game(
        id=1,
        slug="elden-ring",
        title="Elden Ring",
        developers=["From Software"],
        genres=["Action RPG"],
        platforms=["PlayStation 5", "PC"],
        metascore=90,
        tags=["souls-like", "dark-fantasy"],
        genre_detailed="action-rpg",
    )
    close = _game(
        id=2,
        slug="sekiro",
        title="Sekiro",
        developer="FromSoftware",
        genres=["Action"],
        platforms=["PlayStation 5"],
        metascore=88,
        tags=["souls-like"],
        genre_detailed="action-rpg",
    )
    unrelated = _game(
        id=3,
        slug="bus-sim",
        title="Bus Simulator",
        developers=["Stillalive Studios"],
        genres=["Simulation"],
        platforms=["Xbox Series X"],
        metascore=20,
    )
    score, reasons = similarity_score(origin, close)
    # студия 3 + жанр 0.5 + souls-like 2 + общая платформа 0.5
    assert score == 3 + 0.5 + 2 + 0.5
    assert "разработчик" in reasons
    assert "жанр" in reasons
    assert "платформа" in reasons
    assert "близкий рейтинг" not in reasons
    hits = rank_similar(origin, [close, unrelated, origin])
    assert [item["slug"] for item in hits] == ["sekiro"]
    assert hits[0]["score"] == score
    assert hits[0]["chips"] == ["FromSoftware", "souls-like", "action-rpg", "PlayStation 5"]


def test_platform_and_metascore_alone_do_not_match():
    origin = _game(id=1, slug="a", metascore=80, platforms=["PC", "PS5"])
    same_platform = _game(id=2, slug="b", title="B", metascore=70, platforms=["PC"])
    same_rating = _game(id=3, slug="c", title="C", metascore=75, platforms=["Xbox"])
    score, reasons = similarity_score(origin, same_platform)
    assert score == 0.5
    assert reasons == ["платформа"]
    assert similarity_score(origin, same_rating) == (0, [])
    assert rank_similar(origin, [origin, same_platform, same_rating]) == []


def test_generic_management_tag_is_ignored():
    origin = _game(id=1, slug="brig", tags=["tactical-rpg", "management"])
    other = _game(id=2, slug="bus", title="Bus", tags=["management", "vehicle-sim"])
    score, _reasons = similarity_score(origin, other)
    assert score == 0


def test_valheim_does_not_match_horror_or_vampire_sandbox():
    origin = _game(
        id=1,
        slug="valheim",
        tags=["survival", "sandbox", "crafting", "open-world", "co-op", "base-building"],
        genre_detailed="survival-sandbox",
        metascore=89,
    )
    halloween = _game(
        id=2,
        slug="halloween-the-game",
        title="Halloween",
        tags=["asymmetrical-horror", "multiplayer", "stealth", "survival", "horror"],
        genre_detailed="survival-horror",
        metascore=71,
    )
    dawnwalker = _game(
        id=3,
        slug="the-blood-of-dawnwalker",
        title="Dawnwalker",
        tags=["action-rpg", "dark-fantasy", "sandbox", "decision-system", "combat-parry"],
        genre_detailed="action-rpg",
        metascore=83,
    )
    assert similarity_score(origin, halloween) == (0, [])
    assert similarity_score(origin, dawnwalker) == (0, [])
    assert rank_similar(origin, [halloween, dawnwalker]) == []


def test_action_adventure_and_platform_do_not_make_shady_job():
    origin = _game(
        id=1,
        slug="onimusha",
        tags=["swordplay", "cinematic-combat", "dark-fantasy", "action-adventure", "story-rich"],
        genre_detailed="action-adventure",
        platforms=["PlayStation 5", "PC"],
        target_audience="фанаты серии Onimusha и любители динамичных боевых игр",
        metascore=81,
    )
    shady = _game(
        id=2,
        slug="shady-job",
        title="Shady job",
        tags=["co-op", "construction", "vertical-climbing", "action-adventure"],
        genre_detailed="action-adventure",
        platforms=["PC"],
        target_audience="любители кооперативных игр с необычными механиками",
        metascore=None,
    )
    score, reasons = similarity_score(origin, shady)
    assert score < 3
    assert "action-adventure" not in (reasons or [])
    assert rank_similar(origin, [shady]) == []


def test_audience_adds_two_points_when_close():
    origin = _game(
        id=1,
        slug="a",
        tags=["souls-like"],
        target_audience="любители сложных боёв",
    )
    other = _game(
        id=2,
        slug="b",
        title="B",
        tags=["souls-like"],
        target_audience="для фанатов сложных сражений",
    )
    score, reasons = similarity_score(origin, other)
    assert score == 2 + 2
    assert "аудитория" in reasons


def test_action_adventure_does_not_match_solitaire():
    origin = _game(
        id=1,
        slug="resonance-a-plague-tale-legacy",
        tags=["action-adventure", "exploration", "puzzle", "melee-combat", "story-rich"],
        genre_detailed="action-adventure",
        metascore=78,
    )
    solitaire = _game(
        id=2,
        slug="solitaire-adventure",
        title="Solitaire Adventure",
        tags=["solitaire", "puzzle", "casual", "board-game"],
        genre_detailed="puzzle-board",
        metascore=None,
    )
    assert similarity_score(origin, solitaire) == (0, [])
    assert rank_similar(origin, [solitaire]) == []


def test_mixed_hits_use_heuristic_until_llm_rejects():
    origin = _game(
        id=1,
        slug="a",
        tags=["melee-combat", "stealth-infiltration"],
        genre_detailed="stealth-action",
        platforms=["PC"],
        metascore=80,
    )
    other = _game(
        id=2,
        slug="b",
        title="B",
        tags=["melee-combat", "stealth-infiltration"],
        genre_detailed="stealth-action",
        platforms=["PC"],
        metascore=70,
    )
    hits = rank_similar(origin, [other])
    assert hits and hits[0]["score"] == 1 + 1 + 0.5 + 0.5
    assert is_confident_hit(hits[0]) is False
    shown = apply_mixed_filter(hits, {}, catalog_size=10)
    assert [item["slug"] for item in shown] == ["b"]
    assert shown[0]["via"] == "heuristic"
    assert [item["slug"] for item in apply_mixed_filter(hits, {"b": True})] == ["b"]
    assert apply_mixed_filter(hits, {"b": False}) == []


def test_mixed_heuristic_stricter_in_large_catalog():
    origin = _game(
        id=1,
        slug="a",
        tags=["melee-combat", "stealth-infiltration"],
        genre_detailed="stealth-action",
        platforms=["PC"],
    )
    other = _game(
        id=2,
        slug="b",
        title="B",
        tags=["melee-combat", "stealth-infiltration"],
        genre_detailed="stealth-action",
        platforms=["PC"],
    )
    hits = rank_similar(origin, [other])
    assert hits and hits[0]["score"] == 3
    assert apply_mixed_filter(hits, {}, catalog_size=10)
    assert apply_mixed_filter(hits, {}, catalog_size=80) == []


def test_one_tag_is_below_threshold():
    origin = _game(id=1, slug="a", tags=["melee-combat"], metascore=80)
    other = _game(id=2, slug="b", title="B", tags=["melee-combat"], metascore=70)
    assert similarity_score(origin, other) == (1, ["теги"])
    assert rank_similar(origin, [other]) == []


def test_souls_like_with_genre_and_platform_is_confident():
    origin = _game(
        id=1,
        slug="a",
        tags=["souls-like"],
        genre_detailed="action-rpg",
        platforms=["PC"],
        metascore=90,
    )
    other = _game(
        id=2,
        slug="wo-long",
        title="Wo Long",
        tags=["souls-like"],
        genre_detailed="action-rpg",
        platforms=["PC"],
        metascore=50,
    )
    hits = rank_similar(origin, [other])
    assert hits[0]["score"] == 2 + 0.5 + 0.5
    assert is_confident_hit(hits[0]) is True
    assert [item["slug"] for item in apply_mixed_filter(hits, {})] == ["wo-long"]
    assert hits[0]["chips"] == ["souls-like", "action-rpg", "PC"]


def test_pair_fingerprint_is_symmetric():
    left = _game(id=1, slug="onimusha", tags=["souls-like"], genre_detailed="action-rpg")
    right = _game(id=2, slug="wo-long", tags=["souls-like"], genre_detailed="action-rpg")
    other = _game(id=3, slug="valheim", tags=["base-building"], genre_detailed="survival-sandbox")
    assert pair_fingerprint(left, right) == pair_fingerprint(right, left)
    assert pair_fingerprint(left, right) != pair_fingerprint(left, other)


def test_parse_similar_review_keep_list():
    assert parse_similar_review('{"keep": ["wo-long", "sekiro"]}') == {"wo-long", "sekiro"}
    assert parse_similar_review("не json") == set()


def test_generic_open_world_tag_is_ignored():
    origin = _game(id=1, slug="a", tags=["open-world", "multiplayer"], metascore=90)
    other = _game(id=2, slug="b", title="B", tags=["open-world", "co-op"], metascore=88)
    assert similarity_score(origin, other) == (0, [])


def test_generic_action_genre_is_ignored():
    origin = _game(id=1, slug="a", genres=["Action"], metascore=90, platforms=["PS5"])
    other = _game(id=2, slug="nba", title="NBA", genres=["Action"], metascore=88, platforms=["PS5"])
    score, reasons = similarity_score(origin, other)
    assert score == 0.5
    assert reasons == ["платформа"]
    assert rank_similar(origin, [other]) == []


def test_metacritic_genre_alone_does_not_match():
    origin = _game(
        id=1,
        slug="brigandine-abyss",
        genres=["Turn-Based Tactics", "Strategy"],
        genre_detailed="tactical-rpg",
        tags=["tactical-rpg"],
        metascore=78,
    )
    tbd_new = _game(
        id=2,
        slug="infinity-hexadome",
        title="Infinity HexaDome Tactics",
        genres=["Turn-Based Tactics"],
        metascore=None,
    )
    bus = _game(
        id=3,
        slug="bus-sim",
        title="Bus Simulator",
        genres=["Simulation"],
        tags=["management"],
        metascore=61,
    )
    assert similarity_score(origin, tbd_new) == (0, [])
    assert similarity_score(origin, bus) == (0, [])
    assert rank_similar(origin, [tbd_new, bus]) == []


def test_brigandine_not_similar_to_generic_tactics():
    origin = _game(
        id=1,
        slug="brigandine-abyss",
        tags=["turn-based", "tactical-rpg", "grand-strategy", "character-collection", "voice-acting"],
        genre_detailed="tactical-rpg",
        platforms=["PC"],
        target_audience="фанаты сложных тактических RPG и стратегий",
        metascore=69,
    )
    other = _game(
        id=2,
        slug="asteroid-destroyer-ermu-cz",
        title="Asteroid Destroyer (ermu_cz)",
        tags=["turn-based", "tactical", "top-down", "squad-control", "rpg-elements", "mobile"],
        genre_detailed="tactical-rpg",
        platforms=["Mobile"],
        target_audience="любители тактических стратегий на мобильных устройствах",
        metascore=None,
    )
    same_platform = _game(
        id=3,
        slug="asteroid-pc",
        title="Asteroid Destroyer PC",
        tags=["turn-based", "tactical", "top-down", "squad-control", "rpg-elements", "mobile"],
        genre_detailed="tactical-rpg",
        platforms=["PC"],
        target_audience="любители тактических стратегий на мобильных устройствах",
        metascore=50,
    )
    score, reasons = similarity_score(origin, other)
    assert score == 1 + 0.5
    assert "аудитория" not in reasons
    assert "жанр" in reasons
    assert rank_similar(origin, [other, same_platform]) == []
    assert similarity_score(origin, same_platform)[0] == 1 + 0.5 + 0.5



def test_tbd_or_empty_profile_skipped_in_similar():
    origin = _game(
        id=1,
        slug="a",
        tags=["souls-like"],
        genre_detailed="action-rpg",
        platforms=["PC"],
        metascore=90,
    )
    tbd_unknown = _game(
        id=2,
        slug="hexadome",
        title="Infinity HexaDome Tactics",
        genres=["Turn-Based Tactics"],
        metascore=None,
    )
    empty_scored = _game(id=3, slug="empty", title="Empty", genres=["Action RPG"], metascore=80)
    known = _game(
        id=4,
        slug="known",
        title="Known",
        tags=["souls-like"],
        genre_detailed="action-rpg",
        platforms=["PC"],
        metascore=70,
    )
    tbd_known = _game(
        id=5,
        slug="tbd-known",
        title="TBD Known",
        tags=["souls-like"],
        genre_detailed="action-rpg",
        platforms=["PC"],
        metascore=None,
    )
    hits = rank_similar(origin, [tbd_unknown, empty_scored, known, tbd_known])
    assert [item["slug"] for item in hits] == ["known", "tbd-known"]


def test_content_beats_shared_platform():
    origin = _game(
        id=1,
        slug="a",
        genres=["Action RPG"],
        platforms=["PS5"],
        metascore=90,
        tags=["souls-like"],
        genre_detailed="action-rpg",
    )
    sports = _game(
        id=2,
        slug="nba",
        title="NBA",
        genres=["Sports"],
        platforms=["PS5"],
        metascore=90,
    )
    action = _game(
        id=3,
        slug="wo-long",
        title="Wo Long",
        genres=["Action"],
        platforms=["PS5"],
        metascore=50,
        tags=["souls-like"],
        genre_detailed="action-rpg",
    )
    hits = rank_similar(origin, [sports, action])
    assert [item["slug"] for item in hits] == ["wo-long"]
    assert hits[0]["score"] == 2 + 0.5 + 0.5
    assert "souls-like" in hits[0]["chips"]
    assert hits[0]["chips"][-1] == "PS5"


def test_features_alone_do_not_match():
    origin = _game(
        id=1,
        slug="a",
        key_features=["открытый мир", "сложные боссы"],
        target_audience="любители сложных игр",
        metascore=80,
    )
    close = _game(
        id=2,
        slug="b",
        title="B",
        key_features=["открытый мир"],
        target_audience="любители сложных игр",
        metascore=78,
    )
    score, reasons = similarity_score(origin, close)
    assert score == 0
    assert reasons == []
    assert rank_similar(origin, [origin, close]) == []


def test_audience_plus_platform_is_not_enough():
    origin = _game(
        id=1,
        slug="nba",
        tags=["basketball-sim"],
        platforms=["PC"],
        target_audience="любители баскетбольных симуляторов",
    )
    mall = _game(
        id=2,
        slug="idle-mall",
        title="Idle Mall",
        tags=["idle-economy"],
        platforms=["PC"],
        target_audience="любители экономических симуляторов",
    )
    score, reasons = similarity_score(origin, mall)
    assert score == 0.5
    assert reasons == ["платформа"]
    assert rank_similar(origin, [mall]) == []


def test_low_score_souls_like_still_matches():
    origin = _game(
        id=1,
        slug="elden",
        tags=["souls-like"],
        genre_detailed="action-rpg",
        platforms=["PC"],
        metascore=94,
    )
    other = _game(
        id=2,
        slug="jank",
        title="Jank Souls",
        tags=["souls-like"],
        genre_detailed="action-rpg",
        platforms=["PC"],
        metascore=40,
    )
    score, reasons = similarity_score(origin, other)
    assert score == 2 + 0.5 + 0.5
    assert "близкий рейтинг" not in reasons


def test_parse_ai_tags_ignores_scores():
    text = """```json
    {"summary": "Хвалят бой.", "tags": ["Action RPG", "open world", {"score": 99}, "souls-like"]}
    ```"""
    extracted = parse_summary_payload(text)
    assert extracted.parsed is True
    assert "Хвалят" in extracted.summary
    assert extracted.tags == ["action-rpg", "open-world", "souls-like"]
    assert parse_ai_tags("нет json") == []
    assert parse_ai_tags('{"score": 10}') == []
    broken = parse_summary_payload("это не json, просто текст")
    assert broken.parsed is False
    assert broken.tags == []
    assert "просто текст" in broken.summary


def test_parse_full_analyst_json():
    text = """{
      "summary": "Хвалят боёвку и мир, ругают оптимизацию.",
      "tags": ["souls-like", "open-world", "dark-fantasy", "boss-rush"],
      "genre_detailed": "action-rpg",
      "key_features": ["открытый мир", "сложные боссы"],
      "target_audience": "любители сложных игр"
    }"""
    extracted = parse_summary_payload(text)
    assert extracted.parsed is True
    assert extracted.genre_detailed == "action-rpg"
    assert extracted.key_features == ["открытый мир", "сложные боссы"]
    assert extracted.target_audience == "любители сложных игр"
    assert "souls-like" in extracted.tags


def test_parse_summary_only_json_and_clip():
    import json as json_lib

    from app.llm.prompts import clip_summary_text, looks_russian

    extracted = parse_summary_payload('{"summary": "Игрокам нравится боёвка."}')
    assert extracted.parsed is True
    assert extracted.summary == "Игрокам нравится боёвка."
    assert extracted.tags == []
    assert looks_russian(extracted.summary) is True
    assert looks_russian("Critics praise the combat.") is False
    long = "Да. " * 200
    clipped = parse_summary_payload(json_lib.dumps({"summary": long}, ensure_ascii=False))
    assert len(clipped.summary) <= 600
    assert clip_summary_text("коротко") == "коротко"


def test_structured_extraction_applied_once():
    from app.llm.prompts import SummaryExtraction
    from app.services.pipeline import _apply_llm_extraction, _clear_structured_fields

    game = SimpleNamespace(
        tags=None,
        ai_tags=None,
        ai_tags_fingerprint=None,
        genre_detailed=None,
        key_features=None,
        target_audience=None,
    )
    first = SummaryExtraction(
        summary="Хвалят бой.",
        tags=["souls-like"],
        genre_detailed="action-rpg",
        key_features=["сложные боссы"],
        target_audience="любители сложных игр",
        parsed=True,
    )
    second = SummaryExtraction(
        summary="Игрокам скучно.",
        tags=["open-world", "multiplayer"],
        genre_detailed="fps",
        key_features=["кооператив"],
        target_audience="казуальные игроки",
        parsed=True,
    )
    assert _apply_llm_extraction(game, first) is True
    assert game.tags == ["souls-like"]
    assert game.genre_detailed == "action-rpg"
    assert _apply_llm_extraction(game, second) is False
    assert game.tags == ["souls-like"]
    assert game.genre_detailed == "action-rpg"
    assert game.target_audience == "любители сложных игр"
    _clear_structured_fields(game)
    assert game.ai_tags_fingerprint is None
    assert _apply_llm_extraction(game, second) is True
    assert "open-world" in game.tags


def test_summary_only_prompt_skips_tags():
    from app.llm.prompts import build_user_summary_prompt

    text = build_user_summary_prompt(
        title="Sekiro",
        description="Souls-like action",
        userscore=8.5,
        user_snippets=["Great combat"],
        include_features=False,
    )
    assert "только с полем summary" in text
    assert "Не добавляй tags" in text
