"""Готовые примеры летсплея для проверяющего: сняты с домашнего IP, не с Render."""

from __future__ import annotations

from typing import Any

LETS_PLAY_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "slug": "valheim",
        "title": "Valheim",
        "youtube_id": "IAMPr9kytfk",
        "youtube_url": "https://www.youtube.com/watch?v=IAMPr9kytfk",
        "youtube_title": "VALHEIM Gameplay – Viking RPG Survival Game – Part 1 Walkthrough Guide Review!",
        "youtube_channel": "ESO",
        "youtube_duration_sec": 1769,
        "youtube_views": 3_077_642,
        "source_label": "субтитры",
        "youtube_summary_source": "transcript",
        "youtube_summary": (
            "Автор восхищён масштабом и процедурной генерацией мира Valheim: бесконечные биомы, "
            "огромная карта и возможность строить корабли, дома и крепости. Он отмечает, что даже "
            "простые действия, вроде рубки деревьев, требуют осторожности из‑за физики и опасных "
            "существ, а бой с первыми монстрами (например, грейлингом) уже даёт понять, насколько "
            "игра требовательна. Крафт и развитие навыков (древянный топор, молот, каменный щит) "
            "выглядят интересными и открывают путь к более серьёзным сражениям с боссами. В целом "
            "он считает Valheim захватывающим и обещающим проектом, несмотря на ранний доступ."
        ),
    },
    {
        "slug": "elden-ring-tarnished-edition",
        "title": "Elden Ring: Tarnished Edition",
        "youtube_id": "081FSXiNL2s",
        "youtube_url": "https://www.youtube.com/watch?v=081FSXiNL2s",
        "youtube_title": "How Good Are The New Weapons? Elden Ring Tarnished Edition",
        "youtube_channel": None,
        "youtube_duration_sec": 737,
        "youtube_views": None,
        "source_label": "Whisper",
        "youtube_summary_source": "whisper",
        "youtube_summary": (
            "Автор отмечает, что в DLC добавлены семь новых оружий, но только два — Leontiel's "
            "Greatsword и Golden Order Flail — действительно уникальны и достойны отдельного видео. "
            "Остальные пять инфузируемых оружий доступны уже в начале игры, но они уступают "
            "DLC‑версии по урону, поэтому лучше использовать их только до перехода к более сильным "
            "вариантам. Curved Greatswords и Light Greatswords имеют свои плюсы и минусы, выбор "
            "зависит от стиля и билда. В целом он скептичен к базовым версиям, но признаёт их "
            "практичность на раннем этапе."
        ),
    },
)


def demo_for_slug(slug: str | None) -> dict[str, Any] | None:
    key = (slug or "").strip()
    for item in LETS_PLAY_EXAMPLES:
        if item["slug"] == key:
            return item
    return None
