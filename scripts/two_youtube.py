"""Два летсплея без uvicorn и без очереди APScheduler."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.llm.client import LLMClient
from app.models import Game
from app.services.yt_audio import ffmpeg_path
from app.services.youtube import process_letsplay_slug

SLUGS = ("valheim", "elden-ring-tarnished-edition")


async def reset_youtube(slug: str) -> bool:
    async with SessionLocal() as session:
        game = await session.scalar(select(Game).where(Game.slug == slug))
        if game is None:
            print(f"нет в БД: {slug}", flush=True)
            return False
        game.youtube_url = None
        game.youtube_title = None
        game.youtube_channel = None
        game.youtube_views = None
        game.youtube_duration_sec = None
        game.youtube_kind = None
        game.youtube_transcript_sample = None
        game.youtube_summary = None
        game.youtube_summary_source = None
        await session.commit()
        return True


async def show(slug: str) -> None:
    async with SessionLocal() as session:
        game = await session.scalar(select(Game).where(Game.slug == slug))
        if game is None:
            return
        sample = " ".join((game.youtube_transcript_sample or "").split())
        print(
            {
                "slug": slug,
                "source": game.youtube_summary_source,
                "url": game.youtube_url,
                "title": game.youtube_title,
                "views": game.youtube_views,
                "duration": game.youtube_duration_sec,
                "summary": (game.youtube_summary or "")[:400],
                "transcript_chars": len(sample),
            },
            flush=True,
        )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    print("ffmpeg", ffmpeg_path(), flush=True)
    await init_db()
    llm = LLMClient()
    slugs = tuple(sys.argv[1:]) or SLUGS
    for slug in slugs:
        print(f"======== {slug} ========", flush=True)
        if not await reset_youtube(slug):
            continue
        changed = await process_letsplay_slug(slug, llm=llm)
        print("changed", changed, "llm_errors", llm.final_errors, flush=True)
        await show(slug)


if __name__ == "__main__":
    asyncio.run(main())
