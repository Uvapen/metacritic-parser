"""Три прогона пайплайна для проверки дневного курсора."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

from app.services.pipeline import run_hourly_pipeline  # noqa: E402

DB = Path("data/metacritic.db")


def slugs() -> list[str]:
    if not DB.exists():
        return []
    con = sqlite3.connect(DB)
    rows = [row[0] for row in con.execute("SELECT slug FROM games ORDER BY id")]
    con.close()
    return rows


async def main() -> None:
    for index in range(1, 4):
        print(f"======== RUN {index} ========", flush=True)
        before = set(slugs())
        run = await run_hourly_pipeline()
        after = slugs()
        print(
            f"status={run.status} found={run.games_found} processed={run.games_processed} error={run.error_message}",
            flush=True,
        )
        print(f"db_count={len(after)} new_count={len(set(after) - before)}", flush=True)
        print("new", sorted(set(after) - before), flush=True)
    print("======== DONE ========", flush=True)
    print("all", slugs(), flush=True)
    con = sqlite3.connect(DB)
    print("offset", list(con.execute("SELECT day, offset FROM pipeline_state")), flush=True)
    oni = con.execute(
        "SELECT slug, metascore, userscore FROM games WHERE slug = ?",
        ("onimusha-way-of-the-sword",),
    ).fetchone()
    print("onimusha game", oni, flush=True)
    if oni:
        print(
            "platforms",
            list(
                con.execute(
                    """
                    SELECT platform, metascore, userscore
                    FROM platform_scores
                    WHERE game_id = (SELECT id FROM games WHERE slug = ?)
                    """,
                    ("onimusha-way-of-the-sword",),
                )
            ),
            flush=True,
        )
    con.close()


if __name__ == "__main__":
    asyncio.run(main())
