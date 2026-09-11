"""Один прогон через API, ожидание очереди, сводка по БД."""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
DB = Path(__file__).resolve().parent.parent / "data" / "metacritic.db"


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as response:
        return json.load(response)


def post_run() -> dict:
    request = urllib.request.Request(
        BASE + "/api/pipeline/run",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def latest() -> tuple[dict, dict | None]:
    payload = get("/api/runs")
    items = payload.get("items") or []
    return payload, items[0] if items else None


def wait_health(timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            health = get("/api/health")
            if health.get("status") == "ok":
                print("health ok", flush=True)
                return
        except Exception as exc:
            print(f"  health wait: {exc}", flush=True)
        time.sleep(2)
    raise RuntimeError("сервер не ответил /api/health")


def wait_finished(prev_id: int) -> dict:
    while True:
        payload, last = latest()
        run_id = int(last["id"]) if last else 0
        status = (last or {}).get("status")
        running = bool(payload.get("pipeline_running"))
        print(
            f"  running={running} last=#{run_id} status={status} "
            f"processed={(last or {}).get('games_processed')}",
            flush=True,
        )
        if run_id > prev_id and status and status != "running" and not running:
            return last or {}
        time.sleep(8)


def db_snapshot() -> str:
    if not DB.exists():
        return "db missing"
    con = sqlite3.connect(DB)
    cur = con.cursor()
    games = cur.execute("SELECT count(*) FROM games").fetchone()[0]
    jobs = cur.execute(
        "SELECT kind, status, count(*) FROM pipeline_jobs GROUP BY kind, status"
    ).fetchall()
    yt = cur.execute(
        "SELECT coalesce(youtube_summary_source, 'null'), count(*) "
        "FROM games GROUP BY youtube_summary_source"
    ).fetchall()
    yt_urls = cur.execute(
        "SELECT count(*) FROM games WHERE youtube_url IS NOT NULL AND youtube_url != ''"
    ).fetchone()[0]
    links = cur.execute("SELECT count(*) FROM similar_game_links").fetchone()[0]
    con.close()
    return f"games={games} yt_urls={yt_urls} similar_links={links} yt={yt} jobs={jobs}"


def wait_queue_idle(timeout: int = 5400) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        health = get("/api/health")
        queue = int(health.get("queue_similar") or 0) + int(health.get("queue_youtube") or 0)
        print(
            f"  queue similar={health.get('queue_similar')} youtube={health.get('queue_youtube')} "
            f"enrich={health.get('enrichment_running')} pipe={health.get('pipeline_running')} | "
            f"{db_snapshot()}",
            flush=True,
        )
        if (
            queue == 0
            and not health.get("enrichment_running")
            and not health.get("pipeline_running")
        ):
            return health
        time.sleep(20)
    raise TimeoutError("очередь не опустела за отведённое время")


def print_report() -> None:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    print("======== REPORT ========", flush=True)
    print("games", cur.execute("SELECT count(*) FROM games").fetchone()[0], flush=True)
    print("runs", cur.execute("SELECT id, status, games_found, games_processed, llm_errors FROM run_logs").fetchall(), flush=True)
    print(
        "summaries",
        cur.execute("SELECT kind, count(*) FROM summaries GROUP BY kind").fetchall(),
        flush=True,
    )
    print(
        "critic/user fields",
        cur.execute(
            "SELECT "
            "sum(CASE WHEN summary_fingerprint_critic IS NOT NULL THEN 1 ELSE 0 END), "
            "sum(CASE WHEN summary_fingerprint_user IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM games"
        ).fetchone(),
        flush=True,
    )
    print(
        "youtube sources",
        cur.execute(
            "SELECT coalesce(youtube_summary_source, 'null'), count(*) "
            "FROM games GROUP BY youtube_summary_source"
        ).fetchall(),
        flush=True,
    )
    print(
        "youtube with url",
        cur.execute(
            "SELECT count(*) FROM games WHERE youtube_url IS NOT NULL AND youtube_url != ''"
        ).fetchone()[0],
        flush=True,
    )
    print("similar links", cur.execute("SELECT count(*) FROM similar_game_links").fetchone()[0], flush=True)
    print("similar verdicts", cur.execute("SELECT count(*) FROM similar_verdicts").fetchone()[0], flush=True)
    print(
        "jobs",
        cur.execute("SELECT kind, status, count(*) FROM pipeline_jobs GROUP BY kind, status").fetchall(),
        flush=True,
    )
    print("---- games ----", flush=True)
    for row in cur.execute(
        "SELECT slug, title, metascore, youtube_summary_source, "
        "CASE WHEN youtube_url IS NOT NULL THEN 1 ELSE 0 END AS has_yt, "
        "substr(youtube_title, 1, 80) "
        "FROM games ORDER BY id"
    ):
        print(tuple(row), flush=True)
    print("---- similar pairs (top 20 by score) ----", flush=True)
    for row in cur.execute(
        """
        SELECT a.slug, b.slug, l.score, l.source
        FROM similar_game_links l
        JOIN games a ON a.id = l.game_a_id
        JOIN games b ON b.id = l.game_b_id
        ORDER BY l.score DESC, a.slug
        LIMIT 20
        """
    ):
        print(tuple(row), flush=True)
    print("---- youtube samples ----", flush=True)
    for row in cur.execute(
        "SELECT slug, youtube_summary_source, youtube_views, youtube_duration_sec, "
        "substr(youtube_summary, 1, 160) FROM games "
        "WHERE youtube_summary_source IS NOT NULL ORDER BY id"
    ):
        print(tuple(row), flush=True)
    con.close()


def main() -> None:
    wait_health()
    _, last = latest()
    prev_id = int(last["id"]) if last else 0
    print("======== RUN 1 ========", flush=True)
    while True:
        try:
            accepted = post_run()
            break
        except urllib.error.URLError as exc:
            print(f"  post retry: {exc}", flush=True)
            time.sleep(3)
    print("post", accepted, flush=True)
    finished = wait_finished(prev_id)
    print(
        "finished",
        finished.get("id"),
        finished.get("status"),
        "found",
        finished.get("games_found"),
        "processed",
        finished.get("games_processed"),
        "llm_errors",
        finished.get("llm_errors"),
        finished.get("error_message"),
        flush=True,
    )
    print("======== WAIT QUEUE ========", flush=True)
    wait_queue_idle()
    print_report()
    print("======== DONE ========", flush=True)


if __name__ == "__main__":
    main()
