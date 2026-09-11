"""Три последовательных прогона через локальный API, затем ожидание очереди."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


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


def wait_finished(prev_id: int) -> dict:
    while True:
        payload, last = latest()
        run_id = int(last["id"]) if last else 0
        status = (last or {}).get("status")
        running = bool(payload.get("pipeline_running"))
        processed = (last or {}).get("games_processed")
        print(
            f"  running={running} last=#{run_id} status={status} processed={processed}",
            flush=True,
        )
        if run_id > prev_id and status and status not in {"running", "enriching"} and not running:
            return last or {}
        time.sleep(8)


def wait_queue_idle() -> dict:
    while True:
        health = get("/api/health")
        queue = int(health.get("queue_similar") or 0) + int(health.get("queue_youtube") or 0)
        print(
            f"  queue similar={health.get('queue_similar')} youtube={health.get('queue_youtube')} "
            f"enrich={health.get('enrichment_running')} pipe={health.get('pipeline_running')}",
            flush=True,
        )
        if (
            queue == 0
            and not health.get("enrichment_running")
            and not health.get("pipeline_running")
        ):
            return health
        time.sleep(15)


def main() -> None:
    prev_id = 0
    for index in range(1, 4):
        print(f"======== RUN {index} ========", flush=True)
        _, last = latest()
        prev_id = int(last["id"]) if last else 0
        while True:
            try:
                accepted = post_run()
                break
            except urllib.error.URLError as exc:
                print(f"  post retry: {exc}", flush=True)
                time.sleep(3)
        print("post", accepted, flush=True)
        if accepted.get("status") == "already_running":
            finished = wait_finished(prev_id)
        else:
            finished = wait_finished(prev_id)
        print(
            "finished",
            finished.get("id"),
            finished.get("status"),
            "found",
            finished.get("games_found"),
            "processed",
            finished.get("games_processed"),
            finished.get("error_message"),
            flush=True,
        )
    print("======== WAIT QUEUE ========", flush=True)
    wait_queue_idle()
    games = get("/api/games")
    print("games", len(games.get("items") or []), flush=True)
    print("======== DONE ========", flush=True)


if __name__ == "__main__":
    main()
