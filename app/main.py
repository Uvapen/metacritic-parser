"""Точка входа FastAPI: приложение, БД и APScheduler."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from app.config import get_settings
from app.db import init_db
from app.services.pipeline import (
    recover_enrichment_on_startup,
    run_hourly_pipeline,
    tick_pipeline_stages,
)
from app.web.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Поднимает схему БД и hourly-планировщик."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    await init_db()
    if settings.scheduler_enabled:
        scheduler.configure(event_loop=asyncio.get_running_loop())
        scheduler.add_job(
            run_hourly_pipeline,
            "interval",
            hours=settings.pipeline_interval_hours,
            id="hourly_pipeline",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        tick_seconds = max(5, int(float(settings.enrichment_tick_seconds) or 20))
        scheduler.add_job(
            tick_pipeline_stages,
            "interval",
            seconds=tick_seconds,
            id="stage_tick",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=10),
        )
        scheduler.start()
        await recover_enrichment_on_startup()
        logger.info(
            "Планировщик запущен: интервал %s ч, очередь similar/youtube каждые %s с",
            settings.pipeline_interval_hours,
            tick_seconds,
        )
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)


def _job_run_at_iso(job_id: str) -> str | None:
    job = scheduler.get_job(job_id)
    return _dt_iso(job.next_run_time if job is not None else None)


def _dt_iso(value) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def get_next_pipeline_run_at() -> str | None:
    """ISO UTC следующего hourly-прогона или None."""
    return _job_run_at_iso("hourly_pipeline")


app = FastAPI(
    title=settings.app_name,
    description="Сервис-парсер Metacritic: скрейпинг, хранение и LLM-саммари.",
    lifespan=lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    import os

    if os.environ.get("RUN_PIPELINE", "").lower() in {"1", "true", "yes"}:
        asyncio.run(run_hourly_pipeline())
    else:
        import uvicorn

        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
        )
