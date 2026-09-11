"""HTTP-клиент Metacritic на curl_cffi с TLS-impersonate и ретраями."""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType

from curl_cffi.requests import AsyncSession

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_HEADERS: dict[str, str] = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}

RETRYABLE_STATUS_CODES = {403, 408, 425, 429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """Не удалось загрузить страницу после всех попыток."""

    def __init__(self, url: str, message: str, status_code: int | None = None) -> None:
        self.url = url
        self.status_code = status_code
        super().__init__(f"{url}: {message}")


class MetacriticClient:
    """Асинхронный клиент для HTML-страниц Metacritic.

    Использует curl_cffi с ``impersonate='chrome120'``, чтобы обойти
    простую TLS-фильтрацию Cloudflare.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> MetacriticClient:
        self._session = AsyncSession(
            headers=DEFAULT_HEADERS,
            timeout=self._settings.request_timeout,
            impersonate=self._settings.impersonate or "chrome120",
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _require_session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("MetacriticClient нужно использовать как async context manager")
        return self._session

    async def fetch_page(self, url: str) -> str:
        """Загружает HTML страницы с ретраями при сетевых и retryable HTTP-ошибках."""
        session = self._require_session()
        last_error: Exception | None = None
        attempts = max(1, self._settings.max_retries)

        for attempt in range(1, attempts + 1):
            try:
                response = await session.get(
                    url,
                    impersonate=self._settings.impersonate or "chrome120",
                    timeout=self._settings.request_timeout,
                    headers=DEFAULT_HEADERS,
                )
                if response.status_code == 200:
                    return response.text
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < attempts:
                    delay = self._settings.retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Metacritic %s → HTTP %s, попытка %s/%s, пауза %.1fs",
                        url,
                        response.status_code,
                        attempt,
                        attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise FetchError(
                    url,
                    f"HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            except FetchError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Ошибка запроса %s (%s), попытка %s/%s",
                    url,
                    exc,
                    attempt,
                    attempts,
                )
                if attempt < attempts:
                    delay = self._settings.retry_backoff_seconds * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

        raise FetchError(url, f"исчерпаны попытки: {last_error}")
