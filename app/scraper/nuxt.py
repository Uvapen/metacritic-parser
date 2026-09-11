"""Разбор Nuxt 3 payload (`__NUXT_DATA__`) — логика из recon.py / parse_card.py."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from bs4 import BeautifulSoup

COVER_IMAGE_PREFIX = "https://www.metacritic.com/a/img"


class NuxtDataNotFoundError(RuntimeError):
    """На странице нет тега script#__NUXT_DATA__."""


def rehydrate_nuxt(payload: list[Any], root_index: int = 0) -> Any:
    """Восстанавливает JSON из flat-массива Nuxt 3."""

    def resolve(index: Any) -> Any:
        if not isinstance(index, int) or index < 0 or index >= len(payload):
            return index
        item = payload[index]
        if isinstance(item, list):
            return [resolve(child) for child in item]
        if isinstance(item, dict):
            return {key: resolve(value) for key, value in item.items()}
        return item

    return resolve(root_index)


def walk(node: Any) -> Iterator[Any]:
    """Генератор для обхода всех узлов payload."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def find_component(data: Any, component_name: str) -> dict[str, Any] | None:
    """Ищет компонент по componentName с непустым data['item'] или data['items']."""
    for obj in walk(data):
        meta = obj.get("meta") if isinstance(obj, dict) else None
        if isinstance(meta, dict) and meta.get("componentName") == component_name:
            if "data" in obj and obj["data"]:
                data_obj = obj["data"]
                if isinstance(data_obj, dict):
                    if "item" in data_obj and data_obj["item"]:
                        return obj
                    if "items" in data_obj and data_obj["items"]:
                        return obj
    return None


def cover_url_from_image(image_data: Any) -> str | None:
    """Конвертирует image-объект Nuxt в полный URL обложки."""
    if not image_data or not isinstance(image_data, dict):
        return None
    image_url = image_data.get("imageUrl")
    if isinstance(image_url, str) and image_url.startswith("http"):
        return image_url
    bucket_path = image_data.get("bucketPath")
    if not isinstance(bucket_path, str) or not bucket_path:
        return None
    if bucket_path.startswith("http"):
        return bucket_path
    bucket_type = image_data.get("bucketType")
    if not isinstance(bucket_type, str) or not bucket_type:
        bucket_type = "catalog"
    return f"{COVER_IMAGE_PREFIX}/{bucket_type}{bucket_path}"


def normalize_cover_url(url: str | None) -> str | None:
    """Исправляет старый путь /a/img/provider → /a/img/catalog/provider."""
    if not url:
        return None
    if "/a/img/provider/" in url and "/a/img/catalog/" not in url:
        return url.replace("/a/img/provider/", "/a/img/catalog/provider/")
    return url


def extract_nuxt_data(html: str) -> Any:
    """Достаёт и гидратирует ``__NUXT_DATA__`` из HTML-страницы."""
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NUXT_DATA__")
    if tag is None or not tag.string:
        raise NuxtDataNotFoundError("На странице не найден script#__NUXT_DATA__")
    raw_payload = json.loads(tag.string)
    if not isinstance(raw_payload, list):
        raise NuxtDataNotFoundError("__NUXT_DATA__ имеет неожиданный формат")
    return rehydrate_nuxt(raw_payload)
