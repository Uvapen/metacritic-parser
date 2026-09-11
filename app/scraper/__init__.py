"""Скрейпер Metacritic: HTTP-клиент, Nuxt payload, список и карточка игры."""

from app.scraper.card import GameCard, fetch_game_card, parse_game_card
from app.scraper.client import FetchError, MetacriticClient
from app.scraper.lister import ListedGame, fetch_home_new_releases, fetch_latest_games, fetch_main_new_releases, parse_games_list
from app.scraper.nuxt import NuxtDataNotFoundError, cover_url_from_image, find_component, rehydrate_nuxt, walk

__all__ = [
    "FetchError",
    "GameCard",
    "ListedGame",
    "MetacriticClient",
    "NuxtDataNotFoundError",
    "cover_url_from_image",
    "fetch_game_card",
    "fetch_home_new_releases",
    "fetch_latest_games",
    "fetch_main_new_releases",
    "find_component",
    "parse_game_card",
    "parse_games_list",
    "rehydrate_nuxt",
    "walk",
]
