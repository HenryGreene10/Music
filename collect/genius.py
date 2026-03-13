"""Genius API helpers for revival-tracks."""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from dotenv import find_dotenv, load_dotenv
from fuzzywuzzy import fuzz


BASE_URL = "https://api.genius.com"
RELATIONSHIP_TYPE_MAP = {
    "samples": "samples",
    "interpolates": "interpolates",
    "cover": "cover",
    "covers": "cover",
    "covered_by": "cover",
}
REQUEST_DELAY_SECONDS = 0.5


def _get_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def _genius_get(
    path: str, token: str, params: dict[str, Any] | None = None
) -> requests.Response:
    try:
        response = requests.get(
            f"{BASE_URL}{path}",
            params=params,
            headers=_get_headers(token),
            timeout=20,
        )
        response.raise_for_status()
        return response
    finally:
        time.sleep(REQUEST_DELAY_SECONDS)


def search_song(title: str, artist: str, token: str) -> dict[str, Any] | None:
    """Search Genius and return the first matching song result."""
    response = _genius_get(
        "/search",
        token=token,
        params={"q": f"{title} {artist}"},
    )

    hits = response.json().get("response", {}).get("hits", [])
    if not hits:
        return None

    first_result = hits[0].get("result", {})
    first_result_title = first_result.get("title", "")
    first_result_artist = first_result.get("primary_artist", {}).get("name", "")
    title_similarity = fuzz.partial_ratio(title.lower(), first_result_title.lower())
    artist_similarity = fuzz.partial_ratio(artist.lower(), first_result_artist.lower())

    if title_similarity < 80 or artist_similarity < 60:
        return None

    return {
        "id": first_result.get("id"),
        "title": first_result_title,
        "url": first_result.get("url"),
        "primary_artist": first_result_artist,
    }


def get_song_relationships(song_id: int, token: str) -> list[dict[str, Any]]:
    """Fetch supported Genius song relationships for a song id."""
    response = _genius_get(f"/songs/{song_id}", token=token)

    relationships = (
        response.json()
        .get("response", {})
        .get("song", {})
        .get("song_relationships", [])
    )

    results: list[dict[str, Any]] = []
    for relationship in relationships:
        relationship_type = RELATIONSHIP_TYPE_MAP.get(relationship.get("type"))
        if relationship_type is None:
            continue

        for song in relationship.get("songs", []):
            results.append(
                {
                    "relationship_type": relationship_type,
                    "song_title": song.get("title"),
                    "artist_name": song.get("primary_artist", {}).get("name"),
                    "genius_id": song.get("id"),
                    "url": song.get("url"),
                }
            )

    return results


def _load_token() -> str:
    load_dotenv(find_dotenv())
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token or token == "your_token_here":
        raise ValueError("Missing GENIUS_ACCESS_TOKEN in .env")
    return token


def _print_relationships(title: str, artist: str, token: str) -> None:
    song = search_song(title=title, artist=artist, token=token)
    print(f"{artist} - {title}")

    if song is None:
        print("  No Genius song match found.")
        return

    print(
        "  "
        f"song_id={song['id']} | "
        f"title={song['title']} | "
        f"artist={song['primary_artist']} | "
        f"url={song['url']}"
    )

    relationships = get_song_relationships(song_id=song["id"], token=token)
    if not relationships:
        print("  No supported relationships found.")
        return

    for item in relationships:
        print(
            "  "
            f"{item['relationship_type']}: "
            f"{item['song_title']} - {item['artist_name']} "
            f"(id={item['genius_id']}, url={item['url']})"
        )


if __name__ == "__main__":
    genius_token = _load_token()
    _print_relationships("Sweet Disposition", "The Temper Trap", genius_token)
    _print_relationships("HOPE", "XXXTentacion", genius_token)
