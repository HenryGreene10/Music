"""Spotify collection helpers for revival-tracks."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials


def _create_spotify_client() -> Spotify:
    """Create an authenticated Spotify client using client credentials flow."""
    load_dotenv(find_dotenv())

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in .env"
        )

    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return Spotify(auth_manager=auth)


def _find_best_track_match(
    sp: Spotify, track_name: str, artist: str, limit: int = 10
) -> dict[str, Any] | None:
    """Search Spotify and return the best track match for a track/artist pair."""
    artist_clean = artist.strip()
    query = f"track:{track_name}"
    if artist_clean:
        query = f"{query} artist:{artist_clean}"

    results = sp.search(q=query, type="track", limit=limit)
    items = results.get("tracks", {}).get("items", [])

    if not items:
        return None

    if not artist_clean:
        return items[0]

    artist_lower = artist_clean.lower()
    best_match = next(
        (
            item
            for item in items
            if any(a["name"].strip().lower() == artist_lower for a in item.get("artists", []))
        ),
        items[0],
    )

    return best_match


def get_track_info(track_name: str, artist: str) -> dict[str, Any] | None:
    """Search for a track and return track-level metadata."""
    sp = _create_spotify_client()
    best_match = _find_best_track_match(sp, track_name=track_name, artist=artist)

    if best_match is None:
        return None

    primary_artist = best_match.get("artists", [{}])[0]
    return {
        "track_name": best_match["name"],
        "artist": ", ".join(a["name"] for a in best_match.get("artists", [])),
        "album": best_match.get("album", {}).get("name"),
        "release_date": best_match.get("album", {}).get("release_date"),
        "popularity": best_match.get("popularity"),
        "spotify_id": best_match.get("id"),
        "external_url": best_match.get("external_urls", {}).get("spotify"),
        "artist_id": primary_artist.get("id"),
    }


def get_artist_info(artist_id: str) -> dict[str, Any] | None:
    """Return artist-level metadata for a Spotify artist id."""
    sp = _create_spotify_client()
    artist = sp.artist(artist_id)

    if not artist:
        return None

    return {
        "artist_name": artist.get("name"),
        "genres": artist.get("genres", []),
        "followers": artist.get("followers", {}).get("total"),
        "popularity": artist.get("popularity"),
    }


def _build_comparison_frame(track_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a dataframe comparing track and artist metadata across inputs."""
    rows: list[dict[str, Any]] = []

    for track_name, artist_name in track_pairs:
        track_info = get_track_info(track_name, artist_name)

        if track_info is None:
            rows.append(
                {
                    "track_name": track_name,
                    "artist": artist_name,
                    "album": None,
                    "release_date": None,
                    "popularity": None,
                    "spotify_id": None,
                    "external_url": None,
                    "artist_id": None,
                    "artist_name": None,
                    "genres": None,
                    "followers": None,
                    "artist_popularity": None,
                }
            )
            continue

        artist_info = None
        if track_info.get("artist_id"):
            artist_info = get_artist_info(track_info["artist_id"])

        rows.append(
            {
                **track_info,
                "artist_name": artist_info.get("artist_name") if artist_info else None,
                "genres": artist_info.get("genres") if artist_info else None,
                "followers": artist_info.get("followers") if artist_info else None,
                "artist_popularity": artist_info.get("popularity") if artist_info else None,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    comparisons = [
        ("Sweet Disposition", "The Temper Trap"),
        ("Sweet Disposition", "John Summit"),
        ("Running Up That Hill", "Kate Bush"),
        ("Running Up That Hill", "Meg Myers"),
    ]
    frame = _build_comparison_frame(comparisons)
    print(frame[["track_name", "artist", "release_date", "popularity"]].to_string(index=False))
