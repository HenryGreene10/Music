"""Billboard, Genius, and Spotify revival pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from fuzzywuzzy import fuzz

try:
    from collect.genius import get_song_relationships, search_song
    from collect.spotify import get_track_info
except ModuleNotFoundError:
    from genius import get_song_relationships, search_song
    from spotify import get_track_info


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHARTS_DIR = DATA_DIR / "charts"
DB_PATH = DATA_DIR / "revival.db"
HISTORY_CSV = DATA_DIR / "billboard_history.csv"
ALL_BILLBOARD_JSON = DATA_DIR / "billboard_all.json"
HISTORY_START_YEAR = 1960
HISTORY_END_YEAR = 2025
RECENT_START_YEAR = 2018
RECENT_END_YEAR = 2025
TARGET_RELATIONSHIPS = {"samples", "interpolates", "cover"}
HISTORY_TITLE_THRESHOLD = 80
HISTORY_ARTIST_THRESHOLD = 60
CHECKPOINT_INTERVAL = 50


def _load_genius_token() -> str:
    load_dotenv(find_dotenv())
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token or token == "your_token_here":
        raise ValueError("Missing GENIUS_ACCESS_TOKEN in .env")
    return token


def _extract_year(release_date: str | None) -> int | None:
    if not release_date:
        return None

    year = str(release_date)[:4]
    return int(year) if year.isdigit() else None


def _normalize_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _is_near_identical(left: str, right: str) -> bool:
    return _normalize_key(left) == _normalize_key(right)


def _is_valid_pair(
    revival_song: str,
    revival_artist: str,
    original_song: str,
    original_artist: str,
) -> bool:
    revival_song_norm = _normalize_key(revival_song)
    revival_artist_norm = _normalize_key(revival_artist)
    original_song_norm = _normalize_key(original_song)
    original_artist_norm = _normalize_key(original_artist)

    if revival_song_norm == original_song_norm and revival_artist_norm == original_artist_norm:
        return False

    if "taylor swift" in original_artist_norm and "taylor swift" not in revival_artist_norm:
        return False

    if (
        revival_song_norm == "last christmas"
        and revival_artist_norm == "wham"
        and "barry manilow" in original_artist_norm
        and "cant smile without you" in original_song_norm
    ):
        return False

    return True


def build_billboard_history() -> tuple[pd.DataFrame, int]:
    """Load Billboard history from the downloaded all.json dataset."""
    if not ALL_BILLBOARD_JSON.exists():
        raise FileNotFoundError(f"Missing {ALL_BILLBOARD_JSON}")

    with ALL_BILLBOARD_JSON.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: list[dict[str, Any]] = []
    for chart in payload:
        chart_date = chart.get("date")
        if not isinstance(chart_date, str):
            continue

        year = _extract_year(chart_date)
        if year is None:
            continue

        for entry in chart.get("data", []):
            rows.append(
                {
                    "song": entry.get("song"),
                    "artist": entry.get("artist"),
                    "peak_position": entry.get("peak_position"),
                    "year": year,
                    "date": chart_date,
                }
            )

    frame = pd.DataFrame(rows, columns=["song", "artist", "peak_position", "year", "date"])
    frame = frame[
        frame["year"].between(HISTORY_START_YEAR, HISTORY_END_YEAR)
        & frame["song"].notna()
        & frame["artist"].notna()
        & frame["peak_position"].notna()
    ].copy()

    years_loaded = int(frame["year"].nunique()) if not frame.empty else 0
    if frame.empty:
        frame.to_csv(HISTORY_CSV, index=False)
        return frame, years_loaded

    history = (
        frame.sort_values(["peak_position", "year", "date"], ascending=[True, False, False])
        .drop_duplicates(["song", "artist"], keep="first")
        .reset_index(drop=True)
    )
    history.to_csv(HISTORY_CSV, index=False)
    return history, years_loaded


def _load_recent_candidates(history: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    recent = history[
        (history["year"] >= RECENT_START_YEAR) & (history["year"] <= RECENT_END_YEAR)
    ].copy()
    recent = recent.sort_values(["year", "peak_position", "song"], ascending=[False, True, True])
    if limit is not None:
        recent = recent.head(limit).copy()
    return recent.reset_index(drop=True)


def _match_history_song(
    song: str,
    artist: str,
    history_records: list[dict[str, Any]],
    cache: dict[tuple[str, str], dict[str, Any] | None],
) -> dict[str, Any] | None:
    key = (_normalize_key(song), _normalize_key(artist))
    if key in cache:
        return cache[key]

    best_match: dict[str, Any] | None = None
    best_score = -1

    for row in history_records:
        title_score = fuzz.partial_ratio(song.lower(), str(row["song"]).lower())
        if title_score < HISTORY_TITLE_THRESHOLD:
            continue

        artist_score = fuzz.partial_ratio(artist.lower(), str(row["artist"]).lower())
        if artist_score < HISTORY_ARTIST_THRESHOLD:
            continue

        combined_score = title_score + artist_score
        if best_match is None or combined_score > best_score:
            best_match = {
                "song": row["song"],
                "artist": row["artist"],
                "peak_position": int(row["peak_position"]),
                "year": int(row["year"]),
                "title_score": title_score,
                "artist_score": artist_score,
            }
            best_score = combined_score

    cache[key] = best_match
    return best_match


def _lookup_spotify_track(
    track_name: str,
    artist: str,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (_normalize_key(track_name), _normalize_key(artist))
    if key in cache:
        return cache[key]

    info = None
    spotify_unresolved = False

    try:
        info = get_track_info(track_name, artist)
    except Exception:
        info = None

    if info is None and artist.strip():
        try:
            info = get_track_info(track_name, "")
        except Exception:
            info = None

    if info is None:
        spotify_unresolved = True

    result = {
        "info": info,
        "popularity": info.get("popularity") if info else None,
        "release_year": _extract_year(info.get("release_date")) if info else None,
        "track_name": info.get("track_name") if info else track_name,
        "artist": info.get("artist") if info else artist,
        "spotify_unresolved": spotify_unresolved,
    }
    cache[key] = result
    return result


def _classify_story_type(ever_charted: bool, chart_jump: int | None) -> str:
    if not ever_charted:
        return "resurrection"
    if chart_jump is not None and chart_jump > 50:
        return "underdog"
    return "confirmation"


def _initialize_database() -> set[tuple[str, str]]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='processed_tracks'"
        )
        has_processed_tracks = cursor.fetchone() is not None

        if not has_processed_tracks:
            cursor.execute("DROP TABLE IF EXISTS sample_pairs")
            cursor.execute("DROP TABLE IF EXISTS processed_tracks")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_pairs (
                revival_song TEXT,
                revival_artist TEXT,
                revival_peak INTEGER,
                revival_chart_year INTEGER,
                revival_spotify_popularity INTEGER,
                original_song TEXT,
                original_artist TEXT,
                original_release_year INTEGER,
                original_chart_year INTEGER,
                original_spotify_popularity INTEGER,
                ever_charted BOOLEAN,
                original_peak INTEGER,
                years_between INTEGER,
                chart_jump INTEGER,
                revival_strength INTEGER,
                relationship_type TEXT,
                spotify_unresolved BOOLEAN,
                story_type TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_tracks (
                song TEXT NOT NULL,
                artist TEXT NOT NULL,
                PRIMARY KEY (song, artist)
            )
            """
        )
        cursor.execute("SELECT song, artist FROM processed_tracks")
        processed = {(row[0], row[1]) for row in cursor.fetchall()}
        connection.commit()

    return processed


def _append_checkpoint(
    pending_pairs: list[dict[str, Any]],
    pending_processed: list[tuple[str, str]],
) -> None:
    if not pending_pairs and not pending_processed:
        return

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        if pending_pairs:
            cursor.executemany(
                """
                INSERT INTO sample_pairs (
                    revival_song,
                    revival_artist,
                    revival_peak,
                    revival_chart_year,
                    revival_spotify_popularity,
                    original_song,
                    original_artist,
                    original_release_year,
                    original_chart_year,
                    original_spotify_popularity,
                    ever_charted,
                    original_peak,
                    years_between,
                    chart_jump,
                    revival_strength,
                    relationship_type,
                    spotify_unresolved,
                    story_type
                ) VALUES (
                    :revival_song,
                    :revival_artist,
                    :revival_peak,
                    :revival_chart_year,
                    :revival_spotify_popularity,
                    :original_song,
                    :original_artist,
                    :original_release_year,
                    :original_chart_year,
                    :original_spotify_popularity,
                    :ever_charted,
                    :original_peak,
                    :years_between,
                    :chart_jump,
                    :revival_strength,
                    :relationship_type,
                    :spotify_unresolved,
                    :story_type
                )
                """,
                pending_pairs,
            )
        if pending_processed:
            cursor.executemany(
                """
                INSERT OR IGNORE INTO processed_tracks (song, artist)
                VALUES (?, ?)
                """,
                pending_processed,
            )
        connection.commit()


def _load_all_sample_pairs() -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sample_pairs'"
        )
        if cursor.fetchone() is None:
            return []

        cursor.execute("SELECT * FROM sample_pairs")
        rows = cursor.fetchall()

    return [dict(row) for row in rows]


def _build_sample_pairs(
    recent_candidates: pd.DataFrame,
    history: pd.DataFrame,
    genius_token: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "genius_lookups_completed": 0,
        "pairs_skipped_unresolved_years": 0,
        "spotify_unresolved": 0,
    }

    history_records = history.to_dict("records")
    history_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    spotify_cache: dict[tuple[str, str], dict[str, Any]] = {}
    relationship_cache: dict[int, list[dict[str, Any]]] = {}
    processed_keys = _initialize_database()
    sample_pairs: list[dict[str, Any]] = []
    pending_pairs: list[dict[str, Any]] = []
    pending_processed: list[tuple[str, str]] = []
    recent_records = recent_candidates.to_dict("records")
    remaining_records = [
        candidate
        for candidate in recent_records
        if (candidate["song"], candidate["artist"]) not in processed_keys
    ]
    stats["tracks_skipped_resume"] = len(recent_records) - len(remaining_records)
    total = len(remaining_records)

    for index, candidate in enumerate(remaining_records, start=1):
        revival_song = candidate["song"]
        revival_artist = candidate["artist"]
        revival_peak = int(candidate["peak_position"])
        revival_chart_year = int(candidate["year"])
        processed_key = (revival_song, revival_artist)
        print(f"[{index}/{total}] Checking: {revival_song} - {revival_artist}")

        try:
            genius_song = search_song(revival_song, revival_artist, genius_token)
            stats["genius_lookups_completed"] += 1
        except Exception as exc:
            print(f"  Genius search failed: {exc}")
            continue

        if genius_song is None:
            processed_keys.add(processed_key)
            pending_processed.append(processed_key)
            if index % CHECKPOINT_INTERVAL == 0:
                _append_checkpoint(pending_pairs, pending_processed)
                pending_pairs.clear()
                pending_processed.clear()
            continue

        try:
            if genius_song["id"] not in relationship_cache:
                relationship_cache[genius_song["id"]] = get_song_relationships(
                    genius_song["id"], genius_token
                )
            relationships = relationship_cache[genius_song["id"]]
        except Exception as exc:
            print(f"  Genius relationships failed: {exc}")
            continue

        revival_spotify = _lookup_spotify_track(revival_song, revival_artist, spotify_cache)
        revival_spotify_popularity = revival_spotify["popularity"]

        for relationship in relationships:
            relationship_type = relationship.get("relationship_type")
            if relationship_type not in TARGET_RELATIONSHIPS:
                continue

            original_song = relationship.get("song_title") or ""
            original_artist = relationship.get("artist_name") or ""

            if not _is_valid_pair(revival_song, revival_artist, original_song, original_artist):
                continue

            history_match = _match_history_song(
                original_song,
                original_artist,
                history_records,
                history_cache,
            )
            ever_charted = history_match is not None
            original_peak = history_match["peak_position"] if history_match else None
            original_chart_year = history_match["year"] if history_match else None

            original_spotify = _lookup_spotify_track(original_song, original_artist, spotify_cache)
            spotify_unresolved = bool(original_spotify["spotify_unresolved"])
            if spotify_unresolved:
                stats["spotify_unresolved"] += 1

            original_release_year = original_spotify["release_year"]
            if original_release_year is None:
                original_release_year = original_chart_year

            if original_release_year is None:
                stats["pairs_skipped_unresolved_years"] += 1
                continue

            years_between = revival_chart_year - int(original_release_year)
            if years_between < 5:
                continue

            chart_jump = None
            if ever_charted and original_peak is not None:
                chart_jump = int(original_peak) - revival_peak

            pair = {
                "revival_song": revival_song,
                "revival_artist": revival_artist,
                "revival_peak": revival_peak,
                "revival_chart_year": revival_chart_year,
                "revival_spotify_popularity": revival_spotify_popularity,
                "original_song": original_spotify["track_name"],
                "original_artist": original_spotify["artist"],
                "original_release_year": original_release_year,
                "original_chart_year": original_chart_year,
                "original_spotify_popularity": original_spotify["popularity"],
                "ever_charted": ever_charted,
                "original_peak": original_peak,
                "years_between": years_between,
                "chart_jump": chart_jump,
                "revival_strength": revival_peak,
                "relationship_type": relationship_type,
                "spotify_unresolved": spotify_unresolved,
                "story_type": _classify_story_type(ever_charted, chart_jump),
            }
            sample_pairs.append(pair)
            pending_pairs.append(pair)

        processed_keys.add(processed_key)
        pending_processed.append(processed_key)

        if index % CHECKPOINT_INTERVAL == 0:
            _append_checkpoint(pending_pairs, pending_processed)
            pending_pairs.clear()
            pending_processed.clear()

    _append_checkpoint(pending_pairs, pending_processed)

    return _load_all_sample_pairs(), stats


def _print_table(title: str, frame: pd.DataFrame, columns: list[str]) -> None:
    print(title)
    if frame.empty:
        print("None")
        return
    print(frame[columns].to_string(index=False))


def _print_summary(
    history: pd.DataFrame,
    years_loaded: int,
    recent_candidates: pd.DataFrame,
    sample_pairs: list[dict[str, Any]],
    stats: dict[str, int],
) -> None:
    print(
        f"Billboard history loaded: {len(history)} songs across {years_loaded} years (1960-2025)"
    )
    print(f"Recent candidates (2018-2025): {len(recent_candidates)} unique tracks")
    print(f"Already processed and skipped on resume: {stats.get('tracks_skipped_resume', 0)}")
    print(f"Genius lookups completed: {stats['genius_lookups_completed']}")
    print(f"Revival pairs found: {len(sample_pairs)}")
    print(f"Pairs skipped (unresolved years): {stats['pairs_skipped_unresolved_years']}")
    print(f"Spotify unresolved: {stats['spotify_unresolved']}")
    print()

    frame = pd.DataFrame(sample_pairs)
    if frame.empty:
        print("RESULTS BY STORY TYPE:")
        print("Resurrections (never charted -> now famous): 0")
        print("Underdogs (modest hit -> much bigger revival): 0")
        print("Confirmations (famous -> famous again): 0")
        print()
        _print_table(
            "TOP 10 RESURRECTIONS (by years_between):",
            frame,
            [
                "revival_song",
                "revival_artist",
                "revival_peak",
                "original_song",
                "original_artist",
                "original_release_year",
                "years_between",
            ],
        )
        print()
        _print_table(
            "TOP 10 UNDERDOGS (by chart_jump):",
            frame,
            [
                "revival_song",
                "revival_artist",
                "revival_peak",
                "original_song",
                "original_artist",
                "original_peak",
                "chart_jump",
            ],
        )
        print()
        _print_table(
            "TOP 10 CONFIRMATIONS (by original_peak):",
            frame,
            [
                "revival_song",
                "revival_artist",
                "revival_peak",
                "original_song",
                "original_artist",
                "original_peak",
            ],
        )
        return

    resurrections = frame[frame["story_type"] == "resurrection"].copy()
    underdogs = frame[frame["story_type"] == "underdog"].copy()
    confirmations = frame[frame["story_type"] == "confirmation"].copy()

    print("RESULTS BY STORY TYPE:")
    print(f"Resurrections (never charted -> now famous): {len(resurrections)}")
    print(f"Underdogs (modest hit -> much bigger revival): {len(underdogs)}")
    print(f"Confirmations (famous -> famous again): {len(confirmations)}")
    print()

    _print_table(
        "TOP 10 RESURRECTIONS (by years_between):",
        resurrections.sort_values(["years_between", "revival_peak"], ascending=[False, True]).head(10),
        [
            "revival_song",
            "revival_artist",
            "revival_peak",
            "original_song",
            "original_artist",
            "original_release_year",
            "years_between",
        ],
    )
    print()
    _print_table(
        "TOP 10 UNDERDOGS (by chart_jump):",
        underdogs.sort_values(["chart_jump", "years_between"], ascending=[False, False]).head(10),
        [
            "revival_song",
            "revival_artist",
            "revival_peak",
            "original_song",
            "original_artist",
            "original_peak",
            "chart_jump",
        ],
    )
    print()
    _print_table(
        "TOP 10 CONFIRMATIONS (by original_peak):",
        confirmations.sort_values(["original_peak", "years_between"], ascending=[True, False]).head(10),
        [
            "revival_song",
            "revival_artist",
            "revival_peak",
            "original_song",
            "original_artist",
            "original_peak",
        ],
    )


def run_pipeline(limit: int | None = None) -> None:
    genius_token = _load_genius_token()
    history, years_loaded = build_billboard_history()
    recent_candidates = _load_recent_candidates(history, limit=limit)
    sample_pairs, stats = _build_sample_pairs(recent_candidates, history, genius_token)
    _print_summary(history, years_loaded, recent_candidates, sample_pairs, stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_pipeline(limit=args.limit)
