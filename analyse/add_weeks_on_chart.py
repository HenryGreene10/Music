import json
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
VIZ_DIR = ROOT / "viz"
BILLBOARD_PATH = DATA_DIR / "billboard_all.json"
SAMPLE_PAIRS_PATH = DATA_DIR / "sample_pairs_clean.csv"
VIZ_SAMPLE_PAIRS_PATH = VIZ_DIR / "sample_pairs_clean.csv"

TITLE_THRESHOLD = 80
ARTIST_THRESHOLD = 60
DEFAULT_WEEKS = 1


def normalize(value):
    return str(value).lower().strip()


def build_week_counts():
    with BILLBOARD_PATH.open("r") as handle:
        billboard = json.load(handle)

    week_counts = {}
    for week in billboard:
        for entry in week["data"]:
            key = (normalize(entry["song"]), normalize(entry["artist"]))
            week_counts[key] = week_counts.get(key, 0) + 1

    return week_counts


def build_title_index(week_counts):
    title_index = {}
    for (song, artist), weeks in week_counts.items():
        title_index.setdefault(song, []).append((song, artist, weeks))
    return title_index


def best_match(song, artist, week_counts, title_index, title_choices, title_match_cache, cache):
    key = (normalize(song), normalize(artist))
    if key in cache:
        return cache[key]

    if key in week_counts:
        result = {
            "weeks": week_counts[key],
            "matched": True,
            "matched_song": key[0],
            "matched_artist": key[1],
            "title_score": 100,
            "artist_score": 100,
        }
        cache[key] = result
        return result

    best = None
    best_total = -1

    if key[0] in title_match_cache:
        title_candidates = title_match_cache[key[0]]
    else:
        title_candidates = process.extract(
            key[0],
            title_choices,
            scorer=fuzz.ratio,
            score_cutoff=TITLE_THRESHOLD,
            limit=25,
        )
        title_match_cache[key[0]] = title_candidates

    for candidate_title, title_score, _ in title_candidates:
        for _, candidate_artist, weeks in title_index[candidate_title]:
            artist_score = fuzz.ratio(key[1], candidate_artist)
            if artist_score < ARTIST_THRESHOLD:
                continue

            total = title_score + artist_score
            if total > best_total:
                best_total = total
                best = {
                    "weeks": weeks,
                    "matched": True,
                    "matched_song": candidate_title,
                    "matched_artist": candidate_artist,
                    "title_score": title_score,
                    "artist_score": artist_score,
                }

    if best is None:
        best = {
            "weeks": DEFAULT_WEEKS,
            "matched": False,
            "matched_song": None,
            "matched_artist": None,
            "title_score": None,
            "artist_score": None,
        }

    cache[key] = best
    return best


def main():
    week_counts = build_week_counts()
    title_index = build_title_index(week_counts)
    title_choices = list(title_index.keys())
    df = pd.read_csv(SAMPLE_PAIRS_PATH)

    match_cache = {}
    title_match_cache = {}

    revival_matches = [
        best_match(song, artist, week_counts, title_index, title_choices, title_match_cache, match_cache)
        for song, artist in zip(df["revival_song"], df["revival_artist"])
    ]
    original_matches = [
        best_match(song, artist, week_counts, title_index, title_choices, title_match_cache, match_cache)
        for song, artist in zip(df["original_song"], df["original_artist"])
    ]

    df["revival_weeks_on_chart"] = [match["weeks"] for match in revival_matches]
    df["original_weeks_on_chart"] = [match["weeks"] for match in original_matches]

    df.to_csv(SAMPLE_PAIRS_PATH, index=False)
    df.to_csv(VIZ_SAMPLE_PAIRS_PATH, index=False)

    revival_matched = sum(match["matched"] for match in revival_matches)
    revival_defaulted = len(revival_matches) - revival_matched
    original_matched = sum(match["matched"] for match in original_matches)
    original_defaulted = len(original_matches) - original_matched

    top_10 = sorted(week_counts.items(), key=lambda item: item[1], reverse=True)[:10]

    print("Weeks on chart enrichment complete.")
    print(f"Rows updated: {len(df)}")
    print(
        "Revival matches:"
        f" matched={revival_matched}, defaulted_to_{DEFAULT_WEEKS}={revival_defaulted}"
    )
    print(
        "Original matches:"
        f" matched={original_matched}, defaulted_to_{DEFAULT_WEEKS}={original_defaulted}"
    )
    print()
    print("Top 10 songs by weeks on chart:")
    for index, ((song, artist), weeks) in enumerate(top_10, start=1):
        print(f"{index:>2}. {song} | {artist} | {weeks}")

    print()
    print("Verification sample (10 rows):")
    verification = df[
        [
            "revival_song",
            "revival_weeks_on_chart",
            "original_song",
            "original_weeks_on_chart",
        ]
    ].head(10)
    print(verification.to_string(index=False))


if __name__ == "__main__":
    main()
