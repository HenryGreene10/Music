"""Fetch monthly Billboard Hot 100 snapshots and aggregate them locally."""

from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests


BASE_URL = (
    "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/date/{date}.json"
)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHARTS_DIR = DATA_DIR / "charts"
RECENT_CHARTS_CSV = DATA_DIR / "recent_charts.csv"


def _last_saturday(year: int, month: int) -> date:
    """Return the last Saturday for a given month."""
    month_weeks = calendar.monthcalendar(year, month)
    saturday = calendar.SATURDAY
    day = month_weeks[-1][saturday] or month_weeks[-2][saturday]
    return date(year, month, day)


def iter_monthly_chart_dates(start_year: int = 2022, end_year: int = 2025) -> list[str]:
    """Return one monthly chart date per month as YYYY-MM-DD strings."""
    chart_dates: list[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            chart_dates.append(_last_saturday(year, month).isoformat())
    return chart_dates


def download_monthly_charts(
    start_year: int = 2022,
    end_year: int = 2025,
    force: bool = False,
) -> list[Path]:
    """Download one Billboard Hot 100 chart JSON per month."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[Path] = []

    for chart_date in iter_monthly_chart_dates(start_year, end_year):
        destination = CHARTS_DIR / f"{chart_date}.json"
        if destination.exists() and not force:
            downloaded_files.append(destination)
            continue

        response = requests.get(BASE_URL.format(date=chart_date), timeout=20)
        response.raise_for_status()
        destination.write_text(response.text, encoding="utf-8")
        downloaded_files.append(destination)

    return downloaded_files


def build_recent_charts_csv() -> pd.DataFrame:
    """Aggregate the downloaded monthly chart files into one CSV."""
    rows: list[dict[str, object]] = []

    for path in sorted(CHARTS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        chart_date = payload.get("date")
        for entry in payload.get("data", []):
            rows.append(
                {
                    "chart_date": chart_date,
                    "song": entry.get("song"),
                    "artist": entry.get("artist"),
                    "peak_position": entry.get("peak_position"),
                    "weeks_on_chart": entry.get("weeks_on_chart"),
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(RECENT_CHARTS_CSV, index=False)
    return frame


def main() -> None:
    download_monthly_charts()
    frame = build_recent_charts_csv()
    print(
        f"Downloaded {len(list(CHARTS_DIR.glob('*.json')))} monthly charts "
        f"and aggregated {len(frame)} rows into {RECENT_CHARTS_CSV.name}."
    )


if __name__ == "__main__":
    main()
