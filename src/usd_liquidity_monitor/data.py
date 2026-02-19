"""Data access layer for public USD liquidity inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import io
import time
from typing import Iterable

import pandas as pd
import requests

from .config import DEFAULT_START_DATE, FRED_CSV_URL, NYFED_SOFR_URL, SERIES_SPECS


@dataclass(frozen=True)
class SyncStatus:
    """Status of syncing one canonical series."""

    series_name: str
    source: str
    success: bool
    row_count: int
    latest_date: date | None
    message: str = ""


def _request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str],
    timeout: int = 30,
    attempts: int = 3,
    backoff_seconds: float = 0.8,
) -> requests.Response:
    """Execute HTTP GET with bounded retries for transient request failures."""

    last_error: Exception | None = None
    for idx in range(attempts):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if idx < attempts - 1:
                time.sleep(backoff_seconds * (idx + 1))
    assert last_error is not None
    raise last_error


def _normalize_series_frame(df: pd.DataFrame, date_col: str, value_col: str, source: str) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={date_col: "date", value_col: "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date")
    out["source"] = source
    return out[["date", "value", "source"]]


def _fetch_fred_series(series_id: str, start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    sess = session or requests.Session()
    params = {
        "id": series_id,
        "cosd": start.isoformat(),
        "coed": end.isoformat(),
    }
    response = _request_with_retry(sess, FRED_CSV_URL, params=params)
    if response.text.lstrip().startswith("<!DOCTYPE"):
        raise ValueError(f"FRED returned HTML page for series_id={series_id}")
    raw = pd.read_csv(io.StringIO(response.text))
    if raw.empty:
        return pd.DataFrame(columns=["date", "value", "source"])
    if "observation_date" not in raw.columns:
        raise ValueError(f"Unexpected FRED payload schema for series_id={series_id}")
    date_col, value_col = raw.columns.tolist()[:2]
    return _normalize_series_frame(raw, date_col=date_col, value_col=value_col, source="fred")


def _fetch_nyfed_sofr(start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    sess = session or requests.Session()
    params = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
    }
    response = _request_with_retry(sess, NYFED_SOFR_URL, params=params)
    payload = response.json()
    rows = payload.get("refRates", [])
    if not rows:
        return pd.DataFrame(columns=["date", "value", "source"])
    data = pd.DataFrame(
        {
            "effectiveDate": [row.get("effectiveDate") for row in rows],
            "percentRate": [row.get("percentRate") for row in rows],
        }
    )
    return _normalize_series_frame(data, date_col="effectiveDate", value_col="percentRate", source="nyfed")


def fetch_series(series_name: str, start: date, end: date) -> pd.DataFrame:
    """Fetch one canonical series between start/end dates.

    Returns a DataFrame with schema: `date`, `value`, `source`.
    """

    if start > end:
        raise ValueError("start date must be <= end date")

    spec = SERIES_SPECS.get(series_name)
    if spec is None:
        raise KeyError(f"Unknown series_name: {series_name}")

    primary_error: Exception | None = None
    try:
        if spec.source == "fred":
            return _fetch_fred_series(spec.series_id, start=start, end=end)
        if spec.source == "nyfed" and spec.series_id == "SOFR":
            return _fetch_nyfed_sofr(start=start, end=end)
        raise ValueError(f"Unsupported source mapping: {spec.source}:{spec.series_id}")
    except Exception as exc:
        primary_error = exc

    if spec.fallback_source == "fred" and spec.fallback_series_id:
        return _fetch_fred_series(spec.fallback_series_id, start=start, end=end)

    if spec.fallback_source == "nyfed" and spec.fallback_series_id == "SOFR":
        return _fetch_nyfed_sofr(start=start, end=end)

    raise RuntimeError(f"Failed to fetch {series_name}") from primary_error


def fetch_all_series(start: date, end: date, series_names: Iterable[str] | None = None) -> tuple[pd.DataFrame, dict[str, SyncStatus]]:
    """Fetch many canonical series and return long-form data + statuses."""

    names = list(series_names) if series_names is not None else list(SERIES_SPECS.keys())
    frames: list[pd.DataFrame] = []
    statuses: dict[str, SyncStatus] = {}

    for name in names:
        try:
            frame = fetch_series(name, start=start, end=end)
            frame = frame.copy()
            frame["series_name"] = name
            frames.append(frame[["date", "series_name", "value", "source"]])
            latest = None if frame.empty else frame["date"].max().date()
            source = "unknown" if frame.empty else str(frame["source"].iloc[0])
            statuses[name] = SyncStatus(
                series_name=name,
                source=source,
                success=True,
                row_count=int(frame.shape[0]),
                latest_date=latest,
            )
        except Exception as exc:
            spec = SERIES_SPECS[name]
            statuses[name] = SyncStatus(
                series_name=name,
                source=spec.source,
                success=False,
                row_count=0,
                latest_date=None,
                message=str(exc),
            )
            if spec.required:
                continue

    if not frames:
        empty = pd.DataFrame(columns=["date", "series_name", "value", "source"])
        return empty, statuses

    all_data = pd.concat(frames, ignore_index=True).sort_values(["date", "series_name"])
    return all_data.reset_index(drop=True), statuses


def sync_all(as_of: date) -> dict[str, SyncStatus]:
    """Sync all canonical series to `as_of`, returning per-series status."""

    _, statuses = fetch_all_series(start=DEFAULT_START_DATE, end=as_of)
    return statuses
