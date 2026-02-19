"""Dashboard-oriented helpers and contracts."""

from __future__ import annotations

from datetime import date

import pandas as pd


def build_dashboard_table(raw_df: pd.DataFrame, features_df: pd.DataFrame, ulsi_df: pd.DataFrame) -> pd.DataFrame:
    """Build unified wide table for UI and CSV export.

    Contract output columns include:
    - `date`
    - prefixed raw columns (`raw_*`)
    - prefixed feature columns (`feature_*`)
    - ULSI outputs (`ulsi`, `regime`, `alert_flag`, contributions)
    """

    raw_wide = (
        raw_df.pivot_table(index="date", columns="series_name", values="value", aggfunc="last")
        .add_prefix("raw_")
        .reset_index()
    )
    feature_wide = (
        features_df.pivot_table(index="date", columns="feature_name", values="value", aggfunc="last")
        .add_prefix("feature_")
        .reset_index()
    )

    merged = raw_wide.merge(feature_wide, on="date", how="outer").merge(ulsi_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def summarize_data_quality(raw_df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Return per-series freshness and missingness summary."""

    if raw_df.empty:
        return pd.DataFrame(columns=["series_name", "latest_date", "lag_days", "missing_ratio"])

    work = raw_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")

    start = work["date"].min()
    end = pd.Timestamp(as_of)
    full_dates = pd.date_range(start, end, freq="D")

    rows: list[dict[str, object]] = []
    for name, grp in work.groupby("series_name"):
        observed_dates = pd.to_datetime(grp["date"]).dt.normalize().unique()
        observed_count = len(observed_dates)
        latest = pd.Timestamp(grp["date"].max())
        missing_ratio = 1.0 - (observed_count / len(full_dates)) if len(full_dates) else 0.0
        rows.append(
            {
                "series_name": name,
                "latest_date": latest.date(),
                "lag_days": int((end - latest).days),
                "missing_ratio": round(float(missing_ratio), 4),
            }
        )

    return pd.DataFrame(rows).sort_values(["lag_days", "missing_ratio", "series_name"]).reset_index(drop=True)


def build_alert_objects(ulsi_df: pd.DataFrame) -> list[dict[str, object]]:
    """Create alert records for dates where alert state flips to true."""

    if ulsi_df.empty:
        return []

    work = ulsi_df.sort_values("date").copy()
    alert_flag = work["alert_flag"].eq(True)
    work["prev_alert"] = alert_flag.shift(1, fill_value=False)
    trigger_rows = work[alert_flag & (~work["prev_alert"])]

    contrib_cols = [col for col in work.columns if col.startswith("contrib_")]
    alerts: list[dict[str, object]] = []

    for _, row in trigger_rows.iterrows():
        contrib_values = {
            col.replace("contrib_", ""): float(row[col])
            for col in contrib_cols
            if pd.notna(row[col])
        }
        top = sorted(contrib_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        level = str(row.get("regime", "Unknown"))
        alerts.append(
            {
                "date": pd.Timestamp(row["date"]).date(),
                "level": level,
                "trigger_rules": "ULSI > 1.5 for 3 consecutive days",
                "top_contributors": top,
            }
        )

    return alerts


def format_alerts_for_display(alerts: list[dict[str, object]]) -> pd.DataFrame:
    """Convert alert objects to Arrow-friendly tabular data for Streamlit."""

    if not alerts:
        return pd.DataFrame(columns=["date", "level", "trigger_rules", "top_contributors"])

    rows: list[dict[str, object]] = []
    for item in alerts:
        top = item.get("top_contributors", [])
        if isinstance(top, list):
            parts = []
            for pair in top:
                if isinstance(pair, tuple) and len(pair) == 2:
                    name, value = pair
                    try:
                        parts.append(f"{name}: {float(value):+.3f}")
                    except (TypeError, ValueError):
                        parts.append(f"{name}: {value}")
                else:
                    parts.append(str(pair))
            top_text = "; ".join(parts)
        else:
            top_text = str(top)

        rows.append(
            {
                "date": item.get("date"),
                "level": str(item.get("level", "")),
                "trigger_rules": str(item.get("trigger_rules", "")),
                "top_contributors": top_text,
            }
        )

    return pd.DataFrame(rows)
