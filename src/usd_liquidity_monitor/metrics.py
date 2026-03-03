"""Feature engineering and ULSI calculations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import COMPONENT_WEIGHTS, REGIME_THRESHOLDS

BUSINESS_DAY_FREQ = "B"
ROLLING_WINDOW = 252
DELTA_WINDOW = 20
ALERT_DAYS = 3


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _reindex_business_days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    start = pd.Timestamp(df.index.min()).normalize()
    end = pd.Timestamp(df.index.max()).normalize()
    idx = pd.date_range(start=start, end=end, freq=BUSINESS_DAY_FREQ)
    return df.reindex(idx)


def _rolling_z_winsorized(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    z = (series - mean) / std
    zero_std_mask = std.eq(0) & series.notna() & mean.notna()
    z = z.mask(zero_std_mask, 0.0)
    return z.clip(lower=-5.0, upper=5.0)


def compute_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute long-form feature table.

    Input schema: `date`, `series_name`, `value`.
    Output schema: `date`, `feature_name`, `value`.
    """

    _ensure_columns(raw_df, ["date", "series_name", "value"])

    working = raw_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"])
    if working.empty:
        return pd.DataFrame(columns=["date", "feature_name", "value"])

    wide = (
        working.pivot_table(index="date", columns="series_name", values="value", aggfunc="last")
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
    )
    wide = _reindex_business_days(wide)
    wide = wide.ffill().bfill()

    sofr = _col_or_nan(wide, "sofr")
    iorb = _col_or_nan(wide, "iorb")
    tga = _col_or_nan(wide, "tga")
    reserves = _col_or_nan(wide, "reserves")
    hy_oas = _col_or_nan(wide, "hy_oas")

    reserves_trend_252 = reserves.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    reserves_detrended = reserves - reserves_trend_252

    features = pd.DataFrame(index=wide.index)
    features["funding_spread"] = sofr - iorb
    features["fiscal_delta20_tga"] = tga.diff(DELTA_WINDOW)
    features["reserves_delta20_detrended"] = reserves_detrended.diff(DELTA_WINDOW)
    features["credit_hy_oas"] = hy_oas
    # Optional future factor placeholder; only materializes if upstream data exists.
    if "eurusd_basis" in wide.columns:
        features["fx_eurusd_basis"] = _col_or_nan(wide, "eurusd_basis")

    long_features = (
        features.reset_index(names="date")
        .melt(id_vars=["date"], var_name="feature_name", value_name="value")
        .dropna(subset=["value"])
        .sort_values(["date", "feature_name"])
        .reset_index(drop=True)
    )
    return long_features


def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    required = set(COMPONENT_WEIGHTS.keys())
    given = set(weights.keys())
    if given != required:
        raise ValueError(f"weights keys must be exactly {sorted(required)}")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"weights must sum to 1.0, got {total}")
    return {k: float(v) for k, v in weights.items()}


def _classify_regime_dynamic(ulsi: float, q70: float, q85: float, q95: float) -> str:
    if any(pd.isna(v) for v in (ulsi, q70, q85, q95)):
        return "NA"
    if ulsi < q70:
        return "Normal"
    if ulsi < q85:
        return "Watch"
    if ulsi < q95:
        return "Tight"
    return "Stress"


def compute_ulsi(features_df: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Compute ULSI v2 and component contributions.

    Input schema: `date`, `feature_name`, `value`.
    Output schema: `date`, `ulsi`, `regime`, `alert_flag` (+ factors, z-scores, contributions).
    """

    _ensure_columns(features_df, ["date", "feature_name", "value"])
    final_weights = _validate_weights(weights or COMPONENT_WEIGHTS)

    working = features_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"])
    if working.empty:
        return pd.DataFrame(columns=["date", "ulsi", "ulsi_value", "regime", "alert_flag"])

    wide = (
        working.pivot_table(index="date", columns="feature_name", values="value", aggfunc="last")
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
    )
    wide = _reindex_business_days(wide)

    out = pd.DataFrame(index=wide.index)
    out["F_t"] = _rolling_z_winsorized(_col_or_nan(wide, "funding_spread"))
    out["G_t"] = _rolling_z_winsorized(_col_or_nan(wide, "fiscal_delta20_tga"))
    out["R_t"] = _rolling_z_winsorized(-_col_or_nan(wide, "reserves_delta20_detrended"))
    out["C_t"] = _rolling_z_winsorized(_col_or_nan(wide, "credit_hy_oas"))

    out["z_F"] = out["F_t"]
    out["z_G"] = out["G_t"]
    out["z_R"] = out["R_t"]
    out["z_C"] = out["C_t"]

    factor_cols = {"funding": "F_t", "fiscal": "G_t", "reserves": "R_t", "credit": "C_t"}
    for comp, factor_col in factor_cols.items():
        out[f"contrib_{comp}"] = out[factor_col] * final_weights[comp]

    contrib_cols = [f"contrib_{name}" for name in factor_cols]
    out["ulsi"] = out[contrib_cols].sum(axis=1, min_count=len(contrib_cols))
    out["ulsi_value"] = out["ulsi"]

    out["q70_252"] = out["ulsi"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).quantile(0.70)
    out["q85_252"] = out["ulsi"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).quantile(0.85)
    out["q95_252"] = out["ulsi"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).quantile(0.95)

    out["regime"] = [
        _classify_regime_dynamic(ulsi, q70, q85, q95)
        for ulsi, q70, q85, q95 in zip(out["ulsi"], out["q70_252"], out["q85_252"], out["q95_252"])
    ]
    out["alert_flag"] = (
        (out["ulsi"] > REGIME_THRESHOLDS[1])
        .rolling(ALERT_DAYS, min_periods=ALERT_DAYS)
        .sum()
        .fillna(0)
        .ge(ALERT_DAYS)
    )

    result = (
        out.reset_index(names="date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return result
