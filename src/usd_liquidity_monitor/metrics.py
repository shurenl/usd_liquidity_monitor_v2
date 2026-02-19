"""Feature engineering and ULSI calculations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import COMPONENT_WEIGHTS, REGIME_THRESHOLDS


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _rolling_z(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (series - mean) / std.replace(0, np.nan)
    return z


def compute_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute long-form feature table.

    Input schema: `date`, `series_name`, `value`.
    Output schema: `date`, `feature_name`, `value`.
    """

    _ensure_columns(raw_df, ["date", "series_name", "value"])

    working = raw_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"])

    wide = (
        working.pivot_table(index="date", columns="series_name", values="value", aggfunc="last")
        .sort_index()
        .astype(float)
    )

    effr = _col_or_nan(wide, "effr")
    iorb = _col_or_nan(wide, "iorb")
    sofr = _col_or_nan(wide, "sofr")
    reserves = _col_or_nan(wide, "reserves")
    tga = _col_or_nan(wide, "tga")
    on_rrp = _col_or_nan(wide, "on_rrp")
    fed_assets = _col_or_nan(wide, "fed_assets")
    cp_3m = _col_or_nan(wide, "cp_3m")
    t_bill_3m = _col_or_nan(wide, "t_bill_3m")
    hy_oas = _col_or_nan(wide, "hy_oas")
    ig_oas = _col_or_nan(wide, "ig_oas")
    yield_3m = _col_or_nan(wide, "yield_3m")
    yield_2y = _col_or_nan(wide, "yield_2y")
    yield_10y = _col_or_nan(wide, "yield_10y")
    dxy = _col_or_nan(wide, "dxy")
    vix = _col_or_nan(wide, "vix")
    move_proxy = _col_or_nan(wide, "move_proxy")

    features = pd.DataFrame(index=wide.index)
    features["spread_policy"] = effr - iorb
    features["spread_repo"] = sofr - iorb
    features["pressure_reserve"] = -reserves.diff(20)
    features["pressure_tga"] = tga.diff(20)
    features["pressure_on_rrp"] = -on_rrp.diff(20)
    features["pressure_fed_assets"] = -fed_assets.diff(20)
    features["pressure_cp"] = cp_3m - t_bill_3m
    features["pressure_credit"] = hy_oas / ig_oas.replace(0, np.nan)
    # More inverted curve and rising front-end rates are treated as tighter liquidity conditions.
    features["pressure_curve_inversion"] = -(yield_10y - yield_3m)
    features["pressure_frontend_jump"] = yield_2y.diff(20)
    # Raw market stress drivers; z-standardization is done in compute_ulsi.
    features["risk_vix"] = vix
    features["risk_dxy"] = dxy
    features["risk_move_proxy"] = move_proxy

    long_features = (
        features.reset_index()
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


def _classify_regime(value: float) -> str:
    t1, t2, t3 = REGIME_THRESHOLDS
    if np.isnan(value):
        return "NA"
    if value < t1:
        return "Normal"
    if value < t2:
        return "Watch"
    if value < t3:
        return "Tight"
    return "Stress"


def compute_ulsi(features_df: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Compute ULSI and component contributions.

    Input schema: `date`, `feature_name`, `value`.
    Output schema: `date`, `ulsi`, `regime`, `alert_flag` (+ components and contributions).
    """

    _ensure_columns(features_df, ["date", "feature_name", "value"])
    final_weights = _validate_weights(weights or COMPONENT_WEIGHTS)

    working = features_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"])

    wide = (
        working.pivot_table(index="date", columns="feature_name", values="value", aggfunc="last")
        .sort_index()
        .astype(float)
    )

    z = pd.DataFrame(index=wide.index)
    z["z_spread_policy"] = _rolling_z(_col_or_nan(wide, "spread_policy"))
    z["z_spread_repo"] = _rolling_z(_col_or_nan(wide, "spread_repo"))
    z["z_pressure_reserve"] = _rolling_z(_col_or_nan(wide, "pressure_reserve"))
    z["z_pressure_tga"] = _rolling_z(_col_or_nan(wide, "pressure_tga"))
    z["z_pressure_on_rrp"] = _rolling_z(_col_or_nan(wide, "pressure_on_rrp"))
    z["z_pressure_fed_assets"] = _rolling_z(_col_or_nan(wide, "pressure_fed_assets"))
    z["z_pressure_cp"] = _rolling_z(_col_or_nan(wide, "pressure_cp"))
    z["z_pressure_credit"] = _rolling_z(_col_or_nan(wide, "pressure_credit"))
    z["z_pressure_curve_inversion"] = _rolling_z(_col_or_nan(wide, "pressure_curve_inversion"))
    z["z_pressure_frontend_jump"] = _rolling_z(_col_or_nan(wide, "pressure_frontend_jump"))

    risk_z = pd.DataFrame(index=wide.index)
    risk_z["z_risk_vix"] = _rolling_z(_col_or_nan(wide, "risk_vix"))
    risk_z["z_risk_dxy"] = _rolling_z(_col_or_nan(wide, "risk_dxy"))
    risk_z["z_risk_move_proxy"] = _rolling_z(_col_or_nan(wide, "risk_move_proxy"))
    z["z_pressure_market"] = risk_z.sum(axis=1, min_count=1)

    out = pd.DataFrame(index=wide.index)
    out["funding_price"] = z[
        ["z_spread_policy", "z_spread_repo", "z_pressure_curve_inversion", "z_pressure_frontend_jump"]
    ].mean(axis=1)
    out["liquidity_quantity"] = z[
        ["z_pressure_reserve", "z_pressure_tga", "z_pressure_on_rrp", "z_pressure_fed_assets"]
    ].mean(axis=1)
    out["credit"] = z[["z_pressure_cp", "z_pressure_credit"]].mean(axis=1)
    out["market_spillover"] = z["z_pressure_market"]

    for comp, weight in final_weights.items():
        out[f"contrib_{comp}"] = out[comp] * weight

    contrib_cols = [f"contrib_{name}" for name in final_weights.keys()]
    out["ulsi"] = out[contrib_cols].sum(axis=1, min_count=1)
    out["regime"] = out["ulsi"].apply(_classify_regime)
    out["alert_flag"] = (out["ulsi"] > REGIME_THRESHOLDS[1]).rolling(3, min_periods=3).sum().fillna(0).ge(3)

    result = (
        out.join(z, how="left")
        .reset_index()
        .rename(columns={"index": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return result
