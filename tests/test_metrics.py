import numpy as np
import pandas as pd
import pytest

from usd_liquidity_monitor.metrics import compute_features, compute_ulsi


def _make_raw_df(n: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    idx = np.arange(n, dtype=float)

    payload = {
        "sofr": 5.10 + 0.0004 * idx + 0.01 * np.sin(idx / 19.0),
        "iorb": 5.00 + 0.0002 * idx + 0.004 * np.sin(idx / 23.0),
        "tga": 500 + 1.2 * idx + 25.0 * np.sin(idx / 17.0),
        "reserves": 3500 - 0.6 * idx + 35.0 * np.sin(idx / 31.0),
        "hy_oas": 3.8 + 0.002 * idx + 0.12 * np.cos(idx / 27.0),
    }

    rows: list[dict[str, object]] = []
    for name, values in payload.items():
        for d, v in zip(dates, values):
            rows.append({"date": d, "series_name": name, "value": float(v), "source": "test"})
    return pd.DataFrame(rows)


def test_compute_features_generates_expected_feature_set() -> None:
    raw_df = _make_raw_df()
    features = compute_features(raw_df)

    expected = {
        "funding_spread",
        "fiscal_delta20_tga",
        "reserves_delta20_detrended",
        "credit_hy_oas",
    }
    assert expected.issubset(set(features["feature_name"]))


def test_compute_features_reindexes_business_days_and_fills_missing() -> None:
    dates = pd.to_datetime(["2025-01-06", "2025-01-08"])
    rows: list[dict[str, object]] = []
    for series_name, vals in {
        "sofr": [5.10, 5.30],
        "iorb": [5.00, 5.10],
        "tga": [700.0, 710.0],
        "reserves": [3500.0, 3495.0],
        "hy_oas": [4.0, 4.1],
    }.items():
        for d, v in zip(dates, vals):
            rows.append({"date": d, "series_name": series_name, "value": v, "source": "test"})

    features = compute_features(pd.DataFrame(rows))
    funding = features[features["feature_name"] == "funding_spread"].sort_values("date")

    # Jan 7 (Tue) is missing in the input but should exist after B-day reindex and fill.
    jan7 = funding[funding["date"] == pd.Timestamp("2025-01-07")]
    assert not jan7.empty
    assert jan7["value"].iloc[0] == pytest.approx(0.10)


def test_compute_ulsi_validates_weights_sum() -> None:
    features = compute_features(_make_raw_df())

    with pytest.raises(ValueError):
        compute_ulsi(
            features,
            weights={
                "funding": 0.4,
                "fiscal": 0.2,
                "reserves": 0.2,
                "credit": 0.3,
            },
        )


def test_compute_ulsi_outputs_expected_columns_and_dynamic_quantiles() -> None:
    out = compute_ulsi(compute_features(_make_raw_df(n=1600)))

    expected_cols = {
        "date",
        "ulsi",
        "ulsi_value",
        "F_t",
        "G_t",
        "R_t",
        "C_t",
        "z_F",
        "z_G",
        "z_R",
        "z_C",
        "contrib_funding",
        "contrib_fiscal",
        "contrib_reserves",
        "contrib_credit",
        "q70_252",
        "q85_252",
        "q95_252",
        "regime",
        "alert_flag",
    }
    assert expected_cols.issubset(out.columns)

    valid_q = out.dropna(subset=["q70_252", "q85_252", "q95_252"])
    assert not valid_q.empty
    assert set(out["regime"].dropna().unique()).issubset({"Normal", "Watch", "Tight", "Stress", "NA"})


def test_compute_ulsi_winsorizes_component_zscores_to_plus_minus_5() -> None:
    dates = pd.date_range("2021-01-01", periods=600, freq="B")
    rows: list[dict[str, object]] = []
    features = {
        "funding_spread": np.zeros(len(dates), dtype=float),
        "fiscal_delta20_tga": np.zeros(len(dates), dtype=float),
        "reserves_delta20_detrended": np.zeros(len(dates), dtype=float),
        "credit_hy_oas": np.zeros(len(dates), dtype=float),
    }
    # Add a large shock to all factors so z-scores become extreme and must be clipped.
    for key in features:
        features[key][-5:] = 1_000_000.0

    for name, values in features.items():
        for d, v in zip(dates, values):
            rows.append({"date": d, "feature_name": name, "value": float(v)})

    out = compute_ulsi(pd.DataFrame(rows))

    for col in ["z_F", "z_G", "z_R", "z_C"]:
        assert out[col].dropna().abs().max() <= 5.0 + 1e-12


def test_compute_ulsi_triggers_alert_for_persistent_stress() -> None:
    dates = pd.date_range("2020-01-01", periods=700, freq="B")
    rows: list[dict[str, object]] = []
    feature_names = [
        "funding_spread",
        "fiscal_delta20_tga",
        "reserves_delta20_detrended",
        "credit_hy_oas",
    ]
    for name in feature_names:
        values = np.random.default_rng(42).normal(0, 0.1, len(dates))
        values[-10:] = values[-10:] + 100.0
        for d, v in zip(dates, values):
            rows.append({"date": d, "feature_name": name, "value": float(v)})

    out = compute_ulsi(pd.DataFrame(rows))

    assert bool(out["alert_flag"].tail(1).item()) is True
    assert float(out["ulsi"].tail(1).item()) > 1.5


def test_compute_ulsi_requires_all_four_factors_for_ulsi_value() -> None:
    features = compute_features(_make_raw_df())
    # Drop one factor on the latest date only.
    latest_date = features["date"].max()
    mask = (features["date"] == latest_date) & (features["feature_name"] == "credit_hy_oas")
    out = compute_ulsi(features.loc[~mask].copy())

    latest_row = out[out["date"] == out["date"].max()].tail(1).iloc[0]
    assert pd.isna(latest_row["C_t"])
    assert pd.isna(latest_row["ulsi"])
