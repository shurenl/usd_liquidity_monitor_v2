import numpy as np
import pandas as pd
import pytest

from usd_liquidity_monitor.metrics import compute_features, compute_ulsi


def _make_raw_df(n: int = 320) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    idx = np.arange(n)

    payload = {
        "effr": 5.10 + 0.0005 * idx,
        "iorb": 5.00 + 0.0002 * idx,
        "sofr": 5.12 + 0.0004 * idx,
        "reserves": 3500 - 2.0 * idx,
        "tga": 500 + 1.5 * idx,
        "on_rrp": 250 - 0.8 * idx,
        "fed_assets": 9000 - 3.0 * idx,
        "yield_3m": 5.2 + 0.0004 * idx,
        "yield_2y": 4.9 + 0.0007 * idx,
        "yield_10y": 4.2 + 0.0001 * idx,
        "cp_3m": 5.40 + 0.0005 * idx,
        "t_bill_3m": 5.00 + 0.0003 * idx,
        "hy_oas": 4.0 + 0.005 * idx,
        "ig_oas": 1.2 + 0.001 * idx,
        "dxy": 100 + np.sin(idx / 10),
        "vix": 15 + np.cos(idx / 11),
        "move_proxy": 0.2 + np.sin(idx / 9),
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
        "spread_policy",
        "spread_repo",
        "pressure_reserve",
        "pressure_tga",
        "pressure_on_rrp",
        "pressure_fed_assets",
        "pressure_cp",
        "pressure_credit",
        "pressure_curve_inversion",
        "pressure_frontend_jump",
        "risk_vix",
        "risk_dxy",
        "risk_move_proxy",
    }
    assert expected.issubset(set(features["feature_name"]))


def test_compute_features_pressure_direction() -> None:
    raw_df = _make_raw_df()
    features = compute_features(raw_df)
    reserve = features[features["feature_name"] == "pressure_reserve"].sort_values("date")

    # Reserves are monotonically decreasing in fixture, so pressure should be positive after diff window.
    assert reserve["value"].dropna().tail(10).gt(0).all()


def test_compute_ulsi_validates_weights_sum() -> None:
    raw_df = _make_raw_df()
    features = compute_features(raw_df)

    with pytest.raises(ValueError):
        compute_ulsi(
            features,
            weights={
                "funding_price": 0.3,
                "liquidity_quantity": 0.3,
                "credit": 0.3,
                "market_spillover": 0.3,
            },
        )


def test_compute_ulsi_triggers_alert_for_persistent_stress() -> None:
    dates = pd.date_range("2020-01-01", periods=320, freq="D")
    feature_names = [
        "spread_policy",
        "spread_repo",
        "pressure_reserve",
        "pressure_tga",
        "pressure_on_rrp",
        "pressure_fed_assets",
        "pressure_cp",
        "pressure_credit",
        "pressure_curve_inversion",
        "pressure_frontend_jump",
        "risk_vix",
        "risk_dxy",
        "risk_move_proxy",
    ]
    rows: list[dict[str, object]] = []
    for name in feature_names:
        values = np.zeros(len(dates), dtype=float)
        values[-10:] = 50.0
        for d, v in zip(dates, values):
            rows.append({"date": d, "feature_name": name, "value": float(v)})

    features = pd.DataFrame(rows)
    out = compute_ulsi(features)

    assert out["alert_flag"].tail(1).item() is True
    assert out["ulsi"].tail(1).item() > 1.5
