from datetime import date

import numpy as np
import pandas as pd
import pytest

from usd_liquidity_monitor import lev_etf_fragility as lef


def _sample_config(window: int = 5) -> dict:
    return {
        "sub_index": {
            "max_ffill_days": 2,
            "zscore_window": window,
            "aum_rolling_change_days": 2,
            "min_abs_benchmark_return": 0.0005,
        },
        "factors": {
            "funding_sofr_effr_spread": {
                "source": "fred",
                "sign": 1,
                "params": {"left": "sofr", "right": "effr"},
            },
            "funding_sofr_20d_deviation": {
                "source": "fred",
                "sign": 1,
                "params": {"series": "sofr", "moving_average_days": 3},
            },
            "leveraged_etf_tracking_gap": {
                "source": "yfinance",
                "sign": 1,
                "params": {
                    "pairs": [
                        {"leveraged": "TQQQ", "benchmark": "QQQ", "leverage": 3},
                    ],
                },
            },
            "leveraged_etf_aum_growth": {
                "source": "yfinance",
                "sign": 1,
                "params": {"tickers": ["TQQQ"]},
            },
        },
    }


def test_load_default_config_contains_required_signs() -> None:
    config = lef.load_config()

    assert "lev_etf_fragility" == config["sub_index"]["name"]
    for factor in config["factors"].values():
        assert factor["sign"] in (-1, 1)


def test_limited_ffill_does_not_silently_fill_beyond_limit() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    series = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0], index=idx)

    out = lef._limited_ffill(series, limit=2)

    assert out.iloc[1] == pytest.approx(1.0)
    assert out.iloc[2] == pytest.approx(1.0)
    assert pd.isna(out.iloc[3])


def test_standardize_factors_applies_explicit_sign() -> None:
    idx = pd.date_range("2025-01-01", periods=10, freq="B")
    raw = pd.DataFrame(
        {
            "funding_sofr_effr_spread": np.arange(10, dtype=float),
            "funding_sofr_20d_deviation": np.arange(10, dtype=float),
            "leveraged_etf_tracking_gap": np.arange(10, dtype=float),
            "leveraged_etf_aum_growth": np.arange(10, dtype=float),
        },
        index=idx,
    )
    config = _sample_config(window=5)
    config["factors"]["leveraged_etf_tracking_gap"]["sign"] = -1

    zscores = lef.standardize_factors(raw, config)

    latest = zscores.dropna().iloc[-1]
    assert latest["funding_sofr_effr_spread"] > 0
    assert latest["leveraged_etf_tracking_gap"] < 0


def test_tracking_gap_uses_nominal_minus_realized_leverage() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="B")
    columns = pd.MultiIndex.from_product([["close", "volume"], ["TQQQ", "QQQ"]])
    prices = pd.DataFrame(index=idx, columns=columns, dtype=float)
    prices[("close", "QQQ")] = [100.0, 101.0, 102.0, 103.0]
    prices[("close", "TQQQ")] = [100.0, 102.0, 104.0, 106.0]

    gap = lef.compute_tracking_gap_factor(prices, _sample_config())

    qqq_ret = 101.0 / 100.0 - 1.0
    tqqq_ret = 102.0 / 100.0 - 1.0
    assert gap.iloc[1] == pytest.approx(3.0 - (tqqq_ret / qqq_ret))


def test_aum_growth_falls_back_to_turnover_when_shares_missing(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    columns = pd.MultiIndex.from_product([["close", "volume"], ["TQQQ"]])
    prices = pd.DataFrame(index=idx, columns=columns, dtype=float)
    prices[("close", "TQQQ")] = [100.0, 101.0, 102.0, 103.0, 104.0]
    prices[("volume", "TQQQ")] = [10.0, 11.0, 12.0, 13.0, 14.0]
    monkeypatch.setattr(lef, "fetch_yfinance_shares", lambda tickers: {"TQQQ": None})

    with caplog.at_level("WARNING"):
        factor = lef.compute_aum_growth_factor(prices, _sample_config())

    assert "turnover" in caplog.text
    assert factor.dropna().shape[0] > 0


def test_compose_subindex_uses_available_factors_without_crashing() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="B")
    zscores = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [1.0, 2.0, np.nan],
        },
        index=idx,
    )

    out = lef.compose_subindex(zscores)

    assert out.name == "lev_etf_fragility"
    assert out.iloc[0] == pytest.approx(1.0)
    assert out.iloc[1] == pytest.approx(2.0)
    assert out.iloc[2] == pytest.approx(3.0)


def test_compute_lev_etf_fragility_orchestrates_with_mocked_raw_factors(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2025-01-01", periods=20, freq="B")
    raw = pd.DataFrame(
        {
            "funding_sofr_effr_spread": np.arange(20, dtype=float),
            "funding_sofr_20d_deviation": np.arange(20, dtype=float),
            "leveraged_etf_tracking_gap": np.arange(20, dtype=float),
            "leveraged_etf_aum_growth": np.arange(20, dtype=float),
        },
        index=idx,
    )
    monkeypatch.setattr(lef, "load_config", lambda path=None: _sample_config(window=5))
    monkeypatch.setattr(lef, "compute_raw_factors", lambda start, end, config: raw)

    subindex, zscores, raw_out = lef.compute_lev_etf_fragility(date(2025, 1, 1), date(2025, 1, 31))

    assert subindex.name == "lev_etf_fragility"
    assert not subindex.dropna().empty
    assert set(zscores.columns) == set(raw.columns)
    assert raw_out.equals(raw)
