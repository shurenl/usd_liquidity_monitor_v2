from datetime import date

import numpy as np
import pandas as pd
import pytest

from usd_liquidity_monitor import lev_etf_integration as integration


def test_build_forward_target_uses_future_value_without_lookahead_in_factor() -> None:
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    raw = pd.DataFrame(
        {
            "date": dates,
            "series_name": ["vix"] * len(dates),
            "value": [10, 11, 13, 12, 15, 20],
            "source": ["test"] * len(dates),
        }
    )

    target = integration.build_forward_target(
        raw,
        series_name="vix",
        horizon=2,
        start=date(2025, 1, 1),
        end=date(2025, 1, 8),
        max_ffill_days=0,
    )

    assert target.iloc[0] == pytest.approx(13 - 10)
    assert target.iloc[1] == pytest.approx(12 - 11)
    assert pd.isna(target.iloc[-1])


def test_spearman_ic_returns_rank_correlation() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    signal = pd.Series([1, 2, 3, 4, 5], index=idx)
    target = pd.Series([10, 20, 30, 40, 50], index=idx)

    assert integration.spearman_ic(signal, target, min_observations=5) == pytest.approx(1.0)


def test_fit_ols_residual_reports_high_r2_for_linear_replication() -> None:
    idx = pd.date_range("2025-01-01", periods=20, freq="B")
    x = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.ones(20)}, index=idx)
    y = pd.Series(2.0 * x["a"] + 3.0, index=idx, name="lev_etf_fragility")

    result = integration.fit_ols_residual(y, x)

    assert result.r2 == pytest.approx(1.0)
    assert result.residual.dropna().abs().max() < 1e-10


def test_rolling_spearman_ic_keeps_windowed_index() -> None:
    idx = pd.date_range("2025-01-01", periods=8, freq="B")
    signal = pd.Series(np.arange(8, dtype=float), index=idx)
    target = pd.Series(np.arange(8, dtype=float), index=idx)

    out = integration.rolling_spearman_ic(signal, target, window=4, min_periods=4)

    assert out.index.equals(idx)
    assert out.dropna().iloc[-1] == pytest.approx(1.0)


def test_trimmed_ic_removes_largest_absolute_signal_observation() -> None:
    idx = pd.date_range("2025-01-01", periods=100, freq="B")
    signal = pd.Series(np.arange(100, dtype=float), index=idx)
    target = signal.copy()
    signal.iloc[-1] = 1_000_000.0
    target.iloc[-1] = -1_000_000.0

    full_ic = integration.spearman_ic(signal, target, min_observations=40)
    trimmed_ic = integration.compute_trimmed_ic(signal, target, trim_pct=0.01, min_observations=40)

    assert trimmed_ic > full_ic


def test_compose_candidate_ulsi_with_lev_flips_negative_ic_signal() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="B")
    base = pd.Series([1.0, 1.0, 1.0], index=idx)
    lev = pd.Series([2.0, 2.0, 2.0], index=idx)

    out = integration.compose_candidate_ulsi_with_lev(base, lev, lev_sign=-1, lev_weight=0.10)

    assert out.iloc[0] == pytest.approx(0.9 * 1.0 - 0.1 * 2.0)


def test_run_diagnostic_with_mocked_inputs_writes_report(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2025-01-01", periods=180, freq="B")
    raw = pd.DataFrame(
        {
            "date": idx,
            "series_name": ["vix"] * len(idx),
            "value": np.linspace(10, 30, len(idx)),
            "source": ["test"] * len(idx),
        }
    )
    factors = pd.DataFrame(
        {
            "z_F": np.sin(np.arange(len(idx)) / 10),
            "z_G": np.cos(np.arange(len(idx)) / 10),
            "z_R": np.sin(np.arange(len(idx)) / 7),
            "z_C": np.cos(np.arange(len(idx)) / 7),
        },
        index=idx,
    )
    ulsi = factors.mean(axis=1).rename("ulsi_base")
    lev = (factors["z_F"] * 0.2 + np.linspace(-1, 1, len(idx))).rename("lev_etf_fragility")

    config = integration.load_config()
    config["ic"]["min_observations"] = 20
    config["ic"]["rolling_min_periods"] = 20
    config["ic"]["rolling_window"] = 30
    config["target"]["horizons"] = [5]
    config["target"]["primary_horizon"] = 5

    monkeypatch.setattr(integration, "load_config", lambda path=None: config)
    monkeypatch.setattr(integration, "prepare_base_inputs", lambda start, end: (raw, factors, ulsi))
    monkeypatch.setattr(integration, "compute_lev_etf_fragility", lambda start, end: (lev, pd.DataFrame(), pd.DataFrame()))

    result = integration.run_diagnostic(
        start=date(2025, 1, 1),
        end=date(2025, 9, 1),
        output_dir=tmp_path,
    )

    report_path = result["report_path"]
    assert report_path.exists()
    assert "Leveraged ETF Fragility Integration Diagnostic" in report_path.read_text(encoding="utf-8")
