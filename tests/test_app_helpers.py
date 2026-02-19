import pandas as pd
import pytest

from usd_liquidity_monitor.app import (
    _build_rebased_index,
    _convert_million_to_billion,
    _prepare_tech_impact_frame,
    _safe_linear_slope,
    _to_long_series,
)


def test_convert_million_to_billion() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "Bank Reserves": [3_500_000.0, 3_400_000.0],
            "TGA Balance": [700_000.0, 710_000.0],
        }
    )

    out = _convert_million_to_billion(frame, ["Bank Reserves", "TGA Balance"])

    assert out.loc[0, "Bank Reserves"] == 3500.0
    assert out.loc[1, "TGA Balance"] == 710.0


def test_build_rebased_index_base_100() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "Bank Reserves": [3500.0, 3450.0, 3400.0],
            "TGA Balance": [700.0, 770.0, 735.0],
        }
    )

    out = _build_rebased_index(frame, ["Bank Reserves", "TGA Balance"], base=100.0)

    assert out.loc[0, "Bank Reserves (idx)"] == 100.0
    assert out.loc[1, "Bank Reserves (idx)"] < 100.0
    assert out.loc[1, "TGA Balance (idx)"] == pytest.approx(110.0)


def test_to_long_series_drops_null_rows() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "Bank Reserves": [3500.0, None, 3400.0],
            "TGA Balance": [None, 770.0, None],
        }
    )

    out = _to_long_series(frame, ["Bank Reserves", "TGA Balance"], value_name="bn_usd")

    assert list(out.columns) == ["date", "series", "bn_usd"]
    assert out.shape[0] == 3
    assert set(out["series"]) == {"Bank Reserves", "TGA Balance"}


def test_prepare_tech_impact_frame_outputs_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=10, freq="D"),
            "ulsi": [0.5, 0.6, 0.7, 0.8, 0.75, 0.9, 1.0, 0.95, 1.1, 1.2],
            "raw_nasdaq_composite": [100, 101, 102, 103, 101, 104, 105, 106, 104, 107],
        }
    )

    out = _prepare_tech_impact_frame(frame, equity_column="raw_nasdaq_composite", horizon_days=2)

    assert {"ulsi_change", "equity_return", "forward_equity_return"}.issubset(out.columns)
    assert out.shape[0] == 10


def test_safe_linear_slope_detects_negative_relationship() -> None:
    x = pd.Series([0, 1, 2, 3, 4, 5], dtype=float)
    y = pd.Series([0, -1, -2, -3, -4, -5], dtype=float)

    slope = _safe_linear_slope(x, y, min_points=3)

    assert slope is not None
    assert slope == pytest.approx(-1.0, abs=1e-9)
