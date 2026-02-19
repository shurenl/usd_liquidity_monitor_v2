import pandas as pd

from usd_liquidity_monitor.report import (
    _compute_tech_metrics,
    _normalize_smtp_host,
    _resolve_smtp_port,
    _resolve_timezone,
    generate_daily_report,
)


def test_compute_tech_metrics_returns_values() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=300, freq="D"),
            "ulsi": [0.5 + i * 0.001 + ((i % 7) * 0.0003) for i in range(300)],
            "raw_nasdaq_composite": [10000 + i * 2 - (i % 5) + ((i % 11) * 0.7) for i in range(300)],
        }
    )

    metrics = _compute_tech_metrics(frame, price_col="raw_nasdaq_composite")

    assert set(metrics.keys()) == {"corr_60d", "slope_1y", "downside_hit_ratio"}


def test_generate_daily_report_contains_sections(monkeypatch) -> None:
    from usd_liquidity_monitor import report

    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]),
            "series_name": ["effr", "effr", "effr", "effr"],
            "value": [5.1, 5.1, 5.2, 5.2],
            "source": ["fred", "fred", "fred", "fred"],
        }
    )

    statuses = {
        "effr": type("S", (), {"success": True})(),
    }

    features_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]),
            "feature_name": ["spread_policy", "spread_policy", "spread_policy", "spread_policy"],
            "value": [0.1, 0.2, 0.2, 0.3],
        }
    )

    ulsi_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]),
            "ulsi": [0.2, 0.3, 0.4, 0.5],
            "regime": ["Normal", "Normal", "Normal", "Watch"],
            "alert_flag": [False, False, False, False],
        }
    )

    table_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]),
            "ulsi": [0.2, 0.3, 0.4, 0.5],
            "raw_nasdaq_composite": [10000, 10050, 10100, 10130],
        }
    )

    monkeypatch.setattr(report, "fetch_all_series", lambda start, end: (raw_df, statuses))
    monkeypatch.setattr(report, "compute_features", lambda raw: features_df)
    monkeypatch.setattr(report, "compute_ulsi", lambda feat: ulsi_df)
    monkeypatch.setattr(report, "build_dashboard_table", lambda raw, feat, ulsi: table_df)

    text = generate_daily_report(as_of=pd.Timestamp("2025-01-06").date(), lookback_days=10)

    assert "[ULSI Snapshot]" in text
    assert "[Tech Equity Impact]" in text
    assert "[Data Sync Summary]" in text


def test_resolve_timezone_falls_back_for_empty_input() -> None:
    tz = _resolve_timezone("")
    assert tz.key in {"Asia/Shanghai", "UTC"}


def test_resolve_timezone_falls_back_for_invalid_input() -> None:
    tz = _resolve_timezone("Not/A_Real_Zone")
    assert tz.key in {"Asia/Shanghai", "UTC"}


def test_resolve_smtp_port_fallback_for_invalid_input() -> None:
    assert _resolve_smtp_port("***", default=587) == 587
    assert _resolve_smtp_port("abc", default=587) == 587


def test_resolve_smtp_port_accepts_valid_port() -> None:
    assert _resolve_smtp_port("2525", default=587) == 2525


def test_normalize_smtp_host_strips_scheme_and_port() -> None:
    assert _normalize_smtp_host("https://smtp.gmail.com:587") == "smtp.gmail.com"
    assert _normalize_smtp_host("smtp.gmail.com:587") == "smtp.gmail.com"
    assert _normalize_smtp_host("smtp.gmail.com") == "smtp.gmail.com"
