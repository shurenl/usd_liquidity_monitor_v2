import pandas as pd

from usd_liquidity_monitor.dashboard import format_alerts_for_display


def test_format_alerts_for_display_converts_nested_contributors_to_string() -> None:
    alerts = [
        {
            "date": pd.Timestamp("2026-02-10").date(),
            "level": "Tight",
            "trigger_rules": "ULSI > 1.5 for 3 consecutive days",
            "top_contributors": [("funding_price", 0.4567), ("credit", -0.1234)],
        }
    ]

    out = format_alerts_for_display(alerts)

    assert list(out.columns) == ["date", "level", "trigger_rules", "top_contributors"]
    assert out.loc[0, "top_contributors"] == "funding_price: +0.457; credit: -0.123"
    assert isinstance(out.loc[0, "top_contributors"], str)
