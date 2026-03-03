import pandas as pd
import pytest

from usd_liquidity_monitor.report import (
    _compute_tech_metrics,
    _normalize_smtp_host,
    _resolve_smtp_port,
    _resolve_timezone,
    generate_pdf_report,
    generate_daily_report,
    send_email_report,
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
            "F_t": [0.1, 0.2, 0.25, 0.3],
            "G_t": [0.0, 0.1, 0.1, 0.15],
            "R_t": [-0.1, -0.05, 0.0, 0.05],
            "C_t": [0.2, 0.2, 0.25, 0.3],
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
    assert "[ULSI Components]" in text
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


def test_generate_pdf_report_returns_pdf_bytes() -> None:
    impact = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            "ulsi": [0.5 + i * 0.002 + ((i % 6) * 0.0007) for i in range(80)],
            "raw_nasdaq_composite": [10000 + i * 4 + ((i % 9) * 1.1) for i in range(80)],
        }
    )
    impact["ulsi_change"] = impact["ulsi"].diff(5)
    impact["equity_return"] = impact["raw_nasdaq_composite"].pct_change(5)
    impact["forward_equity_return"] = impact["equity_return"].shift(-5)

    bundle = {
        "report_text": "USD Liquidity Daily Report\\n[ULSI Snapshot]\\n- Current ULSI: 0.9",
        "ulsi_df": pd.DataFrame(
            {
                "date": impact["date"],
                "ulsi": impact["ulsi"],
                "F_t": impact["ulsi"] * 0.5,
                "G_t": impact["ulsi"] * 0.2,
                "R_t": impact["ulsi"] * -0.1,
                "C_t": impact["ulsi"] * 0.4,
            }
        ),
        "analyses": [
            {
                "label": "NASDAQ Composite",
                "column": "raw_nasdaq_composite",
                "metrics": {"corr_60d": -0.2, "slope_1y": -0.01, "downside_hit_ratio": 0.55},
                "impact_frame": impact,
            }
        ],
        "table_df": pd.DataFrame(
            {
                "date": impact["date"],
                "raw_sofr": 5.1,
                "raw_effr": 5.05,
                "raw_iorb": 5.0,
                "raw_reserves": 3_500_000.0,
                "raw_tga": 700_000.0,
                "raw_fed_assets": 8_500_000.0,
                "raw_on_rrp": 250_000.0,
            }
        ),
        "quality_df": pd.DataFrame({"series_name": ["sofr"], "latest_date": [pd.Timestamp("2025-03-21").date()], "lag_days": [0], "missing_ratio": [0.0]}),
        "alerts": [],
        "statuses": {},
        "as_of": pd.Timestamp("2025-03-21").date(),
    }

    payload = generate_pdf_report(bundle)

    assert payload.startswith(b"%PDF")
    assert len(payload) > 1000


def test_send_email_report_attaches_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = {}

    class DummySMTP:
        def __init__(self, host, port, timeout=60):
            sent["host"] = host
            sent["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self):
            sent["starttls"] = True

        def login(self, user, password):
            sent["user"] = user
            sent["password"] = password

        def send_message(self, msg):
            sent["message"] = msg

    monkeypatch.setattr("usd_liquidity_monitor.report.smtplib.SMTP", DummySMTP)

    send_email_report(
        subject="test",
        body="hello",
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_user="user@gmail.com",
        smtp_password="app-pwd",
        to_email="to@gmail.com",
        attachments=[("report.pdf", b"%PDF-1.4\\n", "application/pdf")],
    )

    message = sent["message"]
    attachments = list(message.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "report.pdf"
