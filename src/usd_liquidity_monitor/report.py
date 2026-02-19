"""Daily report generation and email delivery for USD liquidity monitor."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from email.message import EmailMessage
import os
import socket
import smtplib
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import fetch_all_series
from .dashboard import build_dashboard_table
from .metrics import compute_features, compute_ulsi


def _safe_linear_slope(x: pd.Series, y: pd.Series, min_points: int = 20) -> float | None:
    pairs = pd.concat([x, y], axis=1).dropna()
    if pairs.shape[0] < min_points:
        return None
    x_arr = pairs.iloc[:, 0].to_numpy(dtype=float)
    y_arr = pairs.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(x_arr) == 0:
        return None
    slope, _ = np.polyfit(x_arr, y_arr, deg=1)
    return float(slope)


def _prepare_impact_frame(frame: pd.DataFrame, price_col: str, horizon_days: int = 5) -> pd.DataFrame:
    required = {"date", "ulsi", price_col}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"Missing required columns: {missing}")

    out = frame[["date", "ulsi", price_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ulsi"] = pd.to_numeric(out["ulsi"], errors="coerce")
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out["ulsi_change"] = out["ulsi"].diff(horizon_days)
    out["equity_return"] = out[price_col].pct_change(periods=horizon_days, fill_method=None)
    out["forward_equity_return"] = out[price_col].pct_change(periods=horizon_days, fill_method=None).shift(-horizon_days)
    return out


def _compute_tech_metrics(frame: pd.DataFrame, price_col: str) -> dict[str, float | None]:
    impact = _prepare_impact_frame(frame, price_col=price_col, horizon_days=5)
    pairs = impact[["ulsi_change", "equity_return"]].dropna()

    corr_60d = None
    if pairs.shape[0] >= 5:
        corr_60d = float(pairs.tail(60).corr().iloc[0, 1])

    slope_1y = _safe_linear_slope(pairs["ulsi_change"].tail(252), pairs["equity_return"].tail(252), min_points=20)

    stress_subset = impact[impact["ulsi_change"] > 0]["forward_equity_return"].dropna()
    downside_hit = None
    if not stress_subset.empty:
        downside_hit = float((stress_subset < 0).mean())

    return {
        "corr_60d": corr_60d,
        "slope_1y": slope_1y,
        "downside_hit_ratio": downside_hit,
    }


def generate_daily_report(as_of: date, lookback_days: int = 365 * 3) -> str:
    start = as_of - timedelta(days=lookback_days)
    raw_df, statuses = fetch_all_series(start=start, end=as_of)
    features_df = compute_features(raw_df)
    ulsi_df = compute_ulsi(features_df)
    table_df = build_dashboard_table(raw_df, features_df, ulsi_df)

    lines: list[str] = []
    lines.append(f"USD Liquidity Daily Report ({as_of.isoformat()})")
    lines.append("=" * 64)

    latest = ulsi_df.dropna(subset=["ulsi"]).tail(1)
    if latest.empty:
        lines.append("No valid ULSI value found in current lookback window.")
        return "\n".join(lines)

    latest_row = latest.iloc[0]
    ulsi_value = float(latest_row["ulsi"])
    regime = str(latest_row.get("regime", "NA"))
    alert = bool(latest_row.get("alert_flag", False))

    ulsi_hist = ulsi_df["ulsi"].dropna()
    delta_20d = float(ulsi_hist.iloc[-1] - ulsi_hist.iloc[-21]) if ulsi_hist.shape[0] >= 21 else np.nan

    lines.append("[ULSI Snapshot]")
    lines.append(f"- Current ULSI: {ulsi_value:.3f}")
    lines.append(f"- Regime: {regime}")
    lines.append(f"- Alert: {'ON' if alert else 'OFF'}")
    lines.append(f"- 20D change: {delta_20d:+.3f}" if not np.isnan(delta_20d) else "- 20D change: NA")

    lines.append("")
    lines.append("[Tech Equity Impact]")

    tech_candidates = {
        "NASDAQ Composite": "raw_nasdaq_composite",
        "NASDAQ-100": "raw_nasdaq100",
    }

    found_any = False
    for label, col in tech_candidates.items():
        if col not in table_df.columns:
            continue
        found_any = True
        metrics = _compute_tech_metrics(table_df[["date", "ulsi", col]].copy(), price_col=col)
        corr_60d = metrics["corr_60d"]
        slope_1y = metrics["slope_1y"]
        downside_hit = metrics["downside_hit_ratio"]

        lines.append(f"- {label}:")
        lines.append(f"  - 60D corr(ULSI Δ, 5D return): {corr_60d:.3f}" if corr_60d is not None else "  - 60D corr(ULSI Δ, 5D return): NA")
        lines.append(f"  - 1Y slope(return ~ ULSI Δ): {slope_1y:.5f}" if slope_1y is not None else "  - 1Y slope(return ~ ULSI Δ): NA")
        lines.append(
            f"  - Downside hit ratio (ULSI Δ>0): {downside_hit:.1%}"
            if downside_hit is not None
            else "  - Downside hit ratio (ULSI Δ>0): NA"
        )

    if not found_any:
        lines.append("- No Nasdaq series available in current dataset.")

    lines.append("")
    lines.append("[Data Sync Summary]")
    failures = [name for name, status in statuses.items() if not status.success]
    lines.append(f"- Total series: {len(statuses)}")
    lines.append(f"- Failed series count: {len(failures)}")
    if failures:
        lines.append(f"- Failed series: {', '.join(sorted(failures))}")

    return "\n".join(lines)


def send_email_report(subject: str, body: str, smtp_host: str, smtp_port: int, smtp_user: str, smtp_password: str, to_email: str, from_email: str | None = None) -> None:
    sender = from_email or smtp_user

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
    except socket.gaierror as exc:
        raise ValueError(
            "SMTP host cannot be resolved. Check SMTP_HOST secret format (use host only, e.g. smtp.gmail.com)."
        ) from exc


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def _resolve_smtp_port(raw_value: str | None, default: int = 587) -> int:
    """Parse SMTP port with safe fallback for invalid inputs."""

    candidate = (raw_value or "").strip()
    if not candidate:
        return default
    try:
        port = int(candidate)
    except ValueError:
        return default
    if 1 <= port <= 65535:
        return port
    return default


def _normalize_smtp_host(raw_value: str) -> str:
    """Normalize SMTP host from common misconfigurations."""

    candidate = (raw_value or "").strip().strip("'").strip('"')
    if not candidate:
        return ""

    if "://" in candidate:
        parsed = urlparse(candidate)
        candidate = parsed.hostname or ""
    else:
        candidate = candidate.split("/", 1)[0]
        if candidate.count(":") == 1:
            host_part, port_part = candidate.rsplit(":", 1)
            if port_part.isdigit():
                candidate = host_part

    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]

    return candidate.strip()


def _resolve_timezone(timezone_name: str | None, fallback: str = "Asia/Shanghai") -> ZoneInfo:
    """Resolve timezone safely, with fallback for empty/invalid input."""

    try:
        fallback_tz = ZoneInfo(fallback)
    except Exception:
        fallback_tz = ZoneInfo("UTC")

    candidate = (timezone_name or "").strip()
    if not candidate:
        return fallback_tz

    try:
        return ZoneInfo(candidate)
    except Exception:
        return fallback_tz


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and send daily ULSI report")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today(), help="Report date in YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=365 * 3)
    parser.add_argument("--timezone", type=str, default=None, help="IANA timezone, e.g. Asia/Shanghai")
    parser.add_argument("--dry-run", action="store_true", help="Print report only, do not send email")
    args = parser.parse_args()

    report_text = generate_daily_report(as_of=args.as_of, lookback_days=args.lookback_days)
    tz = _resolve_timezone(args.timezone or os.getenv("REPORT_TIMEZONE"), fallback="Asia/Shanghai")
    subject_date = datetime.now(tz).strftime("%Y-%m-%d")
    subject = f"[ULSI Daily] {subject_date}"

    if args.dry_run:
        print(subject)
        print()
        print(report_text)
        return

    smtp_host_raw = _required_env("SMTP_HOST")
    smtp_host = _normalize_smtp_host(smtp_host_raw)
    if not smtp_host:
        raise ValueError("Invalid SMTP_HOST format. Use host only, e.g. smtp.gmail.com")
    smtp_port = _resolve_smtp_port(os.getenv("SMTP_PORT"), default=587)
    smtp_user = _required_env("SMTP_USER")
    smtp_password = _required_env("SMTP_PASSWORD")
    to_email = _required_env("REPORT_TO")
    from_email = os.getenv("REPORT_FROM", "").strip() or None

    send_email_report(
        subject=subject,
        body=report_text,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        to_email=to_email,
        from_email=from_email,
    )
    print(f"Report sent to {to_email}")


if __name__ == "__main__":
    main()
