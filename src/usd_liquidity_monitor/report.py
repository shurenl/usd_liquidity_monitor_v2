"""Daily report generation and email delivery for USD liquidity monitor."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from email.message import EmailMessage
import io
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


def _extract_tech_analyses(table_df: pd.DataFrame) -> list[dict[str, object]]:
    tech_candidates = {
        "NASDAQ Composite": "raw_nasdaq_composite",
        "NASDAQ-100": "raw_nasdaq100",
    }

    analyses: list[dict[str, object]] = []
    for label, col in tech_candidates.items():
        if col not in table_df.columns:
            continue
        base = table_df[["date", "ulsi", col]].copy()
        impact = _prepare_impact_frame(base, price_col=col, horizon_days=5)
        metrics = _compute_tech_metrics(base, price_col=col)
        analyses.append(
            {
                "label": label,
                "column": col,
                "metrics": metrics,
                "impact_frame": impact,
            }
        )
    return analyses


def _build_report_bundle(as_of: date, lookback_days: int = 365 * 3) -> dict[str, object]:
    start = as_of - timedelta(days=lookback_days)
    raw_df, statuses = fetch_all_series(start=start, end=as_of)
    features_df = compute_features(raw_df)
    ulsi_df = compute_ulsi(features_df)
    table_df = build_dashboard_table(raw_df, features_df, ulsi_df)
    analyses = _extract_tech_analyses(table_df)

    lines: list[str] = []
    lines.append(f"USD Liquidity Daily Report ({as_of.isoformat()})")
    lines.append("=" * 64)

    latest = ulsi_df.dropna(subset=["ulsi"]).tail(1)
    if latest.empty:
        lines.append("No valid ULSI value found in current lookback window.")
        report_text = "\n".join(lines)
        return {
            "report_text": report_text,
            "ulsi_df": ulsi_df,
            "table_df": table_df,
            "analyses": analyses,
            "statuses": statuses,
            "as_of": as_of,
        }

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
    if analyses:
        for item in analyses:
            label = str(item["label"])
            metrics = item["metrics"]
            corr_60d = metrics["corr_60d"]
            slope_1y = metrics["slope_1y"]
            downside_hit = metrics["downside_hit_ratio"]
            lines.append(f"- {label}:")
            lines.append(
                f"  - 60D corr(ULSI Δ, 5D return): {corr_60d:.3f}"
                if corr_60d is not None
                else "  - 60D corr(ULSI Δ, 5D return): NA"
            )
            lines.append(
                f"  - 1Y slope(return ~ ULSI Δ): {slope_1y:.5f}"
                if slope_1y is not None
                else "  - 1Y slope(return ~ ULSI Δ): NA"
            )
            lines.append(
                f"  - Downside hit ratio (ULSI Δ>0): {downside_hit:.1%}"
                if downside_hit is not None
                else "  - Downside hit ratio (ULSI Δ>0): NA"
            )
    else:
        lines.append("- No Nasdaq series available in current dataset.")

    lines.append("")
    lines.append("[Data Sync Summary]")
    failures = [name for name, status in statuses.items() if not status.success]
    lines.append(f"- Total series: {len(statuses)}")
    lines.append(f"- Failed series count: {len(failures)}")
    if failures:
        lines.append(f"- Failed series: {', '.join(sorted(failures))}")

    report_text = "\n".join(lines)
    return {
        "report_text": report_text,
        "ulsi_df": ulsi_df,
        "table_df": table_df,
        "analyses": analyses,
        "statuses": statuses,
        "as_of": as_of,
    }


def generate_daily_report(as_of: date, lookback_days: int = 365 * 3) -> str:
    return str(_build_report_bundle(as_of=as_of, lookback_days=lookback_days)["report_text"])


def _render_summary_page(pdf, bundle: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    report_text = str(bundle["report_text"])
    ulsi_df = pd.DataFrame(bundle["ulsi_df"]).copy()
    ulsi_df["date"] = pd.to_datetime(ulsi_df["date"], errors="coerce")
    ulsi_df = ulsi_df.dropna(subset=["date", "ulsi"]).sort_values("date")

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("USD Liquidity Daily Report", fontsize=16, fontweight="bold")

    text_ax = fig.add_axes([0.05, 0.55, 0.90, 0.35])
    text_ax.axis("off")
    text_lines = report_text.splitlines()
    text_ax.text(0.0, 1.0, "\n".join(text_lines[:28]), va="top", ha="left", fontsize=9, family="monospace")

    chart_ax = fig.add_axes([0.08, 0.10, 0.86, 0.35])
    if not ulsi_df.empty:
        last_year = ulsi_df.tail(252)
        chart_ax.plot(last_year["date"], last_year["ulsi"], color="#1f77b4", linewidth=1.8, label="ULSI")
        for threshold, color in [(0.5, "#7fbf7b"), (1.5, "#fdae61"), (2.5, "#d7191c")]:
            chart_ax.axhline(threshold, linestyle="--", linewidth=1.0, color=color)
        chart_ax.set_title("ULSI Trend (Last 252 observations)")
        chart_ax.set_xlabel("Date")
        chart_ax.set_ylabel("ULSI")
        chart_ax.legend(loc="best")
    else:
        chart_ax.text(0.5, 0.5, "No valid ULSI series", ha="center", va="center")
        chart_ax.set_axis_off()

    pdf.savefig(fig)
    plt.close(fig)


def _render_tech_page(pdf, analysis: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    label = str(analysis["label"])
    col = str(analysis["column"])
    metrics = dict(analysis["metrics"])
    impact = pd.DataFrame(analysis["impact_frame"]).copy()
    impact["date"] = pd.to_datetime(impact["date"], errors="coerce")
    impact = impact.dropna(subset=["date"]).sort_values("date")

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle(f"Tech Equity Impact: {label}", fontsize=15, fontweight="bold")

    # Top-left: rebased ULSI vs equity index
    ax = axes[0, 0]
    base_df = impact[["date", "ulsi", col]].dropna()
    if not base_df.empty:
        ulsi_base = float(base_df["ulsi"].iloc[0]) if float(base_df["ulsi"].iloc[0]) != 0 else np.nan
        eq_base = float(base_df[col].iloc[0]) if float(base_df[col].iloc[0]) != 0 else np.nan
        if np.isfinite(ulsi_base) and np.isfinite(eq_base):
            ax.plot(base_df["date"], base_df["ulsi"] / ulsi_base * 100.0, label="ULSI (rebased=100)", linewidth=1.5)
            ax.plot(base_df["date"], base_df[col] / eq_base * 100.0, label=f"{label} (rebased=100)", linewidth=1.5)
            ax.set_title("Normalized Trend")
            ax.set_ylabel("Index (base=100)")
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Insufficient base values", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "No overlap data", ha="center", va="center")
        ax.set_axis_off()

    # Top-right: shock scatter
    ax = axes[0, 1]
    scatter = impact[["ulsi_change", "equity_return"]].dropna()
    if not scatter.empty:
        ax.scatter(scatter["ulsi_change"], scatter["equity_return"] * 100.0, s=10, alpha=0.6)
        slope = _safe_linear_slope(scatter["ulsi_change"], scatter["equity_return"], min_points=20)
        if slope is not None:
            x = np.linspace(scatter["ulsi_change"].min(), scatter["ulsi_change"].max(), 100)
            intercept = float(scatter["equity_return"].mean()) - slope * float(scatter["ulsi_change"].mean())
            ax.plot(x, (slope * x + intercept) * 100.0, color="red", linewidth=1.2, linestyle="--")
        ax.set_title("Shock Map (5D return vs 5D ULSI change)")
        ax.set_xlabel("ULSI change (5D)")
        ax.set_ylabel(f"{label} return (5D, %)")
    else:
        ax.text(0.5, 0.5, "No valid scatter points", ha="center", va="center")
        ax.set_axis_off()

    # Bottom-left: rolling correlation
    ax = axes[1, 0]
    roll = impact[["date", "ulsi_change", "equity_return"]].dropna().sort_values("date")
    if roll.shape[0] >= 20:
        roll["rolling_corr_60d"] = roll["ulsi_change"].rolling(window=60, min_periods=20).corr(roll["equity_return"])
        roll = roll.dropna(subset=["rolling_corr_60d"])
        if not roll.empty:
            ax.plot(roll["date"], roll["rolling_corr_60d"], linewidth=1.6)
            ax.axhline(0, color="gray", linewidth=1.0, linestyle="--")
            ax.set_ylim(-1, 1)
            ax.set_title("Rolling Correlation (60D)")
            ax.set_ylabel("corr")
            ax.set_xlabel("Date")
        else:
            ax.text(0.5, 0.5, "No valid rolling correlation", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Insufficient points for rolling corr", ha="center", va="center")
        ax.set_axis_off()

    # Bottom-right: metrics text
    ax = axes[1, 1]
    ax.axis("off")
    corr_60d = metrics.get("corr_60d")
    slope_1y = metrics.get("slope_1y")
    downside = metrics.get("downside_hit_ratio")
    lines = [
        "Summary Metrics",
        "",
        f"60D corr(ULSI Δ, 5D return): {corr_60d:.3f}" if corr_60d is not None else "60D corr(ULSI Δ, 5D return): NA",
        f"1Y slope(return ~ ULSI Δ): {slope_1y:.5f}" if slope_1y is not None else "1Y slope(return ~ ULSI Δ): NA",
        f"Downside hit ratio (ULSI Δ>0): {downside:.1%}" if downside is not None else "Downside hit ratio (ULSI Δ>0): NA",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf_report(bundle: dict[str, object]) -> bytes:
    """Generate PDF bytes with visualized daily report."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    output = io.BytesIO()
    with PdfPages(output) as pdf:
        _render_summary_page(pdf, bundle)
        analyses = list(bundle.get("analyses", []))
        if analyses:
            for analysis in analyses:
                _render_tech_page(pdf, analysis)
    output.seek(0)
    return output.read()


def send_email_report(
    subject: str,
    body: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    from_email: str | None = None,
    attachments: list[tuple[str, bytes, str]] | None = None,
) -> None:
    sender = from_email or smtp_user

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)

    for filename, payload, mime_type in attachments or []:
        if "/" in mime_type:
            maintype, subtype = mime_type.split("/", 1)
        else:
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(payload, maintype=maintype, subtype=subtype, filename=filename)

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
    parser.add_argument("--save-pdf", type=str, default="", help="Optional local path to save generated PDF")
    args = parser.parse_args()

    bundle = _build_report_bundle(as_of=args.as_of, lookback_days=args.lookback_days)
    report_text = str(bundle["report_text"])
    report_pdf = generate_pdf_report(bundle)

    tz = _resolve_timezone(args.timezone or os.getenv("REPORT_TIMEZONE"), fallback="Asia/Shanghai")
    subject_date = datetime.now(tz).strftime("%Y-%m-%d")
    subject = f"[ULSI Daily] {subject_date}"
    pdf_filename = f"ulsi_daily_report_{args.as_of.isoformat()}.pdf"

    if args.save_pdf:
        with open(args.save_pdf, "wb") as fp:
            fp.write(report_pdf)

    if args.dry_run:
        print(subject)
        print()
        print(report_text)
        print()
        print(f"PDF generated ({len(report_pdf)} bytes)")
        if args.save_pdf:
            print(f"PDF saved to: {args.save_pdf}")
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
        attachments=[(pdf_filename, report_pdf, "application/pdf")],
    )
    print(f"Report sent to {to_email} with attachment {pdf_filename}")


if __name__ == "__main__":
    main()
