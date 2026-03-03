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
from .dashboard import build_alert_objects, build_dashboard_table, format_alerts_for_display, summarize_data_quality
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


def _extract_component_snapshot(ulsi_df: pd.DataFrame) -> list[dict[str, object]]:
    component_map = {
        "F_t": "Funding Factor (F)",
        "G_t": "Fiscal Factor (G)",
        "R_t": "Reserves Factor (R)",
        "C_t": "Credit Factor (C)",
    }

    if ulsi_df.empty:
        return []

    items: list[dict[str, object]] = []
    for col, label in component_map.items():
        if col not in ulsi_df.columns:
            continue
        series = pd.to_numeric(ulsi_df[col], errors="coerce").dropna()
        latest = float(series.iloc[-1]) if not series.empty else None
        delta = float(series.iloc[-1] - series.iloc[-2]) if series.shape[0] > 1 else None
        items.append({"column": col, "label": label, "latest": latest, "delta": delta})
    return items


def _build_rebased_index(frame: pd.DataFrame, columns: list[str], base: float = 100.0) -> pd.DataFrame:
    out = pd.DataFrame({"date": frame["date"]})
    for col in columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            out[col] = np.nan
            continue
        first_value = series.loc[first_valid_idx]
        if pd.isna(first_value) or float(first_value) == 0.0:
            out[col] = np.nan
            continue
        out[col] = series / float(first_value) * base
    return out


def _build_report_bundle(as_of: date, lookback_days: int = 365 * 3) -> dict[str, object]:
    start = as_of - timedelta(days=lookback_days)
    raw_df, statuses = fetch_all_series(start=start, end=as_of)
    features_df = compute_features(raw_df)
    ulsi_df = compute_ulsi(features_df)
    table_df = build_dashboard_table(raw_df, features_df, ulsi_df)
    analyses = _extract_tech_analyses(table_df)
    component_snapshot = _extract_component_snapshot(ulsi_df)
    quality_df = summarize_data_quality(raw_df, as_of=as_of)
    alerts = build_alert_objects(ulsi_df)

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
            "components": component_snapshot,
            "quality_df": quality_df,
            "alerts": alerts,
            "statuses": statuses,
            "as_of": as_of,
        }

    latest_row = latest.iloc[0]
    ulsi_value = float(latest_row["ulsi"])
    regime = str(latest_row.get("regime", "NA"))
    alert = bool(latest_row.get("alert_flag", False))
    q70 = pd.to_numeric(pd.Series([latest_row.get("q70_252")]), errors="coerce").iloc[0]
    q85 = pd.to_numeric(pd.Series([latest_row.get("q85_252")]), errors="coerce").iloc[0]
    q95 = pd.to_numeric(pd.Series([latest_row.get("q95_252")]), errors="coerce").iloc[0]

    ulsi_hist = ulsi_df["ulsi"].dropna()
    delta_20d = float(ulsi_hist.iloc[-1] - ulsi_hist.iloc[-21]) if ulsi_hist.shape[0] >= 21 else np.nan

    lines.append("[ULSI Snapshot]")
    lines.append(f"- Current ULSI: {ulsi_value:.3f}")
    lines.append(f"- Regime: {regime}")
    lines.append(
        f"- Rolling 252D quantiles (q70/q85/q95): {q70:.3f} / {q85:.3f} / {q95:.3f}"
        if pd.notna(q70) and pd.notna(q85) and pd.notna(q95)
        else "- Rolling 252D quantiles (q70/q85/q95): NA"
    )
    lines.append(f"- Alert: {'ON' if alert else 'OFF'}")
    lines.append(f"- 20D change: {delta_20d:+.3f}" if not np.isnan(delta_20d) else "- 20D change: NA")

    lines.append("")
    lines.append("[ULSI Components]")
    if component_snapshot:
        for item in component_snapshot:
            latest_value = item["latest"]
            delta_value = item["delta"]
            latest_text = f"{latest_value:.3f}" if latest_value is not None else "NA"
            delta_text = f"{delta_value:+.3f}" if delta_value is not None else "NA"
            lines.append(f"- {item['label']}: current={latest_text}, delta={delta_text}")
    else:
        lines.append("- No valid component values available.")

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
        "components": component_snapshot,
        "quality_df": quality_df,
        "alerts": alerts,
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
        for col, label, color in [
            ("q70_252", "q70 (252D)", "#7fbf7b"),
            ("q85_252", "q85 (252D)", "#fdae61"),
            ("q95_252", "q95 (252D)", "#d7191c"),
        ]:
            if col in last_year.columns:
                vals = pd.to_numeric(last_year[col], errors="coerce")
                if vals.notna().any():
                    chart_ax.plot(last_year["date"], vals, linestyle="--", linewidth=1.0, color=color, label=label)
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


def _render_components_page(pdf, bundle: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    ulsi_df = pd.DataFrame(bundle["ulsi_df"]).copy()
    ulsi_df["date"] = pd.to_datetime(ulsi_df["date"], errors="coerce")
    ulsi_df = ulsi_df.dropna(subset=["date"]).sort_values("date")

    component_map = {
        "F_t": "Funding Factor (F)",
        "G_t": "Fiscal Factor (G)",
        "R_t": "Reserves Factor (R)",
        "C_t": "Credit Factor (C)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("ULSI Component Trends", fontsize=15, fontweight="bold")

    for ax, (col, label) in zip(axes.flatten(), component_map.items()):
        if col not in ulsi_df.columns:
            ax.text(0.5, 0.5, "Series unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue

        series_df = ulsi_df[["date", col]].dropna().tail(252)
        if series_df.empty:
            ax.text(0.5, 0.5, "No valid observations", ha="center", va="center")
            ax.set_axis_off()
            continue

        latest = float(series_df[col].iloc[-1])
        delta = float(series_df[col].iloc[-1] - series_df[col].iloc[-2]) if series_df.shape[0] > 1 else np.nan
        ax.plot(series_df["date"], series_df[col], linewidth=1.5)
        ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
        title = f"{label}\nLatest={latest:.3f}  Delta={delta:+.3f}" if np.isfinite(delta) else f"{label}\nLatest={latest:.3f}  Delta=NA"
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _render_contributions_page(pdf, bundle: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    ulsi_df = pd.DataFrame(bundle["ulsi_df"]).copy()
    ulsi_df["date"] = pd.to_datetime(ulsi_df["date"], errors="coerce")
    ulsi_df = ulsi_df.dropna(subset=["date"]).sort_values("date")

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), gridspec_kw={"height_ratios": [1, 1.4]})
    fig.suptitle("Contributions", fontsize=15, fontweight="bold")

    contrib_cols = [c for c in ulsi_df.columns if c.startswith("contrib_")]
    ax = axes[0]
    latest = ulsi_df[["date"] + contrib_cols].dropna(subset=contrib_cols, how="all").tail(1) if contrib_cols else pd.DataFrame()
    if not latest.empty:
        values = latest.iloc[0][contrib_cols].astype(float)
        labels = [c.replace("contrib_", "") for c in contrib_cols]
        ax.bar(labels, values.values)
        ax.axhline(0, color="gray", linewidth=1.0)
        ax.set_title("Latest Component Contributions")
        ax.set_ylabel("Contribution")
    else:
        ax.text(0.5, 0.5, "No contribution data", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    z_cols = [c for c in ["z_F", "z_G", "z_R", "z_C"] if c in ulsi_df.columns]
    if z_cols:
        heat_df = ulsi_df[["date"] + z_cols].tail(60).set_index("date").T.astype(float)
        im = ax.imshow(heat_df.values, aspect="auto", cmap="RdYlBu_r")
        ax.set_yticks(range(len(heat_df.index)))
        ax.set_yticklabels(heat_df.index)
        x_positions = np.linspace(0, max(len(heat_df.columns) - 1, 0), min(6, len(heat_df.columns))).astype(int)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([pd.Timestamp(heat_df.columns[i]).strftime("%Y-%m-%d") for i in x_positions], rotation=30, ha="right")
        ax.set_title("Last 60 Days Z-score Heatmap")
        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    else:
        ax.text(0.5, 0.5, "No z-score data", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _render_funding_page(pdf, bundle: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    table_df = pd.DataFrame(bundle["table_df"]).copy()
    table_df["date"] = pd.to_datetime(table_df["date"], errors="coerce")
    table_df = table_df.dropna(subset=["date"]).sort_values("date").tail(252)

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.suptitle("Funding", fontsize=15, fontweight="bold")

    rate_cols = {"raw_sofr": "SOFR", "raw_effr": "EFFR", "raw_iorb": "IORB"}
    ax = axes[0]
    plotted = False
    for col, label in rate_cols.items():
        if col in table_df.columns:
            series = pd.to_numeric(table_df[col], errors="coerce")
            if series.notna().any():
                ax.plot(table_df["date"], series, label=label, linewidth=1.5)
                plotted = True
    if plotted:
        ax.set_title("SOFR / EFFR / IORB")
        ax.set_ylabel("Rate (%)")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No funding rate data", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    if {"raw_sofr", "raw_effr", "raw_iorb"}.issubset(table_df.columns):
        spread_df = table_df.copy()
        spread_df["EFFR-IORB"] = pd.to_numeric(spread_df["raw_effr"], errors="coerce") - pd.to_numeric(spread_df["raw_iorb"], errors="coerce")
        spread_df["SOFR-IORB"] = pd.to_numeric(spread_df["raw_sofr"], errors="coerce") - pd.to_numeric(spread_df["raw_iorb"], errors="coerce")
        ax.plot(spread_df["date"], spread_df["EFFR-IORB"], label="EFFR-IORB", linewidth=1.5)
        ax.plot(spread_df["date"], spread_df["SOFR-IORB"], label="SOFR-IORB", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=1.0, linestyle="--")
        ax.set_title("Key Funding Spreads")
        ax.set_ylabel("Spread")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No spread data", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _render_liquidity_page(pdf, bundle: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    table_df = pd.DataFrame(bundle["table_df"]).copy()
    table_df["date"] = pd.to_datetime(table_df["date"], errors="coerce")
    table_df = table_df.dropna(subset=["date"]).sort_values("date").tail(252)

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.suptitle("Liquidity", fontsize=15, fontweight="bold")

    raw_map = {
        "raw_reserves": "Bank Reserves",
        "raw_tga": "TGA Balance",
        "raw_fed_assets": "Fed Total Assets",
        "raw_on_rrp": "ON RRP Usage",
    }

    ax = axes[0]
    plotted = False
    for col, label in raw_map.items():
        if col in table_df.columns:
            series = pd.to_numeric(table_df[col], errors="coerce") / 1000.0
            if series.notna().any():
                ax.plot(table_df["date"], series, label=label, linewidth=1.5)
                plotted = True
    if plotted:
        ax.set_title("Liquidity Quantity Dynamics (bn USD)")
        ax.set_ylabel("bn USD")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No liquidity series", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    available = [col for col in raw_map if col in table_df.columns]
    if available:
        idx_df = _build_rebased_index(table_df[["date"] + available], available, base=100.0)
        plotted = False
        for col in available:
            series = pd.to_numeric(idx_df[col], errors="coerce")
            if series.notna().any():
                ax.plot(idx_df["date"], series, label=raw_map[col], linewidth=1.5)
                plotted = True
        if plotted:
            ax.set_title("Liquidity Quantity Dynamics (Rebased Index, start=100)")
            ax.set_ylabel("Index")
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No rebased liquidity series", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "No liquidity series", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _render_alerts_page(pdf, bundle: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    ulsi_df = pd.DataFrame(bundle["ulsi_df"]).copy()
    ulsi_df["date"] = pd.to_datetime(ulsi_df["date"], errors="coerce")
    ulsi_df = ulsi_df.dropna(subset=["date"]).sort_values("date")
    alert_df = format_alerts_for_display(list(bundle.get("alerts", [])))

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), gridspec_kw={"height_ratios": [1.2, 1]})
    fig.suptitle("Alerts", fontsize=15, fontweight="bold")

    ax = axes[0]
    if not ulsi_df.empty and "alert_flag" in ulsi_df.columns:
        ax.plot(ulsi_df["date"], pd.to_numeric(ulsi_df["ulsi"], errors="coerce"), label="ULSI", linewidth=1.5)
        active = ulsi_df[ulsi_df["alert_flag"].eq(True)]
        if not active.empty:
            ax.scatter(active["date"], active["ulsi"], color="red", s=18, label="Alert ON")
        ax.set_title("ULSI with Alert States")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No alert timeline data", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    ax.axis("off")
    if alert_df.empty:
        ax.text(0.02, 0.98, "No new alert event was triggered in the selected period.", va="top", ha="left")
    else:
        view = alert_df.tail(8).copy()
        table = ax.table(cellText=view.values, colLabels=view.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _render_data_quality_page(pdf, bundle: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    quality_df = pd.DataFrame(bundle.get("quality_df", pd.DataFrame())).copy()
    status_rows = []
    for name, status in dict(bundle.get("statuses", {})).items():
        status_rows.append(
            {
                "Series": name,
                "Source": status.source,
                "Success": status.success,
                "Rows": status.row_count,
                "Latest Date": status.latest_date,
                "Message": status.message,
            }
        )
    status_df = pd.DataFrame(status_rows)

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.suptitle("Data Quality", fontsize=15, fontweight="bold")

    for ax, df, title in [
        (axes[0], quality_df.tail(12), "Freshness and Missingness"),
        (axes[1], status_df.tail(12), "Sync Status"),
    ]:
        ax.axis("off")
        if df.empty:
            ax.text(0.02, 0.98, f"No {title.lower()} data.", va="top", ha="left")
            continue
        table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.3)
        ax.set_title(title)


def generate_pdf_report(bundle: dict[str, object]) -> bytes:
    """Generate PDF bytes with visualized daily report."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    output = io.BytesIO()
    with PdfPages(output) as pdf:
        _render_summary_page(pdf, bundle)
        _render_components_page(pdf, bundle)
        _render_contributions_page(pdf, bundle)
        _render_funding_page(pdf, bundle)
        _render_liquidity_page(pdf, bundle)
        analyses = list(bundle.get("analyses", []))
        if analyses:
            for analysis in analyses:
                _render_tech_page(pdf, analysis)
        _render_alerts_page(pdf, bundle)
        _render_data_quality_page(pdf, bundle)
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
