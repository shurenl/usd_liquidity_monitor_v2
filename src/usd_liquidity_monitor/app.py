"""Streamlit app entrypoint."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from usd_liquidity_monitor.dashboard import (
    build_alert_objects,
    build_dashboard_table,
    format_alerts_for_display,
    summarize_data_quality,
)
from usd_liquidity_monitor.data import fetch_all_series
from usd_liquidity_monitor.metrics import compute_features, compute_ulsi

WINDOW_OPTIONS: dict[str, int | None] = {
    "1M": 31,
    "3M": 93,
    "1Y": 366,
    "3Y": 1096,
    "ALL": None,
}

REGIME_LABELS: dict[str, str] = {
    "Normal": "Normal",
    "Watch": "Watch",
    "Tight": "Tight",
    "Stress": "Stress",
    "NA": "No Data",
}

COMPONENT_LABELS: dict[str, str] = {
    "funding_price": "Funding Price",
    "liquidity_quantity": "Liquidity Quantity",
    "credit": "Credit",
    "market_spillover": "Market Spillover",
}

ZSCORE_LABELS: dict[str, str] = {
    "z_spread_policy": "Policy Spread Z",
    "z_spread_repo": "Repo Spread Z",
    "z_pressure_reserve": "Reserve Pressure Z",
    "z_pressure_tga": "TGA Pressure Z",
    "z_pressure_on_rrp": "ON RRP Pressure Z",
    "z_pressure_fed_assets": "Fed Assets Pressure Z",
    "z_pressure_cp": "CP Funding Pressure Z",
    "z_pressure_credit": "Credit Spread Pressure Z",
    "z_pressure_curve_inversion": "Curve Inversion Pressure Z",
    "z_pressure_frontend_jump": "Front-End Yield Jump Pressure Z",
    "z_pressure_market": "Market Spillover Pressure Z",
}


@st.cache_data(ttl=6 * 60 * 60)
def _load_data(start: date, end: date) -> dict[str, object]:
    raw_df, statuses = fetch_all_series(start=start, end=end)
    features_df = compute_features(raw_df)
    ulsi_df = compute_ulsi(features_df)
    table_df = build_dashboard_table(raw_df, features_df, ulsi_df)
    quality_df = summarize_data_quality(raw_df, as_of=end)
    alerts = build_alert_objects(ulsi_df)
    return {
        "raw": raw_df,
        "features": features_df,
        "ulsi": ulsi_df,
        "table": table_df,
        "quality": quality_df,
        "alerts": alerts,
        "statuses": statuses,
    }


def _window_filter(df: pd.DataFrame, end: date, window: str) -> pd.DataFrame:
    if df.empty:
        return df
    days = WINDOW_OPTIONS[window]
    if days is None:
        return df
    start = pd.Timestamp(end) - pd.Timedelta(days=days)
    return df[df["date"] >= start]


def _build_overview_figure(ulsi_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ulsi_df["date"], y=ulsi_df["ulsi"], mode="lines", name="ULSI"))
    for threshold, color in [(0.5, "#7fbf7b"), (1.5, "#fdae61"), (2.5, "#d7191c")]:
        fig.add_hline(y=threshold, line_dash="dot", line_color=color)
    fig.update_layout(title="USD Liquidity Stress Index (ULSI)", xaxis_title="Date", yaxis_title="ULSI")
    return fig


def _safe_png_bytes(fig: go.Figure) -> bytes | None:
    try:
        return fig.to_image(format="png", width=1200, height=650, scale=2)
    except Exception:
        return None


def _regime_label(value: object) -> str:
    return REGIME_LABELS.get(str(value), str(value))


def _convert_million_to_billion(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns from million USD to billion USD."""

    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") / 1000.0
    return out


def _build_rebased_index(frame: pd.DataFrame, columns: list[str], base: float = 100.0) -> pd.DataFrame:
    """Create base-indexed series for each column using first non-null point."""

    out = pd.DataFrame({"date": frame["date"]})
    for col in columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            out[f"{col} (idx)"] = pd.NA
            continue
        first_value = series.loc[first_valid_idx]
        if pd.isna(first_value) or float(first_value) == 0.0:
            out[f"{col} (idx)"] = pd.NA
            continue
        out[f"{col} (idx)"] = series / float(first_value) * base
    return out


def _to_long_series(frame: pd.DataFrame, columns: list[str], value_name: str = "value") -> pd.DataFrame:
    """Convert wide series table into long format and drop null observations."""

    if "date" not in frame.columns:
        raise ValueError("frame must contain a date column")
    cols = [c for c in columns if c in frame.columns]
    if not cols:
        return pd.DataFrame(columns=["date", "series", value_name])
    long_df = frame[["date"] + cols].melt(id_vars="date", var_name="series", value_name=value_name)
    return long_df.dropna(subset=[value_name]).sort_values(["series", "date"]).reset_index(drop=True)


def _prepare_tech_impact_frame(frame: pd.DataFrame, equity_column: str, horizon_days: int = 5) -> pd.DataFrame:
    """Build impact analysis dataset between ULSI and one equity index."""

    required_cols = {"date", "ulsi", equity_column}
    if not required_cols.issubset(frame.columns):
        missing = sorted(required_cols - set(frame.columns))
        raise ValueError(f"Missing required columns: {missing}")

    out = frame[["date", "ulsi", equity_column]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ulsi"] = pd.to_numeric(out["ulsi"], errors="coerce")
    out[equity_column] = pd.to_numeric(out[equity_column], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out["ulsi_change"] = out["ulsi"].diff(horizon_days)
    out["equity_return"] = out[equity_column].pct_change(periods=horizon_days, fill_method=None)
    out["forward_equity_return"] = out[equity_column].pct_change(periods=horizon_days, fill_method=None).shift(-horizon_days)
    return out


def _safe_linear_slope(x: pd.Series, y: pd.Series, min_points: int = 20) -> float | None:
    """Return linear slope y = a*x + b when sufficient valid points exist."""

    pairs = pd.concat([x, y], axis=1).dropna()
    if pairs.shape[0] < min_points:
        return None
    x_arr = pairs.iloc[:, 0].to_numpy(dtype=float)
    y_arr = pairs.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(x_arr) == 0:
        return None
    slope, _ = np.polyfit(x_arr, y_arr, deg=1)
    return float(slope)


def main() -> None:
    st.set_page_config(page_title="USD Liquidity Stress Monitor", layout="wide")
    st.title("USD Liquidity Stress Monitor")
    st.caption("Daily public-data monitor for ULSI, component attribution, alerts, and data quality.")

    today = date.today()
    default_start = today - timedelta(days=365 * 3)

    with st.sidebar:
        st.header("Controls")
        start = st.date_input("Start date", value=default_start)
        end = st.date_input("End date", value=today)
        window = st.selectbox("Chart window", options=list(WINDOW_OPTIONS.keys()), index=2)
        if st.button("Refresh cache"):
            _load_data.clear()

    if start > end:
        st.error("Start date must be before end date.")
        return

    with st.spinner("Loading data and computing ULSI..."):
        result = _load_data(start=start, end=end)

    ulsi_df = result["ulsi"]
    table_df = result["table"]
    quality_df = result["quality"]
    alerts = result["alerts"]

    if ulsi_df.empty:
        st.warning("No ULSI data could be computed for the selected range.")
        return

    filtered_ulsi = _window_filter(ulsi_df, end=end, window=window)
    latest = filtered_ulsi.dropna(subset=["ulsi"]).tail(1)

    if latest.empty:
        st.warning("No non-null ULSI value is available in the current window.")
        return

    latest_row = latest.iloc[0]
    delta = float(latest_row["ulsi"] - filtered_ulsi["ulsi"].dropna().tail(2).head(1).squeeze()) if filtered_ulsi["ulsi"].dropna().shape[0] > 1 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Current ULSI", f"{latest_row['ulsi']:.2f}", f"{delta:+.2f}")
    c1.caption("Interpretation: higher ULSI implies stronger USD liquidity stress; delta is vs. the previous observation.")
    c2.metric("Regime", _regime_label(latest_row["regime"]))
    c2.caption("Thresholds: <0.5 Normal, 0.5-1.5 Watch, 1.5-2.5 Tight, >2.5 Stress.")
    c3.metric("Alert", "ON" if bool(latest_row["alert_flag"]) else "OFF")
    c3.caption("Rule: alert triggers when ULSI stays above 1.5 for 3 consecutive days.")

    tabs = st.tabs(["Overview", "Contributions", "Funding", "Liquidity", "Tech Equities Impact", "Alerts", "Data Quality"])

    with tabs[0]:
        fig = _build_overview_figure(filtered_ulsi)
        st.plotly_chart(fig, width="stretch")
        st.caption("Chart note: ULSI trend with regime thresholds to show current stress zone and trend direction.")
        png = _safe_png_bytes(fig)
        if png is not None:
            st.download_button("Download overview PNG", data=png, file_name="ulsi_overview.png", mime="image/png")

    with tabs[1]:
        contrib_cols = [col for col in filtered_ulsi.columns if col.startswith("contrib_")]
        if contrib_cols:
            contrib_latest = (
                filtered_ulsi[["date"] + contrib_cols]
                .dropna(subset=contrib_cols, how="all")
                .tail(1)
                .melt(id_vars="date", var_name="component", value_name="contribution")
            )
            if not contrib_latest.empty:
                contrib_latest["component"] = contrib_latest["component"].str.replace("contrib_", "", regex=False)
                contrib_latest["component"] = contrib_latest["component"].map(COMPONENT_LABELS).fillna(contrib_latest["component"])
                st.plotly_chart(
                    px.bar(contrib_latest, x="component", y="contribution", title="Latest component contributions"),
                    width="stretch",
                )
                st.caption("Chart note: larger positive bars indicate stronger upward pressure on ULSI; negative bars offset stress.")

        z_cols = [col for col in filtered_ulsi.columns if col.startswith("z_")]
        if z_cols:
            z_view = filtered_ulsi[["date"] + z_cols].tail(60).set_index("date").T
            z_view.index = [ZSCORE_LABELS.get(name, name) for name in z_view.index]
            st.plotly_chart(px.imshow(z_view, aspect="auto", title="Last 60 days Z-score heatmap"), width="stretch")
            st.caption("Chart note: warmer colors indicate larger positive deviations from historical norms and stronger stress signals.")

    with tabs[2]:
        funding_cols = ["raw_sofr", "raw_effr", "raw_iorb"]
        available = [c for c in funding_cols if c in table_df.columns]
        if available:
            funding_df = _window_filter(table_df[["date"] + available].dropna(how="all", subset=available), end=end, window=window)
            funding_plot_df = funding_df.rename(
                columns={
                    "raw_sofr": "SOFR",
                    "raw_effr": "EFFR",
                    "raw_iorb": "IORB",
                }
            )
            plot_cols = [c for c in ["SOFR", "EFFR", "IORB"] if c in funding_plot_df.columns]
            st.plotly_chart(px.line(funding_plot_df, x="date", y=plot_cols, title="SOFR / EFFR / IORB"), width="stretch")
            st.caption("Chart note: compares overnight funding rates to policy corridor anchors to track near-term funding tightness.")

            if all(col in funding_df.columns for col in ["raw_effr", "raw_iorb", "raw_sofr"]):
                spread_df = funding_df.copy()
                spread_df["EFFR-IORB spread"] = spread_df["raw_effr"] - spread_df["raw_iorb"]
                spread_df["SOFR-IORB spread"] = spread_df["raw_sofr"] - spread_df["raw_iorb"]
                st.plotly_chart(
                    px.line(spread_df, x="date", y=["EFFR-IORB spread", "SOFR-IORB spread"], title="Key funding spreads"),
                    width="stretch",
                )
                st.caption("Chart note: widening spreads usually indicate stronger demand for cash funding or reduced liquidity supply.")

    with tabs[3]:
        liquidity_cols = ["raw_reserves", "raw_tga", "raw_fed_assets", "raw_on_rrp"]
        available = [c for c in liquidity_cols if c in table_df.columns]
        if available:
            liq_df = _window_filter(table_df[["date"] + available].dropna(how="all", subset=available), end=end, window=window)
            liq_plot_df = liq_df.rename(
                columns={
                    "raw_reserves": "Bank Reserves",
                    "raw_tga": "TGA Balance",
                    "raw_fed_assets": "Fed Total Assets",
                    "raw_on_rrp": "ON RRP Usage",
                }
            )
            plot_cols = [c for c in ["Bank Reserves", "TGA Balance", "Fed Total Assets", "ON RRP Usage"] if c in liq_plot_df.columns]
            liq_levels_df = _convert_million_to_billion(liq_plot_df, plot_cols)
            liq_levels_long = _to_long_series(liq_levels_df, plot_cols, value_name="bn_usd")
            if not liq_levels_long.empty:
                fig = px.line(
                    liq_levels_long,
                    x="date",
                    y="bn_usd",
                    color="series",
                    title="Liquidity Quantity Dynamics (bn USD)",
                    markers=True,
                )
                fig.update_yaxes(title_text="bn USD")
                quarter_ends = pd.date_range(
                    liq_levels_long["date"].min(),
                    liq_levels_long["date"].max(),
                    freq="QE",
                )
                for qd in quarter_ends:
                    fig.add_vline(x=qd, line_dash="dot", line_color="gray", opacity=0.2)
                st.plotly_chart(fig, width="stretch")
                st.caption("Chart note: absolute levels in bn USD. This view compares raw system liquidity quantities.")

            liq_index_df = _build_rebased_index(liq_levels_df, plot_cols, base=100.0)
            index_cols = [col for col in liq_index_df.columns if col != "date"]
            if index_cols:
                liq_index_long = _to_long_series(liq_index_df, index_cols, value_name="index_value")
                liq_index_long["series"] = liq_index_long["series"].str.replace(" (idx)", "", regex=False)
                st.plotly_chart(
                    px.line(
                        liq_index_long,
                        x="date",
                        y="index_value",
                        color="series",
                        title="Liquidity Quantity Dynamics (Rebased Index, start=100)",
                        markers=True,
                    ),
                    width="stretch",
                )
                st.caption("Chart note: indexed view highlights relative changes even when absolute magnitudes differ.")

    with tabs[4]:
        tech_candidates = {
            "NASDAQ Composite": "raw_nasdaq_composite",
            "NASDAQ-100": "raw_nasdaq100",
        }
        available_tech = {name: col for name, col in tech_candidates.items() if col in table_df.columns}

        if not available_tech:
            st.info("No Nasdaq series is currently available. Please check sync status in the Data Quality tab.")
        else:
            selected_tech_name = st.selectbox("Tech equity benchmark", options=list(available_tech.keys()), index=0)
            selected_tech_col = available_tech[selected_tech_name]

            tech_input = table_df[["date", "ulsi", selected_tech_col]].copy()
            tech_impact_all = _prepare_tech_impact_frame(tech_input, equity_column=selected_tech_col, horizon_days=5)
            tech_impact_view = _window_filter(tech_impact_all, end=end, window=window)

            if tech_impact_view.empty:
                st.warning("No valid overlap between ULSI and selected tech index in this window.")
            else:
                corr_sample = tech_impact_view[["ulsi_change", "equity_return"]].dropna()
                corr_60d = corr_sample.tail(60).corr().iloc[0, 1] if corr_sample.shape[0] >= 5 else np.nan
                beta_1y = _safe_linear_slope(
                    corr_sample["ulsi_change"].tail(252),
                    corr_sample["equity_return"].tail(252),
                    min_points=20,
                )
                stress_subset = tech_impact_view[tech_impact_view["ulsi_change"] > 0]["forward_equity_return"].dropna()
                downside_hit_ratio = float((stress_subset < 0).mean()) if not stress_subset.empty else np.nan

                m1, m2, m3 = st.columns(3)
                m1.metric("60D corr (ULSI Δ vs 5D equity return)", f"{corr_60d:.2f}" if pd.notna(corr_60d) else "NA")
                m2.metric("1Y slope (return ~ ULSI Δ)", f"{beta_1y:.4f}" if beta_1y is not None else "NA")
                m3.metric("Downside hit ratio (ULSI Δ>0)", f"{downside_hit_ratio:.1%}" if pd.notna(downside_hit_ratio) else "NA")

                trend_base = tech_impact_view[["date", "ulsi", selected_tech_col]].rename(
                    columns={"ulsi": "ULSI", selected_tech_col: selected_tech_name}
                )
                trend_rebased = _build_rebased_index(trend_base, ["ULSI", selected_tech_name], base=100.0)
                trend_cols = [col for col in trend_rebased.columns if col != "date"]
                trend_long = _to_long_series(trend_rebased, trend_cols, value_name="index_value")
                trend_long["series"] = trend_long["series"].str.replace(" (idx)", "", regex=False)
                st.plotly_chart(
                    px.line(
                        trend_long,
                        x="date",
                        y="index_value",
                        color="series",
                        title=f"ULSI vs {selected_tech_name} (Rebased Index, start=100)",
                    ),
                    width="stretch",
                )
                st.caption("Chart note: compares normalized paths to show whether tighter USD liquidity aligns with weaker tech equity performance.")

                scatter_df = tech_impact_view[["date", "ulsi_change", "equity_return"]].dropna().copy()
                scatter_df["equity_return_pct"] = scatter_df["equity_return"] * 100
                st.plotly_chart(
                    px.scatter(
                        scatter_df,
                        x="ulsi_change",
                        y="equity_return_pct",
                        title=f"Shock Map: 5D {selected_tech_name} Return vs 5D ULSI Change",
                        labels={
                            "ulsi_change": "5D ULSI Change",
                            "equity_return_pct": f"5D {selected_tech_name} Return (%)",
                        },
                    ),
                    width="stretch",
                )
                st.caption("Chart note: each dot is one 5-day period. A more negative slope indicates stronger downside sensitivity of tech equities to liquidity tightening.")

                rolling_pairs = tech_impact_view[["date", "ulsi_change", "equity_return"]].dropna().sort_values("date")
                if rolling_pairs.shape[0] >= 20:
                    rolling_pairs["rolling_corr_60d"] = (
                        rolling_pairs["ulsi_change"].rolling(window=60, min_periods=20).corr(rolling_pairs["equity_return"])
                    )
                    rolling_plot = rolling_pairs.dropna(subset=["rolling_corr_60d"])
                    if not rolling_plot.empty:
                        fig_corr = px.line(
                            rolling_plot,
                            x="date",
                            y="rolling_corr_60d",
                            title=f"Rolling 60D Correlation: ULSI Change vs {selected_tech_name} Return",
                        )
                        fig_corr.update_yaxes(range=[-1, 1])
                        st.plotly_chart(fig_corr, width="stretch")
                        st.caption("Chart note: tracks how stable or regime-dependent the liquidity-to-tech-equity relationship is over time.")
                    else:
                        st.info("Rolling correlation has no valid points in this window (insufficient overlapping observations).")
                else:
                    st.info("Rolling correlation needs at least 20 overlapping observations in the selected window.")

    with tabs[5]:
        if not alerts:
            st.info("No new alert event was triggered in the selected period.")
        else:
            alert_df = format_alerts_for_display(alerts).rename(
                columns={
                    "date": "Date",
                    "level": "Level",
                    "trigger_rules": "Trigger Rule",
                    "top_contributors": "Top Contributors",
                }
            )
            st.dataframe(alert_df, width="stretch")
            st.caption("Table note: lists alert dates, severity levels, and the main contributing components for post-event review.")

    with tabs[6]:
        st.subheader("Data Quality")
        quality_view = quality_df.rename(
            columns={
                "series_name": "Series",
                "latest_date": "Latest Date",
                "lag_days": "Lag Days",
                "missing_ratio": "Missing Ratio",
            }
        )
        st.dataframe(quality_view, width="stretch")
        st.caption("Table note: freshness and completeness checks for each raw input series.")

        status_rows = []
        for name, status in result["statuses"].items():
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
        st.dataframe(pd.DataFrame(status_rows), width="stretch")
        st.caption("Table note: sync status for this run, useful for tracing upstream fetch failures.")

    st.download_button(
        "Download unified CSV",
        data=table_df.to_csv(index=False).encode("utf-8"),
        file_name=f"ulsi_unified_{end.isoformat()}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
